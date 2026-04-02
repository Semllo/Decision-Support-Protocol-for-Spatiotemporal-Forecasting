from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import benchmark_selection_rules, metric_summary, summarize_reference_instability
from .baselines import BASELINE_FUNCTIONS
from .synthetic import (
    SyntheticConfig,
    build_graph,
    build_split_labels,
    make_timestamps,
    materialize_references,
    row_normalize,
    simulate_clean_signal,
)


@dataclass(frozen=True)
class ExperimentConfig:
    num_seeds: int = 5
    input_length: int = 24
    output_length: int = 12
    horizons: tuple[int, ...] = (1, 3, 6, 12)
    plausible_perturbations: tuple[str, ...] = ("node_shift_1hop",)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)


def build_window_start_indices(
    split_labels: np.ndarray, split_name: str, input_length: int, output_length: int
) -> list[int]:
    starts: list[int] = []
    max_start = len(split_labels) - input_length - output_length + 1
    for start in range(max_start):
        target_start = start + input_length
        target_end = target_start + output_length
        target_labels = split_labels[target_start:target_end]
        if not np.all(target_labels == split_name):
            continue
        if split_name == "train" and not np.all(split_labels[start:target_end] == "train"):
            continue
        starts.append(start)
    return starts


def build_target_timestamps(
    timestamps: pd.DatetimeIndex, start_indices: list[int], input_length: int, output_length: int
) -> np.ndarray:
    rows = []
    for start in start_indices:
        target_slice = slice(start + input_length, start + input_length + output_length)
        rows.append(timestamps[target_slice].to_numpy(dtype="datetime64[ns]"))
    return np.stack(rows, axis=0) if rows else np.empty((0, output_length), dtype="datetime64[ns]")


def build_prediction_bundles(
    clean: np.ndarray,
    split_labels: np.ndarray,
    timestamps: pd.DatetimeIndex,
    input_length: int,
    output_length: int,
    shift_map: np.ndarray,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    bundles: dict[str, dict[str, np.ndarray]] = {}
    split_to_indices = {
        split_name: build_window_start_indices(split_labels, split_name, input_length, output_length)
        for split_name in ("val", "test")
    }

    for baseline_name, predict_fn in BASELINE_FUNCTIONS.items():
        bundle: dict[str, np.ndarray] = {}
        for split_name, start_indices in split_to_indices.items():
            preds = []
            targets = []
            for start in start_indices:
                context = clean[start : start + input_length]
                target = clean[start + input_length : start + input_length + output_length]
                preds.append(predict_fn(context, output_length, shift_map))
                targets.append(target.astype(np.float32))

            pred_array = (
                np.stack(preds, axis=0).astype(np.float32)
                if preds
                else np.empty((0, output_length, clean.shape[1]), dtype=np.float32)
            )
            target_array = (
                np.stack(targets, axis=0).astype(np.float32)
                if targets
                else np.empty((0, output_length, clean.shape[1]), dtype=np.float32)
            )

            bundle[f"pred_{split_name}"] = pred_array
            bundle[f"target_clean_{split_name}"] = target_array
            bundle[f"timestamps_{split_name}"] = build_target_timestamps(
                timestamps, start_indices, input_length, output_length
            )
            bundle[f"sample_seed_{split_name}"] = np.full(pred_array.shape[0], seed, dtype=np.int32)
        bundles[baseline_name] = bundle
    return bundles


def fetch_reference_windows(
    reference: np.ndarray, split_labels: np.ndarray, split_name: str, input_length: int, output_length: int
) -> np.ndarray:
    start_indices = build_window_start_indices(split_labels, split_name, input_length, output_length)
    rows = [
        reference[start + input_length : start + input_length + output_length].astype(np.float32)
        for start in start_indices
    ]
    if not rows:
        return np.empty((0, output_length, reference.shape[1]), dtype=np.float32)
    return np.stack(rows, axis=0)


def evaluate_condition_rows(
    prediction_bundles: dict[str, dict[str, np.ndarray]],
    references: dict[str, np.ndarray],
    split_labels: np.ndarray,
    input_length: int,
    output_length: int,
    horizons: tuple[int, ...],
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split_name in ("val", "test"):
        reference_windows = {
            perturbation: fetch_reference_windows(reference, split_labels, split_name, input_length, output_length)
            for perturbation, reference in references.items()
        }
        for perturbation, target_windows in reference_windows.items():
            for baseline_name, bundle in prediction_bundles.items():
                pred = bundle[f"pred_{split_name}"]
                for horizon in horizons:
                    h_idx = horizon - 1
                    metrics = metric_summary(target_windows[:, h_idx, :], pred[:, h_idx, :])
                    rows.append(
                        {
                            "seed": int(seed),
                            "split": split_name,
                            "perturbation": perturbation,
                            "baseline": baseline_name,
                            "horizon": int(horizon),
                            **metrics,
                        }
                    )
    return rows


def run_experiment(config: ExperimentConfig, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_cfg = config.synthetic
    positions, adjacency, shift_map = build_graph(synthetic_cfg.num_nodes, synthetic_cfg.graph_seed)
    adj_norm = row_normalize(adjacency)
    split_labels = build_split_labels(synthetic_cfg.num_steps, synthetic_cfg.train_ratio, synthetic_cfg.val_ratio)
    timestamps = make_timestamps(synthetic_cfg.num_steps)

    np.savez_compressed(
        output_dir / "graph_artifacts.npz",
        positions=positions,
        adjacency=adjacency,
        shift_map=shift_map,
        timestamps=timestamps.to_numpy(dtype="datetime64[ns]"),
        split_labels=np.array(split_labels, dtype="U8"),
    )

    node_metadata = pd.DataFrame(
        {
            "node_id": [f"node_{idx:02d}" for idx in range(synthetic_cfg.num_nodes)],
            "x": positions[:, 0],
            "y": positions[:, 1],
            "shift_target": [f"node_{int(idx):02d}" for idx in shift_map],
        }
    )
    node_metadata.to_csv(output_dir / "node_metadata.csv", index=False, encoding="utf-8")

    eval_rows: list[dict[str, object]] = []
    for seed in range(config.num_seeds):
        clean = simulate_clean_signal(synthetic_cfg.num_steps, adj_norm, seed=10_000 + seed)
        references = materialize_references(clean, shift_map, seed=20_000 + seed)
        np.savez_compressed(output_dir / f"dataset_seed_{seed:02d}.npz", **references)

        bundles = build_prediction_bundles(
            clean=clean,
            split_labels=split_labels,
            timestamps=timestamps,
            input_length=config.input_length,
            output_length=config.output_length,
            shift_map=shift_map,
            seed=seed,
        )
        for baseline_name, bundle in bundles.items():
            np.savez_compressed(
                output_dir / f"prediction_bundle_{baseline_name}_seed_{seed:02d}.npz", **bundle
            )

        eval_rows.extend(
            evaluate_condition_rows(
                prediction_bundles=bundles,
                references=references,
                split_labels=split_labels,
                input_length=config.input_length,
                output_length=config.output_length,
                horizons=config.horizons,
                seed=seed,
            )
        )

    eval_df = pd.DataFrame(eval_rows).sort_values(
        ["seed", "split", "perturbation", "baseline", "horizon"]
    ).reset_index(drop=True)
    eval_df.to_csv(output_dir / "evaluation_by_condition.csv", index=False, encoding="utf-8")

    detail_df, reference_summary_df = summarize_reference_instability(eval_df, split_name="test")
    detail_df.to_csv(output_dir / "reference_instability_detail.csv", index=False, encoding="utf-8")
    reference_summary_df.to_csv(output_dir / "reference_instability_summary.csv", index=False, encoding="utf-8")

    benchmark = benchmark_selection_rules(
        eval_df=eval_df,
        plausible_perturbations=list(config.plausible_perturbations),
    )
    benchmark.condition_df.to_csv(output_dir / "rule_condition_summary.csv", index=False, encoding="utf-8")
    benchmark.test_detail_df.to_csv(output_dir / "rule_test_detail.csv", index=False, encoding="utf-8")
    benchmark.selected_condition_df.to_csv(
        output_dir / "rule_selected_condition_summary.csv", index=False, encoding="utf-8"
    )
    benchmark.selected_detail_df.to_csv(
        output_dir / "rule_selected_test_detail.csv", index=False, encoding="utf-8"
    )
    benchmark.rule_summary_df.to_csv(output_dir / "rule_benchmark_summary.csv", index=False, encoding="utf-8")

    report = {
        "config": {
            "num_seeds": config.num_seeds,
            "input_length": config.input_length,
            "output_length": config.output_length,
            "horizons": list(config.horizons),
            "plausible_perturbations": list(config.plausible_perturbations),
            "synthetic": asdict(synthetic_cfg),
        },
        "top_reference_rows": reference_summary_df.to_dict(orient="records"),
        "top_rule_rows": benchmark.rule_summary_df.to_dict(orient="records"),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
