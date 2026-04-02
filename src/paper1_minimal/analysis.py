from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n_eval = int(mask.sum())
    if n_eval == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "coverage": 0.0, "n_eval": 0}
    err = y_pred[mask] - y_true[mask]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "coverage": float(n_eval / mask.size),
        "n_eval": n_eval,
    }


def pairwise_instability(clean_metric: dict[str, float], pert_metric: dict[str, float]) -> float:
    baselines = sorted(set(clean_metric) & set(pert_metric))
    flips = 0
    total = 0
    for idx, left in enumerate(baselines):
        for right in baselines[idx + 1 :]:
            clean_delta = clean_metric[left] - clean_metric[right]
            pert_delta = pert_metric[left] - pert_metric[right]
            if abs(clean_delta) < 1e-12 or abs(pert_delta) < 1e-12:
                continue
            total += 1
            if np.sign(clean_delta) != np.sign(pert_delta):
                flips += 1
    return 0.0 if total == 0 else float(flips / total)


def pick_winner(score_by_baseline: dict[str, float], clean_error_by_baseline: dict[str, float]) -> str:
    ordered = sorted(
        score_by_baseline.items(),
        key=lambda item: (item[1], clean_error_by_baseline.get(item[0], item[1]), item[0]),
    )
    return ordered[0][0]


def summarize_reference_instability(eval_df: pd.DataFrame, split_name: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = eval_df[eval_df["split"] == split_name].copy()
    detail_rows: list[dict[str, object]] = []

    for (seed, horizon), group in subset.groupby(["seed", "horizon"], sort=True):
        clean_group = group[group["perturbation"] == "clean"]
        clean_metric = {row["baseline"]: float(row["mae"]) for _, row in clean_group.iterrows()}
        clean_winner = min(clean_metric, key=clean_metric.get)
        clean_winner_metric_clean = clean_metric[clean_winner]

        for perturbation, pert_group in group.groupby("perturbation", sort=True):
            pert_metric = {row["baseline"]: float(row["mae"]) for _, row in pert_group.iterrows()}
            pert_winner = min(pert_metric, key=pert_metric.get)
            selection_regret = float(pert_metric[clean_winner] - pert_metric[pert_winner])
            normalized_regret = (
                float(selection_regret / clean_winner_metric_clean)
                if clean_winner_metric_clean > 1e-12
                else float("nan")
            )
            detail_rows.append(
                {
                    "seed": int(seed),
                    "split": split_name,
                    "horizon": int(horizon),
                    "perturbation": str(perturbation),
                    "winner_clean": clean_winner,
                    "winner_current": pert_winner,
                    "winner_changed_vs_clean": bool(pert_winner != clean_winner),
                    "pairwise_instability": pairwise_instability(clean_metric, pert_metric),
                    "selection_regret_vs_clean": selection_regret,
                    "normalized_regret_vs_clean": normalized_regret,
                    "clean_winner_metric_clean": clean_winner_metric_clean,
                }
            )

    detail_df = pd.DataFrame(detail_rows).sort_values(["seed", "horizon", "perturbation"]).reset_index(drop=True)

    noise_rows: list[dict[str, object]] = []
    clean_only = subset[subset["perturbation"] == "clean"].copy()
    for horizon, group in clean_only.groupby("horizon", sort=True):
        std_values = group.groupby("baseline")["mae"].std().fillna(0.0).to_numpy(dtype=float)
        positive = std_values[std_values > 1e-12]
        seed_noise_scale = float(np.median(positive)) if positive.size else float(np.max(std_values, initial=0.0))
        noise_rows.append({"horizon": int(horizon), "seed_noise_scale": seed_noise_scale})
    noise_df = pd.DataFrame(noise_rows)
    detail_df = detail_df.merge(noise_df, on="horizon", how="left")
    detail_df["effect_ratio_vs_seed_noise"] = (
        detail_df["selection_regret_vs_clean"] / detail_df["seed_noise_scale"].clip(lower=1e-8)
    )

    clean_winners = detail_df[detail_df["perturbation"] == "clean"][["seed", "horizon", "winner_current"]].copy()
    clean_flip_rows: list[dict[str, object]] = []
    for horizon, group in clean_winners.groupby("horizon", sort=True):
        counts = group["winner_current"].value_counts()
        modal_winner = sorted(counts[counts == counts.max()].index.tolist())[0]
        clean_flip_rows.append(
            {
                "horizon": int(horizon),
                "modal_clean_winner": modal_winner,
                "clean_flip_baseline_rate": float((group["winner_current"] != modal_winner).mean()),
            }
        )
    clean_flip_df = pd.DataFrame(clean_flip_rows)
    detail_df = detail_df.merge(clean_flip_df, on="horizon", how="left")

    summary_rows: list[dict[str, object]] = []
    for perturbation, group in detail_df.groupby("perturbation", sort=True):
        selection_vals = group["selection_regret_vs_clean"].to_numpy(dtype=float)
        norm_vals = group["normalized_regret_vs_clean"].to_numpy(dtype=float)
        effect_vals = group["effect_ratio_vs_seed_noise"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "split": split_name,
                "perturbation": str(perturbation),
                "num_conditions": int(len(group)),
                "num_runs": int(group["seed"].nunique()),
                "winner_flip_rate": float(group["winner_changed_vs_clean"].mean()),
                "mean_pairwise_instability": float(group["pairwise_instability"].mean()),
                "median_selection_regret": float(np.nanmedian(selection_vals)),
                "p95_selection_regret": float(np.nanpercentile(selection_vals, 95)),
                "median_normalized_regret": float(np.nanmedian(norm_vals)),
                "p95_normalized_regret": float(np.nanpercentile(norm_vals, 95)),
                "median_effect_ratio_vs_seed_noise": float(np.nanmedian(effect_vals)),
                "clean_flip_baseline_rate": float(group["clean_flip_baseline_rate"].mean()),
                "excess_flip_over_clean": float(
                    group["winner_changed_vs_clean"].mean() - group["clean_flip_baseline_rate"].mean()
                ),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("perturbation").reset_index(drop=True)
    return detail_df, summary_df


@dataclass(frozen=True)
class RuleBenchmarkResult:
    condition_df: pd.DataFrame
    test_detail_df: pd.DataFrame
    selected_condition_df: pd.DataFrame
    selected_detail_df: pd.DataFrame
    rule_summary_df: pd.DataFrame


def average_rank_scores(frame: pd.DataFrame, baselines: list[str]) -> dict[str, float]:
    rank_frame = frame.rank(axis=1, method="average", ascending=True)
    return {baseline: float(rank_frame[baseline].mean()) for baseline in baselines}


def benchmark_selection_rules(
    eval_df: pd.DataFrame,
    plausible_perturbations: list[str],
    lambdas: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0),
) -> RuleBenchmarkResult:
    subset = eval_df[eval_df["perturbation"].isin(["clean", *plausible_perturbations])].copy()
    condition_rows: list[dict[str, object]] = []
    test_detail_rows: list[dict[str, object]] = []
    lambda_records: list[dict[str, object]] = []

    for (seed, horizon), group in subset.groupby(["seed", "horizon"], sort=True):
        val_df = group[group["split"] == "val"]
        test_df = group[group["split"] == "test"]
        if val_df.empty or test_df.empty:
            continue

        val_pivot = val_df.pivot_table(index="perturbation", columns="baseline", values="mae", aggfunc="first")
        test_pivot = test_df.pivot_table(index="perturbation", columns="baseline", values="mae", aggfunc="first")
        if "clean" not in val_pivot.index or "clean" not in test_pivot.index:
            continue

        common_plausible = [
            name for name in plausible_perturbations if name in val_pivot.index and name in test_pivot.index
        ]
        if not common_plausible:
            continue

        baselines = sorted(set(val_pivot.columns) & set(test_pivot.columns))
        usable_baselines = [
            baseline
            for baseline in baselines
            if not pd.isna(val_pivot.at["clean", baseline]) and not pd.isna(test_pivot.at["clean", baseline])
        ]
        if len(usable_baselines) < 2:
            continue

        val_frame = val_pivot.loc[common_plausible, usable_baselines]
        test_frame = test_pivot.loc[common_plausible, usable_baselines]
        if val_frame.isna().any().any() or test_frame.isna().any().any():
            continue

        val_clean_errors = {baseline: float(val_pivot.at["clean", baseline]) for baseline in usable_baselines}
        test_clean_errors = {baseline: float(test_pivot.at["clean", baseline]) for baseline in usable_baselines}
        best_val = val_frame.min(axis=1, skipna=True)
        best_test = test_frame.min(axis=1, skipna=True)
        standard_winner = pick_winner(val_clean_errors, val_clean_errors)

        rule_to_winner: dict[str, str] = {
            "clean": standard_winner,
            "mean_error_over_p": pick_winner(
                {baseline: float(val_frame[baseline].mean()) for baseline in usable_baselines},
                val_clean_errors,
            ),
            "average_rank_over_p": pick_winner(average_rank_scores(val_frame, usable_baselines), val_clean_errors),
            "worst_case_over_p": pick_winner(
                {baseline: float(val_frame[baseline].max()) for baseline in usable_baselines},
                val_clean_errors,
            ),
        }
        penalty_by_baseline = {
            baseline: float(np.mean((val_frame[baseline] - best_val).to_numpy(dtype=float)))
            for baseline in usable_baselines
        }
        for lambda_value in lambdas:
            scores = {
                baseline: val_clean_errors[baseline] + lambda_value * penalty_by_baseline[baseline]
                for baseline in usable_baselines
            }
            rule_to_winner[f"robust_lambda_{lambda_value:g}"] = pick_winner(scores, val_clean_errors)

        condition_id = f"seed={seed}|h={horizon}"
        for rule_name, winner in rule_to_winner.items():
            val_regrets = (val_frame[winner] - best_val).to_numpy(dtype=float)
            test_regrets = (test_frame[winner] - best_test).to_numpy(dtype=float)
            selected_lambda = (
                float(rule_name.replace("robust_lambda_", ""))
                if rule_name.startswith("robust_lambda_")
                else np.nan
            )
            condition_rows.append(
                {
                    "seed": int(seed),
                    "horizon": int(horizon),
                    "condition_id": condition_id,
                    "rule": rule_name,
                    "selected_winner": winner,
                    "standard_winner": standard_winner,
                    "winner_changed_from_standard": bool(winner != standard_winner),
                    "selected_lambda": selected_lambda,
                    "median_val_decision_regret": float(np.median(val_regrets)),
                    "median_test_decision_regret": float(np.median(test_regrets)),
                    "median_val_clean_tradeoff": float(
                        val_clean_errors[winner] - val_clean_errors[standard_winner]
                    ),
                    "clean_tradeoff_test": float(
                        test_clean_errors[winner] - test_clean_errors[standard_winner]
                    ),
                }
            )
            for perturbation in common_plausible:
                test_detail_rows.append(
                    {
                        "seed": int(seed),
                        "horizon": int(horizon),
                        "condition_id": condition_id,
                        "rule": rule_name,
                        "perturbation": perturbation,
                        "selected_winner": winner,
                        "standard_winner": standard_winner,
                        "decision_regret": float(test_frame.at[perturbation, winner] - best_test[perturbation]),
                    }
                )
            if not np.isnan(selected_lambda):
                lambda_records.append(
                    {
                        "lambda": selected_lambda,
                        "condition_id": condition_id,
                        "median_val_decision_regret": float(np.median(val_regrets)),
                    }
                )

    condition_df = pd.DataFrame(condition_rows)
    test_detail_df = pd.DataFrame(test_detail_rows)
    lambda_df = pd.DataFrame(lambda_records)
    if condition_df.empty or test_detail_df.empty or lambda_df.empty:
        raise ValueError("No valid rule-benchmark records were generated.")

    lambda_choice = (
        lambda_df.groupby("lambda", as_index=False)["median_val_decision_regret"].median()
        .sort_values(["median_val_decision_regret", "lambda"], ascending=[True, True])
        .iloc[0]
    )
    chosen_lambda = float(lambda_choice["lambda"])
    chosen_rule_name = f"robust_lambda_{chosen_lambda:g}"

    selected_rule_map = {
        "clean": "clean",
        "mean_error_over_p": "mean_error_over_p",
        "average_rank_over_p": "average_rank_over_p",
        "worst_case_over_p": "worst_case_over_p",
        "robust_selected": chosen_rule_name,
    }

    selected_condition_parts: list[pd.DataFrame] = []
    selected_detail_parts: list[pd.DataFrame] = []
    for rule_label, source_rule in selected_rule_map.items():
        condition_part = condition_df[condition_df["rule"] == source_rule].copy()
        detail_part = test_detail_df[test_detail_df["rule"] == source_rule].copy()
        condition_part["rule"] = rule_label
        detail_part["rule"] = rule_label
        selected_condition_parts.append(condition_part)
        selected_detail_parts.append(detail_part)

    selected_condition_df = pd.concat(selected_condition_parts, axis=0, ignore_index=True)
    selected_detail_df = pd.concat(selected_detail_parts, axis=0, ignore_index=True)

    standard_detail_df = selected_detail_df[selected_detail_df["rule"] == "clean"].rename(
        columns={"decision_regret": "standard_decision_regret"}
    )
    rule_summary_rows: list[dict[str, object]] = []
    for rule_name, detail_group in selected_detail_df.groupby("rule", sort=True):
        merged = detail_group.merge(
            standard_detail_df[["condition_id", "perturbation", "standard_decision_regret"]],
            on=["condition_id", "perturbation"],
            how="inner",
        )
        delta = merged["decision_regret"].to_numpy(dtype=float) - merged[
            "standard_decision_regret"
        ].to_numpy(dtype=float)
        condition_group = selected_condition_df[selected_condition_df["rule"] == rule_name]
        modal_rule_winner = (
            condition_group["selected_winner"].mode().iloc[0]
            if not condition_group["selected_winner"].mode().empty
            else None
        )
        rule_summary_rows.append(
            {
                "rule": rule_name,
                "num_conditions": int(condition_group["condition_id"].nunique()),
                "num_test_rows": int(len(merged)),
                "selected_lambda": float(condition_group["selected_lambda"].iloc[0])
                if "selected_lambda" in condition_group.columns
                and condition_group["selected_lambda"].notna().any()
                else np.nan,
                "modal_standard_winner": str(condition_group["standard_winner"].mode().iloc[0]),
                "modal_rule_winner": modal_rule_winner,
                "median_test_standard_decision_regret": float(np.median(merged["standard_decision_regret"])),
                "median_test_candidate_rule_decision_regret": float(np.median(merged["decision_regret"])),
                "median_test_decision_regret_delta": float(np.median(delta)),
                "candidate_better_rate_test": float((delta < -1e-12).mean()),
                "candidate_equal_rate_test": float((np.abs(delta) <= 1e-12).mean()),
                "candidate_worse_rate_test": float((delta > 1e-12).mean()),
                "winner_change_rate_test": float(condition_group["winner_changed_from_standard"].mean()),
                "median_clean_tradeoff_test": float(np.median(condition_group["clean_tradeoff_test"])),
            }
        )
    rule_summary_df = pd.DataFrame(rule_summary_rows).sort_values("rule").reset_index(drop=True)

    return RuleBenchmarkResult(
        condition_df=condition_df,
        test_detail_df=test_detail_df,
        selected_condition_df=selected_condition_df,
        selected_detail_df=selected_detail_df,
        rule_summary_df=rule_summary_df,
    )
