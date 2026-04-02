from __future__ import annotations

import unittest

import pandas as pd

from paper1_minimal.analysis import benchmark_selection_rules, summarize_reference_instability


class CoreAnalysisTests(unittest.TestCase):
    def test_reference_instability_detects_flip(self) -> None:
        rows = [
            {"seed": 0, "split": "test", "perturbation": "clean", "baseline": "a", "horizon": 1, "mae": 1.0},
            {"seed": 0, "split": "test", "perturbation": "clean", "baseline": "b", "horizon": 1, "mae": 1.4},
            {"seed": 0, "split": "test", "perturbation": "node_shift_1hop", "baseline": "a", "horizon": 1, "mae": 2.0},
            {"seed": 0, "split": "test", "perturbation": "node_shift_1hop", "baseline": "b", "horizon": 1, "mae": 1.1},
            {"seed": 1, "split": "test", "perturbation": "clean", "baseline": "a", "horizon": 1, "mae": 1.1},
            {"seed": 1, "split": "test", "perturbation": "clean", "baseline": "b", "horizon": 1, "mae": 1.5},
            {"seed": 1, "split": "test", "perturbation": "node_shift_1hop", "baseline": "a", "horizon": 1, "mae": 2.1},
            {"seed": 1, "split": "test", "perturbation": "node_shift_1hop", "baseline": "b", "horizon": 1, "mae": 1.0},
        ]
        detail_df, summary_df = summarize_reference_instability(pd.DataFrame(rows), split_name="test")
        self.assertFalse(detail_df.empty)
        shifted = summary_df.loc[summary_df["perturbation"] == "node_shift_1hop"].iloc[0]
        self.assertGreater(shifted["winner_flip_rate"], 0.0)
        self.assertGreater(shifted["median_selection_regret"], 0.0)

    def test_rule_benchmark_can_choose_robust_lambda(self) -> None:
        rows = []
        for seed in (0, 1):
            rows.extend(
                [
                    {"seed": seed, "split": "val", "perturbation": "clean", "baseline": "clean_model", "horizon": 1, "mae": 1.0},
                    {"seed": seed, "split": "val", "perturbation": "clean", "baseline": "robust_model", "horizon": 1, "mae": 1.2},
                    {"seed": seed, "split": "val", "perturbation": "node_shift_1hop", "baseline": "clean_model", "horizon": 1, "mae": 2.5},
                    {"seed": seed, "split": "val", "perturbation": "node_shift_1hop", "baseline": "robust_model", "horizon": 1, "mae": 1.1},
                    {"seed": seed, "split": "test", "perturbation": "clean", "baseline": "clean_model", "horizon": 1, "mae": 1.0},
                    {"seed": seed, "split": "test", "perturbation": "clean", "baseline": "robust_model", "horizon": 1, "mae": 1.2},
                    {"seed": seed, "split": "test", "perturbation": "node_shift_1hop", "baseline": "clean_model", "horizon": 1, "mae": 2.8},
                    {"seed": seed, "split": "test", "perturbation": "node_shift_1hop", "baseline": "robust_model", "horizon": 1, "mae": 1.0},
                ]
            )
        result = benchmark_selection_rules(pd.DataFrame(rows), plausible_perturbations=["node_shift_1hop"])
        robust_selected = result.rule_summary_df.loc[result.rule_summary_df["rule"] == "robust_selected"].iloc[0]
        self.assertGreaterEqual(robust_selected["selected_lambda"], 1.0)
        self.assertLess(robust_selected["median_test_decision_regret_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
