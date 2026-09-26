from __future__ import annotations

import unittest

import torch

from evaluate.calibration import collect_emission_diagnostics, compute_calibration_metrics, summarize_rows


class CalibrationTests(unittest.TestCase):
    def test_emission_diagnostic_uses_emission_argmax_for_confidence_and_correctness(self):
        id_to_label = {0: "O", 1: "B-PS", 2: "I-PS"}
        emissions = torch.tensor(
            [
                [0.0, 6.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        )

        rows = collect_emission_diagnostics(
            emissions,
            char_indices=[0, 1],
            gold_char_labels=[1, 2],
            id_to_label=id_to_label,
        )

        self.assertEqual([row["pred_label"] for row in rows], ["B-PS", "O"])
        self.assertEqual([row["correct"] for row in rows], [1, 0])
        self.assertGreater(rows[0]["confidence"], 0.99)

    def test_emission_diagnostic_keeps_spaces_and_skips_duplicate_subtokens(self):
        id_to_label = {0: "O", 1: "B-LC", 2: "I-LC"}
        emissions = torch.tensor(
            [
                [7.0, 0.0, 0.0],
                [0.0, 7.0, 0.0],
                [0.0, 0.0, 7.0],
                [0.0, 7.0, 0.0],
            ]
        )

        rows = collect_emission_diagnostics(
            emissions,
            char_indices=[None, 0, 0, 1],
            gold_char_labels=[1, 1],
            id_to_label=id_to_label,
        )

        self.assertEqual([row["token_index"] for row in rows], [1, 3])
        self.assertEqual([row["char_index"] for row in rows], [0, 1])
        self.assertEqual([row["correct"] for row in rows], [1, 1])

    def test_summary_separates_o_heavy_overall_from_entity_only_counts(self):
        rows = [
            {"confidence": 0.9, "correct": 1, "gold_group": "O", "gold_entity_type": None, "pred_label": "O"},
            {"confidence": 0.8, "correct": 1, "gold_group": "O", "gold_entity_type": None, "pred_label": "O"},
            {
                "confidence": 0.7,
                "correct": 0,
                "gold_group": "entity",
                "gold_entity_type": "PS",
                "pred_label": "O",
            },
        ]

        summary = summarize_rows(rows, num_bins=2)

        self.assertEqual(summary["diagnostic"], "emission_classifier_only")
        self.assertIn("do not calibrate final CRF", summary["note"])
        self.assertEqual(summary["overall"]["total_tokens"], 3)
        self.assertEqual(summary["overall"]["gold_o_tokens"], 2)
        self.assertEqual(summary["overall"]["gold_entity_tokens"], 1)
        self.assertEqual(summary["entity_only"]["total_tokens"], 1)
        self.assertEqual(summary["per_entity"]["PS"]["count"], 1)

    def test_zero_confidence_bin_is_included(self):
        metrics = compute_calibration_metrics([0.0, 1.0], [1, 1], num_bins=2)

        self.assertEqual(metrics["bin_stats"][0]["count"], 1)
        self.assertEqual(metrics["bin_stats"][1]["count"], 1)


if __name__ == "__main__":
    unittest.main()
