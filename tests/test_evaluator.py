from __future__ import annotations

import importlib.util
import pathlib
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from korean_ner import DEFAULT_LABEL_LIST, SPACE_TOKEN, build_label_maps, score_strict_iob2


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EVALUATOR_PATH = PROJECT_ROOT / "evaluate" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("korean_ner_evaluator", EVALUATOR_PATH)
evaluator = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(evaluator)


class DummyTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0
    unk_token_id = 100
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    pad_token = "[PAD]"
    unk_token = "[UNK]"

    def __init__(self) -> None:
        self.vocab = {
            "[PAD]": 0,
            "[UNK]": 100,
            "[CLS]": 101,
            "[SEP]": 102,
            SPACE_TOKEN: 1,
            "A": 11,
            "B": 12,
        }
        self.inverse = {value: key for key, value in self.vocab.items()}

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [[self.vocab.get(char, self.unk_token_id)] for char in text]}
        return {"input_ids": [self.vocab.get(text, self.unk_token_id)]}

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, -1)

    def convert_ids_to_tokens(self, ids):
        return [self.inverse.get(int(token_id), "[UNK]") for token_id in ids]


class FakeModel(torch.nn.Module):
    def __init__(self, predictions: list[list[int]]) -> None:
        super().__init__()
        self.predictions = predictions

    def forward(self, input_ids, attention_mask):
        return self.predictions


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.maps = build_label_maps(DEFAULT_LABEL_LIST)
        self.tokenizer = DummyTokenizer()

    def test_build_eval_records_preserves_original_klue_spaces(self):
        labels = [
            self.maps.label_to_id["B-LC"],
            self.maps.label_to_id["I-LC"],
            self.maps.label_to_id["I-LC"],
        ]

        records = evaluator.build_eval_records(
            [{"tokens": ["A", " ", "B"], "ner_tags": labels}],
            self.tokenizer,
            dict(self.maps.label_to_id),
            dict(self.maps.id_to_label),
            max_length=5,
        )

        record = records[0]
        self.assertEqual(record["text"], "A B")
        self.assertEqual(record["char_indices"], [None, 0, 1, 2, None])
        self.assertEqual(record["true_char_labels"], ["B-LC", "I-LC", "I-LC"])
        self.assertEqual(record["input_ids"].tolist(), [101, 11, 1, 12, 102])

    def test_build_eval_records_raises_instead_of_truncating(self):
        labels = [
            self.maps.label_to_id["B-LC"],
            self.maps.label_to_id["I-LC"],
            self.maps.label_to_id["I-LC"],
        ]

        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            evaluator.build_eval_records(
                [{"tokens": ["A", " ", "B"], "ner_tags": labels}],
                self.tokenizer,
                dict(self.maps.label_to_id),
                dict(self.maps.id_to_label),
                max_length=4,
            )

    def test_normalized_rows_keeps_zero_support_rows_at_zero(self):
        matrix = np.array([[0, 0], [1, 1]], dtype=float)

        normalized = evaluator.normalized_rows(matrix)

        self.assertFalse(np.isnan(normalized).any())
        np.testing.assert_array_equal(normalized[0], np.array([0.0, 0.0]))
        np.testing.assert_allclose(normalized[1], np.array([0.5, 0.5]))

    def test_run_model_predictions_maps_tokens_back_to_character_labels(self):
        label_to_id = self.maps.label_to_id
        id_to_label = dict(self.maps.id_to_label)
        records = evaluator.build_eval_records(
            [
                {
                    "tokens": ["A", " ", "B"],
                    "ner_tags": [label_to_id["B-LC"], label_to_id["I-LC"], label_to_id["I-LC"]],
                }
            ],
            self.tokenizer,
            dict(label_to_id),
            id_to_label,
            max_length=5,
        )
        predictions = [[
            label_to_id["O"],
            label_to_id["B-LC"],
            label_to_id["I-LC"],
            label_to_id["I-LC"],
            label_to_id["O"],
        ]]
        loader = DataLoader(records, batch_size=1, collate_fn=evaluator.collate_eval_batch)

        true_labels, pred_labels, _tokens, texts, char_pairs, benchmark_true, benchmark_pred = evaluator.run_model_predictions(
            FakeModel(predictions),
            loader,
            id_to_label,
            device=torch.device("cpu"),
        )

        self.assertEqual(texts, ["A B"])
        self.assertEqual(true_labels, [["B-LC", "I-LC", "I-LC"]])
        self.assertEqual(pred_labels, [["B-LC", "I-LC", "I-LC"]])
        self.assertEqual(char_pairs, [("B-LC", "B-LC"), ("I-LC", "I-LC"), ("I-LC", "I-LC")])
        self.assertEqual(benchmark_true, ["O", "B-LC", "I-LC", "O"])
        self.assertEqual(benchmark_pred, ["O", "B-LC", "I-LC", "O"])

    def test_run_model_predictions_flattens_benchmark_labels_with_raw_sentinels_and_spaces_removed(self):
        label_to_id = self.maps.label_to_id
        id_to_label = dict(self.maps.id_to_label)
        records = evaluator.build_eval_records(
            [
                {
                    "tokens": ["A", " ", "B"],
                    "ner_tags": [label_to_id["B-LC"], label_to_id["I-LC"], label_to_id["I-LC"]],
                },
                {
                    "tokens": ["A"],
                    "ner_tags": [label_to_id["B-PS"]],
                },
            ],
            self.tokenizer,
            dict(label_to_id),
            id_to_label,
            max_length=5,
        )
        predictions = [
            [
                label_to_id["B-OG"],
                label_to_id["B-LC"],
                label_to_id["B-QT"],
                label_to_id["I-LC"],
                label_to_id["I-OG"],
            ],
            [
                label_to_id["O"],
                label_to_id["I-PS"],
                label_to_id["B-TI"],
                label_to_id["O"],
                label_to_id["O"],
            ],
        ]
        loader = DataLoader(records, batch_size=2, collate_fn=evaluator.collate_eval_batch)

        _true_labels, pred_labels, _tokens, _texts, _char_pairs, benchmark_true, benchmark_pred = evaluator.run_model_predictions(
            FakeModel(predictions),
            loader,
            id_to_label,
            device=torch.device("cpu"),
        )

        self.assertEqual(pred_labels[1], ["B-PS"])
        self.assertEqual(benchmark_true, ["O", "B-LC", "I-LC", "O", "O", "B-PS", "O"])
        self.assertEqual(benchmark_pred, ["B-OG", "B-LC", "I-LC", "I-OG", "O", "I-PS", "B-TI"])

    def test_metrics_payload_uses_entity_macro_f1_as_primary(self):
        scores = score_strict_iob2(
            [["B-PS", "I-PS", "O"], ["B-LC"], ["B-OG"]],
            [["B-PS", "I-PS", "O"], ["B-OG"], ["O"]],
        )

        payload = evaluator.metrics_payload(
            split="validation",
            checkpoint=pathlib.Path("weights/best_model.pt"),
            max_length=192,
            sample_count=3,
            scores=scores,
            klue_benchmark_entity_macro_f1=0.25,
        )

        self.assertEqual(payload["primary_metric"], "klue_official_entity_macro_f1")
        self.assertAlmostEqual(payload["overall"]["entity_macro_f1_strict_iob2"], 1 / 3)
        self.assertAlmostEqual(payload["overall"]["entity_micro_f1_strict_iob2"], 0.4)
        self.assertAlmostEqual(payload["overall"]["klue_official_entity_macro_f1"], 0.25)
        self.assertAlmostEqual(payload["overall"]["klue_official_entity_macro_f1_percent"], 25.0)
        self.assertIn("klue_official_entity_macro_f1", payload["metric_definitions"])
        self.assertEqual(payload["split"], "validation")
        self.assertEqual(payload["overall"]["support"], 3)


if __name__ == "__main__":
    unittest.main()
