from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

from evaluate.error_analysis import analyze_errors


DECODE_PATH = Path(__file__).resolve().parents[1] / "korean_ner" / "decode.py"
DECODE_SPEC = importlib.util.spec_from_file_location("korean_ner_decode_for_error_analysis_tests", DECODE_PATH)
decode = importlib.util.module_from_spec(DECODE_SPEC)
assert DECODE_SPEC.loader is not None
sys.modules[DECODE_SPEC.name] = decode
DECODE_SPEC.loader.exec_module(decode)
EntitySpan = decode.EntitySpan
token_predictions_to_char_labels = decode.token_predictions_to_char_labels


def ent(start: int, end: int, label: str, text: str) -> dict:
    return {"start": start, "end": end, "type": label, "text": text[start:end]}


def error_types(errors):
    return [error["type"] for error in errors]


class ErrorAnalysisTests(unittest.TestCase):
    def test_exact_boundary_wrong_label_is_type_confusion_only(self):
        text = "abcdef"
        true_entities = [ent(1, 4, "PS", text)]
        pred_entities = [ent(1, 4, "OG", text)]

        errors = analyze_errors(true_entities, pred_entities, text)

        self.assertEqual(error_types(errors), ["type_confusion"])
        self.assertEqual(errors[0]["true_entity"], true_entities[0])
        self.assertEqual(errors[0]["pred_entity"], pred_entities[0])

    def test_contained_prediction_counts_as_single_boundary_error(self):
        text = "abcdef"
        true_entities = [EntitySpan(1, 5, "OG", text[1:5])]
        pred_entities = [EntitySpan(2, 4, "OG", text[2:4])]

        errors = analyze_errors(true_entities, pred_entities, text)

        self.assertEqual(error_types(errors), ["boundary_error"])
        self.assertEqual(errors[0]["true_entity"], ent(1, 5, "OG", text))
        self.assertEqual(errors[0]["pred_entity"], ent(2, 4, "OG", text))

    def test_overlapping_prediction_is_paired_once_then_remaining_true_is_missed(self):
        text = "abcdef"
        true_entities = [ent(0, 2, "PS", text), ent(2, 4, "LC", text)]
        pred_entities = [ent(1, 3, "PS", text)]

        errors = analyze_errors(true_entities, pred_entities, text)

        self.assertEqual(error_types(errors), ["boundary_error", "missed_entity"])
        self.assertEqual(errors[0]["pred_entity"], pred_entities[0])
        self.assertEqual(errors[1]["true_entity"], true_entities[1])

    def test_non_overlapping_prediction_is_spurious_and_true_is_missed(self):
        text = "abcdef"
        true_entities = [ent(0, 2, "PS", text)]
        pred_entities = [ent(4, 6, "PS", text)]

        errors = analyze_errors(true_entities, pred_entities, text)

        self.assertEqual(error_types(errors), ["missed_entity", "spurious_entity"])

    def test_first_subword_o_is_not_overwritten_by_later_subword_entity(self):
        id_to_label = {0: "O", 1: "B-PS", 2: "I-PS"}

        labels = token_predictions_to_char_labels(
            predictions=[0, 1, 2],
            char_indices=[0, 0, 1],
            id_to_label=id_to_label,
            text_length=2,
        )

        self.assertEqual(labels, ["O", "B-PS"])
