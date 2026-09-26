from __future__ import annotations

import json
import io
import unittest
from unittest.mock import patch

import torch

from korean_ner import (
    DEFAULT_LABEL_LIST,
    IGNORE_INDEX,
    SPACE_TOKEN,
    align_text,
    build_label_maps,
    extract_bio_spans,
    extract_model_state_dict,
    load_checkpoint,
    repair_bio_sequence,
    score_strict_iob2,
    token_predictions_to_char_labels,
)
from korean_ner.decode import token_predictions_to_raw_char_labels
from korean_ner.metrics import score_klue_benchmark_entity_macro_f1


class DummyTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    pad_token_id = 0
    unk_token_id = 100
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    pad_token = "[PAD]"
    unk_token = "[UNK]"

    def __init__(self, *, space_is_unknown: bool = False) -> None:
        self.calls = []
        self.vocab = {
            "[PAD]": 0,
            "[UNK]": 100,
            "[CLS]": 101,
            "[SEP]": 102,
            SPACE_TOKEN: 100 if space_is_unknown else 1,
            "한": 11,
            "국": 12,
            "서": 13,
            "울": 14,
            ".": 15,
            "A": 16,
            "##A": 17,
        }
        self.inverse = {value: key for key, value in self.vocab.items()}

    def __call__(self, text, add_special_tokens=False):
        self.calls.append(text)
        if isinstance(text, list):
            return {"input_ids": [self._encode_one(item) for item in text]}
        return {"input_ids": self._encode_one(text)}

    def _encode_one(self, text):
        if text.isspace():
            return []
        if text == "쌍":
            return [16, 17]
        return [self.vocab.get(text, self.unk_token_id)]

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, -1)

    def convert_ids_to_tokens(self, ids):
        return [self.inverse.get(int(token_id), "[UNK]") for token_id in ids]


class CoreTests(unittest.TestCase):
    def test_alignment_preserves_korean_space_with_unused0_id_and_index_map(self):
        tokenizer = DummyTokenizer()
        maps = build_label_maps(DEFAULT_LABEL_LIST)
        labels = [
            maps.label_to_id["B-LC"],
            maps.label_to_id["I-LC"],
            maps.label_to_id["I-LC"],
        ]

        encoding = align_text(
            "한 국",
            tokenizer,
            labels,
            max_length=8,
            label_to_id=maps.label_to_id,
            id_to_label=maps.id_to_label,
        )

        self.assertEqual(encoding.input_ids[:5], [101, 11, 1, 12, 102])
        self.assertEqual(encoding.tokens[:5], ["[CLS]", "한", SPACE_TOKEN, "국", "[SEP]"])
        self.assertEqual(encoding.char_indices[:5], [None, 0, 1, 2, None])
        self.assertEqual(encoding.labels[:5], [IGNORE_INDEX, labels[0], labels[1], labels[2], IGNORE_INDEX])
        self.assertEqual(tokenizer.calls, [["한", "국"]])

    def test_alignment_preserves_tab_and_newline_indices_as_unused0(self):
        tokenizer = DummyTokenizer()

        encoding = align_text("한\t\n국", tokenizer, max_length=8)

        self.assertEqual(encoding.input_ids[:6], [101, 11, 1, 1, 12, 102])
        self.assertEqual(encoding.tokens[:6], ["[CLS]", "한", SPACE_TOKEN, SPACE_TOKEN, "국", "[SEP]"])
        self.assertEqual(encoding.char_indices[:6], [None, 0, 1, 2, 3, None])
        self.assertEqual(tokenizer.calls, [["한", "국"]])

    def test_alignment_rejects_tokenizer_without_distinct_unused0_id(self):
        tokenizer = DummyTokenizer(space_is_unknown=True)

        with self.assertRaisesRegex(ValueError, "unknown-token id"):
            align_text("한 국", tokenizer, max_length=8)

    def test_alignment_raises_on_overflow_instead_of_truncating(self):
        tokenizer = DummyTokenizer()

        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            align_text("한국서울", tokenizer, max_length=5)

    def test_alignment_batches_non_space_chars_and_handles_multipiece_char(self):
        tokenizer = DummyTokenizer()
        maps = build_label_maps(DEFAULT_LABEL_LIST)
        label_id = maps.label_to_id["B-OG"]

        encoding = align_text(
            "쌍 한",
            tokenizer,
            [label_id, maps.label_to_id["O"], maps.label_to_id["O"]],
            max_length=8,
            label_to_id=maps.label_to_id,
            id_to_label=maps.id_to_label,
        )

        self.assertEqual(tokenizer.calls, [["쌍", "한"]])
        self.assertEqual(encoding.tokens[:6], ["[CLS]", "A", "##A", SPACE_TOKEN, "한", "[SEP]"])
        self.assertEqual(encoding.char_indices[:6], [None, 0, 0, 1, 2, None])
        self.assertEqual(
            encoding.labels[:6],
            [
                IGNORE_INDEX,
                maps.label_to_id["B-OG"],
                maps.label_to_id["I-OG"],
                maps.label_to_id["O"],
                maps.label_to_id["O"],
                IGNORE_INDEX,
            ],
        )

    def test_bio_repair_and_span_extraction_cover_invalid_i_transitions(self):
        labels = ["I-PS", "I-PS", "I-OG", "O", "B-LC", "I-LC"]

        self.assertEqual(repair_bio_sequence(labels), ["B-PS", "I-PS", "B-OG", "O", "B-LC", "I-LC"])
        spans = extract_bio_spans("abcdef", labels)
        self.assertEqual(
            [(span.start, span.end, span.label, span.text) for span in spans],
            [
                (0, 2, "PS", "ab"),
                (2, 3, "OG", "c"),
                (4, 6, "LC", "ef"),
            ],
        )

    def test_span_extraction_keeps_internal_whitespace(self):
        spans = extract_bio_spans("한 국", ["B-LC", "I-LC", "I-LC"])

        self.assertEqual([(span.start, span.end, span.label, span.text) for span in spans], [(0, 3, "LC", "한 국")])

    def test_token_predictions_map_final_korean_char_to_original_index_two(self):
        tokenizer = DummyTokenizer()
        maps = build_label_maps(DEFAULT_LABEL_LIST)
        encoding = align_text("한 국", tokenizer, max_length=8)
        predictions = [
            maps.label_to_id["O"],
            maps.label_to_id["B-LC"],
            maps.label_to_id["I-LC"],
            maps.label_to_id["I-LC"],
            maps.label_to_id["O"],
            maps.label_to_id["O"],
            maps.label_to_id["O"],
            maps.label_to_id["O"],
        ]

        char_labels = token_predictions_to_char_labels(
            predictions,
            encoding.char_indices,
            maps.id_to_label,
            text_length=3,
        )

        self.assertEqual(char_labels, ["B-LC", "I-LC", "I-LC"])
        self.assertEqual(extract_bio_spans("한 국", char_labels)[0].text, "한 국")

    def test_raw_token_predictions_preserve_invalid_bio_for_benchmark(self):
        tokenizer = DummyTokenizer()
        maps = build_label_maps(DEFAULT_LABEL_LIST)
        encoding = align_text("A", tokenizer, max_length=4)
        predictions = [
            maps.label_to_id["O"],
            maps.label_to_id["I-LC"],
            maps.label_to_id["O"],
            maps.label_to_id["O"],
        ]

        raw_labels = token_predictions_to_raw_char_labels(
            predictions,
            encoding.char_indices,
            maps.id_to_label,
            text_length=1,
        )
        repaired_labels = token_predictions_to_char_labels(
            predictions,
            encoding.char_indices,
            maps.id_to_label,
            text_length=1,
        )

        self.assertEqual(raw_labels, ["I-LC"])
        self.assertEqual(repaired_labels, ["B-LC"])

    def test_strict_iob2_metric_micro_and_macro_known_example(self):
        y_true = [["B-PS", "I-PS", "O"], ["B-LC", "O"], ["B-OG"]]
        y_pred = [["B-PS", "I-PS", "O"], ["B-OG", "O"], ["O"]]

        scores = score_strict_iob2(y_true, y_pred)

        self.assertAlmostEqual(scores.f1_micro, 0.4)
        self.assertAlmostEqual(scores.f1_macro, 1 / 3)
        self.assertEqual(scores.report["micro avg"]["support"], 3)
        self.assertIsInstance(scores.report["micro avg"]["support"], int)
        json.dumps(scores.report)
        self.assertIs(type(scores.f1_micro), float)
        self.assertIs(type(scores.precision_micro), float)
        self.assertIs(type(scores.recall_micro), float)
        buffer = io.BytesIO()
        torch.save(scores.__dict__, buffer)
        buffer.seek(0)
        self.assertEqual(torch.load(buffer, weights_only=True)["f1_micro"], scores.f1_micro)

    def test_klue_benchmark_entity_macro_f1_flattens_one_sequence(self):
        score = score_klue_benchmark_entity_macro_f1(
            ["O", "B-PS", "O", "B-LC", "O"],
            ["O", "B-PS", "B-LC", "I-LC", "O"],
        )

        self.assertAlmostEqual(score, 0.5)
        self.assertIs(type(score), float)
        json.dumps({"klue_official_entity_macro_f1": score})

    def test_extract_model_state_dict_accepts_wrapped_and_raw_checkpoints(self):
        raw = {"classifier.weight": object()}

        self.assertIs(extract_model_state_dict(raw), raw)
        self.assertIs(extract_model_state_dict({"model_state_dict": raw}), raw)

        with self.assertRaises(ValueError):
            extract_model_state_dict(["not", "a", "state"])

    def test_load_checkpoint_always_uses_weights_only(self):
        with patch("korean_ner.checkpoint.torch.load", return_value={"ok": True}) as torch_load:
            self.assertEqual(load_checkpoint("weights.pt"), {"ok": True})

        torch_load.assert_called_once()
        self.assertTrue(torch_load.call_args.kwargs["weights_only"])
        self.assertEqual(torch_load.call_args.kwargs["map_location"], "cpu")


if __name__ == "__main__":
    unittest.main()
