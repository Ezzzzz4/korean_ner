from __future__ import annotations

import unittest

from evaluate.attention_viz import parse_args as parse_attention_args
from evaluate.attention_viz import example_sentences, resolve_entity_span, token_indices_for_char_span, valid_token_count
from evaluate.benchmark import parse_args as parse_benchmark_args
from evaluate.benchmark import percentile, test_texts as benchmark_texts


class VisualizationTests(unittest.TestCase):
    def test_attention_cli_defaults_to_cpu(self):
        args = parse_attention_args([])

        self.assertEqual(args.device, "cpu")

    def test_benchmark_cli_defaults_to_cpu_and_declares_run_shape(self):
        args = parse_benchmark_args(["--warmup", "2", "--runs", "3"])

        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.warmup, 2)
        self.assertEqual(args.runs, 3)

    def test_token_indices_for_char_span_preserves_space_alignment(self):
        char_indices = [None, 0, 1, 2, None, None]

        self.assertEqual(token_indices_for_char_span(char_indices, 0, 3), [1, 2, 3])
        self.assertEqual(token_indices_for_char_span(char_indices, 1, 2), [2])

    def test_resolve_entity_span_repairs_unique_handwritten_offset(self):
        text = "abc def abc"
        entity = {"text": "def", "type": "OG", "start": 0, "end": 3}

        self.assertEqual(resolve_entity_span(text, entity), {"text": "def", "type": "OG", "start": 4, "end": 7})

    def test_resolve_entity_span_rejects_ambiguous_text_annotation(self):
        text = "abc def abc"
        entity = {"text": "abc", "type": "OG", "start": 4, "end": 7}

        self.assertIsNone(resolve_entity_span(text, entity))

    def test_valid_token_count_stops_at_first_padding_token(self):
        self.assertEqual(valid_token_count(["[CLS]", "a", "[PAD]", "[PAD]"]), 2)
        self.assertEqual(valid_token_count(["[CLS]", "a"]), 2)

    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([10.0, 20.0, 30.0], 50), 20.0)
        self.assertEqual(percentile([10.0, 20.0], 95), 19.5)

    def test_examples_are_korean_and_annotation_offsets_are_valid(self):
        texts = benchmark_texts()
        examples = example_sentences()
        self.assertTrue(all(any(0xAC00 <= ord(char) <= 0xD7A3 for char in text) for text in texts))
        self.assertTrue(all(any(0xAC00 <= ord(char) <= 0xD7A3 for char in example["text"]) for example in examples))
        for example in examples:
            for entity in example["entities"]:
                self.assertEqual(example["text"][entity["start"]:entity["end"]], entity["text"])
