import unittest

import torch

from korean_ner.deberta_runtime import DebertaRuntime


class TinyTokenizer:
    def __call__(self, text, **kwargs):
        self.text = text
        offsets = [(0, 0), (0, 2), (3, 6), (0, 0)] if text == "서울 대학교" else [(0, 0), (0, 1), (1, 2), (0, 0)]
        return {
            "input_ids": [101, 11, 12, 102],
            "attention_mask": [1, 1, 1, 1],
            "token_type_ids": [0, 0, 0, 0],
            "special_tokens_mask": [1, 0, 0, 1],
            "offset_mapping": offsets,
        }


class TinyModel(torch.nn.Module):
    def forward(self, **kwargs):
        logits = torch.zeros((1, 4, 3))
        logits[0, 1, 1] = 5.0
        logits[0, 2, 2] = 5.0
        return type("Output", (), {"logits": logits})()


class DebertaRuntimeTests(unittest.TestCase):
    def test_prediction_projects_offsets_to_original_text(self):
        runtime = DebertaRuntime(
            TinyTokenizer(), TinyModel(), {0: "O", 1: "B-PS", 2: "I-PS"}, 128, torch.device("cpu")
        )
        self.assertEqual(runtime.predict_labels("김철"), ["B-PS", "I-PS"])
        self.assertEqual(runtime.predict_entities("김철"), [
            {"text": "김철", "label": "PS", "start": 0, "end": 2}
        ])

    def test_multiword_entity_keeps_internal_space_and_original_offsets(self):
        runtime = DebertaRuntime(
            TinyTokenizer(), TinyModel(), {0: "O", 1: "B-LC", 2: "I-LC"}, 128, torch.device("cpu")
        )
        self.assertEqual(runtime.predict_labels("서울 대학교"), [
            "B-LC", "I-LC", "I-LC", "I-LC", "I-LC", "I-LC"
        ])
        self.assertEqual(runtime.predict_entities("서울 대학교"), [
            {"text": "서울 대학교", "label": "LC", "start": 0, "end": 6}
        ])


if __name__ == "__main__":
    unittest.main()
