from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from korean_ner.labels import DEFAULT_LABEL_LIST, build_label_maps
from korean_ner.metrics import score_klue_benchmark_entity_macro_f1
from korean_ner.subword import (
    IGNORE_INDEX,
    benchmark_labels_from_prediction,
    encode_subword_offsets,
    project_token_predictions_to_chars,
)
from train_deberta import (
    SELECTION_METRIC,
    TrainConfig,
    atomic_torch_save,
    collate_batch,
    load_encoder_only_weights,
    load_initial_model_weights,
    restore_training_state,
    validate_config,
)


class DummyFastTokenizer:
    is_fast = True

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=False,
    ):
        assert add_special_tokens
        assert return_offsets_mapping
        assert return_special_tokens_mask
        assert truncation is False
        return {
            "input_ids": [101, 11, 12, 102],
            "attention_mask": [1, 1, 1, 1],
            "token_type_ids": [0, 0, 0, 0],
            "offset_mapping": [(0, 0), (0, 2), (2, len(text)), (0, 0)],
            "special_tokens_mask": [1, 0, 0, 1],
        }


class TinyEncoderModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deberta = torch.nn.Linear(2, 2)
        self.classifier = torch.nn.Linear(2, 3)


class SubwordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.maps = build_label_maps(DEFAULT_LABEL_LIST)

    def test_encode_aligns_token_label_from_first_non_space_char(self):
        labels = [
            self.maps.label_to_id["B-PS"],
            self.maps.label_to_id["I-PS"],
            self.maps.label_to_id["O"],
            self.maps.label_to_id["B-LC"],
        ]

        encoded = encode_subword_offsets(
            "AB C",
            DummyFastTokenizer(),
            labels,
            label_to_id=self.maps.label_to_id,
            id_to_label=self.maps.id_to_label,
            max_length=4,
        )

        self.assertEqual(encoded.labels, [self.maps.label_to_id["O"], self.maps.label_to_id["B-PS"], self.maps.label_to_id["B-LC"], self.maps.label_to_id["O"]])

    def test_encode_raises_on_overflow(self):
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            encode_subword_offsets("AB C", DummyFastTokenizer(), max_length=3)

    def test_project_expands_b_label_inside_offset_and_skips_spaces(self):
        predicted = project_token_predictions_to_chars(
            [self.maps.label_to_id["O"], self.maps.label_to_id["B-LC"], self.maps.label_to_id["O"]],
            [(0, 0), (0, 3), (0, 0)],
            [1, 0, 1],
            self.maps.id_to_label,
            "A B",
        )

        self.assertEqual(predicted, ["B-LC", "O", "I-LC"])

    def test_benchmark_flattening_includes_special_predictions_and_matches_metric_contract(self):
        gold, pred = benchmark_labels_from_prediction(
            [
                self.maps.label_to_id["B-OG"],
                self.maps.label_to_id["B-LC"],
                self.maps.label_to_id["I-OG"],
            ],
            [self.maps.label_to_id["B-LC"], self.maps.label_to_id["I-LC"], self.maps.label_to_id["I-LC"]],
            [(0, 0), (0, 3), (0, 0)],
            [1, 0, 1],
            self.maps.id_to_label,
            "A B",
        )

        self.assertEqual(gold, ["O", "B-LC", "I-LC", "O"])
        self.assertEqual(pred, ["B-OG", "B-LC", "I-LC", "I-OG"])
        self.assertAlmostEqual(score_klue_benchmark_entity_macro_f1(gold, pred), 0.5)

    def test_load_encoder_only_weights_uses_weights_only_and_rejects_missing_encoder_keys(self):
        model = TinyEncoderModel()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pytorch_model.bin"
            torch.save({"deberta.weight": torch.ones_like(model.deberta.weight)}, checkpoint)
            with self.assertRaisesRegex(ValueError, "missing encoder keys"):
                load_encoder_only_weights(model, checkpoint)

            torch.save(
                {
                    "deberta.weight": torch.ones_like(model.deberta.weight),
                    "deberta.bias": torch.ones_like(model.deberta.bias),
                    "deberta.embeddings.position_embeddings.weight": torch.ones(4, 2),
                    "classifier.weight": torch.ones_like(model.classifier.weight),
                },
                checkpoint,
            )
            with patch("train_deberta.torch.load", wraps=torch.load) as torch_load:
                load_encoder_only_weights(model, checkpoint)
            self.assertTrue(torch_load.call_args.kwargs["weights_only"])

    def test_dynamic_padding_does_not_mark_pad_as_scored_special_token(self):
        batch = collate_batch(
            [
                {
                    "input_ids": [101, 11, 102],
                    "attention_mask": [1, 1, 1],
                    "token_type_ids": [0, 0, 0],
                    "labels": [self.maps.label_to_id["O"], self.maps.label_to_id["O"], self.maps.label_to_id["O"]],
                    "offset_mapping": [(0, 0), (0, 1), (0, 0)],
                    "special_tokens_mask": [1, 0, 1],
                    "text": "A",
                    "gold_char_labels": [self.maps.label_to_id["O"]],
                },
                {
                    "input_ids": [101, 12, 13, 102],
                    "attention_mask": [1, 1, 1, 1],
                    "token_type_ids": [0, 0, 0, 0],
                    "labels": [self.maps.label_to_id["O"], self.maps.label_to_id["O"], self.maps.label_to_id["O"], self.maps.label_to_id["O"]],
                    "offset_mapping": [(0, 0), (0, 1), (1, 2), (0, 0)],
                    "special_tokens_mask": [1, 0, 0, 1],
                    "text": "AB",
                    "gold_char_labels": [self.maps.label_to_id["O"], self.maps.label_to_id["O"]],
                },
            ]
        )

        self.assertEqual(batch["special_tokens_mask"][0], [1, 0, 1, 0])
        self.assertEqual(batch["attention_mask"][0].tolist(), [1, 1, 1, 0])
        self.assertEqual(batch["labels"][0].tolist()[-1], IGNORE_INDEX)

    def test_atomic_save_leaves_safe_checkpoint_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            atomic_torch_save({"metric": 1.0}, path)

            self.assertEqual(torch.load(path, weights_only=True)["metric"], 1.0)
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_new_finetuning_stage_loads_weights_without_optimizer(self):
        config = TrainConfig(init_checkpoint="previous.pt", epochs=1, lr=5e-6)
        model = TinyEncoderModel()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "previous.pt"
            old = TinyEncoderModel()
            with torch.no_grad():
                old.deberta.weight.fill_(3.0)
            torch.save(
                {
                    "model_state_dict": old.state_dict(),
                    "label_names": list(DEFAULT_LABEL_LIST),
                    "config": {"model_name": config.model_name, "model_revision": config.model_revision},
                },
                path,
            )
            load_initial_model_weights(model, path, config, list(DEFAULT_LABEL_LIST))
        self.assertTrue(torch.equal(model.deberta.weight, old.deberta.weight))
        with self.assertRaisesRegex(ValueError, "choose either"):
            validate_config(TrainConfig(init_checkpoint="old.pt", resume_checkpoint="last.pt"))

    def test_resume_restores_strict_training_state_and_next_epoch(self):
        config = TrainConfig(epochs=3)
        label_names = list(DEFAULT_LABEL_LIST)
        model = TinyEncoderModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        with torch.no_grad():
            model.deberta.weight.fill_(7.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_last.pt"
            torch.save(
                {
                    "epoch": 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_selection_metric": 0.42,
                    "selection_metric": SELECTION_METRIC,
                    "best_metrics": {SELECTION_METRIC: 0.42},
                    "stale_epochs": 1,
                    "label_names": label_names,
                    "config": config.__dict__,
                },
                path,
            )

            fresh_model = TinyEncoderModel()
            fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            fresh_scheduler = torch.optim.lr_scheduler.LambdaLR(fresh_optimizer, lambda step: 1.0)
            with patch("train_deberta.torch.load", wraps=torch.load) as torch_load:
                start_epoch, best_metric, best_metrics, stale_epochs = restore_training_state(
                    fresh_model,
                    fresh_optimizer,
                    fresh_scheduler,
                    path,
                    config,
                    label_names,
                )

        self.assertTrue(torch_load.call_args.kwargs["weights_only"])
        self.assertEqual(start_epoch, 2)
        self.assertAlmostEqual(best_metric, 0.42)
        self.assertEqual(best_metrics[SELECTION_METRIC], 0.42)
        self.assertEqual(stale_epochs, 1)
        self.assertTrue(torch.equal(fresh_model.deberta.weight, torch.full_like(fresh_model.deberta.weight, 7.0)))

    def test_resume_rejects_schedule_config_mismatch(self):
        config = TrainConfig(epochs=4)
        checkpoint = {
            "epoch": 1,
            "model_state_dict": TinyEncoderModel().state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "best_selection_metric": 0.0,
            "selection_metric": SELECTION_METRIC,
            "label_names": list(DEFAULT_LABEL_LIST),
            "config": TrainConfig(epochs=3).__dict__,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_last.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "differs from checkpoint for epochs"):
                restore_training_state(
                    TinyEncoderModel(),
                    torch.optim.AdamW(TinyEncoderModel().parameters(), lr=config.lr),
                    torch.optim.lr_scheduler.LambdaLR(torch.optim.AdamW(TinyEncoderModel().parameters(), lr=config.lr), lambda step: 1.0),
                    path,
                    config,
                    list(DEFAULT_LABEL_LIST),
                )


if __name__ == "__main__":
    unittest.main()
