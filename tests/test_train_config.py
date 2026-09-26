from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

import train


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.electra = torch.nn.Module()
        self.electra.embeddings = torch.nn.Module()
        self.electra.embeddings.word_embeddings = torch.nn.Embedding(4, 2)
        self.electra.embeddings.position_embeddings = torch.nn.Embedding(4, 2)
        self.classifier = torch.nn.Linear(2, 3)


class DummyScheduler:
    def state_dict(self):
        return {"scheduler": "state"}


class TrainConfigTests(unittest.TestCase):
    def test_parser_defaults_to_cpu_safe_config(self):
        args = train.build_arg_parser().parse_args([])
        config = train.config_from_args(args)

        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.max_length, 192)
        self.assertIsNone(config.train_limit)
        self.assertIsNone(config.eval_limit)
        self.assertFalse(config.bf16)
        self.assertFalse(config.use_fgm)
        self.assertFalse(config.use_rdrop)

    def test_parser_accepts_bounded_smoke_limits(self):
        args = train.build_arg_parser().parse_args(["--train-limit", "8", "--eval-limit", "4"])
        config = train.config_from_args(args)

        self.assertEqual(config.train_limit, 8)
        self.assertEqual(config.eval_limit, 4)

    def test_cuda_requires_explicit_device_argument(self):
        args = train.build_arg_parser().parse_args(["--device", "cuda"])
        config = train.config_from_args(args)

        self.assertEqual(config.device, "cuda")

    def test_bf16_is_rejected_on_cpu(self):
        args = train.build_arg_parser().parse_args(["--bf16"])

        with self.assertRaisesRegex(ValueError, "bf16"):
            train.config_from_args(args)

    def test_max_length_is_locked_to_corrected_default(self):
        args = train.build_arg_parser().parse_args(["--max-length", "256"])

        with self.assertRaisesRegex(ValueError, "max_length"):
            train.config_from_args(args)

    def test_init_from_legacy_and_resume_are_mutually_exclusive(self):
        args = train.build_arg_parser().parse_args([
            "--init-from-legacy-weights",
            "legacy.pt",
            "--resume-checkpoint",
            "resume.pt",
        ])

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            train.config_from_args(args)

    def test_cpu_seed_does_not_probe_cuda(self):
        with patch("train.torch.cuda.is_available", side_effect=AssertionError("cuda probed")):
            train.set_seed(123, "cpu")

    def test_optimizer_uses_separate_encoder_and_head_learning_rates(self):
        config = train.TrainConfig(encoder_lr=1e-5, head_lr=2e-4)
        optimizer = train.build_optimizer(DummyModel(), config)

        self.assertEqual([group["name"] for group in optimizer.param_groups], ["encoder", "head"])
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-5, 2e-4])

    def test_masked_symmetric_kl_ignores_inactive_positions(self):
        emissions_a = torch.tensor([[[2.0, 0.0], [100.0, -100.0]]], dtype=torch.bfloat16)
        emissions_b = torch.tensor([[[0.0, 2.0], [-100.0, 100.0]]], dtype=torch.bfloat16)
        masked = torch.tensor([[True, False]])
        unmasked = torch.tensor([[True, True]])

        masked_loss = train.masked_symmetric_kl(emissions_a, emissions_b, masked)
        self.assertEqual(masked_loss.dtype, torch.float32)
        self.assertGreater(train.masked_symmetric_kl(emissions_a, emissions_b, unmasked).item(), 1.0)
        self.assertLess(masked_loss.item(), 2.0)

    def test_fgm_only_perturbs_word_embeddings(self):
        model = DummyModel()
        for _, param in model.named_parameters():
            param.grad = torch.ones_like(param)
        before_word = model.electra.embeddings.word_embeddings.weight.detach().clone()
        before_position = model.electra.embeddings.position_embeddings.weight.detach().clone()

        fgm = train.FGM(model, epsilon=0.1)
        fgm.attack()

        self.assertFalse(torch.equal(model.electra.embeddings.word_embeddings.weight, before_word))
        self.assertTrue(torch.equal(model.electra.embeddings.position_embeddings.weight, before_position))
        fgm.restore()
        self.assertTrue(torch.equal(model.electra.embeddings.word_embeddings.weight, before_word))

    def test_manifest_contains_metric_definitions_and_label_metadata(self):
        config = train.TrainConfig(output_dir="out", train_limit=8, eval_limit=4)
        metrics = {"validation_klue_official_entity_macro_f1": 0.5}

        with patch("train.torch.cuda.is_available", side_effect=AssertionError("cuda probed")):
            manifest = train.build_manifest(config, ["B-PS", "I-PS", "O"], metrics)

        self.assertEqual(manifest["dataset"]["validation_split"], "validation")
        self.assertEqual(manifest["dataset"]["limits"], {"train": 8, "validation": 4})
        self.assertEqual(manifest["label_names"], ["B-PS", "I-PS", "O"])
        self.assertEqual(manifest["selection_metric"], "validation_klue_official_entity_macro_f1")
        self.assertEqual(manifest["runtime"]["configured_device"], "cpu")
        self.assertIsNone(manifest["runtime"]["cuda_available"])
        self.assertIn("validation_klue_official_entity_macro_f1", manifest["metric_definitions"])

    def test_save_training_artifacts_writes_best_and_resume_checkpoints(self):
        config = train.TrainConfig(output_dir="unused")
        model = DummyModel()
        optimizer = train.build_optimizer(model, config)
        metrics = {
            "validation_klue_official_entity_macro_f1": 0.61,
            "validation_entity_macro_f1": 0.6,
            "validation_entity_micro_f1": 0.7,
            "split": "validation",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            train.save_training_artifacts(
                Path(tmpdir),
                model,
                optimizer,
                DummyScheduler(),
                epoch=1,
                best_metric=0.61,
                config=config,
                label_names=["B-PS", "I-PS", "O"],
                metrics=metrics,
                is_best=True,
                stale_epochs=2,
                best_metrics=metrics,
            )
            manifest = json.loads((Path(tmpdir) / "run_manifest.json").read_text(encoding="utf-8"))
            best_metrics = json.loads((Path(tmpdir) / "best_metrics.json").read_text(encoding="utf-8"))
            last_metrics = json.loads((Path(tmpdir) / "last_metrics.json").read_text(encoding="utf-8"))
            checkpoint = torch.load(Path(tmpdir) / "checkpoint_last.pt", map_location="cpu", weights_only=True)

            self.assertTrue((Path(tmpdir) / "checkpoint_last.pt").exists())
            self.assertTrue((Path(tmpdir) / "best_model.pt").exists())
            self.assertEqual(manifest["metrics"]["validation_entity_macro_f1"], 0.6)
            self.assertEqual(best_metrics["split"], "validation")
            self.assertEqual(last_metrics["validation_entity_micro_f1"], 0.7)
            self.assertEqual(checkpoint["stale_epochs"], 2)
            self.assertEqual(checkpoint["best_metrics"]["validation_entity_macro_f1"], 0.6)
            self.assertEqual(checkpoint["selection_metric"], train.SELECTION_METRIC)
            self.assertEqual(checkpoint["best_selection_metric"], 0.61)

    def test_training_evaluation_uses_nonspace_raw_bio_for_selection(self):
        class FixedModel:
            def eval(self):
                return self

            def __call__(self, **_kwargs):
                return [[0, 1, 0, 2, 0]]

        batch = {
            "input_ids": torch.zeros((1, 5), dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "labels": torch.tensor([[-100, 1, 2, 2, -100]], dtype=torch.long),
            "char_indices": [[None, 0, 1, 2, None]],
            "text": ["a b"],
        }
        scores = train.evaluate_model(FixedModel(), [batch], {0: "O", 1: "B-LC", 2: "I-LC"}, torch.device("cpu"))
        self.assertEqual(scores["validation_klue_official_entity_macro_f1"], 1.0)
        self.assertEqual(scores["validation_entity_macro_f1"], 0.0)

    def test_interrupted_checkpoint_save_preserves_previous_file(self):
        def interrupted_save(_payload, path):
            Path(path).write_bytes(b"partial")
            raise OSError("simulated interruption")

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "checkpoint_last.pt"
            target.write_bytes(b"previous checkpoint")
            with patch("train.torch.save", side_effect=interrupted_save):
                with self.assertRaisesRegex(OSError, "simulated interruption"):
                    train.atomic_torch_save({"epoch": 2}, target)
            self.assertEqual(target.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(Path(tmpdir).glob("*.tmp")), [])

    def test_resume_rejects_changed_schedule_and_completed_run(self):
        config = train.TrainConfig(epochs=3)
        checkpoint = {"config": asdict(config), "epoch": 1}
        train.validate_resume_config(config, checkpoint)
        with self.assertRaisesRegex(ValueError, "epochs"):
            train.validate_resume_config(train.TrainConfig(epochs=4), checkpoint)
        with self.assertRaisesRegex(ValueError, "already reached"):
            train.validate_resume_config(config, {"config": asdict(config), "epoch": 3})


if __name__ == "__main__":
    unittest.main()
