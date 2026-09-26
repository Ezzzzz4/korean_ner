import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from ablation import train as ablation_train


class AblationRunnerTests(unittest.TestCase):
    def parse_args(self, *args):
        return ablation_train.build_arg_parser().parse_args(list(args))

    def test_variants_are_regularization_only(self):
        variants = ablation_train.VARIANTS

        self.assertEqual(set(variants), {"base", "fgm", "rdrop", "fgm_rdrop"})
        self.assertFalse(variants["base"].use_fgm)
        self.assertFalse(variants["base"].use_rdrop)
        self.assertTrue(variants["fgm"].use_fgm)
        self.assertFalse(variants["fgm"].use_rdrop)
        self.assertFalse(variants["rdrop"].use_fgm)
        self.assertTrue(variants["rdrop"].use_rdrop)
        self.assertTrue(variants["fgm_rdrop"].use_fgm)
        self.assertTrue(variants["fgm_rdrop"].use_rdrop)

        for variant in variants.values():
            self.assertIn("KoELECTRA + BiLSTM + CRF", variant.description)

    def test_build_train_command_defaults_to_cpu_and_adds_only_variant_toggles(self):
        args = self.parse_args("--output-dir", "out", "--epochs", "1")

        base = ablation_train.build_train_command(args, ablation_train.VARIANTS["base"])
        fgm_rdrop = ablation_train.build_train_command(args, ablation_train.VARIANTS["fgm_rdrop"])

        self.assertIn("--device", base)
        self.assertEqual(base[base.index("--device") + 1], "cpu")
        self.assertNotIn("--use-fgm", base)
        self.assertNotIn("--use-rdrop", base)
        self.assertIn("--use-fgm", fgm_rdrop)
        self.assertIn("--use-rdrop", fgm_rdrop)
        self.assertIn("--max-length", base)
        self.assertEqual(base[base.index("--max-length") + 1], "192")

    def test_manifest_states_scope_metric_and_variant(self):
        args = self.parse_args("--output-dir", "out", "--variant", "fgm")
        variant = ablation_train.VARIANTS["fgm"]
        command = ablation_train.build_train_command(args, variant)

        manifest = ablation_train.build_ablation_manifest(args, variant, command, status="planned")

        self.assertEqual(manifest["architecture"], "KoELECTRA + BiLSTM + CRF")
        self.assertIn("regularization toggles only", manifest["ablation_scope"])
        self.assertEqual(manifest["selection_metric"], "validation_klue_official_entity_macro_f1")
        self.assertTrue(manifest["variant"]["use_fgm"])
        self.assertFalse(manifest["variant"]["use_rdrop"])

    def test_status_does_not_report_stale_metrics_without_completed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            root_manifest = output_dir / "base" / ablation_train.ROOT_MANIFEST_NAME
            ablation_train.write_json(
                root_manifest,
                {"metrics": {"validation_klue_official_entity_macro_f1": 0.8123}},
            )

            records = ablation_train.collect_variant_records(output_dir)
            base = next(record for record in records if record["name"] == "base")

            self.assertIsNone(base["validation_klue_official_entity_macro_f1"])
            self.assertEqual(base["status"], "not_started")

            ablation_train.write_json(
                ablation_train.manifest_path(output_dir, ablation_train.VARIANTS["base"]),
                {"status": "completed", "return_code": 0},
            )
            records = ablation_train.collect_variant_records(output_dir)
            base = next(record for record in records if record["name"] == "base")
            self.assertEqual(base["validation_klue_official_entity_macro_f1"], 0.8123)

    def test_dry_run_writes_manifest_without_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.parse_args("--output-dir", tmp, "--variant", "base", "--dry-run")

            with redirect_stdout(StringIO()):
                code = ablation_train.main(
                    ["--output-dir", tmp, "--variant", "base", "--dry-run", "--epochs", str(args.epochs)]
                )
            manifest = ablation_train.read_json(Path(tmp) / "base" / ablation_train.MANIFEST_NAME)

            self.assertEqual(code, 0)
            self.assertEqual(manifest["status"], "dry_run")
            self.assertFalse(manifest["variant"]["use_fgm"])
            self.assertFalse(manifest["variant"]["use_rdrop"])


if __name__ == "__main__":
    unittest.main()
