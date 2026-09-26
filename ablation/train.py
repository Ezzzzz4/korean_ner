"""Honest ablation runner for the corrected Korean NER training pipeline.

This script does not implement separate model architectures. It runs the root
``train.py`` entry point with the fixed KoELECTRA + BiLSTM + CRF architecture
and varies only regularization toggles: FGM and R-Drop.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
ROOT_TRAIN = ROOT_DIR / "train.py"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs"
MANIFEST_NAME = "ablation_run_manifest.json"
ROOT_MANIFEST_NAME = "run_manifest.json"


@dataclass(frozen=True)
class Variant:
    name: str
    use_fgm: bool
    use_rdrop: bool
    description: str


VARIANTS: dict[str, Variant] = {
    "base": Variant("base", False, False, "KoELECTRA + BiLSTM + CRF, no FGM, no R-Drop"),
    "fgm": Variant("fgm", True, False, "KoELECTRA + BiLSTM + CRF + FGM"),
    "rdrop": Variant("rdrop", False, True, "KoELECTRA + BiLSTM + CRF + R-Drop"),
    "fgm_rdrop": Variant("fgm_rdrop", True, True, "KoELECTRA + BiLSTM + CRF + FGM + R-Drop"),
}


PASS_THROUGH_OPTIONS = (
    "model_name",
    "seed",
    "max_length",
    "train_batch_size",
    "eval_batch_size",
    "epochs",
    "encoder_lr",
    "head_lr",
    "weight_decay",
    "warmup_ratio",
    "gradient_clip",
    "early_stopping_patience",
    "num_workers",
    "fgm_epsilon",
    "rdrop_alpha",
    "init_from_legacy_weights",
    "resume_checkpoint",
    "train_limit",
    "eval_limit",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def root_train_source() -> str:
    return ROOT_TRAIN.read_text(encoding="utf-8")


def root_train_supports(flag: str) -> bool:
    source = root_train_source()
    return f'"{flag}"' in source or f"'{flag}'" in source


def validate_root_train_args(args: argparse.Namespace) -> None:
    missing = []
    if args.train_limit is not None and not root_train_supports("--train-limit"):
        missing.append("--train-limit")
    if args.eval_limit is not None and not root_train_supports("--eval-limit"):
        missing.append("--eval-limit")
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"root train.py does not support {joined} yet; update train.py before using those limits")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run regularization ablations through root train.py")
    parser.add_argument("--variant", choices=tuple(VARIANTS), help="Run one variant instead of all variants")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory containing per-variant run dirs")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="CPU by default; CUDA only when explicit")
    parser.add_argument("--status", action="store_true", help="Show existing per-variant manifest status")
    parser.add_argument("--report", action="store_true", help="Print a report from existing manifests")
    parser.add_argument("--dry-run", action="store_true", help="Write planned manifests and print commands without training")

    parser.add_argument("--model-name", default="monologg/koelectra-base-v3-discriminator")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", help="Passed to train.py only with --device cuda")
    parser.add_argument("--fgm-epsilon", type=float, default=0.5)
    parser.add_argument("--rdrop-alpha", type=float, default=1.0)
    parser.add_argument("--init-from-legacy-weights")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--train-limit", type=int, help="Pass through when root train.py supports it")
    parser.add_argument("--eval-limit", type=int, help="Pass through when root train.py supports it")
    return parser


def selected_variants(args: argparse.Namespace) -> list[Variant]:
    if args.variant:
        return [VARIANTS[args.variant]]
    return list(VARIANTS.values())


def flag_name(option_name: str) -> str:
    return "--" + option_name.replace("_", "-")


def append_option(command: list[str], option_name: str, value: Any) -> None:
    if value is None:
        return
    command.extend([flag_name(option_name), str(value)])


def build_train_command(args: argparse.Namespace, variant: Variant) -> list[str]:
    variant_dir = Path(args.output_dir) / variant.name
    command = [
        sys.executable,
        str(ROOT_TRAIN),
        "--output-dir",
        str(variant_dir),
        "--device",
        args.device,
    ]

    for option_name in PASS_THROUGH_OPTIONS:
        append_option(command, option_name, getattr(args, option_name))

    if args.device == "cuda" and args.bf16:
        command.append("--bf16")
    if variant.use_fgm:
        command.append("--use-fgm")
    if variant.use_rdrop:
        command.append("--use-rdrop")
    return command


def manifest_path(output_dir: str | Path, variant: Variant) -> Path:
    return Path(output_dir) / variant.name / MANIFEST_NAME


def root_manifest_path(output_dir: str | Path, variant: Variant) -> Path:
    return Path(output_dir) / variant.name / ROOT_MANIFEST_NAME


def build_ablation_manifest(
    args: argparse.Namespace,
    variant: Variant,
    command: Sequence[str],
    *,
    status: str,
    return_code: int | None = None,
) -> dict[str, Any]:
    variant_dir = Path(args.output_dir) / variant.name
    shared_config = {
        option_name: getattr(args, option_name)
        for option_name in PASS_THROUGH_OPTIONS
        if getattr(args, option_name) is not None
    }
    shared_config["device"] = args.device
    shared_config["bf16"] = bool(args.bf16 and args.device == "cuda")
    return {
        "created_at": utc_now(),
        "status": status,
        "return_code": return_code,
        "variant": asdict(variant),
        "architecture": "KoELECTRA + BiLSTM + CRF",
        "ablation_scope": "regularization toggles only; no BiLSTM or CRF architectural ablations",
        "selection_metric": "validation_klue_official_entity_macro_f1",
        "metric_policy": "KLUE baseline-compatible validation entity macro F1, same as root train.py",
        "root_train": str(ROOT_TRAIN),
        "variant_output_dir": str(variant_dir),
        "command": list(command),
        "shared_config": shared_config,
    }


def run_variant(args: argparse.Namespace, variant: Variant) -> int:
    command = build_train_command(args, variant)
    path = manifest_path(args.output_dir, variant)
    write_json(path, build_ablation_manifest(args, variant, command, status="planned"))
    if args.dry_run:
        print(" ".join(command))
        write_json(path, build_ablation_manifest(args, variant, command, status="dry_run"))
        return 0

    completed = subprocess.run(command, cwd=ROOT_DIR, check=False)
    status = "completed" if completed.returncode == 0 else "failed"
    write_json(path, build_ablation_manifest(args, variant, command, status=status, return_code=completed.returncode))
    return completed.returncode


def collect_variant_records(output_dir: str | Path) -> list[dict[str, Any]]:
    records = []
    for variant in VARIANTS.values():
        ablation_manifest = read_json(manifest_path(output_dir, variant))
        root_manifest = read_json(root_manifest_path(output_dir, variant))
        status = (ablation_manifest or {}).get("status", "not_started")
        completed = status == "completed" and (ablation_manifest or {}).get("return_code") == 0
        root_metrics = (root_manifest or {}).get("metrics", {}) if completed else {}
        records.append(
            {
                "name": variant.name,
                "description": variant.description,
                "use_fgm": variant.use_fgm,
                "use_rdrop": variant.use_rdrop,
                "status": status,
                "return_code": (ablation_manifest or {}).get("return_code"),
                "validation_klue_official_entity_macro_f1": root_metrics.get("validation_klue_official_entity_macro_f1"),
                "manifest": str(manifest_path(output_dir, variant)),
                "root_manifest": str(root_manifest_path(output_dir, variant)),
            }
        )
    return records


def print_status(output_dir: str | Path) -> None:
    print("Regularization ablation status")
    print("Architecture fixed: KoELECTRA + BiLSTM + CRF")
    for record in collect_variant_records(output_dir):
        metric = record["validation_klue_official_entity_macro_f1"]
        metric_text = "n/a" if metric is None else f"{float(metric):.6f}"
        print(f"{record['name']:<10} {record['status']:<12} klue_macro_f1={metric_text}")


def print_report(output_dir: str | Path) -> None:
    records = collect_variant_records(output_dir)
    completed = [record for record in records if record["status"] == "completed" and record["validation_klue_official_entity_macro_f1"] is not None]
    if not completed:
        print("No completed root run manifests found.")
        print_status(output_dir)
        return

    completed.sort(key=lambda record: float(record["validation_klue_official_entity_macro_f1"]), reverse=True)
    print("Regularization ablation report")
    print("Metric: validation_klue_official_entity_macro_f1")
    print("| Variant | FGM | R-Drop | Macro F1 |")
    print("|---|---:|---:|---:|")
    for record in completed:
        print(
            f"| {record['name']} | {record['use_fgm']} | {record['use_rdrop']} | "
            f"{float(record['validation_klue_official_entity_macro_f1']):.6f} |"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.status:
        print_status(args.output_dir)
        return 0
    if args.report:
        print_report(args.output_dir)
        return 0

    validate_root_train_args(args)
    exit_code = 0
    for variant in selected_variants(args):
        variant_code = run_variant(args, variant)
        if variant_code != 0:
            exit_code = variant_code
            break
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
