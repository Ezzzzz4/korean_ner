"""KF-DeBERTa-base subword KLUE NER experiment.

This script is isolated from the corrected KoELECTRA character pipeline. It is
CPU-first and uses CUDA only when explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from korean_ner.labels import build_label_maps, labels_from_klue_dataset
from korean_ner.metrics import score_klue_benchmark_entity_macro_f1
from korean_ner.subword import IGNORE_INDEX, benchmark_labels_from_prediction, encode_subword_offsets


MODEL_NAME = "kakaobank/kf-deberta-base"
MODEL_REVISION = "363b171d71443b0874b0bf9cea053eb5b1650633"
SELECTION_METRIC = "validation_klue_official_entity_macro_f1"
METRIC_DEFINITIONS = {
    SELECTION_METRIC: "KLUE-compatible strict IOB2 entity macro F1: examples flattened, ASCII spaces removed, predicted CLS/SEP scored against gold O",
}


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = MODEL_NAME
    model_revision: str = MODEL_REVISION
    pretrained_bin: str | None = None
    output_dir: str = "runs/kf_deberta_base_ner"
    resume_checkpoint: str | None = None
    init_checkpoint: str | None = None
    device: str = "cpu"
    seed: int = 42
    max_length: int = 128
    train_batch_size: int = 16
    eval_batch_size: int = 32
    train_limit: int | None = None
    eval_limit: int | None = None
    epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    early_stopping_patience: int = 2
    num_workers: int = 0
    bf16: bool = False


class KlueSubwordDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        tokenizer: object,
        *,
        label_to_id: Mapping[str, int],
        id_to_label: Mapping[int, str],
        max_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.id_to_label = id_to_label
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        text = "".join(row["tokens"])
        encoding = encode_subword_offsets(
            text,
            self.tokenizer,
            row["ner_tags"],
            label_to_id=self.label_to_id,
            id_to_label=self.id_to_label,
            max_length=self.max_length,
        )
        if encoding.labels is None:
            raise ValueError("subword encoding did not produce labels")
        return {
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
            "token_type_ids": encoding.token_type_ids,
            "labels": encoding.labels,
            "offset_mapping": encoding.offset_mapping,
            "special_tokens_mask": encoding.special_tokens_mask,
            "text": text,
            "gold_char_labels": list(row["ner_tags"]),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train KF-DeBERTa-base on KLUE NER")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--pretrained-bin", default=None, help="Optional local pytorch_model.bin path")
    parser.add_argument("--output-dir", default="runs/kf_deberta_base_ner")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--init-checkpoint", default=None, help="Fine-tune from model weights with a fresh optimizer schedule")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", help="Enable CUDA bfloat16 autocast")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig(**vars(args))
    validate_config(config)
    return config


def validate_config(config: TrainConfig) -> None:
    if config.resume_checkpoint and config.init_checkpoint:
        raise ValueError("choose either --resume-checkpoint or --init-checkpoint")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if config.bf16 and config.device != "cuda":
        raise ValueError("--bf16 is only valid with --device cuda")
    if config.max_length < 8:
        raise ValueError("max_length is too small for KLUE NER examples")
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.train_batch_size < 1 or config.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if config.train_limit is not None and config.train_limit < 1:
        raise ValueError("train_limit must be positive when provided")
    if config.eval_limit is not None and config.eval_limit < 1:
        raise ValueError("eval_limit must be positive when provided")
    if not 0 <= config.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")


def set_seed(seed: int, device_intent: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device_intent == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(config: TrainConfig) -> torch.device:
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cpu")


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise ValueError("pretrained payload does not contain a state dict")
    return {str(key).removeprefix("module."): value for key, value in payload.items()}


def safe_torch_load(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    return torch.load(path, map_location=map_location, weights_only=True)


def _encoder_prefix(model_state: Mapping[str, torch.Tensor]) -> str:
    for prefix in ("deberta.", "bert.", "roberta.", "electra."):
        if any(key.startswith(prefix) for key in model_state):
            return prefix
    raise ValueError("could not identify encoder prefix in token-classification model")


def load_encoder_only_weights(model: torch.nn.Module, pretrained_bin: str | Path, *, map_location: str | torch.device = "cpu") -> None:
    """Load encoder weights from a safe PyTorch checkpoint, leaving head random."""

    raw_state = _extract_state_dict(safe_torch_load(pretrained_bin, map_location=map_location))
    model_state = model.state_dict()
    prefix = _encoder_prefix(model_state)
    encoder_keys = {key for key in model_state if key.startswith(prefix)}
    loadable: dict[str, torch.Tensor] = {}
    shape_errors: list[str] = []
    allowed_unexpected = {prefix + "embeddings.position_embeddings.weight"}
    unexpected_encoder = sorted(
        key for key in raw_state
        if key.startswith(prefix) and key not in model_state and key not in allowed_unexpected
    )

    for key in encoder_keys:
        tensor = raw_state.get(key)
        if tensor is None:
            continue
        if tuple(tensor.shape) != tuple(model_state[key].shape):
            shape_errors.append(f"{key}: checkpoint {tuple(tensor.shape)} != model {tuple(model_state[key].shape)}")
            continue
        loadable[key] = tensor

    missing_encoder = sorted(encoder_keys - set(loadable))
    if unexpected_encoder:
        raise ValueError(f"pretrained checkpoint has unexpected encoder keys: {unexpected_encoder[:5]}")
    if missing_encoder:
        raise ValueError(f"pretrained checkpoint is missing encoder keys: {missing_encoder[:5]} ({len(missing_encoder)} total)")
    if shape_errors:
        raise ValueError("pretrained checkpoint has encoder shape mismatches: " + "; ".join(shape_errors[:5]))

    incompatible = model.load_state_dict(loadable, strict=False)
    still_missing_encoder = [key for key in incompatible.missing_keys if key.startswith(prefix)]
    if still_missing_encoder:
        raise ValueError(f"encoder keys were not loaded: {still_missing_encoder[:5]}")


def build_model(
    config: TrainConfig,
    label_names: Sequence[str],
    *,
    load_pretrained_encoder: bool = True,
) -> torch.nn.Module:
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForTokenClassification

    hf_config = AutoConfig.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        num_labels=len(label_names),
        id2label={index: label for index, label in enumerate(label_names)},
        label2id={label: index for index, label in enumerate(label_names)},
    )
    model = AutoModelForTokenClassification.from_config(hf_config)
    if load_pretrained_encoder:
        pretrained_bin = Path(config.pretrained_bin) if config.pretrained_bin else Path(
            hf_hub_download(config.model_name, filename="pytorch_model.bin", revision=config.model_revision)
        )
        load_encoder_only_weights(model, pretrained_bin, map_location="cpu")
    return model


def load_initial_model_weights(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    config: TrainConfig,
    label_names: Sequence[str],
) -> None:
    """Start a new fine-tuning stage from trusted model weights only."""
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    saved_config = checkpoint.get("config", {})
    if saved_config.get("model_name") != config.model_name or saved_config.get("model_revision") != config.model_revision:
        raise ValueError("initial checkpoint model identity differs from the requested encoder")
    if list(checkpoint.get("label_names", [])) != list(label_names):
        raise ValueError("initial checkpoint label names do not match the dataset")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def pad_list(values: Sequence[int], length: int, pad_value: int) -> list[int]:
    return list(values) + [pad_value] * (length - len(values))


def collate_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    max_length = max(len(item["input_ids"]) for item in items)
    return {
        "input_ids": torch.tensor([pad_list(item["input_ids"], max_length, 0) for item in items], dtype=torch.long),
        "attention_mask": torch.tensor([pad_list(item["attention_mask"], max_length, 0) for item in items], dtype=torch.long),
        "token_type_ids": torch.tensor([pad_list(item["token_type_ids"], max_length, 0) for item in items], dtype=torch.long),
        "labels": torch.tensor([pad_list(item["labels"], max_length, IGNORE_INDEX) for item in items], dtype=torch.long),
        "offset_mapping": [item["offset_mapping"] + [(0, 0)] * (max_length - len(item["offset_mapping"])) for item in items],
        "special_tokens_mask": [pad_list(item["special_tokens_mask"], max_length, 0) for item in items],
        "text": [item["text"] for item in items],
        "gold_char_labels": [item["gold_char_labels"] for item in items],
    }


def limit_dataset(split: Any, limit: int | None) -> Any:
    if limit is None:
        return split
    bounded = min(limit, len(split))
    if hasattr(split, "select"):
        return split.select(range(bounded))
    return split[:bounded]


def autocast_context(config: TrainConfig):
    if config.device == "cuda" and config.bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def model_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }
    token_type_ids = batch["token_type_ids"]
    if token_type_ids.ne(0).any():
        inputs["token_type_ids"] = token_type_ids.to(device)
    return inputs


def make_scheduler(optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_ratio: float):
    from transformers import get_linear_schedule_with_warmup

    warmup_steps = int(total_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def validate_resume_config(config: TrainConfig, checkpoint: Mapping[str, Any]) -> None:
    saved = checkpoint.get("config")
    if not isinstance(saved, Mapping):
        raise ValueError("resume checkpoint has no training config")
    required_keys = (
        "model_name",
        "model_revision",
        "pretrained_bin",
        "max_length",
        "train_batch_size",
        "eval_batch_size",
        "train_limit",
        "eval_limit",
        "epochs",
        "lr",
        "weight_decay",
        "warmup_ratio",
        "gradient_clip",
    )
    for key in required_keys:
        if saved.get(key) != getattr(config, key):
            raise ValueError(f"resume config differs from checkpoint for {key}")
    if int(checkpoint.get("epoch", 0)) >= config.epochs:
        raise ValueError("resume checkpoint already reached the requested number of epochs")


def restore_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    checkpoint_path: str | Path,
    config: TrainConfig,
    label_names: Sequence[str],
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[int, float, dict[str, Any], int]:
    checkpoint = safe_torch_load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("resume checkpoint is not a mapping")
    if checkpoint.get("selection_metric") != SELECTION_METRIC:
        raise ValueError("resume checkpoint was selected using a different metric")
    if list(checkpoint.get("label_names", [])) != list(label_names):
        raise ValueError("resume checkpoint label names do not match the dataset")
    validate_resume_config(config, checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_metric = float(checkpoint.get("best_selection_metric", -1.0))
    best_metrics = dict(checkpoint.get("best_metrics", checkpoint.get("metrics", {})))
    stale_epochs = int(checkpoint.get("stale_epochs", 0))
    return start_epoch, best_metric, best_metrics, stale_epochs


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: TrainConfig,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(config):
            outputs = model(**model_inputs(batch, device))
            loss = outputs.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite training loss: {float(loss.detach().cpu())}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    id_to_label: Mapping[int, str],
    config: TrainConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    benchmark_true: list[str] = []
    benchmark_pred: list[str] = []
    for batch in dataloader:
        inputs = model_inputs({**batch, "labels": batch["labels"]}, device)
        inputs.pop("labels")
        with autocast_context(config):
            logits = model(**inputs).logits
        predictions = logits.argmax(dim=-1).detach().cpu().tolist()
        for pred_ids, offsets, special_mask, text, gold_labels in zip(
            predictions,
            batch["offset_mapping"],
            batch["special_tokens_mask"],
            batch["text"],
            batch["gold_char_labels"],
        ):
            gold, pred = benchmark_labels_from_prediction(
                pred_ids,
                gold_labels,
                offsets,
                special_mask,
                id_to_label,
                text,
            )
            benchmark_true.extend(gold)
            benchmark_pred.extend(pred)
    metric = score_klue_benchmark_entity_macro_f1(benchmark_true, benchmark_pred)
    return {
        SELECTION_METRIC: metric,
        "validation_klue_official_entity_macro_f1_percent": metric * 100.0,
        "validation_flat_label_count": len(benchmark_true),
    }


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_manifest(config: TrainConfig, label_names: Sequence[str], metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "klue",
            "subset": "ner",
            "train_split": "train",
            "validation_split": "validation",
            "limits": {"train": config.train_limit, "validation": config.eval_limit},
        },
        "model": {"name": config.model_name, "architecture": "AutoModelForTokenClassification encoder + random token-classification head"},
        "config": asdict(config),
        "label_names": list(label_names),
        "metric_definitions": METRIC_DEFINITIONS,
        "selection_metric": SELECTION_METRIC,
        "metrics": dict(metrics),
        "runtime": {
            "python": ".".join(map(str, os.sys.version_info[:3])),
            "torch": torch.__version__,
            "configured_device": config.device,
            "cuda_available": torch.cuda.is_available() if config.device == "cuda" else None,
        },
    }


def save_training_artifacts(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    epoch: int,
    best_metric: float,
    config: TrainConfig,
    label_names: Sequence[str],
    metrics: Mapping[str, Any],
    is_best: bool,
    stale_epochs: int,
    best_metrics: Mapping[str, Any],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_selection_metric": best_metric,
        "selection_metric": SELECTION_METRIC,
        "label_names": list(label_names),
        "config": asdict(config),
        "metric_definitions": METRIC_DEFINITIONS,
        "metrics": dict(metrics),
        "best_metrics": dict(best_metrics),
        "stale_epochs": stale_epochs,
    }
    if is_best:
        atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "label_names": list(label_names),
                "config": asdict(config),
                "metric_definitions": METRIC_DEFINITIONS,
                "metrics": dict(metrics),
            },
            output_dir / "best_model.pt",
        )
        write_json(output_dir / "best_metrics.json", dict(metrics))
        write_json(output_dir / "run_manifest.json", build_manifest(config, label_names, metrics))
    # Publish the best weights before the resume checkpoint. A crash between the
    # two leaves a best model ahead of the last checkpoint, which resume can
    # reconcile; the opposite order can silently strand a stale best model.
    atomic_torch_save(checkpoint, output_dir / "checkpoint_last.pt")
    write_json(output_dir / "last_metrics.json", dict(metrics))
    write_json(output_dir / "config.json", asdict(config))


def run_training(config: TrainConfig) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    validate_config(config)
    set_seed(config.seed, config.device)
    device = resolve_device(config)
    dataset = load_dataset("klue", "ner")
    label_names = labels_from_klue_dataset(dataset)
    label_maps = build_label_maps(label_names)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(f"{config.model_name} must provide a fast tokenizer with offset mappings")

    train_rows = limit_dataset(dataset["train"], config.train_limit)
    valid_rows = limit_dataset(dataset["validation"], config.eval_limit)
    train_dataset = KlueSubwordDataset(
        train_rows,
        tokenizer,
        label_to_id=label_maps.label_to_id,
        id_to_label=label_maps.id_to_label,
        max_length=config.max_length,
    )
    valid_dataset = KlueSubwordDataset(
        valid_rows,
        tokenizer,
        label_to_id=label_maps.label_to_id,
        id_to_label=label_maps.id_to_label,
        max_length=config.max_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
    )

    model = build_model(
        config,
        label_names,
        load_pretrained_encoder=not (config.init_checkpoint or config.resume_checkpoint),
    )
    if config.init_checkpoint:
        load_initial_model_weights(model, config.init_checkpoint, config, label_names)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = make_scheduler(
        optimizer,
        total_steps=max(1, math.ceil(len(train_loader)) * config.epochs),
        warmup_ratio=config.warmup_ratio,
    )

    output_dir = Path(config.output_dir)
    best_metric = -1.0
    best_metrics: dict[str, Any] = {}
    stale_epochs = 0
    start_epoch = 1
    if config.resume_checkpoint:
        start_epoch, best_metric, best_metrics, stale_epochs = restore_training_state(
            model,
            optimizer,
            scheduler,
            config.resume_checkpoint,
            config,
            label_names,
            map_location=device,
        )
        best_path = output_dir / "best_model.pt"
        if best_path.exists():
            best_artifact = safe_torch_load(best_path, map_location="cpu")
            if list(best_artifact.get("label_names", [])) != list(label_names):
                raise ValueError("best model label names do not match the dataset")
            artifact_metrics = dict(best_artifact.get("metrics", {}))
            artifact_score = float(artifact_metrics.get(SELECTION_METRIC, -1.0))
            if artifact_score > best_metric:
                best_metric = artifact_score
                best_metrics = artifact_metrics
                stale_epochs = 0
            elif artifact_score < best_metric:
                raise ValueError("best model is older than the resume checkpoint; recover the artifact before resuming")

    for epoch in range(start_epoch, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, config, device)
        metrics = evaluate_model(model, valid_loader, label_maps.id_to_label, config, device)
        metrics = {
            **metrics,
            "epoch": epoch,
            "train_loss": train_loss,
            "split": "validation",
            "selection_metric": SELECTION_METRIC,
            "metric_definitions": METRIC_DEFINITIONS,
        }
        current_metric = float(metrics[SELECTION_METRIC])
        is_best = current_metric > best_metric
        if is_best:
            best_metric = current_metric
            best_metrics = dict(metrics)
            stale_epochs = 0
        else:
            stale_epochs += 1
        save_training_artifacts(
            output_dir,
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            best_metric=best_metric,
            config=config,
            label_names=label_names,
            metrics=metrics,
            is_best=is_best,
            stale_epochs=stale_epochs,
            best_metrics=best_metrics,
        )
        print(
            f"epoch={epoch} loss={train_loss:.6f} "
            f"validation_klue_macro_f1={metrics[SELECTION_METRIC]:.6f}",
            flush=True,
        )
        if stale_epochs >= config.early_stopping_patience:
            break
    return best_metrics


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_arg_parser().parse_args(argv))
    metrics = run_training(config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
