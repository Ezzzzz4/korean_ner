"""Corrected KLUE NER training entry point.

The CLI is CPU-first by design. CUDA is used only when explicitly requested with
``--device cuda``.
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
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

import korean_ner as core
from korean_ner import (
    DEFAULT_MAX_LENGTH,
    align_text,
    build_label_maps,
    extract_model_state_dict,
    labels_from_klue_dataset,
    load_checkpoint,
    score_strict_iob2,
    token_predictions_to_char_labels,
)
from korean_ner.decode import token_predictions_to_raw_char_labels
from korean_ner.metrics import score_klue_benchmark_entity_macro_f1


MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
SELECTION_METRIC = "validation_klue_official_entity_macro_f1"
METRIC_DEFINITIONS = {
    "validation_klue_official_entity_macro_f1": "KLUE baseline-compatible strict IOB2 entity macro F1: raw decoded character labels, ASCII spaces removed, predicted CLS/SEP labels against gold O, examples flattened",
    "validation_entity_macro_f1": "strict IOB2 entity macro F1 on all original validation characters, including spaces, with BIO repair",
    "validation_entity_micro_f1": "strict IOB2 entity micro F1 on all original validation characters, including spaces, with BIO repair",
}


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = MODEL_NAME
    output_dir: str = "runs/koelectra_ner"
    device: str = "cpu"
    seed: int = 42
    max_length: int = DEFAULT_MAX_LENGTH
    train_batch_size: int = 16
    eval_batch_size: int = 32
    train_limit: int | None = None
    eval_limit: int | None = None
    epochs: int = 10
    encoder_lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    early_stopping_patience: int = 3
    num_workers: int = 0
    bf16: bool = False
    use_fgm: bool = False
    fgm_epsilon: float = 0.5
    use_rdrop: bool = False
    rdrop_alpha: float = 1.0
    init_from_legacy_weights: str | None = None
    resume_checkpoint: str | None = None


class KlueNerDataset(Dataset):
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
        encoding = align_text(
            text,
            self.tokenizer,
            row["ner_tags"],
            max_length=self.max_length,
            label_to_id=self.label_to_id,
            id_to_label=self.id_to_label,
        )
        if encoding.labels is None:
            raise ValueError("training alignment did not produce labels")
        return {
            "input_ids": torch.tensor(encoding.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(encoding.attention_mask, dtype=torch.long),
            "labels": torch.tensor(encoding.labels, dtype=torch.long),
            "char_indices": encoding.char_indices,
            "text": text,
        }


class FGM:
    def __init__(self, model: torch.nn.Module, epsilon: float) -> None:
        self.model = model
        self.epsilon = epsilon
        self.backup: dict[str, torch.Tensor] = {}

    def attack(self) -> None:
        for name, param in self.model.named_parameters():
            if name != "electra.embeddings.word_embeddings.weight" or param.grad is None or not param.requires_grad:
                continue
            norm = torch.norm(param.grad)
            if torch.isfinite(norm) and norm != 0:
                self.backup[name] = param.data.clone()
                param.data.add_(self.epsilon * param.grad / norm)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train corrected KLUE NER model")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", default="runs/koelectra_ner")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--train-limit", type=int, default=None, help="Limit train examples for bounded smoke runs")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit validation examples for bounded smoke runs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", help="Enable CUDA bfloat16 autocast; ignored on CPU")
    parser.add_argument("--use-fgm", action="store_true")
    parser.add_argument("--fgm-epsilon", type=float, default=0.5)
    parser.add_argument("--use-rdrop", action="store_true")
    parser.add_argument("--rdrop-alpha", type=float, default=1.0)
    parser.add_argument("--init-from-legacy-weights")
    parser.add_argument("--resume-checkpoint")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig(**vars(args))
    validate_config(config)
    return config


def validate_config(config: TrainConfig) -> None:
    if config.max_length != DEFAULT_MAX_LENGTH:
        raise ValueError("max_length must remain 192 until the corrected alignment pipeline is revalidated")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if config.bf16 and config.device != "cuda":
        raise ValueError("--bf16 is only valid with --device cuda")
    if not 0 <= config.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.train_batch_size < 1 or config.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if config.train_limit is not None and config.train_limit < 1:
        raise ValueError("train_limit must be positive when provided")
    if config.eval_limit is not None and config.eval_limit < 1:
        raise ValueError("eval_limit must be positive when provided")
    if config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    if config.init_from_legacy_weights and config.resume_checkpoint:
        raise ValueError("--init-from-legacy-weights cannot be combined with --resume-checkpoint")


def set_seed(seed: int, device_intent: str = "cpu") -> None:
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


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    encoder_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("electra."):
            encoder_params.append(param)
        else:
            head_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config.encoder_lr, "name": "encoder"},
            {"params": head_params, "lr": config.head_lr, "name": "head"},
        ],
        weight_decay=config.weight_decay,
    )


def make_linear_scheduler(optimizer: torch.optim.Optimizer, *, total_steps: int, warmup_ratio: float):
    from transformers import get_linear_schedule_with_warmup

    warmup_steps = int(total_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def masked_symmetric_kl(emissions_a: torch.Tensor, emissions_b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask.bool()
    if active.sum() == 0:
        return emissions_a.new_tensor(0.0)
    emissions_a = emissions_a.float()
    emissions_b = emissions_b.float()
    log_prob_a = F.log_softmax(emissions_a, dim=-1)
    log_prob_b = F.log_softmax(emissions_b, dim=-1)
    prob_a = log_prob_a.exp()
    prob_b = log_prob_b.exp()
    kl_ab = F.kl_div(log_prob_a, prob_b, reduction="none").sum(dim=-1)
    kl_ba = F.kl_div(log_prob_b, prob_a, reduction="none").sum(dim=-1)
    return ((kl_ab + kl_ba) * 0.5)[active].mean()


def autocast_context(config: TrainConfig):
    if config.device == "cuda" and config.bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


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
    fgm = FGM(model, config.fgm_epsilon) if config.use_fgm else None

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast_context(config):
            loss, emissions = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            if config.use_rdrop:
                loss_2, emissions_2 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                active_mask = attention_mask.bool() & labels.ne(-100)
                loss = (loss + loss_2) * 0.5 + config.rdrop_alpha * masked_symmetric_kl(emissions, emissions_2, active_mask)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite training loss: {float(loss.detach().cpu())}")
        loss.backward()

        if fgm is not None:
            fgm.attack()
            with autocast_context(config):
                adv_loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            adv_loss.backward()
            fgm.restore()

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
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    all_true: list[list[str]] = []
    all_pred: list[list[str]] = []
    benchmark_true: list[str] = []
    benchmark_pred: list[str] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)

        for pred_ids, label_ids, char_indices, text in zip(predictions, labels.tolist(), batch["char_indices"], batch["text"]):
            char_pred = token_predictions_to_char_labels(pred_ids, char_indices, id_to_label, text_length=len(text))
            raw_char_pred = token_predictions_to_raw_char_labels(pred_ids, char_indices, id_to_label, text_length=len(text))
            char_true = ["O"] * len(text)
            seen_true_indices: set[int] = set()
            for label_id, char_index in zip(label_ids, char_indices):
                if char_index is not None and label_id != -100 and char_index not in seen_true_indices:
                    char_true[char_index] = id_to_label[int(label_id)]
                    seen_true_indices.add(char_index)
            all_true.append(char_true)
            all_pred.append(char_pred)
            benchmark_true.append("O")
            benchmark_pred.append(id_to_label[int(pred_ids[0])])
            for char, true_label, raw_label in zip(text, char_true, raw_char_pred):
                if char != " ":
                    benchmark_true.append(true_label)
                    benchmark_pred.append(raw_label)
            benchmark_true.append("O")
            benchmark_pred.append(id_to_label[int(pred_ids[-1])])

    scores = score_strict_iob2(all_true, all_pred)
    return {
        "validation_klue_official_entity_macro_f1": score_klue_benchmark_entity_macro_f1(benchmark_true, benchmark_pred),
        "validation_entity_macro_f1": scores.f1_macro,
        "validation_entity_micro_f1": scores.f1_micro,
        "validation_precision_micro": scores.precision_micro,
        "validation_recall_micro": scores.recall_micro,
        "all_character_report": scores.report,
    }


def collate_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = ("input_ids", "attention_mask", "labels")
    batch = {key: torch.stack([item[key] for item in items]) for key in tensor_keys}
    batch["char_indices"] = [item["char_indices"] for item in items]
    batch["text"] = [item["text"] for item in items]
    return batch


def limit_dataset(split: Any, limit: int | None) -> Any:
    if limit is None:
        return split
    bounded = min(limit, len(split))
    if hasattr(split, "select"):
        return split.select(range(bounded))
    return split[:bounded]


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
        "model": {"name": config.model_name, "architecture": "KoELECTRA + BiLSTM + CRF"},
        "config": asdict(config),
        "label_names": list(label_names),
        "metric_definitions": METRIC_DEFINITIONS,
        "selection_metric": SELECTION_METRIC,
        "metrics": metrics,
        "runtime": {
            "python": ".".join(map(str, os.sys.version_info[:3])),
            "torch": torch.__version__,
            "configured_device": config.device,
            "cuda_available": torch.cuda.is_available() if config.device == "cuda" else None,
        },
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_resume_config(config: TrainConfig, checkpoint: Mapping[str, Any]) -> None:
    """Reject resumes that would invalidate the saved optimizer schedule or metric."""
    saved = checkpoint.get("config")
    if not isinstance(saved, Mapping):
        raise ValueError("resume checkpoint has no training config")
    required_keys = (
        "model_name", "seed", "max_length", "train_batch_size", "train_limit",
        "eval_limit", "epochs", "encoder_lr", "head_lr", "weight_decay",
        "warmup_ratio", "gradient_clip", "use_fgm", "fgm_epsilon",
        "use_rdrop", "rdrop_alpha",
    )
    for key in required_keys:
        if saved.get(key) != getattr(config, key):
            raise ValueError(f"resume config differs from checkpoint for {key}")
    if int(checkpoint.get("epoch", 0)) >= config.epochs:
        raise ValueError("resume checkpoint already reached the requested number of epochs")


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    """Keep the previous checkpoint usable if serialization is interrupted."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    stale_epochs: int = 0,
    best_metrics: Mapping[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_selection_metric": best_metric,
        "selection_metric": SELECTION_METRIC,
        "best_metrics": dict(best_metrics or metrics),
        "stale_epochs": stale_epochs,
        "label_names": list(label_names),
        "config": asdict(config),
        "metric_definitions": METRIC_DEFINITIONS,
    }
    atomic_torch_save(checkpoint, output_dir / "checkpoint_last.pt")
    write_json(output_dir / "last_metrics.json", dict(metrics))
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
    write_json(output_dir / "config.json", asdict(config))


def run_training(config: TrainConfig) -> dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    validate_config(config)
    set_seed(config.seed, config.device)
    device = resolve_device(config)
    output_dir = Path(config.output_dir)

    dataset = load_dataset("klue", "ner")
    label_names = labels_from_klue_dataset(dataset)
    label_maps = build_label_maps(label_names)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_rows = limit_dataset(dataset["train"], config.train_limit)
    valid_rows = limit_dataset(dataset["validation"], config.eval_limit)

    train_dataset = KlueNerDataset(
        train_rows,
        tokenizer,
        label_to_id=label_maps.label_to_id,
        id_to_label=label_maps.id_to_label,
        max_length=config.max_length,
    )
    valid_dataset = KlueNerDataset(
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

    model = core.KoElectraNER(config.model_name, len(label_names), o_label_id=label_maps.label_to_id["O"])
    model.to(device)
    if config.init_from_legacy_weights:
        state = extract_model_state_dict(load_checkpoint(config.init_from_legacy_weights, map_location=device))
        model.load_state_dict(state, strict=True)

    optimizer = build_optimizer(model, config)
    total_steps = max(1, math.ceil(len(train_loader)) * config.epochs)
    scheduler = make_linear_scheduler(optimizer, total_steps=total_steps, warmup_ratio=config.warmup_ratio)

    start_epoch = 1
    best_metric = -1.0
    best_metrics: dict[str, Any] = {}
    stale_epochs = 0
    if config.resume_checkpoint:
        checkpoint = load_checkpoint(config.resume_checkpoint, map_location=device)
        validate_resume_config(config, checkpoint)
        if checkpoint.get("selection_metric") != SELECTION_METRIC:
            raise ValueError("resume checkpoint was selected using a different metric")
        model.load_state_dict(extract_model_state_dict(checkpoint), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_metric = float(checkpoint.get("best_selection_metric", best_metric))
        best_metrics = dict(checkpoint.get("best_metrics", {}))
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    for epoch in range(start_epoch, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, config, device)
        metrics = evaluate_model(model, valid_loader, label_maps.id_to_label, device)
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
            f"validation_klue_macro_f1={metrics['validation_klue_official_entity_macro_f1']:.6f} "
            f"validation_macro_f1={metrics['validation_entity_macro_f1']:.6f} "
            f"validation_micro_f1={metrics['validation_entity_micro_f1']:.6f}",
            flush=True,
        )
        if stale_epochs >= config.early_stopping_patience:
            break

    return best_metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = config_from_args(args)
    metrics = run_training(config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
