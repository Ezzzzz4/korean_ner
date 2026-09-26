"""Evaluate the Korean NER model on original KLUE character sequences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from korean_ner import (
    DEFAULT_MAX_LENGTH,
    align_text,
    build_label_maps,
    labels_from_klue_dataset,
    load_model_state,
    score_strict_iob2,
    token_predictions_to_char_labels,
)
from korean_ner.decode import token_predictions_to_raw_char_labels
from korean_ner.metrics import score_klue_benchmark_entity_macro_f1


DEFAULT_MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights" / "best_model.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "evaluation"
DEFAULT_BATCH_SIZE = 32


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu", help="Torch device. Defaults to CPU and never auto-selects CUDA.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--limit", type=int, default=None, help="Optional small-sample smoke limit.")
    return parser.parse_args()


def klue_text(example: dict[str, Any]) -> str:
    """Return the original KLUE character sequence, preserving spaces."""

    tokens = example["tokens"]
    if isinstance(tokens, str):
        return tokens
    return "".join(tokens)


def label_names_from_ids(label_ids: Iterable[int], id_to_label: dict[int, str]) -> list[str]:
    return [id_to_label[int(label_id)] for label_id in label_ids]


def build_eval_records(
    examples: Sequence[dict[str, Any]],
    tokenizer: Any,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> list[dict[str, Any]]:
    """Tokenize KLUE examples without dropping spaces or silently truncating."""

    records: list[dict[str, Any]] = []
    for example in examples:
        text = klue_text(example)
        label_ids = [int(label_id) for label_id in example["ner_tags"]]
        encoding = align_text(
            text,
            tokenizer,
            label_ids,
            max_length=max_length,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
        )
        records.append(
            {
                "input_ids": torch.tensor(encoding.input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(encoding.attention_mask, dtype=torch.long),
                "labels": torch.tensor(encoding.labels or [], dtype=torch.long),
                "char_indices": encoding.char_indices,
                "text": text,
                "true_char_labels": label_names_from_ids(label_ids, id_to_label),
                "tokens": encoding.tokens,
            }
        )
    return records


def collate_eval_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "char_indices": [item["char_indices"] for item in batch],
        "texts": [item["text"] for item in batch],
        "true_char_labels": [item["true_char_labels"] for item in batch],
        "tokens": [item["tokens"] for item in batch],
    }


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows while keeping all-zero rows at zero."""

    matrix = matrix.astype(float, copy=False)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)


def confusion_matrix_from_pairs(pairs: Iterable[tuple[str, str]], labels: Sequence[str]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for true_label, pred_label in pairs:
        matrix[label_to_index[true_label], label_to_index[pred_label]] += 1
    return matrix


def entity_type(label: str) -> str:
    return label.split("-", 1)[1] if "-" in label else label


def run_model_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    id_to_label: dict[int, str],
    *,
    device: torch.device,
) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[str], list[tuple[str, str]], list[str], list[str]]:
    all_true: list[list[str]] = []
    all_pred: list[list[str]] = []
    all_tokens: list[list[str]] = []
    texts: list[str] = []
    char_pairs: list[tuple[str, str]] = []
    benchmark_true: list[str] = []
    benchmark_pred: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            predictions = model(input_ids, attention_mask)

            for pred_ids, char_indices, text, true_labels, tokens, attention_mask_row in zip(
                predictions,
                batch["char_indices"],
                batch["texts"],
                batch["true_char_labels"],
                batch["tokens"],
                batch["attention_mask"].tolist(),
            ):
                active_length = int(sum(attention_mask_row))
                if active_length < 2:
                    raise ValueError("encoded prediction must include CLS and SEP tokens")
                active_pred_ids = list(pred_ids[:active_length])
                raw_pred_labels = token_predictions_to_raw_char_labels(
                    active_pred_ids,
                    char_indices[:active_length],
                    id_to_label,
                    text_length=len(text),
                )
                pred_labels = token_predictions_to_char_labels(
                    active_pred_ids,
                    char_indices[:active_length],
                    id_to_label,
                    text_length=len(text),
                )
                all_true.append(true_labels)
                all_pred.append(pred_labels)
                all_tokens.append(tokens)
                texts.append(text)
                char_pairs.extend(zip(true_labels, pred_labels))
                benchmark_true.append("O")
                benchmark_pred.append(id_to_label[int(active_pred_ids[0])])
                for char, true_label, raw_pred_label in zip(text, true_labels, raw_pred_labels):
                    if char == " ":
                        continue
                    benchmark_true.append(true_label)
                    benchmark_pred.append(raw_pred_label)
                benchmark_true.append("O")
                benchmark_pred.append(id_to_label[int(active_pred_ids[-1])])

    return all_true, all_pred, all_tokens, texts, char_pairs, benchmark_true, benchmark_pred


def metrics_payload(
    *,
    split: str,
    checkpoint: Path,
    max_length: int,
    sample_count: int,
    scores: Any,
    klue_benchmark_entity_macro_f1: float,
) -> dict[str, Any]:
    micro_report = scores.report.get("micro avg", {})
    macro_report = scores.report.get("macro avg", {})
    weighted_report = scores.report.get("weighted avg", {})
    per_entity = {
        label: values
        for label, values in scores.report.items()
        if label not in {"micro avg", "macro avg", "weighted avg"}
    }
    return {
        "split": split,
        "checkpoint": str(checkpoint),
        "max_length": max_length,
        "sample_count": sample_count,
        "primary_metric": "klue_official_entity_macro_f1",
        "metric_definitions": {
            "entity_macro_f1_strict_iob2": "All-character entity macro F1 from seqeval strict IOB2 after BIO repair of CRF predictions; includes spaces.",
            "entity_micro_f1_strict_iob2": "All-character entity micro F1 from seqeval strict IOB2 after BIO repair of CRF predictions; includes spaces.",
            "klue_official_entity_macro_f1": (
                "Official KLUE baseline-compatible entity macro F1: raw CRF labels before BIO repair, "
                "literal ASCII spaces removed, predicted CLS/SEP labels kept against gold O, "
                "all examples flattened into one strict IOB2 seqeval sequence."
            ),
            "character_confusion": "One BIO tag per original KLUE character, including spaces.",
        },
        "overall": {
            "entity_macro_f1_strict_iob2": float(scores.f1_macro),
            "entity_micro_f1_strict_iob2": float(scores.f1_micro),
            "klue_official_entity_macro_f1": float(klue_benchmark_entity_macro_f1),
            "klue_official_entity_macro_f1_percent": float(klue_benchmark_entity_macro_f1 * 100.0),
            "entity_micro_precision_strict_iob2": float(scores.precision_micro),
            "entity_micro_recall_strict_iob2": float(scores.recall_micro),
            "support": int(micro_report.get("support", 0)),
            "macro_precision": float(macro_report.get("precision", 0.0)),
            "macro_recall": float(macro_report.get("recall", 0.0)),
            "weighted_f1": float(weighted_report.get("f1-score", 0.0)),
        },
        "per_entity": per_entity,
    }


def write_classification_report(output_dir: Path, metrics: dict[str, Any], report: dict[str, Any]) -> None:
    lines = [
        "=" * 72,
        "KLUE NER EVALUATION RESULTS",
        "=" * 72,
        "",
        f"Split: {metrics['split']}",
        f"Checkpoint: {metrics['checkpoint']}",
        f"Samples: {metrics['sample_count']}",
        "",
        "PRIMARY METRIC (KLUE BASELINE-COMPATIBLE)",
        f"KLUE official-compatible entity macro F1: {metrics['overall']['klue_official_entity_macro_f1']:.6f}",
        f"KLUE official-compatible entity macro F1 (%): {metrics['overall']['klue_official_entity_macro_f1_percent']:.4f}",
        "",
        "SECONDARY METRICS (ALL CHARACTERS, INCLUDING SPACES)",
        f"Entity macro F1, strict IOB2: {metrics['overall']['entity_macro_f1_strict_iob2']:.6f}",
        "",
        f"Entity micro F1, strict IOB2: {metrics['overall']['entity_micro_f1_strict_iob2']:.6f}",
        "",
        "PER-ENTITY REPORT",
        json.dumps(report, ensure_ascii=False, indent=2),
    ]
    (output_dir / "classification_report.txt").write_text("\n".join(lines), encoding="utf-8")


def save_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_per_entity_report(output_dir: Path, report: dict[str, Any], entity_types: Sequence[str], macro_f1: float, split: str) -> None:
    entities = [entity for entity in entity_types if entity in report]
    precisions = [report[entity]["precision"] for entity in entities]
    recalls = [report[entity]["recall"] for entity in entities]
    f1_scores = [report[entity]["f1-score"] for entity in entities]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(entities))
    width = 0.25
    bars1 = ax.bar(x - width, precisions, width, label="Precision", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x, recalls, width, label="Recall", color="#2ecc71", alpha=0.85)
    bars3 = ax.bar(x + width, f1_scores, width, label="F1", color="#e74c3c", alpha=0.85)
    ax.set_xlabel("Entity Type", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{split} All-Character Entity F1 by Type (BIO-Repaired Predictions)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(entities, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=macro_f1, color="#9b59b6", linestyle="--", linewidth=2,
               label=f"All-character macro F1 ({macro_f1:.3f})")
    ax.legend(loc="lower right", fontsize=10)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(output_dir / "per_entity_f1.png", bbox_inches="tight", facecolor="white")
    plt.close()


def plot_confusion_matrices(
    output_dir: Path,
    *,
    char_pairs: Sequence[tuple[str, str]],
    label_names: Sequence[str],
    entity_types: Sequence[str],
    split: str,
) -> None:
    label_matrix = normalized_rows(confusion_matrix_from_pairs(char_pairs, label_names))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        label_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        cbar_kws={"label": "Proportion of True Character Labels"},
    )
    ax.set_xlabel("Predicted BIO Label", fontsize=12)
    ax.set_ylabel("True BIO Label", fontsize=12)
    ax.set_title(f"{split} Character-Level BIO Label Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight", facecolor="white")
    plt.close()

    entity_labels = ["O", *entity_types]
    entity_pairs = [(entity_type(true), entity_type(pred)) for true, pred in char_pairs]
    entity_matrix = normalized_rows(confusion_matrix_from_pairs(entity_pairs, entity_labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        entity_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        xticklabels=entity_labels,
        yticklabels=entity_labels,
        ax=ax,
        cbar_kws={"label": "Proportion of True Character Types"},
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted Entity Type", fontsize=12)
    ax.set_ylabel("True Entity Type", fontsize=12)
    ax.set_title(f"{split} Character-Level Entity-Type Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "entity_confusion_matrix.png", bbox_inches="tight", facecolor="white")
    plt.close()


def write_examples(
    output_dir: Path,
    texts: Sequence[str],
    predictions: Sequence[Sequence[str]],
    labels: Sequence[Sequence[str]],
    *,
    max_examples: int = 3,
) -> None:
    correct: list[tuple[str, Sequence[str], Sequence[str]]] = []
    errors: list[tuple[str, Sequence[str], Sequence[str]]] = []
    for text, pred, gold in zip(texts, predictions, labels):
        has_entity = any(label != "O" for label in gold)
        if not has_entity:
            continue
        if list(pred) == list(gold) and len(correct) < max_examples:
            correct.append((text, pred, gold))
        elif list(pred) != list(gold) and len(errors) < max_examples:
            errors.append((text, pred, gold))

    lines = ["=" * 70, "CHARACTER-LEVEL EXAMPLE PREDICTIONS", "=" * 70, ""]
    for title, examples in [("CORRECT PREDICTIONS", correct), ("PREDICTIONS WITH ERRORS", errors)]:
        lines.extend([f"--- {title} ---", ""])
        for text, pred, gold in examples:
            chars = list(text)
            limit = min(80, len(chars))
            lines.append(f"Text:   {text[:80]}")
            lines.append(f"Chars:  {' '.join(chars[:limit])}")
            lines.append(f"True:   {' '.join(gold[:limit])}")
            lines.append(f"Pred:   {' '.join(pred[:limit])}")
            if title.endswith("ERRORS"):
                diff = ["^^^" if pred_label != true_label else "..." for pred_label, true_label in zip(pred[:limit], gold[:limit])]
                lines.append(f"Diff:   {' '.join(diff)}")
            lines.append("")
    (output_dir / "example_predictions.txt").write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataset = load_dataset("klue", "ner")
    if args.split not in dataset:
        raise ValueError(f"unknown split {args.split!r}; available splits: {', '.join(dataset.keys())}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    label_maps = build_label_maps(labels_from_klue_dataset(dataset))
    entity_types = sorted({label.split("-", 1)[1] for label in label_maps.names if label != "O"})

    split_dataset = dataset[args.split]
    if args.limit is not None:
        split_dataset = split_dataset.select(range(min(args.limit, len(split_dataset))))
    records = build_eval_records(
        list(split_dataset),
        tokenizer,
        dict(label_maps.label_to_id),
        dict(label_maps.id_to_label),
        max_length=args.max_length,
    )
    loader = DataLoader(records, batch_size=args.batch_size, collate_fn=collate_eval_batch)

    from korean_ner import KoElectraNER

    model = KoElectraNER(
        args.model_name,
        len(label_maps.names),
        o_label_id=label_maps.label_to_id["O"],
        num_samples=5,
    )
    load_model_state(model, args.checkpoint, map_location=device)
    model.to(device)

    true_labels, pred_labels, _tokens, texts, char_pairs, benchmark_true, benchmark_pred = run_model_predictions(
        model,
        loader,
        dict(label_maps.id_to_label),
        device=device,
    )
    scores = score_strict_iob2(true_labels, pred_labels)
    benchmark_f1 = score_klue_benchmark_entity_macro_f1(benchmark_true, benchmark_pred)
    metrics = metrics_payload(
        split=args.split,
        checkpoint=args.checkpoint,
        max_length=args.max_length,
        sample_count=len(records),
        scores=scores,
        klue_benchmark_entity_macro_f1=benchmark_f1,
    )
    save_metrics(output_dir, metrics)
    write_classification_report(output_dir, metrics, scores.report)
    plot_per_entity_report(output_dir, scores.report, entity_types, scores.f1_macro, args.split)
    plot_confusion_matrices(
        output_dir,
        char_pairs=char_pairs,
        label_names=label_maps.names,
        entity_types=entity_types,
        split=args.split,
    )
    write_examples(output_dir, texts, pred_labels, true_labels)
    return metrics


def main() -> None:
    args = parse_args()
    print("=" * 72)
    print("KOREAN NER VALIDATION: KLUE-COMPATIBLE AND ALL-CHARACTER METRICS")
    print("=" * 72)
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    metrics = evaluate(args)
    print(f"Primary KLUE-compatible entity macro F1: {metrics['overall']['klue_official_entity_macro_f1']:.6f}")
    print(f"All-character entity macro F1: {metrics['overall']['entity_macro_f1_strict_iob2']:.6f}")
    print(f"All-character entity micro F1: {metrics['overall']['entity_micro_f1_strict_iob2']:.6f}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
