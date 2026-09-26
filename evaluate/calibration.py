"""Emission-only confidence diagnostics for Korean NER.

This script measures whether the token classifier emissions are calibrated.
It intentionally does not calibrate the final CRF-decoded sequence, because
CRF transitions can choose a different label than the local emission argmax.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from korean_ner import (
    DEFAULT_LABEL_LIST,
    DEFAULT_MAX_LENGTH,
    align_text,
    build_label_maps,
    labels_from_klue_dataset,
    load_model_state,
)


MODEL_PATH = REPO_ROOT / "weights" / "best_model.pt"
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
OUTPUT_DIR = REPO_ROOT / "runs" / "calibration"


def compute_calibration_metrics(confidences: Sequence[float], accuracies: Sequence[int], num_bins: int = 10) -> dict:
    """Compute Expected Calibration Error (ECE), MCE, and bin statistics."""

    confidences = np.asarray(confidences, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)
    if confidences.shape != accuracies.shape:
        raise ValueError("confidences and accuracies must have the same shape")
    if confidences.size == 0:
        return {"ece": 0.0, "mce": 0.0, "bin_stats": []}

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_stats = []

    for bin_index, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
        if bin_index == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(in_bin.mean())

        if prop_in_bin:
            avg_confidence = float(confidences[in_bin].mean())
            avg_accuracy = float(accuracies[in_bin].mean())
            bin_error = abs(avg_accuracy - avg_confidence)
            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)

            bin_stats.append(
                {
                    "bin_lower": float(bin_lower),
                    "bin_upper": float(bin_upper),
                    "avg_confidence": avg_confidence,
                    "avg_accuracy": avg_accuracy,
                    "count": int(in_bin.sum()),
                    "proportion": prop_in_bin,
                }
            )
        else:
            bin_stats.append(
                {
                    "bin_lower": float(bin_lower),
                    "bin_upper": float(bin_upper),
                    "avg_confidence": None,
                    "avg_accuracy": None,
                    "count": 0,
                    "proportion": 0.0,
                }
            )

    return {"ece": float(ece), "mce": float(mce), "bin_stats": bin_stats}


def collect_emission_diagnostics(
    emissions: torch.Tensor,
    char_indices: Sequence[int | None],
    gold_char_labels: Sequence[int],
    id_to_label: dict[int, str],
) -> list[dict]:
    """Collect confidence/correctness pairs for first subtokens of original characters."""

    if emissions.ndim != 2:
        raise ValueError("emissions must have shape (seq_len, num_labels)")
    if len(char_indices) != emissions.shape[0]:
        raise ValueError("char_indices length must match emissions sequence length")

    probs = F.softmax(emissions, dim=-1)
    confidences, pred_ids = probs.max(dim=-1)
    rows = []
    seen_chars: set[int] = set()

    for token_index, char_index in enumerate(char_indices):
        if char_index is None or char_index in seen_chars:
            continue
        seen_chars.add(char_index)
        if char_index >= len(gold_char_labels):
            continue

        pred_id = int(pred_ids[token_index].item())
        gold_id = int(gold_char_labels[char_index])
        gold_label = id_to_label[gold_id]
        pred_label = id_to_label[pred_id]
        rows.append(
            {
                "token_index": token_index,
                "char_index": char_index,
                "confidence": float(confidences[token_index].item()),
                "correct": int(pred_id == gold_id),
                "gold_id": gold_id,
                "pred_id": pred_id,
                "gold_label": gold_label,
                "pred_label": pred_label,
                "gold_group": "O" if gold_label == "O" else "entity",
                "gold_entity_type": None if gold_label == "O" else gold_label.split("-", 1)[1],
            }
        )

    return rows


def summarize_rows(rows: Sequence[dict], *, num_bins: int = 10) -> dict:
    """Summarize emission diagnostics for all tokens and entity-only tokens."""

    confidences = np.array([row["confidence"] for row in rows], dtype=float)
    correct = np.array([row["correct"] for row in rows], dtype=float)
    groups = Counter(row["gold_group"] for row in rows)
    pred_groups = Counter("O" if row["pred_label"] == "O" else "entity" for row in rows)

    overall = compute_calibration_metrics(confidences, correct, num_bins=num_bins)
    overall.update(
        {
            "accuracy": float(correct.mean()) if len(correct) else 0.0,
            "mean_confidence": float(confidences.mean()) if len(confidences) else 0.0,
            "total_tokens": int(len(rows)),
            "gold_o_tokens": int(groups["O"]),
            "gold_entity_tokens": int(groups["entity"]),
            "pred_o_tokens": int(pred_groups["O"]),
            "pred_entity_tokens": int(pred_groups["entity"]),
        }
    )

    entity_rows = [row for row in rows if row["gold_group"] == "entity"]
    entity_conf = np.array([row["confidence"] for row in entity_rows], dtype=float)
    entity_corr = np.array([row["correct"] for row in entity_rows], dtype=float)
    entity_only = compute_calibration_metrics(entity_conf, entity_corr, num_bins=num_bins)
    entity_only.update(
        {
            "accuracy": float(entity_corr.mean()) if len(entity_corr) else 0.0,
            "mean_confidence": float(entity_conf.mean()) if len(entity_conf) else 0.0,
            "total_tokens": int(len(entity_rows)),
        }
    )

    per_entity = {}
    by_entity_type: dict[str, list[dict]] = defaultdict(list)
    for row in entity_rows:
        by_entity_type[row["gold_entity_type"]].append(row)
    for entity_type, entity_type_rows in sorted(by_entity_type.items()):
        conf = np.array([row["confidence"] for row in entity_type_rows], dtype=float)
        corr = np.array([row["correct"] for row in entity_type_rows], dtype=float)
        metrics = compute_calibration_metrics(conf, corr, num_bins=num_bins)
        per_entity[entity_type] = {
            "ece": metrics["ece"],
            "mce": metrics["mce"],
            "mean_confidence": float(conf.mean()),
            "accuracy": float(corr.mean()),
            "count": int(len(entity_type_rows)),
        }

    return {
        "diagnostic": "emission_classifier_only",
        "note": (
            "Confidence is max softmax over local token emissions and correctness is the "
            "emission argmax against the original character gold label. These numbers do "
            "not calibrate final CRF-decoded sequence predictions."
        ),
        "overall": overall,
        "entity_only": entity_only,
        "per_entity": per_entity,
    }


def plot_reliability_diagram(bin_stats: Sequence[dict], output_path: str | Path, *, title: str, label: str) -> None:
    """Create an emission reliability diagram."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    confidences = []
    accuracies = []
    for bin_stat in bin_stats:
        if bin_stat["avg_confidence"] is not None:
            confidences.append(bin_stat["avg_confidence"])
            accuracies.append(bin_stat["avg_accuracy"])

    if confidences:
        ax1.bar(confidences, accuracies, width=0.08, alpha=0.7, color="#45B7D1", edgecolor="black", label=label)
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.set_xlabel("Mean emission confidence")
    ax1.set_ylabel("Emission argmax accuracy")
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    bin_centers = [(bs["bin_lower"] + bs["bin_upper"]) / 2 for bs in bin_stats]
    bin_counts = [bs["count"] for bs in bin_stats]
    ax2.bar(bin_centers, bin_counts, width=0.09, color="#96CEB4", edgecolor="black")
    ax2.set_xlabel("Emission confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence distribution")
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_confidence_accuracy_curve(
    confidences: Sequence[float],
    accuracies: Sequence[int],
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot emission argmax accuracy as a function of confidence threshold."""

    confidences = np.asarray(confidences, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)
    thresholds = np.linspace(0, 0.99, 50)

    accs_at_threshold = []
    coverage_at_threshold = []
    for thresh in thresholds:
        mask = confidences >= thresh
        if mask.sum() > 0:
            accs_at_threshold.append(float(accuracies[mask].mean()))
            coverage_at_threshold.append(float(mask.mean()))
        else:
            accs_at_threshold.append(np.nan)
            coverage_at_threshold.append(0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(thresholds, accs_at_threshold, "b-", linewidth=2)
    ax1.set_xlabel("Emission confidence threshold")
    ax1.set_ylabel("Emission argmax accuracy above threshold")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.plot(thresholds, coverage_at_threshold, "g-", linewidth=2)
    ax2.set_xlabel("Emission confidence threshold")
    ax2.set_ylabel("Coverage fraction above threshold")
    ax2.set_title("Coverage vs emission confidence")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def iter_limited(dataset: Iterable, limit: int | None) -> Iterable:
    for index, example in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield index, example


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--limit", type=int, default=None, help="Optionally analyze only the first N validation samples.")
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"), help="Defaults to CPU for safer diagnostics.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from korean_ner import KoElectraNER

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("Emission-only confidence diagnostics for Korean NER")
    print("This does not calibrate final CRF-decoded sequence predictions.")
    print("=" * 72)
    print(f"Device: {device}")
    if args.limit is not None:
        print(f"Limit: first {args.limit} validation samples")

    print("\nLoading KLUE NER validation set...")
    dataset = load_dataset("klue", "ner")
    val_data = dataset["validation"]
    label_maps = build_label_maps(labels_from_klue_dataset(dataset) or DEFAULT_LABEL_LIST)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = KoElectraNER(args.model_name, len(label_maps.names), o_label_id=label_maps.label_to_id["O"])
    load_model_state(model, args.model_path, map_location=device)
    model.to(device)
    model.eval()

    rows = []
    skipped_overflow = 0
    total_to_process = min(len(val_data), args.limit) if args.limit is not None else len(val_data)
    print(f"\nAnalyzing {total_to_process} samples...")

    for index, example in iter_limited(val_data, args.limit):
        if index % 500 == 0:
            print(f"  Processing {index}/{total_to_process}...")

        chars = example["tokens"]
        true_labels = [int(label) for label in example["ner_tags"]]
        text = "".join(chars)

        try:
            encoding = align_text(
                text,
                tokenizer,
                true_labels,
                max_length=args.max_length,
                label_to_id=label_maps.label_to_id,
                id_to_label=label_maps.id_to_label,
            )
        except ValueError as exc:
            if "exceeds max_length" not in str(exc):
                raise
            skipped_overflow += 1
            continue

        input_ids = torch.tensor([encoding.input_ids], dtype=torch.long, device=device)
        attention_mask = torch.tensor([encoding.attention_mask], dtype=torch.long, device=device)

        with torch.no_grad():
            emissions = model.emissions(input_ids, attention_mask)[0].detach().cpu()

        rows.extend(collect_emission_diagnostics(emissions, encoding.char_indices, true_labels, label_maps.id_to_label))

    summary = summarize_rows(rows, num_bins=args.num_bins)
    summary["skipped_overflow_samples"] = skipped_overflow

    print("\n" + "=" * 40)
    print("EMISSION CLASSIFIER CALIBRATION")
    print("=" * 40)
    print(summary["note"])
    print(f"\nTotal original-character tokens analyzed: {summary['overall']['total_tokens']}")
    print(f"Gold O tokens: {summary['overall']['gold_o_tokens']}")
    print(f"Gold entity tokens: {summary['overall']['gold_entity_tokens']}")
    print(f"Skipped over-length samples: {skipped_overflow}")
    print(f"Overall emission argmax accuracy: {summary['overall']['accuracy']:.4f}")
    print(f"Overall mean emission confidence: {summary['overall']['mean_confidence']:.4f}")
    print(f"Overall emission ECE: {summary['overall']['ece']:.4f}")
    print(f"Entity-only emission ECE: {summary['entity_only']['ece']:.4f}")

    print("\nPer-entity emission calibration:")
    for entity_type, metrics in summary["per_entity"].items():
        print(
            f"  {entity_type}: ECE={metrics['ece']:.4f}, "
            f"acc={metrics['accuracy']:.4f}, conf={metrics['mean_confidence']:.4f}, count={metrics['count']}"
        )

    all_confidences = [row["confidence"] for row in rows]
    all_correct = [row["correct"] for row in rows]
    entity_rows = [row for row in rows if row["gold_group"] == "entity"]
    entity_confidences = [row["confidence"] for row in entity_rows]
    entity_correct = [row["correct"] for row in entity_rows]

    print("\n" + "=" * 40)
    print("GENERATING EMISSION-ONLY VISUALIZATIONS")
    print("=" * 40)
    plot_reliability_diagram(
        summary["overall"]["bin_stats"],
        output_dir / "calibration_emission_reliability_all.png",
        title="Emission classifier reliability (all original-character tokens)",
        label="Emission classifier",
    )
    plot_confidence_accuracy_curve(
        all_confidences,
        all_correct,
        output_dir / "calibration_emission_accuracy_curve_all.png",
        title="Emission argmax accuracy vs confidence (all tokens)",
    )
    if entity_rows:
        plot_reliability_diagram(
            summary["entity_only"]["bin_stats"],
            output_dir / "calibration_emission_reliability_entity_only.png",
            title="Emission classifier reliability (entity-only gold tokens)",
            label="Emission classifier, entity-only",
        )
        plot_confidence_accuracy_curve(
            entity_confidences,
            entity_correct,
            output_dir / "calibration_emission_accuracy_curve_entity_only.png",
            title="Emission argmax accuracy vs confidence (entity-only)",
        )

    with open(output_dir / "calibration_results.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_dir / 'calibration_results.json'}")


if __name__ == "__main__":
    main()
