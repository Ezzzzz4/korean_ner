"""Reload a KF-DeBERTa checkpoint and score a KLUE NER data split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from korean_ner.labels import build_label_maps, labels_from_klue_dataset
from train_deberta import KlueSubwordDataset, TrainConfig, collate_batch, evaluate_model, safe_torch_load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch size and limit must be positive")

    from datasets import load_dataset
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    checkpoint = safe_torch_load(args.checkpoint, map_location="cpu")
    saved_config = TrainConfig(**checkpoint["config"])
    dataset = load_dataset("klue", "ner")
    label_names = labels_from_klue_dataset(dataset)
    if list(checkpoint["label_names"]) != list(label_names):
        raise ValueError("checkpoint labels differ from KLUE labels")
    label_maps = build_label_maps(label_names)
    tokenizer = AutoTokenizer.from_pretrained(
        saved_config.model_name, revision=saved_config.model_revision, use_fast=True
    )
    hf_config = AutoConfig.from_pretrained(
        saved_config.model_name,
        revision=saved_config.model_revision,
        num_labels=len(label_names),
        id2label={index: label for index, label in enumerate(label_names)},
        label2id={label: index for index, label in enumerate(label_names)},
    )
    model = AutoModelForTokenClassification.from_config(hf_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    rows = dataset[args.split]
    if args.limit is not None:
        if args.split == "train":
            rows = rows.shuffle(seed=42)
        rows = rows.select(range(min(args.limit, len(rows))))
    loader = DataLoader(
        KlueSubwordDataset(
            rows,
            tokenizer,
            label_to_id=label_maps.label_to_id,
            id_to_label=label_maps.id_to_label,
            max_length=saved_config.max_length,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    eval_config = TrainConfig(**{**checkpoint["config"], "device": args.device, "bf16": False})
    metrics = evaluate_model(model, loader, label_maps.id_to_label, eval_config, device)
    if args.split == "train":
        metrics = {key.replace("validation_", "train_", 1): value for key, value in metrics.items()}
    with args.checkpoint.open("rb") as checkpoint_file:
        checkpoint_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "dataset": "klue/ner",
        "split": args.split,
        "examples": len(rows),
        "sample_seed": 42 if args.split == "train" and args.limit is not None else None,
        "saved_epoch": checkpoint.get("metrics", {}).get("epoch"),
        **metrics,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
