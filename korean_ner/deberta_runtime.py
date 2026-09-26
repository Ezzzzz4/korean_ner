"""Safe, explicit inference for the fine-tuned KF-DeBERTa KLUE NER model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .decode import extract_bio_spans, repair_bio_sequence
from .labels import build_label_maps
from .subword import encode_subword_offsets, project_token_predictions_to_chars


def _labels_with_entity_spaces(text: str, raw_labels: list[str]) -> list[str]:
    """Restore whitespace inside continuous entities for original-text spans."""
    positions = [index for index, char in enumerate(text) if not char.isspace()]
    compact = repair_bio_sequence([raw_labels[index] for index in positions])
    labels = ["O"] * len(text)
    for position, label in zip(positions, compact):
        labels[position] = label
    for index in range(len(positions) - 1):
        left_position = positions[index]
        right_position = positions[index + 1]
        if right_position == left_position + 1:
            continue
        left_label = compact[index]
        right_label = compact[index + 1]
        if left_label != "O" and right_label.startswith("I-") and left_label[2:] == right_label[2:]:
            for position in range(left_position + 1, right_position):
                labels[position] = right_label
    return labels


@dataclass
class DebertaRuntime:
    tokenizer: Any
    model: torch.nn.Module
    id_to_label: dict[int, str]
    max_length: int
    device: torch.device

    @torch.no_grad()
    def predict_labels(self, text: str) -> list[str]:
        encoding = encode_subword_offsets(text, self.tokenizer, max_length=self.max_length)
        inputs = {
            "input_ids": torch.tensor([encoding.input_ids], device=self.device),
            "attention_mask": torch.tensor([encoding.attention_mask], device=self.device),
        }
        if any(encoding.token_type_ids):
            inputs["token_type_ids"] = torch.tensor([encoding.token_type_ids], device=self.device)
        predictions = self.model(**inputs).logits.argmax(dim=-1)[0].tolist()
        labels = project_token_predictions_to_chars(
            predictions,
            encoding.offset_mapping,
            encoding.special_tokens_mask,
            self.id_to_label,
            text,
        )
        return _labels_with_entity_spaces(text, labels)

    def predict_entities(self, text: str) -> list[dict[str, Any]]:
        return [
            {"text": entity.text, "label": entity.label, "start": entity.start, "end": entity.end}
            for entity in extract_bio_spans(text, self.predict_labels(text), repair=False)
        ]


def load_deberta_runtime(checkpoint_path: str | Path, *, device: str = "cpu") -> DebertaRuntime:
    """Load trusted local fine-tuned weights without unpickling arbitrary objects."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = payload["config"]
    model_name = config["model_name"]
    revision = config["model_revision"]
    if model_name != "kakaobank/kf-deberta-base" or revision != "363b171d71443b0874b0bf9cea053eb5b1650633":
        raise ValueError("checkpoint model identity differs from the supported KF-DeBERTa revision")
    label_names = list(payload["label_names"])
    label_maps = build_label_maps(label_names)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("KF-DeBERTa tokenizer must provide offset mappings")
    hf_config = AutoConfig.from_pretrained(
        model_name,
        revision=revision,
        num_labels=len(label_names),
        id2label=label_maps.id_to_label,
        label2id=label_maps.label_to_id,
    )
    model = AutoModelForTokenClassification.from_config(hf_config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    target = torch.device(device)
    model.to(target).eval()
    return DebertaRuntime(tokenizer, model, label_maps.id_to_label, int(config["max_length"]), target)
