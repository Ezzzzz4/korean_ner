"""Subword offset alignment for KLUE NER experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


IGNORE_INDEX = -100


@dataclass(frozen=True)
class SubwordEncoding:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    labels: list[int] | None
    offset_mapping: list[tuple[int, int]]
    special_tokens_mask: list[int]
    text: str


def _to_int_list(values: object) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [int(value) for value in values]  # type: ignore[arg-type]


def _normalize_offsets(offsets: object) -> list[tuple[int, int]]:
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    return [(int(start), int(end)) for start, end in offsets]  # type: ignore[union-attr]


def _first_non_space_index(text: str, start: int, end: int) -> int | None:
    for index in range(max(0, start), min(len(text), end)):
        if not text[index].isspace():
            return index
    return None


def encode_subword_offsets(
    text: str,
    tokenizer: object,
    labels: Sequence[int] | None = None,
    *,
    label_to_id: Mapping[str, int] | None = None,
    id_to_label: Mapping[int, str] | None = None,
    max_length: int = 128,
) -> SubwordEncoding:
    """Tokenize full text with offsets and align labels from original chars.

    Each content token receives the label of the first non-space original
    character covered by its offset. CLS/SEP are trained as O, matching the
    KLUE baseline; all-space offsets are ignored for training.
    """

    if labels is not None and len(labels) != len(text):
        raise ValueError(f"labels length {len(labels)} does not match text length {len(text)}")

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=False,
    )
    input_ids = _to_int_list(encoded["input_ids"])
    attention_mask = _to_int_list(encoded.get("attention_mask", [1] * len(input_ids)))
    token_type_ids = _to_int_list(encoded.get("token_type_ids", [0] * len(input_ids)))
    special_tokens_mask = _to_int_list(encoded.get("special_tokens_mask", [0] * len(input_ids)))
    offsets = _normalize_offsets(encoded["offset_mapping"])

    if not (len(input_ids) == len(attention_mask) == len(token_type_ids) == len(special_tokens_mask) == len(offsets)):
        raise ValueError("tokenizer returned inconsistent sequence field lengths")
    if len(input_ids) > max_length:
        raise ValueError(
            f"encoded length {len(input_ids)} exceeds max_length {max_length}; "
            "increase max_length instead of silently truncating"
        )

    aligned_labels: list[int] | None = None
    if labels is not None:
        if label_to_id is None or id_to_label is None:
            raise ValueError("label_to_id and id_to_label are required when labels are provided")
        aligned_labels = []
        for offset, special in zip(offsets, special_tokens_mask):
            if special:
                aligned_labels.append(int(label_to_id["O"]))
                continue
            char_index = _first_non_space_index(text, *offset)
            if char_index is None:
                aligned_labels.append(IGNORE_INDEX)
                continue
            label_id = int(labels[char_index])
            label = id_to_label[label_id]
            aligned_labels.append(int(label_to_id[label]))

    return SubwordEncoding(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=aligned_labels,
        offset_mapping=offsets,
        special_tokens_mask=special_tokens_mask,
        text=text,
    )


def expand_label_over_offset(label: str, text: str, start: int, end: int) -> list[tuple[int, str]]:
    """Expand one token label to non-space character positions in its offset."""

    positions = [index for index in range(max(0, start), min(len(text), end)) if not text[index].isspace()]
    if label == "O":
        return [(index, "O") for index in positions]
    if "-" not in label:
        raise ValueError(f"invalid BIO label: {label}")
    prefix, entity_type = label.split("-", 1)
    if prefix not in {"B", "I"}:
        raise ValueError(f"invalid BIO label: {label}")
    expanded: list[tuple[int, str]] = []
    for sequence_index, position in enumerate(positions):
        if prefix == "B" and sequence_index > 0:
            expanded.append((position, "I-" + entity_type))
        else:
            expanded.append((position, label))
    return expanded


def project_token_predictions_to_chars(
    predictions: Sequence[int],
    offset_mapping: Sequence[tuple[int, int]],
    special_tokens_mask: Sequence[int],
    id_to_label: Mapping[int, str],
    text: str,
) -> list[str]:
    """Project token predictions to original character labels using offsets."""

    if not (len(predictions) == len(offset_mapping) == len(special_tokens_mask)):
        raise ValueError("predictions, offsets, and special mask must have the same length")
    char_labels = ["O"] * len(text)
    written: set[int] = set()
    for pred_id, offset, special in zip(predictions, offset_mapping, special_tokens_mask):
        if special:
            continue
        label = id_to_label[int(pred_id)]
        for char_index, char_label in expand_label_over_offset(label, text, *offset):
            if char_index in written:
                continue
            char_labels[char_index] = char_label
            written.add(char_index)
    return char_labels


def benchmark_labels_from_prediction(
    predictions: Sequence[int],
    gold_labels: Sequence[int],
    offset_mapping: Sequence[tuple[int, int]],
    special_tokens_mask: Sequence[int],
    id_to_label: Mapping[int, str],
    text: str,
) -> tuple[list[str], list[str]]:
    """Build KLUE benchmark-compatible flattened labels for one example."""

    if len(predictions) != len(offset_mapping) or len(predictions) != len(special_tokens_mask):
        raise ValueError("prediction fields must have matching lengths")
    if len(gold_labels) != len(text):
        raise ValueError("gold label count must match text length")

    gold: list[str] = []
    pred: list[str] = []
    char_pred = project_token_predictions_to_chars(predictions, offset_mapping, special_tokens_mask, id_to_label, text)

    special_positions = [index for index, special in enumerate(special_tokens_mask) if special]
    if special_positions:
        gold.append("O")
        pred.append(id_to_label[int(predictions[special_positions[0]])])

    for char, gold_id, pred_label in zip(text, gold_labels, char_pred):
        if char == " ":
            continue
        gold.append(id_to_label[int(gold_id)])
        pred.append(pred_label)

    if len(special_positions) > 1:
        gold.append("O")
        pred.append(id_to_label[int(predictions[special_positions[-1]])])
    return gold, pred
