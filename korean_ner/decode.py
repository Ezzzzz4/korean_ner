"""BIO decoding helpers for character-level NER output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, order=True)
class EntitySpan:
    start: int
    end: int
    label: str
    text: str = ""


def repair_bio_sequence(labels: Sequence[str]) -> list[str]:
    """Repair invalid BIO transitions for display and span extraction.

    A leading ``I-X`` or an ``I-X`` after ``O`` or a different entity type is
    treated as ``B-X``. Legal IOB2 labels are returned unchanged.
    """

    repaired: list[str] = []
    previous_type: str | None = None
    previous_inside = False
    for label in labels:
        if label == "O":
            repaired.append(label)
            previous_type = None
            previous_inside = False
            continue
        if "-" not in label:
            raise ValueError(f"invalid BIO label: {label}")
        prefix, entity_type = label.split("-", 1)
        if prefix == "B":
            repaired.append(label)
            previous_type = entity_type
            previous_inside = True
        elif prefix == "I":
            if previous_inside and previous_type == entity_type:
                repaired.append(label)
            else:
                repaired.append("B-" + entity_type)
            previous_type = entity_type
            previous_inside = True
        else:
            raise ValueError(f"invalid BIO label: {label}")
    return repaired


def extract_bio_spans(text: str | Sequence[str], labels: Sequence[str], *, repair: bool = True) -> list[EntitySpan]:
    """Extract entity spans from character labels using end-exclusive offsets."""

    chars = list(text)
    if len(chars) != len(labels):
        raise ValueError(f"text length {len(chars)} does not match labels length {len(labels)}")
    sequence = repair_bio_sequence(labels) if repair else list(labels)
    spans: list[EntitySpan] = []
    start: int | None = None
    current_type: str | None = None

    for index, label in enumerate(sequence):
        if label == "O":
            if start is not None and current_type is not None:
                spans.append(EntitySpan(start, index, current_type, "".join(chars[start:index])))
            start = None
            current_type = None
            continue

        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or current_type != entity_type:
            if start is not None and current_type is not None:
                spans.append(EntitySpan(start, index, current_type, "".join(chars[start:index])))
            start = index
            current_type = entity_type

    if start is not None and current_type is not None:
        spans.append(EntitySpan(start, len(sequence), current_type, "".join(chars[start:])))
    return spans


def token_predictions_to_raw_char_labels(
    predictions: Sequence[int],
    char_indices: Sequence[int | None],
    id_to_label: Mapping[int, str],
    *,
    text_length: int,
) -> list[str]:
    """Map token predictions back to one raw label per original character."""

    char_labels = ["O"] * text_length
    seen: set[int] = set()
    for pred_id, char_index in zip(predictions, char_indices):
        if char_index is None or char_index in seen:
            continue
        char_labels[char_index] = id_to_label[int(pred_id)]
        seen.add(char_index)
    return char_labels


def token_predictions_to_char_labels(
    predictions: Sequence[int],
    char_indices: Sequence[int | None],
    id_to_label: Mapping[int, str],
    *,
    text_length: int,
) -> list[str]:
    """Map token predictions back to one repaired label per original character."""

    char_labels = token_predictions_to_raw_char_labels(
        predictions,
        char_indices,
        id_to_label,
        text_length=text_length,
    )
    return repair_bio_sequence(char_labels)
