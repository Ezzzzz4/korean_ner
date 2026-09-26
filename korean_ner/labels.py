"""Label helpers for KLUE NER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


DEFAULT_LABEL_LIST: tuple[str, ...] = (
    "B-DT",
    "I-DT",
    "B-LC",
    "I-LC",
    "B-OG",
    "I-OG",
    "B-PS",
    "I-PS",
    "B-QT",
    "I-QT",
    "B-TI",
    "I-TI",
    "O",
)


@dataclass(frozen=True)
class LabelMaps:
    names: tuple[str, ...]
    label_to_id: Mapping[str, int]
    id_to_label: Mapping[int, str]


def build_label_maps(label_names: Sequence[str] | None = None) -> LabelMaps:
    """Build stable label maps from KLUE metadata or the published fallback order."""

    names = tuple(label_names or DEFAULT_LABEL_LIST)
    return LabelMaps(
        names=names,
        label_to_id={label: index for index, label in enumerate(names)},
        id_to_label={index: label for index, label in enumerate(names)},
    )


def labels_from_klue_dataset(dataset: object) -> tuple[str, ...]:
    """Read KLUE NER label names from a loaded Hugging Face dataset object."""

    try:
        feature = dataset["train"].features["ner_tags"].feature
        names: Iterable[str] = feature.names
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("dataset does not expose KLUE ner_tags feature names") from exc
    return tuple(names)


def to_label_names(ids: Iterable[int], id_to_label: Mapping[int, str]) -> list[str]:
    """Convert label ids to names with an explicit error for unknown ids."""

    labels: list[str] = []
    for label_id in ids:
        try:
            labels.append(id_to_label[int(label_id)])
        except KeyError as exc:
            raise ValueError(f"unknown label id: {label_id}") from exc
    return labels
