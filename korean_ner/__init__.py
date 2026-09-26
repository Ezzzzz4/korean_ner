"""Shared utilities for the Korean NER project."""

from .alignment import DEFAULT_MAX_LENGTH, IGNORE_INDEX, SPACE_TOKEN, AlignedEncoding, align_text
from .checkpoint import extract_model_state_dict, load_checkpoint, load_model_state
from .decode import EntitySpan, extract_bio_spans, repair_bio_sequence, token_predictions_to_char_labels, token_predictions_to_raw_char_labels
from .labels import DEFAULT_LABEL_LIST, LabelMaps, build_label_maps, labels_from_klue_dataset, to_label_names
from .metrics import StrictEntityScores, score_klue_benchmark_entity_macro_f1, score_strict_iob2


def __getattr__(name: str):
    if name == "KoElectraNER":
        from .model import KoElectraNER

        return KoElectraNER
    raise AttributeError(name)

__all__ = [
    "DEFAULT_LABEL_LIST",
    "DEFAULT_MAX_LENGTH",
    "IGNORE_INDEX",
    "SPACE_TOKEN",
    "AlignedEncoding",
    "EntitySpan",
    "KoElectraNER",
    "LabelMaps",
    "StrictEntityScores",
    "align_text",
    "build_label_maps",
    "extract_bio_spans",
    "extract_model_state_dict",
    "labels_from_klue_dataset",
    "load_checkpoint",
    "load_model_state",
    "repair_bio_sequence",
    "score_strict_iob2",
    "score_klue_benchmark_entity_macro_f1",
    "to_label_names",
    "token_predictions_to_char_labels",
    "token_predictions_to_raw_char_labels",
]
