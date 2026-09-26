"""Strict entity metrics for KLUE NER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2


@dataclass(frozen=True)
class StrictEntityScores:
    precision_micro: float
    recall_micro: float
    f1_micro: float
    f1_macro: float
    report: dict


def _plain_numbers(value: Any) -> Any:
    """Convert NumPy scalar values in seqeval reports to JSON-safe Python values."""

    if isinstance(value, dict):
        return {key: _plain_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_numbers(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def score_strict_iob2(y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]]) -> StrictEntityScores:
    """Compute strict IOB2 entity-level micro and macro scores."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must contain the same number of sequences")
    report = _plain_numbers(classification_report(
        y_true,
        y_pred,
        mode="strict",
        scheme=IOB2,
        digits=6,
        output_dict=True,
        zero_division=0,
    ))
    return StrictEntityScores(
        precision_micro=float(precision_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)),
        recall_micro=float(recall_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)),
        f1_micro=float(f1_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)),
        f1_macro=float(report.get("macro avg", {}).get("f1-score", 0.0)),
        report=report,
    )


def score_klue_benchmark_entity_macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Compute the official KLUE NER benchmark entity macro F1.

    The KLUE baseline flattens all validation examples into one sequence and
    calls seqeval's strict IOB2 macro F1 over that single sequence.
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must contain the same number of labels")
    return float(f1_score([list(y_true)], [list(y_pred)], average="macro", mode="strict", scheme=IOB2, zero_division=0))
