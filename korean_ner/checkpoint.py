"""Checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


MODEL_STATE_KEY = "model_state_dict"


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    """Load a checkpoint using PyTorch's safe tensor-only path."""

    path = Path(path)
    return torch.load(path, map_location=map_location, weights_only=True)


def extract_model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Return the model state dict from either a raw or wrapped checkpoint."""

    if isinstance(checkpoint, dict) and MODEL_STATE_KEY in checkpoint:
        state = checkpoint[MODEL_STATE_KEY]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a model state dict")
    return state


def load_model_state(model: torch.nn.Module, path: str | Path, *, map_location: str | torch.device = "cpu", strict: bool = True) -> Any:
    """Load model weights from a raw or wrapped checkpoint."""

    checkpoint = load_checkpoint(path, map_location=map_location)
    return model.load_state_dict(extract_model_state_dict(checkpoint), strict=strict)
