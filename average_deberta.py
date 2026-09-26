"""Average compatible fine-tuned KF-DeBERTa model weights (no optimizer state)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_deberta import atomic_torch_save, safe_torch_load


def average_checkpoints(paths: list[Path], output: Path) -> None:
    if len(paths) < 2:
        raise ValueError("at least two checkpoints are required")
    if output.resolve() in {path.resolve() for path in paths}:
        raise ValueError("output must differ from each input checkpoint")
    first = safe_torch_load(paths[0], map_location="cpu")
    state = first["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError("first checkpoint has no model state dict")
    original_dtypes = {key: value.dtype for key, value in state.items()}
    count = len(paths)
    for key, value in state.items():
        if value.is_floating_point():
            state[key] = value.float().mul_(1.0 / count)
    for path in paths[1:]:
        other = safe_torch_load(path, map_location="cpu")
        if other["config"]["model_name"] != first["config"]["model_name"]:
            raise ValueError("model names differ")
        if other["config"]["model_revision"] != first["config"]["model_revision"]:
            raise ValueError("model revisions differ")
        if other["label_names"] != first["label_names"]:
            raise ValueError("label mappings differ")
        other_state = other["model_state_dict"]
        if set(other_state) != set(state):
            raise ValueError("model parameter keys differ")
        for key, value in state.items():
            candidate = other_state[key]
            if candidate.shape != value.shape or candidate.dtype != original_dtypes[key]:
                raise ValueError(f"model parameter {key} is incompatible")
            if value.is_floating_point():
                value.add_(candidate.float(), alpha=1.0 / count)
            elif not torch.equal(value, candidate):
                raise ValueError(f"non-floating model parameter {key} differs")
    for key, value in state.items():
        if value.dtype != original_dtypes[key]:
            state[key] = value.to(original_dtypes[key])
    atomic_torch_save(
        {
            "model_state_dict": state,
            "label_names": first["label_names"],
            "config": first["config"],
            "metric_definitions": first.get("metric_definitions", {}),
            "averaged_from": [str(path.resolve()) for path in paths],
            "metrics": {},
        },
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    average_checkpoints(args.checkpoints, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
