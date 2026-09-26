"""Print named entities from a fine-tuned KF-DeBERTa checkpoint as JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from korean_ner.deberta_runtime import load_deberta_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", help="Korean text to analyze")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("KOREAN_NER_DEBERTA_CHECKPOINT", "runs/kf_deberta_best.pt")),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    runtime = load_deberta_runtime(args.checkpoint, device=args.device)
    print(json.dumps(runtime.predict_entities(args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
