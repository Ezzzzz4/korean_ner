"""Local Gradio demo for the fine-tuned KF-DeBERTa KLUE NER model."""

from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path

import torch  # Load PyTorch DLLs before Gradio's transitive imports on Windows.
import gradio as gr

from korean_ner.deberta_runtime import load_deberta_runtime


CHECKPOINT = Path(os.environ.get("KOREAN_NER_DEBERTA_CHECKPOINT", "runs/kf_deberta_best.pt"))
DEVICE = os.environ.get("KOREAN_NER_DEVICE", "cpu").strip().lower()
_RUNTIME = None
_LOAD_LOCK = threading.Lock()


def extract_entities(text: str) -> dict:
    global _RUNTIME
    if not text.strip():
        return {"entities": []}
    try:
        with _LOAD_LOCK:
            if _RUNTIME is None:
                _RUNTIME = load_deberta_runtime(CHECKPOINT, device=DEVICE)
            runtime = _RUNTIME
        return {"entities": runtime.predict_entities(text)}
    except (OSError, EOFError, ValueError, RuntimeError, KeyError, pickle.UnpicklingError) as exc:
        return {"error": str(exc)}


with gr.Blocks(title="Korean NER · KF-DeBERTa") as demo:
    gr.Markdown("# Korean named entity recognition\nFine-tuned KF-DeBERTa on KLUE NER.")
    text = gr.Textbox(label="Korean text", lines=3, placeholder="한국어 문장을 입력하세요")
    output = gr.JSON(label="Entities")
    gr.Button("Extract entities").click(extract_entities, inputs=text, outputs=output)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
