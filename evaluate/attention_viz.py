"""Descriptive KoELECTRA self-attention visualizations for Korean NER."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = REPO_ROOT / "weights" / "best_model.pt"
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
MAX_LENGTH = 192
OUTPUT_DIR = REPO_ROOT / "runs" / "attention"

ENTITY_COLORS = {
    "PS": "#FF6B6B",
    "LC": "#4ECDC4",
    "OG": "#45B7D1",
    "DT": "#96CEB4",
    "TI": "#FFEAA7",
    "QT": "#DDA0DD",
}


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Create descriptive encoder self-attention plots for Korean NER examples."
    )
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to a raw or wrapped model checkpoint.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name or path.")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Aligned token sequence length.")
    parser.add_argument("--device", default="cpu", help="Torch device to use. Defaults to explicit CPU.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for generated attention figures.")
    return parser.parse_args(argv)


def valid_token_count(tokens: Sequence[str]) -> int:
    """Return the non-padding prefix length in an aligned token sequence."""
    try:
        return list(tokens).index("[PAD]")
    except ValueError:
        return len(tokens)


def token_indices_for_char_span(char_indices: Sequence[int | None], start: int, end: int) -> list[int]:
    """Map an end-exclusive original character span to aligned token indices."""
    return [
        token_idx
        for token_idx, char_idx in enumerate(char_indices)
        if char_idx is not None and start <= char_idx < end
    ]


def resolve_entity_span(text: str, entity: dict) -> dict | None:
    """Validate a hand-written annotation or repair it by unique text match."""
    start = int(entity["start"])
    end = int(entity["end"])
    entity_text = entity["text"]
    if 0 <= start < end <= len(text) and text[start:end] == entity_text:
        return dict(entity)

    found = text.find(entity_text)
    if found >= 0 and text.find(entity_text, found + 1) < 0:
        repaired = dict(entity)
        repaired["start"] = found
        repaired["end"] = found + len(entity_text)
        return repaired
    return None


def get_attention_for_text(model, tokenizer, text: str, device, label_maps, max_length: int, torch_module):
    """Get predictions and encoder self-attention for original characters, including spaces."""
    from korean_ner.alignment import align_text
    from korean_ner.decode import token_predictions_to_char_labels

    encoding = align_text(
        text,
        tokenizer,
        max_length=max_length,
        label_to_id=label_maps.label_to_id,
        id_to_label=label_maps.id_to_label,
    )
    input_ids = torch_module.tensor([encoding.input_ids], dtype=torch_module.long, device=device)
    attention_mask = torch_module.tensor([encoding.attention_mask], dtype=torch_module.long, device=device)

    with torch_module.no_grad():
        predictions, attentions = model(input_ids, attention_mask, output_attentions=True)

    pred_labels = token_predictions_to_char_labels(
        predictions[0],
        encoding.char_indices,
        label_maps.id_to_label,
        text_length=len(text),
    )
    all_layer_attention = [att[0].detach().cpu().numpy().mean(axis=0) for att in attentions]

    return {
        "tokens": encoding.tokens,
        "char_indices": encoding.char_indices,
        "text": text,
        "pred_labels": pred_labels,
        "attention": all_layer_attention[-1],
        "all_layer_attention": all_layer_attention,
        "attention_description": "encoder self-attention averaged over heads; descriptive only",
    }


def visualize_attention_heatmap(result, output_path, title="Last-Layer Mean Encoder Self-Attention"):
    """Create a heatmap of last-layer encoder self-attention averaged across heads."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    tokens = result["tokens"]
    attention = result["attention"]
    valid_len = valid_token_count(tokens)
    tokens = tokens[:valid_len]
    attention = attention[:valid_len, :valid_len]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Mean attention weight"},
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Key tokens")
    ax.set_ylabel("Query tokens")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def visualize_entity_attention(result, entity: dict, output_path):
    """Plot mean encoder self-attention from annotated span tokens to all tokens."""
    import matplotlib.pyplot as plt

    tokens = result["tokens"]
    attention = result["attention"]
    valid_len = valid_token_count(tokens)
    tokens = tokens[:valid_len]
    entity_indices = [
        idx for idx in token_indices_for_char_span(result["char_indices"], entity["start"], entity["end"]) if idx < valid_len
    ]
    if not entity_indices:
        print(f"Skipped unsupported annotation without aligned tokens: {entity}")
        return

    entity_attention = attention[entity_indices, :valid_len].mean(axis=0)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#45B7D1" if i in entity_indices else "#E0E0E0" for i in range(len(tokens))]
    bars = ax.bar(range(len(tokens)), entity_attention, color=colors)
    for i in entity_indices:
        bars[i].set_color(ENTITY_COLORS.get(entity["type"], "#FF6B6B"))

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean attention weight")
    ax.set_title(f"Mean encoder self-attention from annotated span '{entity['text']}' ({entity['type']})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def visualize_layer_attention_evolution(result, output_path):
    """Show layer-wise mean encoder self-attention, averaged across heads."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    all_attention = result["all_layer_attention"]
    tokens = result["tokens"]
    valid_len = valid_token_count(tokens)
    num_layers = len(all_attention)
    rows = max(1, (num_layers + 3) // 4)
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

    for layer_idx, ax in enumerate(axes):
        if layer_idx >= num_layers:
            ax.axis("off")
            continue
        layer_attention = all_attention[layer_idx][:valid_len, :valid_len]
        sns.heatmap(layer_attention, ax=ax, cmap="Blues", cbar=False, xticklabels=False, yticklabels=False)
        ax.set_title(f"Layer {layer_idx + 1} mean attention", fontsize=10)

    plt.suptitle("Encoder Self-Attention by Layer (Head Average, Descriptive)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def example_sentences() -> list[dict]:
    samples = [
        ("삼성전자 이재용 회장이 서울에서 회의를 열었다.", [("삼성전자", "OG"), ("이재용", "PS"), ("서울", "LC")]),
        ("2024년 1월 15일 서울에서 행사가 열렸다.", [("2024년 1월 15일", "DT"), ("서울", "LC")]),
        ("네이버와 카카오는 한국에서 새로운 서비스를 발표했다.", [("네이버", "OG"), ("카카오", "OG"), ("한국", "LC")]),
    ]
    examples = []
    for text, spans in samples:
        entities = []
        for entity_text, entity_type in spans:
            if text.count(entity_text) != 1:
                raise ValueError(f"example entity must occur exactly once: {entity_text}")
            start = text.index(entity_text)
            entities.append({"text": entity_text, "type": entity_type, "start": start, "end": start + len(entity_text)})
        examples.append({"text": text, "entities": entities})
    return examples


def main(argv: Sequence[str] | None = None):
    import torch
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    from korean_ner.checkpoint import load_model_state
    from korean_ner.labels import build_label_maps
    from korean_ner.model import KoElectraNER

    args = parse_args(argv)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 60)
    print("Attention Visualization for Korean NER")
    print("=" * 60)
    print(f"Device: {device}")
    print("Attention plots are descriptive encoder self-attention summaries, not causal CRF explanations.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    label_maps = build_label_maps()
    model = KoElectraNER(args.model_name, len(label_maps.names))
    load_model_state(model, args.model_path, map_location=device, strict=True)
    model.to(device)
    model.eval()

    for ex_idx, example in enumerate(example_sentences(), start=1):
        text = example["text"]
        print(f"\nExample {ex_idx}: {text}")
        result = get_attention_for_text(model, tokenizer, text, device, label_maps, args.max_length, torch)

        visualize_attention_heatmap(
            result,
            output_dir / f"attention_heatmap_{ex_idx}.png",
            title=f"Last-Layer Mean Encoder Self-Attention: {text[:30]}...",
        )
        visualize_layer_attention_evolution(result, output_dir / f"layer_evolution_{ex_idx}.png")

        for entity in example["entities"][:1]:
            resolved = resolve_entity_span(text, entity)
            if resolved is None:
                print(f"Skipped unsupported annotation: {entity}")
                continue
            visualize_entity_attention(result, resolved, output_dir / f"entity_attention_{ex_idx}_{resolved['type']}.png")

    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
