"""
Korean NER Demo Application
Uses CHARACTER-LEVEL tokenization to match KLUE training format.
Includes attention visualization for model interpretability.
"""

import torch
from transformers import AutoTokenizer
import gradio as gr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import html
import os
import threading
from pathlib import Path
from PIL import Image

from korean_ner import (
    align_text,
    build_label_maps,
    extract_bio_spans,
    extract_model_state_dict,
    load_checkpoint,
    token_predictions_to_char_labels,
)

# Configure Korean font for matplotlib
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows Korean font
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = os.environ.get(
    "KOREAN_NER_CHECKPOINT",
    str(Path(__file__).resolve().parent / "runs" / "corrected_seed42" / "averaged_e1_e2_e3.pt"),
)
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
MAX_LENGTH = 192
MAX_USER_CHARS = MAX_LENGTH - 2
LABEL_MAPS = build_label_maps()
ID_TO_LABEL = LABEL_MAPS.id_to_label
NUM_LABELS = len(LABEL_MAPS.names)

# Entity type colors and labels
ENTITY_COLORS = {
    "PS": "#FF6B6B",   # Person - Red
    "LC": "#4ECDC4",   # Location - Teal  
    "OG": "#45B7D1",   # Organization - Blue
    "DT": "#96CEB4",   # Date - Green
    "TI": "#FFEAA7",   # Time - Yellow
    "QT": "#DDA0DD",   # Quantity - Plum
}

ENTITY_LABELS = {
    "PS": "Person",
    "LC": "Location",
    "OG": "Organization",
    "DT": "Date",
    "TI": "Time",
    "QT": "Quantity",
}


_TOKENIZER = None
_MODEL = None
_DEVICE = None
_RUNTIME_LOCK = threading.Lock()


def resolve_device():
    requested = os.environ.get("KOREAN_NER_DEVICE", "cpu").strip().lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("KOREAN_NER_DEVICE=cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    if requested not in {"", "cpu"}:
        raise ValueError("KOREAN_NER_DEVICE must be 'cpu' or 'cuda'.")
    return torch.device("cpu")


def get_runtime_components():
    global _TOKENIZER, _MODEL, _DEVICE
    from korean_ner import KoElectraNER

    with _RUNTIME_LOCK:
        if _TOKENIZER is None:
            _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        if _MODEL is None:
            device = resolve_device()
            model = KoElectraNER(MODEL_NAME, NUM_LABELS, num_samples=5)
            checkpoint = load_checkpoint(MODEL_PATH, map_location=device)
            model.load_state_dict(extract_model_state_dict(checkpoint), strict=True)
            model.to(device)
            model.eval()
            _DEVICE = device
            _MODEL = model
    return _TOKENIZER, _MODEL, _DEVICE

# ============================================================================
# Inference Function
# ============================================================================
def extract_entities(text):
    """Extract named entities from Korean text."""
    if not text.strip():
        return "Please enter some Korean text.", ""

    chars = list(text)
    if not chars:
        return "Could not tokenize the text.", ""

    if len(chars) > MAX_USER_CHARS:
        return (
            create_length_error_html(len(chars)),
            f"Input is too long: {len(chars)} characters. Please use {MAX_USER_CHARS} characters or fewer."
        )

    tokenizer, model, device = get_runtime_components()
    try:
        encoding = align_text(text, tokenizer, max_length=MAX_LENGTH, pad_to_max_length=True)
    except ValueError as exc:
        return create_alignment_error_html(exc), str(exc)

    input_ids = torch.tensor([encoding.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([encoding.attention_mask], dtype=torch.long, device=device)

    with torch.no_grad():
        predictions = model(input_ids, attention_mask)[0]

    char_labels = token_predictions_to_char_labels(
        predictions,
        encoding.char_indices,
        ID_TO_LABEL,
        text_length=len(chars),
    )
    entities = extract_bio_spans(text, char_labels)

    html = create_output_html(text, chars, char_labels, entities)
    breakdown = create_entity_breakdown(entities)

    return html, breakdown


def _escape_text(value):
    """Escape user/model text before inserting it into Gradio HTML."""
    return html.escape(str(value), quote=True)


def _escape_markdown_inline_code(value):
    return _escape_text(value).replace("`", r"\`")


def create_length_error_html(actual_length):
    safe_actual = _escape_text(actual_length)
    safe_limit = _escape_text(MAX_USER_CHARS)
    return (
        "<p style='font-size: 18px; color: #b00020;'>"
        f"Input is too long: {safe_actual} characters. "
        f"Please use {safe_limit} characters or fewer."
        "</p>"
    )


def create_alignment_error_html(error):
    safe_error = _escape_text(error)
    return (
        "<p style='font-size: 18px; color: #b00020;'>"
        "Could not align the full text with the model input window. "
        f"{safe_error}"
        "</p>"
    )


def _label_at(char_labels, index):
    if hasattr(char_labels, "get"):
        return char_labels.get(index, "O")
    if index < len(char_labels):
        return char_labels[index]
    return "O"


def _entity_parts(entity):
    if hasattr(entity, "text") and hasattr(entity, "label"):
        return entity.text, entity.label
    entity_text, etype, _ = entity
    return entity_text, etype


def create_output_html(original_text, chars, char_labels, entities):
    """Create HTML with highlighted entities in original text."""
    if not entities:
        safe_text = _escape_text(original_text)
        return f"<p style='font-size: 18px;'>{safe_text}</p><p style='color: #888;'><i>No entities detected.</i></p>"
    
    # Build highlighted text character by character
    html_chars = []
    in_entity = False
    current_type = None
    
    for idx, char in enumerate(chars):
        label = _label_at(char_labels, idx)
        
        if label.startswith('B-'):
            # Close previous entity span if open
            if in_entity:
                html_chars.append('</span>')
            # Start new entity
            etype = label[2:]
            color = ENTITY_COLORS.get(etype, "#888")
            html_chars.append(f"<span style='background: {color}40; border-bottom: 3px solid {color}; padding: 2px 1px; border-radius: 3px;'>")
            html_chars.append(_escape_text(char))
            in_entity = True
            current_type = etype
            
        elif label.startswith('I-'):
            etype = label[2:]
            if in_entity and current_type == etype:
                # Continue same entity
                html_chars.append(_escape_text(char))
            else:
                # I- without matching B-, or type mismatch - start new highlight anyway
                if in_entity:
                    html_chars.append('</span>')
                color = ENTITY_COLORS.get(etype, "#888")
                html_chars.append(f"<span style='background: {color}40; border-bottom: 3px solid {color}; padding: 2px 1px; border-radius: 3px;'>")
                html_chars.append(_escape_text(char))
                in_entity = True
                current_type = etype
            
        else:
            if in_entity:
                html_chars.append('</span>')
                in_entity = False
                current_type = None
            html_chars.append(_escape_text(char))
    
    if in_entity:
        html_chars.append('</span>')
    
    highlighted = ''.join(html_chars)
    
    # Entity cards
    entity_html = "<div style='margin-top: 20px;'><strong>🎯 Detected Entities:</strong><br><br>"
    for entity in entities:
        entity_text, etype = _entity_parts(entity)
        color = ENTITY_COLORS.get(etype, "#888")
        label = _escape_text(ENTITY_LABELS.get(etype, etype))
        safe_entity_text = _escape_text(entity_text)
        entity_html += f"""
        <span style='display: inline-block; margin: 5px; padding: 8px 16px;
                     background: {color}22; border: 2px solid {color}; border-radius: 25px;'>
            {label}: <strong>{safe_entity_text}</strong>
        </span>
        """
    entity_html += "</div>"
    
    return f"<p style='font-size: 18px; line-height: 2;'>{highlighted}</p>{entity_html}"


def create_entity_breakdown(entities):
    """Create text summary of entities."""
    if not entities:
        return "No entities detected."
    
    by_type = {}
    for entity in entities:
        text, etype = _entity_parts(entity)
        if etype not in by_type:
            by_type[etype] = []
        by_type[etype].append(text)
    
    lines = []
    for etype, texts in sorted(by_type.items()):
        label = ENTITY_LABELS.get(etype, etype)
        lines.append(f"{label}: {', '.join(texts)}")
    
    return "\n".join(lines)


# ============================================================================
# Attention Visualization Function
# ============================================================================
def visualize_attention(text):
    """Generate attention visualization for the input text."""
    if not text.strip():
        return None, "Please enter some Korean text."

    chars = list(text)
    if len(chars) > MAX_USER_CHARS:
        return None, f"Input is too long: {len(chars)} characters. Please use {MAX_USER_CHARS} characters or fewer."

    tokenizer, model, device = get_runtime_components()
    try:
        encoding = align_text(text, tokenizer, max_length=MAX_LENGTH, pad_to_max_length=True)
    except ValueError as exc:
        return None, str(exc)

    input_ids = torch.tensor([encoding.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([encoding.attention_mask], dtype=torch.long, device=device)

    with torch.no_grad():
        _, attentions = model(input_ids, attention_mask, output_attentions=True)

    valid_len = sum(encoding.attention_mask)
    tokens = encoding.tokens[:valid_len]

    # Average attention across all heads in last layer
    last_layer_attention = attentions[-1][0].cpu().numpy()  # (heads, seq, seq)
    avg_attention = last_layer_attention.mean(axis=0)  # (seq, seq)
    avg_attention = avg_attention[:valid_len, :valid_len]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        avg_attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'}
    )

    title_text = text[:40]
    ax.set_title(f'Attention Heatmap (Last Layer Average)\n"{title_text}..."', fontsize=12)
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    # Convert to image for Gradio
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf).copy()
    plt.close(fig)

    # Analysis text
    analysis = f"""
**Attention Visualization Analysis**

**Input Text**: `{_escape_markdown_inline_code(text)}`

**Tokens (corrected character alignment)**: {len(tokens)} tokens

**Visualization**: The heatmap shows how much each token (y-axis) attends to every other token (x-axis) in the last transformer layer. Brighter colors indicate higher attention weights.

**How to interpret**:
- **Diagonal patterns**: Tokens attending to themselves or nearby tokens
- **Vertical lines**: Tokens receiving high average attention from many query tokens
- **Horizontal lines**: Query tokens distributing attention across many key tokens

These averaged transformer weights are descriptive; they do not measure a token's causal contribution to the final BiLSTM-CRF prediction.
"""

    return image, analysis

# ============================================================================
# Gradio Interface
# ============================================================================
EXAMPLES = [
    ["\uc0bc\uc131\uc804\uc790 \uc774\uc7ac\uc6a9 \ud68c\uc7a5\uc774 \uc11c\uc6b8\uc5d0\uc11c \ud68c\uc758\ub97c \uc5f4\uc5c8\ub2e4."],
    ["\ud55c\uad6d\uc740\ud589\uc740 \uc11c\uc6b8\uc5d0\uc11c \uacbd\uc81c \uc804\ub9dd\uc744 \ubc1c\ud45c\ud588\ub2e4."],
    ["\ub124\uc774\ubc84\uc640 \uce74\uce74\uc624\ub294 \uc0c8\ub85c\uc6b4 \uc11c\ube44\uc2a4\ub97c \ucd9c\uc2dc\ud588\ub2e4."],
    ["\uae40\ubbfc\uc218 \uad50\uc218\ub294 \uc11c\uc6b8\ub300\ud559\uad50\uc5d0\uc11c \uc5f0\uad6c\ud55c\ub2e4."],
]

with gr.Blocks(title="Korean NER Demo", theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <div style="text-align:center;padding:20px">
          <h1>Korean Named Entity Recognition</h1>
          <p>KoELECTRA + BiLSTM + CRF</p>
          <p>Character alignment preserves original spaces. Scores must come from an explicit evaluation run.</p>
        </div>
    """)
    with gr.Tabs():
        with gr.TabItem("Entity extraction"):
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(label="Korean text", placeholder="\ud55c\uad6d\uc5b4 \ubb38\uc7a5\uc744 \uc785\ub825\ud558\uc138\uc694", lines=3)
                    submit_btn = gr.Button("Extract entities", variant="primary")
                    gr.Examples(examples=EXAMPLES, inputs=input_text)
                with gr.Column():
                    output_html = gr.HTML(label="Results")
                    output_text = gr.Textbox(label="Summary", lines=4, interactive=False)
            submit_btn.click(fn=extract_entities, inputs=[input_text], outputs=[output_html, output_text])
            input_text.submit(fn=extract_entities, inputs=[input_text], outputs=[output_html, output_text])

        with gr.TabItem("Attention visualization"):
            gr.Markdown("Averaged encoder attention is descriptive and does not explain the final BiLSTM-CRF prediction.")
            with gr.Row():
                with gr.Column():
                    attn_input = gr.Textbox(label="Korean text", placeholder="\ud55c\uad6d\uc5b4 \ubb38\uc7a5\uc744 \uc785\ub825\ud558\uc138\uc694", lines=3)
                    attn_btn = gr.Button("Visualize attention", variant="primary")
                    gr.Examples(examples=EXAMPLES, inputs=attn_input)
                with gr.Column():
                    attn_image = gr.Image(label="Attention heatmap", type="pil")
                    attn_analysis = gr.Markdown(label="Analysis")
            attn_btn.click(fn=visualize_attention, inputs=[attn_input], outputs=[attn_image, attn_analysis])

    gr.Markdown("Entity types: PS person \u00b7 LC location \u00b7 OG organization \u00b7 DT date \u00b7 TI time \u00b7 QT quantity")

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
