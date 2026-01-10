"""
Korean NER Demo Application
Uses CHARACTER-LEVEL tokenization to match KLUE training format.
Includes attention visualization for model interpretability.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
import gradio as gr
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64

# Configure Korean font for matplotlib
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows Korean font
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = "weights/best_model.pt"
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    "PS": "👤 Person",
    "LC": "📍 Location",
    "OG": "🏢 Organization", 
    "DT": "📅 Date",
    "TI": "⏰ Time",
    "QT": "🔢 Quantity",
}

# KLUE NER Labels - CORRECT ORDER!
# 0: B-DT, 1: I-DT, 2: B-LC, 3: I-LC, 4: B-OG, 5: I-OG, 6: B-PS, 7: I-PS, 8: B-QT, 9: I-QT, 10: B-TI, 11: I-TI, 12: O
LABEL_LIST = ['B-DT', 'I-DT', 'B-LC', 'I-LC', 'B-OG', 'I-OG', 'B-PS', 'I-PS', 'B-QT', 'I-QT', 'B-TI', 'I-TI', 'O']
ID_TO_LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)


# ============================================================================
# Model Definition
# ============================================================================
class KoElectraNER(nn.Module):
    def __init__(self, model_name, num_labels, num_samples=5):
        super().__init__()
        self.num_labels = num_labels
        self.num_samples = num_samples
        
        self.electra = AutoModel.from_pretrained(model_name, output_attentions=True)
        hidden = self.electra.config.hidden_size
        
        self.lstm = nn.LSTM(hidden, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(512, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask, labels=None, output_attentions=False):
        enc = self.electra(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
        seq_out = enc.last_hidden_state
        lstm_out, _ = self.lstm(seq_out)
        emissions = self.classifier(self.dropout(lstm_out))
        mask = attention_mask.bool()
        
        if labels is not None:
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0
            loss = -self.crf(emissions, labels_crf, mask=mask, reduction='mean')
            return loss, emissions
        
        predictions = self.crf.decode(emissions, mask=mask)
        if output_attentions:
            return predictions, enc.attentions
        return predictions


# ============================================================================
# Character-Level Tokenization (matches KLUE format!)
# ============================================================================
def char_tokenize(text):
    """
    KLUE NER uses character-level tokenization where each character is a token,
    including spaces as separate tokens.
    
    Example:
        "김민수 교수" → ['김', '민', '수', ' ', '교', '수']
    """
    return list(text)


# ============================================================================
# Load Model
# ============================================================================
print("Loading NER model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = KoElectraNER(MODEL_NAME, NUM_LABELS, num_samples=5)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}")


# ============================================================================
# Inference Function
# ============================================================================
def extract_entities(text):
    """Extract named entities from Korean text."""
    if not text.strip():
        return "Please enter some Korean text.", ""
    
    # Character-level tokenization (matches KLUE!)
    chars = char_tokenize(text)
    
    if not chars:
        return "Could not tokenize the text.", ""
    
    print(f"Characters: {chars[:30]}...")
    
    # Tokenize with BERT (is_split_into_words=True)
    inputs = tokenizer(
        chars,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )
    
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)
    word_ids = inputs.word_ids(batch_index=0)
    
    # Predict
    with torch.no_grad():
        predictions = model(input_ids, attention_mask)[0]
    
    # Map predictions to characters
    char_labels = {}
    for idx, (pred_id, word_idx) in enumerate(zip(predictions, word_ids)):
        if word_idx is None:
            continue
        if word_idx not in char_labels:
            char_labels[word_idx] = ID_TO_LABEL[pred_id]
    
    print(f"Char labels sample: {[(chars[i], char_labels.get(i, 'O')) for i in range(min(20, len(chars)))]}")
    
    # Extract entities by grouping consecutive B-/I- tags
    entities = []
    current_type = None
    current_chars = []
    current_start = 0
    
    for idx, char in enumerate(chars):
        label = char_labels.get(idx, 'O')
        
        if label.startswith('B-'):
            # Save previous entity
            if current_type and current_chars:
                entity_text = ''.join(current_chars)
                if entity_text.strip():  # Don't save empty entities
                    entities.append((entity_text, current_type, current_start))
            # Start new entity
            current_type = label[2:]
            current_chars = [char]
            current_start = idx
            
        elif label.startswith('I-'):
            etype = label[2:]
            if current_type == etype:
                current_chars.append(char)
            else:
                # Type mismatch - save and start new
                if current_type and current_chars:
                    entity_text = ''.join(current_chars)
                    if entity_text.strip():
                        entities.append((entity_text, current_type, current_start))
                current_type = etype
                current_chars = [char]
                current_start = idx
        else:
            # O tag
            if current_type and current_chars:
                entity_text = ''.join(current_chars)
                if entity_text.strip():
                    entities.append((entity_text, current_type, current_start))
            current_type = None
            current_chars = []
    
    # Don't forget last entity
    if current_type and current_chars:
        entity_text = ''.join(current_chars)
        if entity_text.strip():
            entities.append((entity_text, current_type, current_start))
    
    print(f"Entities: {entities}")
    
    # Create output
    html = create_output_html(text, chars, char_labels, entities)
    breakdown = create_entity_breakdown(entities)
    
    return html, breakdown


def create_output_html(original_text, chars, char_labels, entities):
    """Create HTML with highlighted entities in original text."""
    if not entities:
        return f"<p style='font-size: 18px;'>{original_text}</p><p style='color: #888;'><i>No entities detected.</i></p>"
    
    # Build highlighted text character by character
    html_chars = []
    in_entity = False
    current_type = None
    
    for idx, char in enumerate(chars):
        label = char_labels.get(idx, 'O')
        
        if label.startswith('B-'):
            # Close previous entity span if open
            if in_entity:
                html_chars.append('</span>')
            # Start new entity
            etype = label[2:]
            color = ENTITY_COLORS.get(etype, "#888")
            html_chars.append(f"<span style='background: {color}40; border-bottom: 3px solid {color}; padding: 2px 1px; border-radius: 3px;'>")
            html_chars.append(char)
            in_entity = True
            current_type = etype
            
        elif label.startswith('I-'):
            etype = label[2:]
            if in_entity and current_type == etype:
                # Continue same entity
                html_chars.append(char)
            else:
                # I- without matching B-, or type mismatch - start new highlight anyway
                if in_entity:
                    html_chars.append('</span>')
                color = ENTITY_COLORS.get(etype, "#888")
                html_chars.append(f"<span style='background: {color}40; border-bottom: 3px solid {color}; padding: 2px 1px; border-radius: 3px;'>")
                html_chars.append(char)
                in_entity = True
                current_type = etype
            
        else:
            if in_entity:
                html_chars.append('</span>')
                in_entity = False
                current_type = None
            html_chars.append(char)
    
    if in_entity:
        html_chars.append('</span>')
    
    highlighted = ''.join(html_chars)
    
    # Entity cards
    entity_html = "<div style='margin-top: 20px;'><strong>🎯 Detected Entities:</strong><br><br>"
    for entity_text, etype, _ in entities:
        color = ENTITY_COLORS.get(etype, "#888")
        label = ENTITY_LABELS.get(etype, etype)
        entity_html += f"""
        <span style='display: inline-block; margin: 5px; padding: 8px 16px;
                     background: {color}22; border: 2px solid {color}; border-radius: 25px;'>
            {label}: <strong>{entity_text}</strong>
        </span>
        """
    entity_html += "</div>"
    
    return f"<p style='font-size: 18px; line-height: 2;'>{highlighted}</p>{entity_html}"


def create_entity_breakdown(entities):
    """Create text summary of entities."""
    if not entities:
        return "No entities detected."
    
    by_type = {}
    for text, etype, _ in entities:
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
    
    inputs = tokenizer(
        chars,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )
    
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        predictions, attentions = model(input_ids, attention_mask, output_attentions=True)
    
    # Get tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    
    # Filter out padding
    valid_len = sum(1 for t in tokens if t != '[PAD]')
    tokens = tokens[:valid_len]
    
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
    
    ax.set_title(f'Attention Heatmap (Last Layer Average)\n"{text[:40]}..."', fontsize=12)
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # Convert to image for Gradio
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    # Analysis text
    analysis = f"""
**Attention Visualization Analysis**

**Input Text**: {text}

**Tokens (after subword tokenization)**: {len(tokens)} tokens

**Visualization**: The heatmap shows how much each token (y-axis) attends to every other token (x-axis) in the last transformer layer. Brighter colors indicate higher attention weights.

**How to interpret**:
- **Diagonal patterns**: Tokens attending to themselves or nearby tokens
- **Vertical lines**: Tokens that many other tokens attend to (important words)
- **Horizontal lines**: Tokens that attend to many others (contextual aggregation)
"""
    
    return buf, analysis


# ============================================================================
# Gradio Interface
# ============================================================================
EXAMPLES = [
    ["삼성전자 이재용 회장이 2024년 1월 15일 서울에서 기자회견을 열었다."],
    ["대한민국 서울특별시 강남구에서 열린 행사에 약 500명이 참석했습니다."],
    ["카카오와 네이버는 한국의 대표적인 IT 기업입니다."],
    ["김민수 교수는 서울대학교에서 인공지능을 연구하고 있다."],
]

with gr.Blocks(title="Korean NER Demo", theme=gr.themes.Soft()) as demo:
    gr.HTML("""
        <div style='text-align: center; padding: 20px;'>
            <h1>🇰🇷 Korean Named Entity Recognition</h1>
            <p style='color: #666;'>KoELECTRA + BiLSTM + CRF | F1: 85.9% on KLUE NER</p>
            <p style='color: #888; font-size: 12px;'>Character-level tokenization (matches KLUE format)</p>
        </div>
    """)
    
    with gr.Tabs():
        # Tab 1: Entity Extraction
        with gr.TabItem("🔍 Entity Extraction"):
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="📝 Enter Korean Text",
                        placeholder="한국어 텍스트를 입력하세요...",
                        lines=3
                    )
                    submit_btn = gr.Button("🔍 Extract Entities", variant="primary")
                    gr.Examples(examples=EXAMPLES, inputs=input_text, label="📌 Try these examples")
                
                with gr.Column():
                    output_html = gr.HTML(label="Results")
                    output_text = gr.Textbox(label="📊 Summary", lines=4, interactive=False)
            
            submit_btn.click(fn=extract_entities, inputs=[input_text], outputs=[output_html, output_text])
            input_text.submit(fn=extract_entities, inputs=[input_text], outputs=[output_html, output_text])
        
        # Tab 2: Attention Visualization
        with gr.TabItem("🧠 Attention Visualization"):
            gr.Markdown("""
            ### Visualize Model Attention Patterns
            See which tokens the model focuses on when making predictions. This provides insight into the model's decision-making process.
            """)
            
            with gr.Row():
                with gr.Column():
                    attn_input = gr.Textbox(
                        label="📝 Enter Korean Text",
                        placeholder="한국어 텍스트를 입력하세요...",
                        lines=3
                    )
                    attn_btn = gr.Button("🧠 Visualize Attention", variant="primary")
                    gr.Examples(examples=EXAMPLES, inputs=attn_input, label="📌 Try these examples")
                
                with gr.Column():
                    attn_image = gr.Image(label="Attention Heatmap", type="filepath")
                    attn_analysis = gr.Markdown(label="Analysis")
            
            attn_btn.click(fn=visualize_attention, inputs=[attn_input], outputs=[attn_image, attn_analysis])
    
    gr.HTML("""
        <div style='margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; color: white;'>
            <h4 style='margin: 0 0 10px 0;'>🏷️ Entity Types</h4>
            <div style='display: flex; flex-wrap: wrap; gap: 8px;'>
                <span style='background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 12px;'>👤 PS (Person)</span>
                <span style='background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 12px;'>📍 LC (Location)</span>
                <span style='background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 12px;'>🏢 OG (Organization)</span>
                <span style='background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 12px;'>📅 DT (Date)</span>
                <span style='background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 12px;'>⏰ TI (Time)</span>
                <span style='background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 12px;'>🔢 QT (Quantity)</span>
            </div>
        </div>
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
