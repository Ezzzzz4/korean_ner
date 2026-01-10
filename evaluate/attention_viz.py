"""
Attention Visualization for Korean NER
Extracts and visualizes attention patterns from the KoELECTRA encoder.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import numpy as np
import os

# Configure Korean font for matplotlib
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows Korean font
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign rendering

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = "../weights/best_model.pt"
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "../assets/attention"

# KLUE NER Labels
LABEL_LIST = ['B-DT', 'I-DT', 'B-LC', 'I-LC', 'B-OG', 'I-OG', 'B-PS', 'I-PS', 'B-QT', 'I-QT', 'B-TI', 'I-TI', 'O']
ID_TO_LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

ENTITY_COLORS = {
    "PS": "#FF6B6B",
    "LC": "#4ECDC4",
    "OG": "#45B7D1",
    "DT": "#96CEB4",
    "TI": "#FFEAA7",
    "QT": "#DDA0DD",
}


# ============================================================================
# Model with Attention Output
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
            if output_attentions:
                return loss, emissions, enc.attentions
            return loss, emissions
        
        if output_attentions:
            return self.crf.decode(emissions, mask=mask), enc.attentions
        return self.crf.decode(emissions, mask=mask)


def get_attention_for_text(model, tokenizer, text, device):
    """Get attention weights and predictions for input text."""
    chars = list(text)
    
    inputs = tokenizer(
        chars,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    word_ids = inputs.word_ids(batch_index=0)
    
    with torch.no_grad():
        predictions, attentions = model(input_ids, attention_mask, output_attentions=True)
    
    # Get tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    
    # Map predictions to characters
    pred_labels = []
    for tok_idx, word_idx in enumerate(word_ids):
        if word_idx is not None and tok_idx < len(predictions[0]):
            pred_labels.append(ID_TO_LABEL[predictions[0][tok_idx]])
        else:
            pred_labels.append(None)
    
    # Average attention across all heads in last layer
    # Shape: (batch, heads, seq_len, seq_len)
    last_layer_attention = attentions[-1][0].cpu().numpy()  # (heads, seq, seq)
    avg_attention = last_layer_attention.mean(axis=0)  # (seq, seq)
    
    return {
        'tokens': tokens,
        'chars': chars,
        'word_ids': word_ids,
        'predictions': predictions[0],
        'pred_labels': pred_labels,
        'attention': avg_attention,
        'all_layer_attention': [att[0].cpu().numpy().mean(axis=0) for att in attentions]
    }


def visualize_attention_heatmap(result, output_path, title="Attention Heatmap"):
    """Create attention heatmap visualization."""
    tokens = result['tokens']
    attention = result['attention']
    
    # Filter out padding tokens
    valid_len = sum(1 for t in tokens if t != '[PAD]')
    tokens = tokens[:valid_len]
    attention = attention[:valid_len, :valid_len]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        attention,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_entity_attention(result, entity_indices, entity_text, entity_type, output_path):
    """Visualize attention from entity tokens to all other tokens."""
    tokens = result['tokens']
    attention = result['attention']
    
    # Filter valid tokens
    valid_len = sum(1 for t in tokens if t != '[PAD]')
    tokens = tokens[:valid_len]
    
    # Average attention from entity tokens
    entity_attention = attention[entity_indices, :valid_len].mean(axis=0)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 5))
    
    colors = ['#45B7D1' if i in entity_indices else '#E0E0E0' for i in range(len(tokens))]
    
    bars = ax.bar(range(len(tokens)), entity_attention, color=colors)
    
    # Highlight entity bars
    for i in entity_indices:
        if i < len(bars):
            bars[i].set_color(ENTITY_COLORS.get(entity_type, '#FF6B6B'))
    
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Attention Weight')
    ax.set_title(f'Attention from "{entity_text}" ({entity_type}) to Other Tokens')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def visualize_layer_attention_evolution(result, output_path):
    """Show how attention patterns evolve across layers."""
    all_attention = result['all_layer_attention']
    tokens = result['tokens']
    
    valid_len = sum(1 for t in tokens if t != '[PAD]')
    tokens = tokens[:valid_len]
    
    num_layers = len(all_attention)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for layer_idx, att in enumerate(all_attention):
        if layer_idx < 12:
            ax = axes[layer_idx]
            att = att[:valid_len, :valid_len]
            
            sns.heatmap(
                att,
                ax=ax,
                cmap='Blues',
                cbar=False,
                xticklabels=False,
                yticklabels=False
            )
            ax.set_title(f'Layer {layer_idx + 1}', fontsize=10)
    
    plt.suptitle('Attention Evolution Across Layers', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Attention Visualization for Korean NER")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = KoElectraNER(MODEL_NAME, NUM_LABELS)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")
    
    # Example sentences for visualization
    examples = [
        {
            'text': "삼성전자 이재용 회장이 서울에서 기자회견을 열었다.",
            'entities': [
                {'text': '삼성전자', 'type': 'OG', 'start': 0, 'end': 4},
                {'text': '이재용', 'type': 'PS', 'start': 5, 'end': 8},
                {'text': '서울', 'type': 'LC', 'start': 13, 'end': 15},
            ]
        },
        {
            'text': "2024년 1월 15일 대한민국에서 열린 행사",
            'entities': [
                {'text': '2024년 1월 15일', 'type': 'DT', 'start': 0, 'end': 11},
                {'text': '대한민국', 'type': 'LC', 'start': 12, 'end': 16},
            ]
        },
        {
            'text': "네이버와 카카오는 한국의 대표적인 IT 기업입니다.",
            'entities': [
                {'text': '네이버', 'type': 'OG', 'start': 0, 'end': 3},
                {'text': '카카오', 'type': 'OG', 'start': 5, 'end': 8},
                {'text': '한국', 'type': 'LC', 'start': 10, 'end': 12},
            ]
        }
    ]
    
    for ex_idx, example in enumerate(examples):
        text = example['text']
        print(f"\n{'='*60}")
        print(f"Example {ex_idx + 1}: {text}")
        print(f"{'='*60}")
        
        # Get attention
        result = get_attention_for_text(model, tokenizer, text, DEVICE)
        
        # Full attention heatmap
        visualize_attention_heatmap(
            result,
            f"{OUTPUT_DIR}/attention_heatmap_{ex_idx + 1}.png",
            title=f"Attention Heatmap: {text[:30]}..."
        )
        
        # Layer evolution
        visualize_layer_attention_evolution(
            result,
            f"{OUTPUT_DIR}/layer_evolution_{ex_idx + 1}.png"
        )
        
        # Entity-specific attention (for first entity)
        if example['entities']:
            ent = example['entities'][0]
            # Find token indices for entity
            entity_token_indices = []
            for tok_idx, word_idx in enumerate(result['word_ids']):
                if word_idx is not None and ent['start'] <= word_idx < ent['end']:
                    entity_token_indices.append(tok_idx)
            
            if entity_token_indices:
                visualize_entity_attention(
                    result,
                    entity_token_indices,
                    ent['text'],
                    ent['type'],
                    f"{OUTPUT_DIR}/entity_attention_{ex_idx + 1}_{ent['type']}.png"
                )
    
    print(f"\n\nAll visualizations saved to {OUTPUT_DIR}/")
    print("\nGenerated files:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
