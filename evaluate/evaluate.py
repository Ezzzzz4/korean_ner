"""
Comprehensive Evaluation Script for Korean NER Model
Generates all metrics, visualizations, and analysis for portfolio presentation.

Run from project root: python evaluate/evaluate.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torchcrf import CRF
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import os
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
# Get the script's directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_PATH = PROJECT_ROOT / "weights" / "best_model.pt"
OUTPUT_DIR = SCRIPT_DIR  # Save outputs in evaluate folder
BATCH_SIZE = 32
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# ============================================================================
# Model Definition (must match training)
# ============================================================================
class KoElectraNER(nn.Module):
    def __init__(self, model_name, num_labels, num_samples=5):
        super().__init__()
        self.num_labels = num_labels
        self.num_samples = num_samples
        
        self.electra = AutoModel.from_pretrained(model_name)
        hidden = self.electra.config.hidden_size
        
        self.lstm = nn.LSTM(hidden, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(512, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask, labels=None):
        enc = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = enc.last_hidden_state
        lstm_out, _ = self.lstm(seq_out)
        emissions = self.classifier(self.dropout(lstm_out))
        mask = attention_mask.bool()
        
        if labels is not None:
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0
            loss = -self.crf(emissions, labels_crf, mask=mask, reduction='mean')
            return loss, emissions
        return self.crf.decode(emissions, mask=mask)


# ============================================================================
# Main Evaluation
# ============================================================================
def main():
    print("=" * 60)
    print("KOREAN NER MODEL EVALUATION")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")

    print("\n[1/6] Loading dataset and tokenizer...")
    dataset = load_dataset("klue", "ner")
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    label_list = dataset['train'].features['ner_tags'].feature.names
    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {i: l for i, l in enumerate(label_list)}
    NUM_LABELS = len(label_list)

    entity_types = sorted(set(l.split('-')[1] for l in label_list if l != 'O'))
    print(f"   Labels: {label_list}")
    print(f"   Entity types: {entity_types}")

    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples['tokens'], truncation=True, is_split_into_words=True,
            max_length=MAX_LENGTH, padding='max_length'
        )
        labels = []
        for idx, tags in enumerate(examples['ner_tags']):
            word_ids = tokenized.word_ids(batch_index=idx)
            prev_word = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != prev_word:
                    label_ids.append(tags[word_idx])
                else:
                    orig = tags[word_idx]
                    name = id_to_label[orig]
                    if name.startswith('B-'):
                        i_name = name.replace('B-', 'I-')
                        label_ids.append(label_to_id.get(i_name, orig))
                    else:
                        label_ids.append(orig)
                prev_word = word_idx
            labels.append(label_ids)
        tokenized['labels'] = labels
        return tokenized

    tokenized = dataset.map(tokenize_and_align, batched=True)
    tokenized = tokenized.remove_columns(dataset['train'].column_names)
    tokenized.set_format('torch')

    val_loader = DataLoader(tokenized['validation'], batch_size=BATCH_SIZE)
    print(f"   Validation samples: {len(tokenized['validation'])}")

    # Load Model
    print("\n[2/6] Loading model...")
    model = KoElectraNER("monologg/koelectra-base-v3-discriminator", NUM_LABELS, num_samples=5)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Run Predictions
    print("\n[3/6] Running predictions...")
    all_preds = []
    all_labels = []
    all_tokens = []
    raw_predictions = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            batch_labels = batch['labels']
            
            preds = model(input_ids, attention_mask)
            
            for i, (p, l, m) in enumerate(zip(preds, batch_labels.numpy(), attention_mask.cpu().numpy())):
                length = int(m.sum())
                tokens = tokenizer.convert_ids_to_tokens(input_ids[i].cpu().numpy()[:length])
                
                pred_tags = []
                true_tags = []
                sample_tokens = []
                
                for j, (pred_id, label_id) in enumerate(zip(p[:length], l[:length])):
                    if label_id != -100:
                        pred_tags.append(label_list[pred_id])
                        true_tags.append(label_list[int(label_id)])
                        sample_tokens.append(tokens[j] if j < len(tokens) else '[UNK]')
                        raw_predictions.append((label_list[int(label_id)], label_list[pred_id]))
                
                all_preds.append(pred_tags)
                all_labels.append(true_tags)
                all_tokens.append(sample_tokens)

    # Calculate Metrics
    print("\n[4/6] Calculating metrics...")
    overall_f1 = f1_score(all_labels, all_preds, mode='strict', scheme=IOB2)
    overall_precision = precision_score(all_labels, all_preds, mode='strict', scheme=IOB2)
    overall_recall = recall_score(all_labels, all_preds, mode='strict', scheme=IOB2)

    report_str = classification_report(all_labels, all_preds, mode='strict', scheme=IOB2, digits=4)
    report_dict = classification_report(all_labels, all_preds, mode='strict', scheme=IOB2, digits=4, output_dict=True)

    print(f"\n   Overall F1: {overall_f1:.4f}")
    print(f"   Precision:  {overall_precision:.4f}")
    print(f"   Recall:     {overall_recall:.4f}")

    # Save classification report
    with open(OUTPUT_DIR / "classification_report.txt", 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("KLUE NER EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: KoELECTRA-base-v3 + BiLSTM + CRF\n")
        f.write(f"Validation samples: {len(tokenized['validation'])}\n\n")
        f.write("-" * 60 + "\n")
        f.write("ENTITY-LEVEL METRICS (Strict Evaluation)\n")
        f.write("-" * 60 + "\n\n")
        f.write(report_str)
        f.write("\n" + "-" * 60 + "\n")
        f.write(f"Overall Entity F1: {overall_f1:.4f}\n")
        f.write(f"KLUE Paper Baseline (KoELECTRA-base): 0.8611\n")
        f.write("-" * 60 + "\n")

    # Save metrics as JSON
    metrics = {
        "overall": {
            "f1": float(round(overall_f1, 4)),
            "precision": float(round(overall_precision, 4)),
            "recall": float(round(overall_recall, 4))
        },
        "per_entity": {}
    }

    for entity in entity_types:
        if entity in report_dict:
            metrics["per_entity"][entity] = {
                "f1": float(round(report_dict[entity]['f1-score'], 4)),
                "precision": float(round(report_dict[entity]['precision'], 4)),
                "recall": float(round(report_dict[entity]['recall'], 4)),
                "support": int(report_dict[entity]['support'])
            }

    with open(OUTPUT_DIR / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Generate Visualizations
    print("\n[5/6] Generating visualizations...")

    # 1. Per-Entity F1 Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    entities = []
    f1_scores = []
    precisions = []
    recalls = []

    for entity in entity_types:
        if entity in report_dict:
            entities.append(entity)
            f1_scores.append(report_dict[entity]['f1-score'])
            precisions.append(report_dict[entity]['precision'])
            recalls.append(report_dict[entity]['recall'])

    x = np.arange(len(entities))
    width = 0.25

    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#2ecc71', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Entity Type', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Named Entity Recognition Performance by Entity Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(entities, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=overall_f1, color='#9b59b6', linestyle='--', linewidth=2)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_entity_f1.png", bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: per_entity_f1.png")

    # 2. Confusion Matrix
    print("   Generating confusion matrix...")
    confusion = defaultdict(lambda: defaultdict(int))
    for true, pred in raw_predictions:
        confusion[true][pred] += 1

    matrix = np.zeros((len(label_list), len(label_list)))
    for i, true_label in enumerate(label_list):
        for j, pred_label in enumerate(label_list):
            matrix[i, j] = confusion[true_label][pred_label]

    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_normalized = np.divide(matrix, row_sums, where=row_sums!=0)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_list, yticklabels=label_list, ax=ax,
                cbar_kws={'label': 'Proportion'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Token-Level Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: confusion_matrix.png")

    # 3. Entity-only confusion matrix
    entity_confusion = defaultdict(lambda: defaultdict(int))
    for true, pred in raw_predictions:
        true_entity = true.split('-')[1] if '-' in true else true
        pred_entity = pred.split('-')[1] if '-' in pred else pred
        entity_confusion[true_entity][pred_entity] += 1

    entity_labels = ['O'] + entity_types
    entity_matrix = np.zeros((len(entity_labels), len(entity_labels)))
    for i, true_label in enumerate(entity_labels):
        for j, pred_label in enumerate(entity_labels):
            entity_matrix[i, j] = entity_confusion[true_label][pred_label]

    row_sums = entity_matrix.sum(axis=1, keepdims=True)
    entity_matrix_norm = np.divide(entity_matrix, row_sums, where=row_sums!=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(entity_matrix_norm, annot=True, fmt='.2f', cmap='RdYlGn_r',
                xticklabels=entity_labels, yticklabels=entity_labels, ax=ax,
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
    ax.set_xlabel('Predicted Entity', fontsize=12)
    ax.set_ylabel('True Entity', fontsize=12)
    ax.set_title('Entity-Type Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "entity_confusion_matrix.png", bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved: entity_confusion_matrix.png")

    # Example Predictions
    print("\n[6/6] Generating example predictions...")
    examples_output = []
    examples_output.append("=" * 70)
    examples_output.append("EXAMPLE PREDICTIONS")
    examples_output.append("=" * 70 + "\n")

    correct_examples = []
    error_examples = []

    for i, (tokens, preds, labels) in enumerate(zip(all_tokens, all_preds, all_labels)):
        has_entity = any(l != 'O' for l in labels)
        is_correct = preds == labels
        
        if has_entity:
            if is_correct and len(correct_examples) < 3:
                correct_examples.append((tokens, preds, labels))
            elif not is_correct and len(error_examples) < 3:
                error_examples.append((tokens, preds, labels))

    examples_output.append("--- CORRECT PREDICTIONS ---\n")
    for tokens, preds, labels in correct_examples:
        examples_output.append(f"Tokens: {' '.join(tokens[:50])}")
        examples_output.append(f"True:   {' '.join(labels[:50])}")
        examples_output.append(f"Pred:   {' '.join(preds[:50])}")
        examples_output.append("")

    examples_output.append("\n--- PREDICTIONS WITH ERRORS ---\n")
    for tokens, preds, labels in error_examples:
        examples_output.append(f"Tokens: {' '.join(tokens[:50])}")
        examples_output.append(f"True:   {' '.join(labels[:50])}")
        examples_output.append(f"Pred:   {' '.join(preds[:50])}")
        diffs = ['^^^' if p != l else '   ' for p, l in zip(preds[:50], labels[:50])]
        examples_output.append(f"Diff:   {' '.join(diffs)}")
        examples_output.append("")

    with open(OUTPUT_DIR / "example_predictions.txt", 'w', encoding='utf-8') as f:
        f.write('\n'.join(examples_output))
    print(f"   Saved: example_predictions.txt")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - classification_report.txt")
    print(f"  - metrics.json")
    print(f"  - per_entity_f1.png")
    print(f"  - confusion_matrix.png")
    print(f"  - entity_confusion_matrix.png")
    print(f"  - example_predictions.txt")
    print(f"\n{'='*60}")
    print(f"FINAL SCORE: F1 = {overall_f1:.4f}")
    print(f"KLUE Baseline: F1 = 0.8611")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
