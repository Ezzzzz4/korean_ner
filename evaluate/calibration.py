"""
Confidence Calibration Analysis for Korean NER
Analyzes whether model confidence correlates with prediction correctness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json
import os

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = "../weights/best_model.pt"
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
MAX_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "../assets"

# KLUE NER Labels
LABEL_LIST = ['B-DT', 'I-DT', 'B-LC', 'I-LC', 'B-OG', 'I-OG', 'B-PS', 'I-PS', 'B-QT', 'I-QT', 'B-TI', 'I-TI', 'O']
ID_TO_LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)


# ============================================================================
# Model Definition (with emission output)
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
    
    def forward(self, input_ids, attention_mask, labels=None, return_emissions=False):
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
        
        predictions = self.crf.decode(emissions, mask=mask)
        
        if return_emissions:
            return predictions, emissions
        return predictions


def compute_calibration_metrics(confidences, accuracies, num_bins=10):
    """Compute Expected Calibration Error (ECE) and related metrics."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    mce = 0  # Maximum Calibration Error
    bin_stats = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            bin_error = abs(avg_accuracy - avg_confidence)
            
            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)
            
            bin_stats.append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'avg_confidence': float(avg_confidence),
                'avg_accuracy': float(avg_accuracy),
                'count': int(in_bin.sum()),
                'proportion': float(prop_in_bin)
            })
        else:
            bin_stats.append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'avg_confidence': None,
                'avg_accuracy': None,
                'count': 0,
                'proportion': 0
            })
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'bin_stats': bin_stats
    }


def plot_reliability_diagram(bin_stats, output_path):
    """Create reliability diagram (calibration plot)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Reliability diagram
    confidences = []
    accuracies = []
    counts = []
    
    for bin_stat in bin_stats:
        if bin_stat['avg_confidence'] is not None:
            confidences.append(bin_stat['avg_confidence'])
            accuracies.append(bin_stat['avg_accuracy'])
            counts.append(bin_stat['count'])
    
    if confidences:
        ax1.bar(confidences, accuracies, width=0.08, alpha=0.7, color='#45B7D1', edgecolor='black', label='Model')
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax1.set_xlabel('Mean Predicted Confidence')
        ax1.set_ylabel('Fraction of Correct Predictions')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
    
    # Right: Confidence histogram
    bin_centers = [(bs['bin_lower'] + bs['bin_upper']) / 2 for bs in bin_stats]
    bin_counts = [bs['count'] for bs in bin_stats]
    
    ax2.bar(bin_centers, bin_counts, width=0.09, color='#96CEB4', edgecolor='black')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_confidence_accuracy_curve(confidences, accuracies, output_path):
    """Plot accuracy as a function of confidence threshold."""
    thresholds = np.linspace(0, 0.99, 50)
    
    accs_at_threshold = []
    coverage_at_threshold = []
    
    for thresh in thresholds:
        mask = confidences >= thresh
        if mask.sum() > 0:
            accs_at_threshold.append(accuracies[mask].mean())
            coverage_at_threshold.append(mask.mean())
        else:
            accs_at_threshold.append(np.nan)
            coverage_at_threshold.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Accuracy vs threshold
    ax1.plot(thresholds, accs_at_threshold, 'b-', linewidth=2)
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Accuracy (above threshold)')
    ax1.set_title('Accuracy vs Confidence Threshold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0.5, 1)
    
    # Right: Coverage vs threshold
    ax2.plot(thresholds, coverage_at_threshold, 'g-', linewidth=2)
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Coverage (fraction above threshold)')
    ax2.set_title('Coverage vs Confidence Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Confidence Calibration Analysis for Korean NER")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = KoElectraNER(MODEL_NAME, NUM_LABELS)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(DEVICE)
    model.eval()
    
    # Load validation data
    print("Loading KLUE NER validation set...")
    dataset = load_dataset('klue', 'ner')
    val_data = dataset['validation']
    
    # Collect confidence and correctness for all tokens
    all_confidences = []
    all_correct = []
    entity_confidences = defaultdict(list)
    entity_correct = defaultdict(list)
    
    print(f"\nAnalyzing {len(val_data)} samples...")
    
    for idx, example in enumerate(val_data):
        if idx % 500 == 0:
            print(f"  Processing {idx}/{len(val_data)}...")
        
        chars = example['tokens']
        true_labels = example['ner_tags']
        
        # Get predictions with emissions
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
        
        with torch.no_grad():
            predictions, emissions = model(input_ids, attention_mask, return_emissions=True)
        
        # Convert emissions to probabilities
        probs = F.softmax(emissions[0], dim=-1)  # (seq_len, num_labels)
        
        # For each token, get confidence and correctness
        for tok_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx >= len(chars):
                continue
            
            # Skip if already processed this word
            if tok_idx > 0 and word_ids[tok_idx - 1] == word_idx:
                continue
            
            pred_label = predictions[0][tok_idx]
            true_label = true_labels[word_idx]
            
            # Get confidence (max probability)
            confidence = probs[tok_idx].max().item()
            correct = int(pred_label == true_label)
            
            all_confidences.append(confidence)
            all_correct.append(correct)
            
            # Track by entity type
            true_label_name = ID_TO_LABEL[true_label]
            if true_label_name != 'O':
                entity_type = true_label_name.split('-')[1]
                entity_confidences[entity_type].append(confidence)
                entity_correct[entity_type].append(correct)
    
    all_confidences = np.array(all_confidences)
    all_correct = np.array(all_correct)
    
    print(f"\nTotal tokens analyzed: {len(all_confidences)}")
    print(f"Overall accuracy: {all_correct.mean():.4f}")
    print(f"Mean confidence: {all_confidences.mean():.4f}")
    
    # Compute calibration metrics
    print("\n" + "=" * 40)
    print("CALIBRATION METRICS")
    print("=" * 40)
    
    calibration = compute_calibration_metrics(all_confidences, all_correct, num_bins=10)
    
    print(f"\nExpected Calibration Error (ECE): {calibration['ece']:.4f}")
    print(f"Maximum Calibration Error (MCE): {calibration['mce']:.4f}")
    
    print("\nBin Statistics:")
    print("-" * 60)
    print(f"{'Confidence Range':<20} | {'Accuracy':<10} | {'Confidence':<10} | {'Count':<10}")
    print("-" * 60)
    for bs in calibration['bin_stats']:
        if bs['count'] > 0:
            print(f"[{bs['bin_lower']:.1f}, {bs['bin_upper']:.1f}]" + " " * 10 + 
                  f"| {bs['avg_accuracy']:.4f}" + " " * 4 +
                  f"| {bs['avg_confidence']:.4f}" + " " * 4 +
                  f"| {bs['count']}")
    
    # Per-entity calibration
    print("\n" + "=" * 40)
    print("PER-ENTITY CALIBRATION")
    print("=" * 40)
    
    entity_calibration = {}
    for entity_type in entity_confidences:
        conf = np.array(entity_confidences[entity_type])
        corr = np.array(entity_correct[entity_type])
        ece = compute_calibration_metrics(conf, corr)['ece']
        entity_calibration[entity_type] = {
            'ece': ece,
            'mean_confidence': float(conf.mean()),
            'accuracy': float(corr.mean()),
            'count': len(conf)
        }
        print(f"\n{entity_type}:")
        print(f"  ECE: {ece:.4f}")
        print(f"  Mean Confidence: {conf.mean():.4f}")
        print(f"  Accuracy: {corr.mean():.4f}")
        print(f"  Count: {len(conf)}")
    
    # Generate visualizations
    print("\n" + "=" * 40)
    print("GENERATING VISUALIZATIONS")
    print("=" * 40)
    
    plot_reliability_diagram(
        calibration['bin_stats'],
        f"{OUTPUT_DIR}/calibration_reliability_diagram.png"
    )
    
    plot_confidence_accuracy_curve(
        all_confidences,
        all_correct,
        f"{OUTPUT_DIR}/calibration_accuracy_curve.png"
    )
    
    # Save results
    results = {
        'overall': {
            'ece': calibration['ece'],
            'mce': calibration['mce'],
            'mean_confidence': float(all_confidences.mean()),
            'accuracy': float(all_correct.mean()),
            'total_tokens': len(all_confidences)
        },
        'bin_stats': calibration['bin_stats'],
        'per_entity': entity_calibration
    }
    
    with open(f"{OUTPUT_DIR}/calibration_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR}/calibration_results.json")


if __name__ == "__main__":
    main()
