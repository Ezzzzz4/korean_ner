"""
Error Analysis Script for Korean NER
Analyzes model errors: boundary errors, type confusion, and failure patterns.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
from datasets import load_dataset
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

# KLUE NER Labels (correct order)
LABEL_LIST = ['B-DT', 'I-DT', 'B-LC', 'I-LC', 'B-OG', 'I-OG', 'B-PS', 'I-PS', 'B-QT', 'I-QT', 'B-TI', 'I-TI', 'O']
ID_TO_LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

ENTITY_NAMES = {
    'PS': 'Person',
    'LC': 'Location', 
    'OG': 'Organization',
    'DT': 'Date',
    'TI': 'Time',
    'QT': 'Quantity'
}


# ============================================================================
# Model Definition
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


def extract_entities(chars, labels):
    """Extract entity spans from character-level labels."""
    entities = []
    current_type = None
    current_chars = []
    start_idx = 0
    
    for idx, (char, label) in enumerate(zip(chars, labels)):
        if label.startswith('B-'):
            if current_type and current_chars:
                entities.append({
                    'text': ''.join(current_chars),
                    'type': current_type,
                    'start': start_idx,
                    'end': idx
                })
            current_type = label[2:]
            current_chars = [char]
            start_idx = idx
        elif label.startswith('I-'):
            etype = label[2:]
            if current_type == etype:
                current_chars.append(char)
            else:
                if current_type and current_chars:
                    entities.append({
                        'text': ''.join(current_chars),
                        'type': current_type,
                        'start': start_idx,
                        'end': idx
                    })
                current_type = etype
                current_chars = [char]
                start_idx = idx
        else:
            if current_type and current_chars:
                entities.append({
                    'text': ''.join(current_chars),
                    'type': current_type,
                    'start': start_idx,
                    'end': idx
                })
            current_type = None
            current_chars = []
    
    if current_type and current_chars:
        entities.append({
            'text': ''.join(current_chars),
            'type': current_type,
            'start': start_idx,
            'end': len(chars)
        })
    
    return entities


def analyze_errors(true_entities, pred_entities, text):
    """Categorize errors between true and predicted entities."""
    errors = []
    
    # Create lookup by position
    true_by_pos = {(e['start'], e['end']): e for e in true_entities}
    pred_by_pos = {(e['start'], e['end']): e for e in pred_entities}
    
    # Find missed entities (false negatives)
    for pos, true_ent in true_by_pos.items():
        if pos not in pred_by_pos:
            # Check for partial matches
            partial = None
            for pred_pos, pred_ent in pred_by_pos.items():
                if (pred_pos[0] <= pos[0] < pred_pos[1]) or (pred_pos[0] < pos[1] <= pred_pos[1]):
                    partial = pred_ent
                    break
            
            if partial:
                errors.append({
                    'type': 'boundary_error',
                    'true_entity': true_ent,
                    'pred_entity': partial,
                    'text': text
                })
            else:
                errors.append({
                    'type': 'missed_entity',
                    'true_entity': true_ent,
                    'text': text
                })
    
    # Find spurious entities (false positives)
    for pos, pred_ent in pred_by_pos.items():
        if pos not in true_by_pos:
            # Check if it overlaps with any true entity
            overlaps = False
            for true_pos in true_by_pos:
                if (true_pos[0] <= pos[0] < true_pos[1]) or (true_pos[0] < pos[1] <= true_pos[1]):
                    overlaps = True
                    break
            
            if not overlaps:
                errors.append({
                    'type': 'spurious_entity',
                    'pred_entity': pred_ent,
                    'text': text
                })
    
    # Find type confusions (correct boundary, wrong type)
    for pos in true_by_pos:
        if pos in pred_by_pos:
            true_ent = true_by_pos[pos]
            pred_ent = pred_by_pos[pos]
            if true_ent['type'] != pred_ent['type']:
                errors.append({
                    'type': 'type_confusion',
                    'true_entity': true_ent,
                    'pred_entity': pred_ent,
                    'text': text
                })
    
    return errors


def main():
    print("=" * 60)
    print("Korean NER Error Analysis")
    print("=" * 60)
    
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
    
    # Collect all errors
    all_errors = []
    error_counts = defaultdict(int)
    type_confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    print(f"\nAnalyzing {len(val_data)} samples...")
    
    for idx, example in enumerate(val_data):
        if idx % 500 == 0:
            print(f"  Processing {idx}/{len(val_data)}...")
        
        chars = example['tokens']
        true_labels = [ID_TO_LABEL[t] for t in example['ner_tags']]
        text = ''.join(chars)
        
        # Get predictions
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
            predictions = model(input_ids, attention_mask)[0]
        
        # Map predictions to characters
        pred_labels = ['O'] * len(chars)
        for tok_idx, (pred_id, word_idx) in enumerate(zip(predictions, word_ids)):
            if word_idx is not None and word_idx < len(chars):
                if pred_labels[word_idx] == 'O':  # First subword only
                    pred_labels[word_idx] = ID_TO_LABEL[pred_id]
        
        # Extract entities
        true_entities = extract_entities(chars, true_labels)
        pred_entities = extract_entities(chars, pred_labels)
        
        # Analyze errors
        errors = analyze_errors(true_entities, pred_entities, text)
        all_errors.extend(errors)
        
        for error in errors:
            error_counts[error['type']] += 1
            if error['type'] == 'type_confusion':
                true_type = error['true_entity']['type']
                pred_type = error['pred_entity']['type']
                type_confusion_matrix[true_type][pred_type] += 1
    
    # Generate report
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS RESULTS")
    print("=" * 60)
    
    print("\n## Error Type Distribution")
    print("-" * 40)
    total_errors = sum(error_counts.values())
    for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_errors if total_errors > 0 else 0
        print(f"  {error_type:20s}: {count:5d} ({pct:5.1f}%)")
    print(f"  {'TOTAL':20s}: {total_errors:5d}")
    
    print("\n## Type Confusion Matrix")
    print("-" * 40)
    entity_types = ['PS', 'LC', 'OG', 'DT', 'TI', 'QT']
    print("True\\Pred |", " | ".join(f"{t:>4s}" for t in entity_types))
    print("-" * 50)
    for true_type in entity_types:
        row = [type_confusion_matrix[true_type][pred_type] for pred_type in entity_types]
        print(f"    {true_type:4s}  |", " | ".join(f"{c:4d}" for c in row))
    
    print("\n## Sample Errors")
    print("-" * 40)
    
    # Show examples of each error type
    for error_type in ['missed_entity', 'spurious_entity', 'type_confusion', 'boundary_error']:
        examples = [e for e in all_errors if e['type'] == error_type][:3]
        if examples:
            print(f"\n### {error_type.replace('_', ' ').title()} Examples:")
            for ex in examples:
                print(f"\n  Text: {ex['text'][:80]}...")
                if 'true_entity' in ex:
                    te = ex['true_entity']
                    print(f"  True: '{te['text']}' ({ENTITY_NAMES.get(te['type'], te['type'])})")
                if 'pred_entity' in ex:
                    pe = ex['pred_entity']
                    print(f"  Pred: '{pe['text']}' ({ENTITY_NAMES.get(pe['type'], pe['type'])})")
    
    # Save detailed report
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    report = {
        'total_errors': total_errors,
        'error_counts': dict(error_counts),
        'type_confusion': {k: dict(v) for k, v in type_confusion_matrix.items()},
        'sample_errors': all_errors[:100]  # Save first 100 for reference
    }
    
    with open(f"{OUTPUT_DIR}/error_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Save text report
    with open(f"{OUTPUT_DIR}/error_analysis.txt", 'w', encoding='utf-8') as f:
        f.write("Korean NER Error Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Error Type Distribution:\n")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_errors if total_errors > 0 else 0
            f.write(f"  {error_type:20s}: {count:5d} ({pct:5.1f}%)\n")
        
        f.write(f"\nTotal Errors: {total_errors}\n")
    
    print(f"\n\nReports saved to {OUTPUT_DIR}/")
    print("  - error_analysis.json")
    print("  - error_analysis.txt")


if __name__ == "__main__":
    main()
