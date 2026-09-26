"""Error analysis script for Korean NER."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_PATH = REPO_ROOT / "weights" / "best_model.pt"
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
MAX_LENGTH = 192
OUTPUT_DIR = REPO_ROOT / "runs" / "error_analysis"

ENTITY_NAMES = {
    'PS': 'Person',
    'LC': 'Location', 
    'OG': 'Organization',
    'DT': 'Date',
    'TI': 'Time',
    'QT': 'Quantity'
}


def entity_to_dict(entity: object) -> dict:
    """Return the script's historical entity dict shape."""
    if isinstance(entity, dict):
        return entity
    return {
        'text': entity.text,
        'type': entity.label,
        'start': entity.start,
        'end': entity.end,
    }


def entities_to_dicts(entities: Iterable[object]) -> list[dict]:
    return [entity_to_dict(entity) for entity in entities]


def spans_overlap(left, right):
    """Return True when half-open character spans share at least one char."""
    return max(left['start'], right['start']) < min(left['end'], right['end'])


def overlap_size(left, right):
    """Return the number of shared characters between half-open spans."""
    return max(0, min(left['end'], right['end']) - max(left['start'], right['start']))


def same_boundary(left, right):
    """Return True when two entity spans have identical character boundaries."""
    return left['start'] == right['start'] and left['end'] == right['end']


def analyze_errors(true_entities, pred_entities, text):
    """Categorize errors between true and predicted entities."""
    errors = []
    true_entities = entities_to_dicts(true_entities)
    pred_entities = entities_to_dicts(pred_entities)

    matched_true = set()
    matched_pred = set()

    # Exact boundaries are decided before partial overlaps so wrong labels are
    # counted as type confusions instead of boundary errors.
    for true_idx, true_ent in enumerate(true_entities):
        for pred_idx, pred_ent in enumerate(pred_entities):
            if pred_idx in matched_pred or not same_boundary(true_ent, pred_ent):
                continue

            matched_true.add(true_idx)
            matched_pred.add(pred_idx)
            if true_ent['type'] != pred_ent['type']:
                errors.append({
                    'type': 'type_confusion',
                    'true_entity': true_ent,
                    'pred_entity': pred_ent,
                    'text': text
                })
            break

    overlap_pairs = []
    for true_idx, true_ent in enumerate(true_entities):
        if true_idx in matched_true:
            continue
        for pred_idx, pred_ent in enumerate(pred_entities):
            if pred_idx in matched_pred:
                continue
            overlap = overlap_size(true_ent, pred_ent)
            if overlap > 0:
                overlap_pairs.append((overlap, true_idx, pred_idx))

    for _, true_idx, pred_idx in sorted(overlap_pairs, key=lambda item: (-item[0], item[1], item[2])):
        if true_idx in matched_true or pred_idx in matched_pred:
            continue
        matched_true.add(true_idx)
        matched_pred.add(pred_idx)
        errors.append({
            'type': 'boundary_error',
            'true_entity': true_entities[true_idx],
            'pred_entity': pred_entities[pred_idx],
            'text': text
        })

    for true_idx, true_ent in enumerate(true_entities):
        if true_idx not in matched_true:
            errors.append({
                'type': 'missed_entity',
                'true_entity': true_ent,
                'text': text
            })

    for pred_idx, pred_ent in enumerate(pred_entities):
        if pred_idx not in matched_pred:
            errors.append({
                'type': 'spurious_entity',
                'pred_entity': pred_ent,
                'text': text
            })

    return errors


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Analyze Korean NER model errors on KLUE validation data.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to a raw or wrapped model checkpoint.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Hugging Face model name or path.")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Aligned token sequence length.")
    parser.add_argument("--device", default="cpu", help="Torch device to use. Defaults to explicit CPU.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of validation samples to analyze.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for error_analysis reports.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from korean_ner.alignment import align_text
    from korean_ner.checkpoint import load_model_state
    from korean_ner.decode import extract_bio_spans, token_predictions_to_char_labels
    from korean_ner.labels import build_label_maps, labels_from_klue_dataset, to_label_names
    from korean_ner.model import KoElectraNER

    args = parse_args(argv)
    device = torch.device(args.device)

    print("=" * 60)
    print("Korean NER Error Analysis")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading KLUE NER validation set...")
    dataset = load_dataset('klue', 'ner')
    label_maps = build_label_maps(labels_from_klue_dataset(dataset))
    model = KoElectraNER(args.model_name, len(label_maps.names))
    load_model_state(model, args.model_path, map_location=device)
    model.to(device)
    model.eval()

    val_data = dataset['validation']
    if args.limit is not None:
        val_data = val_data.select(range(min(args.limit, len(val_data))))
    
    # Collect all errors
    all_errors = []
    error_counts = defaultdict(int)
    type_confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    print(f"\nAnalyzing {len(val_data)} samples...")
    
    for idx, example in enumerate(val_data):
        if idx % 500 == 0:
            print(f"  Processing {idx}/{len(val_data)}...")
        
        text = ''.join(example['tokens'])
        true_labels = to_label_names(example['ner_tags'], label_maps.id_to_label)
        
        # Get predictions
        encoding = align_text(
            text,
            tokenizer,
            max_length=args.max_length,
            label_to_id=label_maps.label_to_id,
            id_to_label=label_maps.id_to_label,
        )

        input_ids = torch.tensor([encoding.input_ids], dtype=torch.long, device=device)
        attention_mask = torch.tensor([encoding.attention_mask], dtype=torch.long, device=device)
        
        with torch.no_grad():
            predictions = model(input_ids, attention_mask)[0]
        
        # Map predictions to characters
        pred_labels = token_predictions_to_char_labels(
            predictions,
            encoding.char_indices,
            label_maps.id_to_label,
            text_length=len(text),
        )
        
        # Extract entities
        true_entities = extract_bio_spans(text, true_labels)
        pred_entities = extract_bio_spans(text, pred_labels)
        
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
    entity_types = sorted({label.split('-', 1)[1] for label in label_maps.names if '-' in label})
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        'total_errors': total_errors,
        'error_counts': dict(error_counts),
        'type_confusion': {k: dict(v) for k, v in type_confusion_matrix.items()},
        'sample_errors': all_errors[:100]  # Save first 100 for reference
    }
    
    with (output_dir / "error_analysis.json").open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Save text report
    with (output_dir / "error_analysis.txt").open('w', encoding='utf-8') as f:
        f.write("Korean NER Error Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Error Type Distribution:\n")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_errors if total_errors > 0 else 0
            f.write(f"  {error_type:20s}: {count:5d} ({pct:5.1f}%)\n")
        
        f.write(f"\nTotal Errors: {total_errors}\n")
    
    print(f"\n\nReports saved to {output_dir}/")
    print("  - error_analysis.json")
    print("  - error_analysis.txt")


if __name__ == "__main__":
    main()
