"""
Ablation Study for Korean NER
Trains multiple model variants to quantify individual technique contributions.

Features:
- Automatic checkpoint saving after each epoch
- Resume from interruption (continues from last completed epoch)
- Early stopping with configurable patience
- Tracks all experiments in ablation_status.json
- Generates comparison report and visualizations

Experiments:
1. baseline: KoELECTRA + Linear (no BiLSTM, no CRF)
2. +crf: baseline + CRF
3. +bilstm: baseline + BiLSTM (no CRF)
4. full_no_fgm: Full model without FGM
5. full_no_rdrop: Full model without R-Drop
6. full: Complete model (BiLSTM + CRF + FGM + R-Drop)

Setup:
    pip install -r requirements.txt

Usage:
    python train.py                    # Run all experiments
    python train.py --experiment full  # Run specific experiment
    python train.py --status           # Show progress
    python train.py --report           # Generate comparison report
"""

import os
import sys

# Python 3.11 version check
if sys.version_info < (3, 11):
    print(f"ERROR: Python 3.11+ required. You have Python {sys.version_info.major}.{sys.version_info.minor}")
    print("Please install Python 3.11 and run the setup script again.")
    sys.exit(1)

import json
import argparse
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from datasets import load_dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 40
PATIENCE = 10  # Early stopping patience
ENCODER_LR = 3e-5
HEAD_LR = 1e-3
WARMUP_RATIO = 0.1
GRADIENT_CLIP = 1.0
FGM_EPSILON = 0.5
RDROP_ALPHA = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).parent
SAVE_DIR = BASE_DIR / "checkpoints"
STATUS_FILE = BASE_DIR / "ablation_status.json"
RESULTS_DIR = BASE_DIR / "results"
LOG_FILE = BASE_DIR / "log.txt"

# ============================================================================
# Logging Setup
# ============================================================================
def setup_logging():
    """Setup logging to both console and file with session markers."""
    formatter = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger = logging.getLogger('ablation')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Add session separator to log file
    session_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"NEW SESSION STARTED: {session_time}")
    logger.info(f"Device: {DEVICE} | PyTorch: {torch.__version__}")
    logger.info("=" * 60)
    
    return logger

logger = setup_logging()

# KLUE NER Labels
LABEL_LIST = ['B-DT', 'I-DT', 'B-LC', 'I-LC', 'B-OG', 'I-OG', 'B-PS', 'I-PS', 'B-QT', 'I-QT', 'B-TI', 'I-TI', 'O']
ID_TO_LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL_TO_ID = {l: i for i, l in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

# Experiment configurations
EXPERIMENTS = {
    'baseline': {
        'use_bilstm': False,
        'use_crf': False,
        'use_fgm': False,
        'use_rdrop': False,
        'description': 'KoELECTRA + Linear head only'
    },
    '+crf': {
        'use_bilstm': False,
        'use_crf': True,
        'use_fgm': False,
        'use_rdrop': False,
        'description': 'Baseline + CRF layer'
    },
    '+bilstm': {
        'use_bilstm': True,
        'use_crf': False,
        'use_fgm': False,
        'use_rdrop': False,
        'description': 'Baseline + BiLSTM layer'
    },
    'full_no_fgm': {
        'use_bilstm': True,
        'use_crf': True,
        'use_fgm': False,
        'use_rdrop': True,
        'description': 'Full model without FGM'
    },
    'full_no_rdrop': {
        'use_bilstm': True,
        'use_crf': True,
        'use_fgm': True,
        'use_rdrop': False,
        'description': 'Full model without R-Drop'
    },
    'full': {
        'use_bilstm': True,
        'use_crf': True,
        'use_fgm': True,
        'use_rdrop': True,
        'description': 'Complete model with all techniques'
    }
}


# ============================================================================
# Model Definitions
# ============================================================================
try:
    from torchcrf import CRF
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("Warning: pytorch-crf not installed. CRF experiments will fail.")


class AblationNERModel(nn.Module):
    """Configurable NER model for ablation study."""
    
    def __init__(self, model_name, num_labels, use_bilstm=True, use_crf=True):
        super().__init__()
        self.num_labels = num_labels
        self.use_bilstm = use_bilstm
        self.use_crf = use_crf
        
        self.electra = AutoModel.from_pretrained(model_name)
        hidden = self.electra.config.hidden_size  # 768
        
        if use_bilstm:
            self.lstm = nn.LSTM(hidden, 256, num_layers=1, batch_first=True, bidirectional=True)
            classifier_input = 512
        else:
            self.lstm = None
            classifier_input = hidden
        
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(classifier_input, num_labels)
        
        if use_crf and CRF_AVAILABLE:
            self.crf = CRF(num_labels, batch_first=True)
        else:
            self.crf = None
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask, labels=None):
        enc = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        seq_out = enc.last_hidden_state
        
        if self.use_bilstm and self.lstm is not None:
            seq_out, _ = self.lstm(seq_out)
        
        emissions = self.classifier(self.dropout(seq_out))
        mask = attention_mask.bool()
        
        if labels is not None:
            if self.use_crf and self.crf is not None:
                labels_crf = labels.clone()
                labels_crf[labels_crf == -100] = 0
                loss = -self.crf(emissions, labels_crf, mask=mask, reduction='mean')
            else:
                loss = F.cross_entropy(
                    emissions.view(-1, self.num_labels),
                    labels.view(-1),
                    ignore_index=-100
                )
            return loss, emissions
        
        if self.use_crf and self.crf is not None:
            return self.crf.decode(emissions, mask=mask)
        else:
            return emissions.argmax(dim=-1).tolist()


class FGM:
    """Fast Gradient Method for adversarial training."""
    
    def __init__(self, model, epsilon=0.5):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}
    
    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'word_embeddings' in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    perturbation = self.epsilon * param.grad / norm
                    param.data.add_(perturbation)
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================================
# Data Loading
# ============================================================================
def load_data(tokenizer):
    """Load and preprocess KLUE NER dataset."""
    print("Downloading KLUE NER dataset...")
    dataset = load_dataset('klue', 'ner')
    
    def tokenize_and_align(examples):
        tokenized = tokenizer(
            examples['tokens'],
            truncation=True,
            is_split_into_words=True,
            max_length=MAX_LENGTH,
            padding='max_length'
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
                    # Convert B- to I- for subwords
                    orig = tags[word_idx]
                    name = ID_TO_LABEL[orig]
                    if name.startswith('B-'):
                        i_name = name.replace('B-', 'I-')
                        label_ids.append(LABEL_TO_ID.get(i_name, orig))
                    else:
                        label_ids.append(orig)
                prev_word = word_idx
            labels.append(label_ids)
        
        tokenized['labels'] = labels
        return tokenized
    
    print("Preprocessing training data...")
    train_data = dataset['train'].map(tokenize_and_align, batched=True, remove_columns=dataset['train'].column_names)
    print("Preprocessing validation data...")
    val_data = dataset['validation'].map(tokenize_and_align, batched=True, remove_columns=dataset['validation'].column_names)
    
    train_data.set_format('torch')
    val_data.set_format('torch')
    
    return train_data, val_data


# ============================================================================
# Training Functions
# ============================================================================
def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model(input_ids, attention_mask)
            
            for pred, label, mask in zip(predictions, labels, attention_mask):
                pred_tags = []
                true_tags = []
                
                if isinstance(pred, list):
                    pred_seq = pred
                else:
                    pred_seq = pred.tolist()
                
                for p, l, m in zip(pred_seq, label.tolist(), mask.tolist()):
                    if m == 1 and l != -100:
                        pred_tags.append(ID_TO_LABEL[p])
                        true_tags.append(ID_TO_LABEL[l])
                
                if pred_tags:
                    all_preds.append(pred_tags)
                    all_labels.append(true_tags)
    
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    return {'f1': f1, 'precision': precision, 'recall': recall}


def train_epoch(model, train_loader, optimizer, scheduler, scaler, config, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    fgm = FGM(model, FGM_EPSILON) if config['use_fgm'] else None
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            loss, emissions = model(input_ids, attention_mask, labels)
            
            # R-Drop: Forward twice and add KL divergence
            if config['use_rdrop']:
                loss2, emissions2 = model(input_ids, attention_mask, labels)
                
                # KL divergence between two forward passes
                p = F.log_softmax(emissions, dim=-1)
                q = F.softmax(emissions2, dim=-1)
                kl_loss = F.kl_div(p, q, reduction='batchmean')
                
                loss = (loss + loss2) / 2 + RDROP_ALPHA * kl_loss
        
        scaler.scale(loss).backward()
        
        # FGM adversarial training
        if fgm is not None:
            fgm.attack()
            with autocast():
                adv_loss, _ = model(input_ids, attention_mask, labels)
            scaler.scale(adv_loss).backward()
            fgm.restore()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


# ============================================================================
# Experiment Management
# ============================================================================
def load_status():
    """Load experiment status from file."""
    if STATUS_FILE.exists():
        with open(STATUS_FILE, 'r') as f:
            return json.load(f)
    return {'experiments': {}, 'start_time': None}


def save_status(status):
    """Save experiment status to file."""
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, 'w') as f:
        json.dump(status, f, indent=2)


def get_checkpoint_path(exp_name, epoch):
    """Get checkpoint path for experiment and epoch."""
    return SAVE_DIR / exp_name / f'checkpoint_epoch_{epoch}.pt'


def get_best_model_path(exp_name):
    """Get best model path for experiment."""
    return SAVE_DIR / exp_name / 'best_model.pt'


def run_experiment(exp_name, config, tokenizer, train_data, val_data, patience=PATIENCE):
    """Run a single experiment with resume support and early stopping."""
    logger.info(f"{'='*60}")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Config: {config['description']}")
    logger.info(f"Epochs: {EPOCHS} | Patience: {patience}")
    logger.info(f"{'='*60}")
    
    status = load_status()
    
    # Initialize experiment status
    if exp_name not in status['experiments']:
        status['experiments'][exp_name] = {
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'completed_epochs': 0,
            'best_f1': 0.0,
            'best_epoch': 0,
            'epochs_without_improvement': 0,
            'metrics_history': []
        }
        save_status(status)
    
    exp_status = status['experiments'][exp_name]
    
    # Check if already completed
    if exp_status['status'] == 'completed':
        logger.info(f"Experiment already completed. Best F1: {exp_status['best_f1']:.4f}")
        return exp_status['best_f1']
    
    # Create model
    model = AblationNERModel(
        MODEL_NAME, NUM_LABELS,
        use_bilstm=config['use_bilstm'],
        use_crf=config['use_crf']
    ).to(DEVICE)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with differential LR
    encoder_params = [p for n, p in model.named_parameters() if 'electra' in n]
    head_params = [p for n, p in model.named_parameters() if 'electra' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': ENCODER_LR},
        {'params': head_params, 'lr': HEAD_LR}
    ], weight_decay=0.01)
    
    # Data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
    
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = GradScaler()
    
    # Resume from checkpoint if available
    start_epoch = exp_status['completed_epochs']
    best_f1 = exp_status['best_f1']
    epochs_without_improvement = exp_status.get('epochs_without_improvement', 0)
    
    if start_epoch > 0:
        checkpoint_path = get_checkpoint_path(exp_name, start_epoch)
        if checkpoint_path.exists():
            print(f"Resuming from epoch {start_epoch}...")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
            print(f"Loaded checkpoint. Best F1 so far: {best_f1:.4f}")
    
    # Training loop
    for epoch in range(start_epoch, EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, config, DEVICE)
        metrics = evaluate(model, val_loader, DEVICE)
        
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val F1: {metrics['f1']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")
        
        # Update status
        exp_status['completed_epochs'] = epoch + 1
        exp_status['metrics_history'].append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **metrics
        })
        
        # Save checkpoint
        checkpoint_dir = SAVE_DIR / exp_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for improvement
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            exp_status['best_f1'] = best_f1
            exp_status['best_epoch'] = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), get_best_model_path(exp_name))
            logger.info(f"  ★ New best model saved! F1: {best_f1:.4f}")
        else:
            epochs_without_improvement += 1
            logger.info(f"  No improvement for {epochs_without_improvement} epoch(s)")
        
        exp_status['epochs_without_improvement'] = epochs_without_improvement
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'f1': metrics['f1'],
            'best_f1': best_f1,
            'epochs_without_improvement': epochs_without_improvement
        }, get_checkpoint_path(exp_name, epoch + 1))
        
        save_status(status)
        
        # Clean up old checkpoints (keep last 2)
        for old_epoch in range(1, epoch - 1):
            old_ckpt = get_checkpoint_path(exp_name, old_epoch)
            if old_ckpt.exists():
                old_ckpt.unlink()
        
        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"  ⚠ Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Mark as completed
    exp_status['status'] = 'completed'
    exp_status['end_time'] = datetime.now().isoformat()
    save_status(status)
    
    logger.info(f"✓ Experiment completed! Best F1: {best_f1:.4f} (epoch {exp_status['best_epoch']})")
    return best_f1


# ============================================================================
# Report Generation
# ============================================================================
def generate_report():
    """Generate comparison report and visualizations."""
    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        print("Warning: matplotlib not installed. Skipping visualizations.")
    
    status = load_status()
    
    if not status['experiments']:
        print("No experiments found. Run some experiments first.")
        return
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    
    # Collect results
    results = []
    for exp_name, exp_data in status['experiments'].items():
        if exp_data['status'] == 'completed':
            config = EXPERIMENTS.get(exp_name, {})
            results.append({
                'name': exp_name,
                'description': config.get('description', ''),
                'best_f1': exp_data['best_f1'],
                'best_epoch': exp_data['best_epoch'],
                'use_bilstm': config.get('use_bilstm', False),
                'use_crf': config.get('use_crf', False),
                'use_fgm': config.get('use_fgm', False),
                'use_rdrop': config.get('use_rdrop', False),
                'metrics_history': exp_data.get('metrics_history', [])
            })
    
    if not results:
        print("No completed experiments found.")
        return
    
    # Sort by F1
    results.sort(key=lambda x: x['best_f1'], reverse=True)
    
    # Print table
    print("\n### Results Table\n")
    print("| Experiment | F1 | BiLSTM | CRF | FGM | R-Drop |")
    print("|------------|-----|--------|-----|-----|--------|")
    for r in results:
        bilstm = "✓" if r['use_bilstm'] else "✗"
        crf = "✓" if r['use_crf'] else "✗"
        fgm = "✓" if r['use_fgm'] else "✗"
        rdrop = "✓" if r['use_rdrop'] else "✗"
        print(f"| {r['name']:<12} | {r['best_f1']:.4f} | {bilstm} | {crf} | {fgm} | {rdrop} |")
    
    # Calculate contributions
    baseline_f1 = next((r['best_f1'] for r in results if r['name'] == 'baseline'), None)
    full_f1 = next((r['best_f1'] for r in results if r['name'] == 'full'), None)
    
    if baseline_f1 and full_f1:
        print(f"\n### Key Findings\n")
        print(f"- Baseline → Full improvement: {(full_f1 - baseline_f1)*100:.2f}%")
        
        for r in results:
            if r['name'].startswith('full_no_'):
                technique = r['name'].replace('full_no_', '')
                contribution = (full_f1 - r['best_f1']) * 100
                print(f"- {technique.upper()} contribution: {contribution:.2f}%")
    
    if not HAS_MATPLOTLIB:
        return
    
    # Color palette for experiments
    colors = {
        'baseline': '#FF6B6B',
        '+crf': '#4ECDC4', 
        '+bilstm': '#45B7D1',
        'full_no_fgm': '#96CEB4',
        'full_no_rdrop': '#FFEAA7',
        'full': '#6C5CE7'
    }
    
    # ========================================
    # 1. Bar chart - F1 Comparison
    # ========================================
    plt.figure(figsize=(12, 6))
    names = [r['name'] for r in results]
    f1s = [r['best_f1'] for r in results]
    bar_colors = [colors.get(r['name'], '#96CEB4') for r in results]
    
    bars = plt.bar(names, f1s, color=bar_colors, edgecolor='black', linewidth=1.2)
    plt.xlabel('Experiment', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Ablation Study: F1 Comparison', fontsize=14, fontweight='bold')
    plt.ylim(min(f1s) - 0.02, max(f1s) + 0.02)
    
    for i, (name, f1) in enumerate(zip(names, f1s)):
        plt.text(i, f1 + 0.005, f'{f1:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'ablation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'ablation_comparison.png'}")
    
    # ========================================
    # 2. F1 Curves over Epochs
    # ========================================
    plt.figure(figsize=(12, 6))
    
    for r in results:
        if r['metrics_history']:
            epochs = [m['epoch'] for m in r['metrics_history']]
            f1_values = [m['f1'] for m in r['metrics_history']]
            color = colors.get(r['name'], '#96CEB4')
            plt.plot(epochs, f1_values, marker='o', markersize=4, label=r['name'], 
                     color=color, linewidth=2)
            # Mark best epoch
            best_idx = f1_values.index(max(f1_values))
            plt.scatter([epochs[best_idx]], [f1_values[best_idx]], s=100, 
                       color=color, edgecolors='black', linewidths=2, zorder=5)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Validation F1 Score Over Training', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'f1_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'f1_curves.png'}")
    
    # ========================================
    # 3. Training Loss Curves
    # ========================================
    plt.figure(figsize=(12, 6))
    
    for r in results:
        if r['metrics_history']:
            epochs = [m['epoch'] for m in r['metrics_history']]
            loss_values = [m['train_loss'] for m in r['metrics_history']]
            color = colors.get(r['name'], '#96CEB4')
            plt.plot(epochs, loss_values, marker='o', markersize=4, label=r['name'],
                     color=color, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'loss_curves.png'}")
    
    # ========================================
    # 4. Technique Contribution Chart
    # ========================================
    if baseline_f1 and full_f1:
        contributions = []
        
        # CRF contribution: +crf vs baseline
        crf_f1 = next((r['best_f1'] for r in results if r['name'] == '+crf'), None)
        if crf_f1:
            contributions.append(('CRF', (crf_f1 - baseline_f1) * 100))
        
        # BiLSTM contribution: +bilstm vs baseline
        bilstm_f1 = next((r['best_f1'] for r in results if r['name'] == '+bilstm'), None)
        if bilstm_f1:
            contributions.append(('BiLSTM', (bilstm_f1 - baseline_f1) * 100))
        
        # FGM contribution: full vs full_no_fgm
        no_fgm_f1 = next((r['best_f1'] for r in results if r['name'] == 'full_no_fgm'), None)
        if no_fgm_f1:
            contributions.append(('FGM', (full_f1 - no_fgm_f1) * 100))
        
        # R-Drop contribution: full vs full_no_rdrop
        no_rdrop_f1 = next((r['best_f1'] for r in results if r['name'] == 'full_no_rdrop'), None)
        if no_rdrop_f1:
            contributions.append(('R-Drop', (full_f1 - no_rdrop_f1) * 100))
        
        if contributions:
            plt.figure(figsize=(10, 6))
            techniques = [c[0] for c in contributions]
            values = [c[1] for c in contributions]
            bar_colors = ['#4ECDC4', '#45B7D1', '#FF6B6B', '#FFEAA7'][:len(contributions)]
            
            bars = plt.barh(techniques, values, color=bar_colors, edgecolor='black', height=0.6)
            
            for bar, val in zip(bars, values):
                width = bar.get_width()
                plt.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{val:+.2f}%', va='center', fontsize=11, fontweight='bold')
            
            plt.xlabel('F1 Improvement (%)', fontsize=12)
            plt.title('Individual Technique Contributions', fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / 'technique_contributions.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {RESULTS_DIR / 'technique_contributions.png'}")
    
    # ========================================
    # 5. Summary Dashboard (2x2 grid)
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Bar chart
    ax1 = axes[0, 0]
    bar_colors = [colors.get(r['name'], '#96CEB4') for r in results]
    ax1.bar(names, f1s, color=bar_colors, edgecolor='black')
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Final F1 Comparison', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    for i, f1 in enumerate(f1s):
        ax1.text(i, f1 + 0.003, f'{f1:.3f}', ha='center', fontsize=8)
    
    # Top-right: F1 curves
    ax2 = axes[0, 1]
    for r in results:
        if r['metrics_history']:
            epochs = [m['epoch'] for m in r['metrics_history']]
            f1_values = [m['f1'] for m in r['metrics_history']]
            ax2.plot(epochs, f1_values, marker='.', label=r['name'], 
                     color=colors.get(r['name'], '#96CEB4'), linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Over Training', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(alpha=0.3)
    
    # Bottom-left: Loss curves
    ax3 = axes[1, 0]
    for r in results:
        if r['metrics_history']:
            epochs = [m['epoch'] for m in r['metrics_history']]
            loss_values = [m['train_loss'] for m in r['metrics_history']]
            ax3.plot(epochs, loss_values, marker='.', label=r['name'],
                     color=colors.get(r['name'], '#96CEB4'), linewidth=1.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Training Loss')
    ax3.set_title('Training Loss', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(alpha=0.3)
    
    # Bottom-right: Best epochs
    ax4 = axes[1, 1]
    best_epochs = [r['best_epoch'] for r in results]
    ax4.barh(names, best_epochs, color=bar_colors, edgecolor='black', height=0.6)
    ax4.set_xlabel('Best Epoch')
    ax4.set_title('Epoch of Best F1', fontweight='bold')
    for i, (name, epoch) in enumerate(zip(names, best_epochs)):
        ax4.text(epoch + 0.3, i, str(epoch), va='center', fontsize=9)
    
    plt.suptitle('Ablation Study Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'ablation_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'ablation_summary.png'}")
    
    # Save results JSON
    results_for_json = [{k: v for k, v in r.items() if k != 'metrics_history'} for r in results]
    with open(RESULTS_DIR / 'ablation_results.json', 'w') as f:
        json.dump(results_for_json, f, indent=2)
    print(f"Saved: {RESULTS_DIR / 'ablation_results.json'}")
    
    # Save full metrics history
    full_history = {r['name']: r['metrics_history'] for r in results}
    with open(RESULTS_DIR / 'metrics_history.json', 'w') as f:
        json.dump(full_history, f, indent=2)
    print(f"Saved: {RESULTS_DIR / 'metrics_history.json'}")
    
    print("\n" + "="*60)
    print("GENERATED VISUALIZATIONS:")
    print("="*60)
    print("  1. ablation_comparison.png   - Bar chart of final F1 scores")
    print("  2. f1_curves.png             - F1 over epochs for all experiments")
    print("  3. loss_curves.png           - Training loss curves")
    print("  4. technique_contributions.png - Individual technique impact")
    print("  5. ablation_summary.png      - 2x2 dashboard summary")


def show_status():
    """Show current status of all experiments."""
    status = load_status()
    
    print("\n" + "="*60)
    print("ABLATION STUDY STATUS")
    print("="*60)
    print(f"Epochs per experiment: {EPOCHS}")
    print(f"Early stopping patience: {PATIENCE}")
    print()
    
    for exp_name in EXPERIMENTS.keys():
        exp_data = status['experiments'].get(exp_name, {})
        exp_status = exp_data.get('status', 'not started')
        completed = exp_data.get('completed_epochs', 0)
        best_f1 = exp_data.get('best_f1', 0)
        best_epoch = exp_data.get('best_epoch', 0)
        
        if exp_status == 'completed':
            icon = "✓"
        elif exp_status == 'running':
            icon = "►"
        else:
            icon = "○"
        
        print(f"  {icon} {exp_name:<15} | {exp_status:<12} | Epochs: {completed:>2}/{EPOCHS} | Best F1: {best_f1:.4f} @ epoch {best_epoch}")
    
    print()


# ============================================================================
# Main
# ============================================================================
def main():
    global EPOCHS, PATIENCE
    
    parser = argparse.ArgumentParser(description='Ablation Study for Korean NER')
    parser.add_argument('--experiment', type=str, choices=list(EXPERIMENTS.keys()),
                        help='Run specific experiment only')
    parser.add_argument('--status', action='store_true', help='Show status of all experiments')
    parser.add_argument('--report', action='store_true', help='Generate comparison report')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs per experiment (default: 40)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10)')
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    PATIENCE = args.patience
    
    if args.status:
        show_status()
        return
    
    if args.report:
        generate_report()
        return
    
    # System info
    print("="*60)
    print("KOREAN NER ABLATION STUDY")
    print("="*60)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Epochs: {EPOCHS} | Patience: {PATIENCE}")
    
    # Initialize
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data, val_data = load_data(tokenizer)
    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")
    
    # Run experiments
    experiments_to_run = [args.experiment] if args.experiment else list(EXPERIMENTS.keys())
    
    status = load_status()
    if status['start_time'] is None:
        status['start_time'] = datetime.now().isoformat()
        save_status(status)
    
    for exp_name in experiments_to_run:
        config = EXPERIMENTS[exp_name]
        run_experiment(exp_name, config, tokenizer, train_data, val_data, patience=PATIENCE)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*60)
    
    generate_report()


if __name__ == "__main__":
    main()
