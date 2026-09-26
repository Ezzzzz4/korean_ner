"""Rebuild the README summary from saved reports; no model loading or training."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'


def read_report(name):
    return json.loads((REPORTS / name).read_text(encoding='utf-8'))


def main():
    koelectra = read_report('koelectra_average.json')
    epoch5 = read_report('kf_deberta_epoch5_validation.json')
    extra = read_report('kf_deberta_extra_validation.json')
    averaged = read_report('kf_deberta_average_validation.json')
    train5 = read_report('kf_deberta_epoch5_train_sample.json')
    train6 = read_report('kf_deberta_extra_train_sample.json')
    key = 'validation_klue_official_entity_macro_f1_percent'
    train_key = 'train_klue_official_entity_macro_f1_percent'
    validation = [
        {'model': 'KoELECTRA · averaged epochs 1–3', 'f1_percent': koelectra['overall']['klue_official_entity_macro_f1_percent']},
        {'model': 'KF-DeBERTa · epoch 5', 'f1_percent': epoch5[key]},
        {'model': 'KF-DeBERTa · averaged stages', 'f1_percent': averaged[key]},
        {'model': 'KF-DeBERTa · extra epoch', 'f1_percent': extra[key]},
    ]
    gap_rows = [
        {'stage': 'Epoch 5', 'train_f1_percent': train5[train_key], 'validation_f1_percent': epoch5[key], 'gap_pp': train5[train_key] - epoch5[key]},
        {'stage': 'Extra epoch', 'train_f1_percent': train6[train_key], 'validation_f1_percent': extra[key], 'gap_pp': train6[train_key] - extra[key]},
    ]
    summary = {
        'dataset': 'klue/ner',
        'metric': 'KLUE-compatible strict IOB2 entity macro F1; ASCII spaces excluded, character predictions reconstructed',
        'validation_examples': 5000,
        'training_sample_examples': 5000,
        'training_sample_seed': 42,
        'training_seed': 42,
        'validation_results': validation,
        'train_validation_comparison': gap_rows,
        'extra_epoch_validation_gain_pp': extra[key] - epoch5[key],
        'selected_checkpoint': extra['checkpoint'],
        'selected_checkpoint_sha256': extra['checkpoint_sha256'],
        'limitations': [
            'Checkpoint selection, continuation and averaging use the same validation split.',
            'Training scores use already seen data; the gap is consistent with overfitting but does not prove memorization.',
            'One training seed; no confidence interval or hidden-test evaluation.',
            'Encoder and head differ from KoELECTRA; no causal component-gain claim.',
        ],
    }
    (REPORTS / 'verified_results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (scores_ax, gap_ax) = plt.subplots(2, 1, figsize=(11, 8))
    fig.subplots_adjust(left=0.31, right=0.96, top=0.86, bottom=0.15, hspace=0.75)
    fig.suptitle('Korean NER: verified results', fontsize=19, fontweight='bold', y=0.965)
    fig.text(0.5, 0.915, 'KLUE-compatible entity macro F1 · public validation · 5,000 examples', ha='center', color='#475569')

    colors = ['#64748b', '#4078a8', '#4078a8', '#157f61']
    bars = scores_ax.barh([row['model'] for row in validation], [row['f1_percent'] for row in validation], color=colors, height=0.6)
    scores_ax.invert_yaxis()
    scores_ax.set_title('Model selection', loc='left', fontweight='bold', pad=12)
    for bar, row in zip(bars, validation):
        scores_ax.text(row['f1_percent'] - 1.1, bar.get_y() + bar.get_height() / 2, f"{row['f1_percent']:.2f}%", ha='right', va='center', color='white', fontweight='bold')

    for index, row in enumerate(gap_rows):
        for offset, field, color in [(-0.18, 'train_f1_percent', '#64748b'), (0.18, 'validation_f1_percent', '#157f61')]:
            score = row[field]
            gap_ax.barh(index + offset, score, height=0.32, color=color, label=('Seen train sample' if offset < 0 else 'Validation') if index == 0 else None)
            gap_ax.text(score - 1.1, index + offset, f'{score:.2f}%', ha='right', va='center', color='white', fontweight='bold')
    gap_ax.set_yticks(range(2), [f"{row['stage']}\nGap: {row['gap_pp']:.2f} pp" for row in gap_rows])
    gap_ax.invert_yaxis()
    gap_ax.set_title('Train–validation gap', loc='left', fontweight='bold', pad=30)
    gap_ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=9)
    for ax in (scores_ax, gap_ax):
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel('Entity macro F1 (%)')
        ax.grid(axis='x', alpha=0.18)
        ax.set_axisbelow(True)
    fig.text(0.03, 0.07, 'Train: same seeded sample of 5,000 previously seen examples. One training seed.\nValidation was used for model selection; these scores are not a hidden-test result.', fontsize=9, color='#475569')
    fig.savefig(ROOT / 'assets' / 'verified_results.png', dpi=160, facecolor='white')
    plt.close(fig)
    print('Updated reports/verified_results.json and assets/verified_results.png')


if __name__ == '__main__':
    main()
