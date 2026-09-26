# Verified experiment reports

These are snapshots of the completed 24 September 2026 evaluations. Paths were normalized for portability; numerical results were not changed. The original local outputs remain under `runs/`, which is excluded from Git.

| Report | What it measures |
|--------|------------------|
| [verified_results.json](verified_results.json) | Aggregate README numbers, differences, and limitations |
| [kf_deberta_extra_validation.json](kf_deberta_extra_validation.json) | Selected checkpoint: all 5,000 public validation examples |
| [kf_deberta_extra_train_sample.json](kf_deberta_extra_train_sample.json) | Selected checkpoint: 5,000 seen training examples, sample seed 42 |
| [kf_deberta_epoch5_validation.json](kf_deberta_epoch5_validation.json) | Rollback checkpoint: all public validation examples |
| [kf_deberta_epoch5_train_sample.json](kf_deberta_epoch5_train_sample.json) | Rollback checkpoint: the same seen training sample |
| [kf_deberta_average_validation.json](kf_deberta_average_validation.json) | Unselected average of epoch 5 and the extra-epoch weights |
| [koelectra_average.json](koelectra_average.json) | Corrected KoELECTRA average: benchmark-compatible and all-character metrics |
| [kf_deberta_training.json](kf_deberta_training.json) | Initial five-epoch run and one-epoch continuation settings |

## Protocol and limits

The primary metric reconstructs character predictions, drops ordinary ASCII spaces, includes predicted CLS/SEP labels against gold `O`, flattens examples, and computes strict IOB2 entity macro F1. It follows the [KLUE baseline metric](https://github.com/KLUE-benchmark/KLUE-baseline/blob/main/klue_baseline/metrics/functional.py). The implementation uses tokenizer offsets and scores the active SEP position; the original baseline code takes the last entry of its fixed-length prediction array, which may be PAD. Accordingly, the project calls this **KLUE-compatible**, not a bit-for-bit reproduction of that implementation.

The `klue_official_entity_macro_f1` field name is retained from the evaluator. The definition above applies to it. KoELECTRA's separate `entity_macro_f1_strict_iob2` and per-type fields include spaces after BIO repair and must not be substituted for the primary metric.

`saved_epoch: 1` in the selected KF-DeBERTa reports means the first epoch of the continuation stage, after the original five epochs. Training used BF16 autocast; the reports here use FP32 evaluation after reloading weights.

All candidate selection used the same public validation split. The training comparison uses already seen examples. One seed was trained, and the train–validation gap does not by itself prove memorization. No independent test, confidence interval, or official hidden-test submission is claimed.

## Checkpoints

The selected `runs/kf_deberta_best.pt` and rollback `runs/kf_deberta_epoch5.pt` files are excluded from Git. The selected checkpoint is published as [`best_model.pt` on Hugging Face](https://huggingface.co/mrleast/kf-deberta-klue-ner); the rollback remains local. Their reports record SHA-256 hashes, so replacing a file at the same path does not silently change which model a result identifies. The unselected average's original report did not record a checkpoint hash.

After training, a new score can be recorded with:

```bash
python evaluate_deberta.py --checkpoint runs/kf_deberta_best.pt --output runs/recheck/metrics.json
```

Rebuild only the published summary and figure, without loading a model:

```bash
python evaluate/report_results.py
```

The full findings and corrections are documented in [AUDIT_RU.md](../AUDIT_RU.md).
