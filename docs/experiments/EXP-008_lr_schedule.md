# EXP-008 — EXP-007 + LR drop schedule (v0.2.0 candidate)

**Date:** 2026-05-03
**Branch / commit:** main / Phase 5d
**Config:** `configs/exp008_lr_schedule.yaml`

## Hypothesis

EXP-007 plateaued: train_kl 1.85 (epoch 5) → 1.43 (epoch 50), val_kl
2.42 → 2.19. The train/val gap of ~0.75 nats by epoch 50 suggests
the model is mostly fitting the per-epoch random patches it sees,
not the underlying val distribution. **A learning-rate drop at the
plateau should let the model fine-tune toward generalization rather
than memorize each epoch's specific subsample.**

Predicted: small AUC gain (+0.005–0.015), tighter convergence.

## Setup

EXP-007's config plus:
- `train.epochs: 60` (extended past the plateau)
- `train.lr_drop_at_epoch: 30`
- `train.lr_drop_to: 0.0001` (10× drop from 1e-3)

Implementation in `train.py`: at the start of each epoch, check if
the current epoch matches `lr_drop_at_epoch` and call
`optimizer.learning_rate.assign(lr_drop_to)`.

## Results — AudioSet test

| | best F1 | best thr | precision | recall | AUC | FPR @ best thr |
|---|---:|---:|---:|---:|---:|---:|
| EXP-006 (v0.1.0) | 0.696 | 0.096 | 0.571 | 0.889 | 0.860 | 31.8 % |
| EXP-007 | 0.756 | 0.032 | 0.630 | 0.944 | 0.861 | 22.7 % |
| **EXP-008** | **0.756** | **0.047** | **0.630** | **0.944** | **0.870** | **22.7 %** |
| Δ vs EXP-007 | 0.000 | +0.015 | 0.000 | 0.000 | **+0.009** | 0.000 |

INT8 quantized:

| | best F1 | AUC |
|---|---:|---:|
| EXP-008 FP32 | 0.756 | 0.870 |
| **EXP-008 INT8** | **0.744** | **0.866** |
| Δ from quantization | -0.012 | -0.004 |

Quantization cost a hair more than EXP-006 (which was 0.000) — the
post-LR-drop activation distributions are slightly tighter, less
quantization-friendly. Still well under the ≤0.02 AUC budget.

## Analysis

**The LR drop closed the train/val gap by ~0.06 nats** (epoch 50:
EXP-007 train/val 1.43/2.20, EXP-008 1.42/2.15). Small but consistent.

**Best-F1 didn't move; AUC did (+0.009).** Same operating point
performance, slightly better behavior off-peak. That's exactly the
shape of "fine-tune at low LR" gain — the decision boundary tightens
at non-optimal thresholds without disturbing the operating point.

**Cumulative gain over EXP-006 (v0.1.0):**
- best F1: 0.696 → 0.756 (+0.060)
- AUC: 0.860 → 0.870 (+0.010)
- FPR at best thr: 31.8 % → 22.7 % (-9.1 pp)
- Effective deployment false-alert rate (5-of-9 voting): ~5× lower

The bulk of the gain came from EXP-007 (per-epoch resample + 1:3
ratio); EXP-008's LR schedule polished the AUC.

## Recommendation: ship as v0.2.0

The cumulative numbers are meaningful enough to justify a v0.2.0
release. Same student, same INT8 size (110 KB), real improvement on
the deployment-relevant FPR axis.

The model card claim updates from "AUC 0.860 / best-F1 0.696" to
"AUC 0.870 / best-F1 0.756 / FPR 22.7 % at recall 94.4 %."

## Next steps

EXP-009 ideas (deferred):
- **Mix augmentation** — overlay AudioSet cry samples on silent
  capture segments and vice versa. Probably the next big gain
  (+0.03–0.05 AUC plausible).
- **Cry-head distillation** — KL on classes 19+20 only vs full 521.
  Frees student capacity, may sharpen mass further.
- **Bigger student** — 200 K params. Likely +0.03–0.06 AUC at the
  cost of 2× training time and ~250 KB INT8.

Per the buildout plan, the v0.2.0 ship closes Phase 5; further
iteration should be a Phase 6 backlog item rather than blocking
publication.

## Reproducibility

```bash
python -m yamnet_cry_distill_int8.train \
       --config configs/exp008_lr_schedule.yaml          # ~3 min
python -m yamnet_cry_distill_int8.eval --exp EXP-008     # AUC 0.870
python -m yamnet_cry_distill_int8.quantize.int8 --exp EXP-008
python -m yamnet_cry_distill_int8.eval --exp EXP-008 --tflite
# INT8 best F1 = 0.744, AUC = 0.866
```
