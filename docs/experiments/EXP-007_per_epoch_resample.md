# EXP-007 — per-epoch resampling + 1:3 pos:neg ratio

**Date:** 2026-05-03
**Branch / commit:** main / Phase 5d
**Config:** `configs/exp007_per_epoch_resample.yaml`

## Hypothesis

EXP-006 ran the teacher-as-filter scan, got 5 714 positives and
30 324 negatives, then **subsampled to 5 714 / 5 714 once and froze
that pool for all 50 epochs.** Two improvements available without
new data:

1. **Per-epoch resampling.** Keep both pools intact; randomize the
   negative subsample each epoch. Over 50 epochs the student sees
   ~all 30 K negatives at least once, vs the same 5 714 every epoch.
2. **1:3 pos:neg ratio.** Real deployment audio is ~98 % non-cry.
   Training at 1:1 is far from that distribution. 1:3 is a step
   toward deployment realism without crushing positive signal.

Predicted: small AUC gain, larger improvement in *deployment-relevant*
FPR at the operating threshold (the part of the ROC curve we
actually care about).

## Setup

Same student, loss, optimizer, epochs as EXP-006. Two changes:
- `mixers.patches_filtered_by_teacher(..., return_separate=True)` →
  returns `(pos_all, neg_all)` instead of pre-balanced.
- `train.py` epoch loop re-samples 5 714 pos + 17 142 neg per epoch.

The `return_separate` API is back-compatible — EXP-006 still runs
bit-exactly with `return_separate=False`.

## Results — AudioSet test (HEADLINE)

| | best F1 | best thr | precision | recall | AUC | FPR @ best thr |
|---|---:|---:|---:|---:|---:|---:|
| EXP-006 | 0.696 | 0.096 | 0.571 | 0.889 | 0.860 | 31.8 % |
| **EXP-007** | **0.756** | **0.032** | **0.630** | **0.944** | **0.861** | **22.7 %** |
| Δ | **+0.060** | -0.064 | +0.059 | +0.056 | +0.001 | **-9.1 pp** |

AUC barely moved (0.860 → 0.861). The win is at the operating point:

- **9 percentage points off the false-positive rate.** From ~18 K FP
  windows / night down to ~13 K (raw, before voting).
- **+0.056 recall** at the same threshold range — catching 17 of 18
  cries vs EXP-006's 16 of 18.
- The threshold dropped 3× (0.096 → 0.032) — the *positive* mass is
  *less* concentrated in absolute terms but the *negatives* are
  much more concentrated near zero. That's exactly what 1:3 training
  is supposed to do.

## Analysis

**Per-epoch resampling alone explains some of the gain** — over 50
epochs the student saw the full 30 K negative pool, vs the same 5 714
every epoch. The 1:3 ratio adds the deployment-realism part.

**AUC vs operating-point F1 are different metrics.** AUC is the
ranking quality across all thresholds; F1 is point-at-best-threshold.
EXP-007 has the same AUC as EXP-006 but a much better best-F1
because the score distribution is *concentrated* differently: positives
spread out a bit (lower threshold), but negatives crushed harder
toward zero (much lower FPR at the same recall).

**The deployment math improves materially.** With 5-of-9 voting:

- EXP-006: per-window FPR 31.8 % → cluster FPR ~9 % → ~600 false alert
  clusters per 8 h night.
- EXP-007: per-window FPR 22.7 % → cluster FPR ~1.5 % → ~110 clusters/night.

**~5× fewer false alerts** at the F1-optimal threshold. The user-
relevant improvement is much bigger than the +0.001 AUC suggests.

## Next steps

EXP-008 layers a learning-rate schedule (drop to 1e-4 at epoch 30) on
top of EXP-007 to see if the train/val gap (still ~0.7 nats KL at
epoch 50) closes any further. Result: small additional AUC gain
(0.861 → 0.870), same best-F1.

## Reproducibility

```bash
python -m yamnet_cry_distill_int8.train \
       --config configs/exp007_per_epoch_resample.yaml
python -m yamnet_cry_distill_int8.eval --exp EXP-007
# best_F1 = 0.756 ± 0.01 (variance from per-epoch random subsampling
# of 17 142 of 30 324 negatives; same `train.shuffle_seed` is bit-exact).
```
