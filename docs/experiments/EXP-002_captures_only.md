# EXP-002 — captures-only distillation

**Date:** 2026-05-02
**Branch / commit:** main / Phase 2
**Config:** `configs/exp002_captures_only.yaml`

## Hypothesis

Training the DS-CNN student to mimic YAMNet's per-clip soft logits on
the 475 deployed-device captures (no labels, KL only) drives held-out
validation KL down meaningfully from random init. The buildout plan
set a stretch target of **val KL ≤ 0.5 nats**; we expected to land
between 1 and 3 given the student's 80 K parameter budget against a
521-way teacher.

## Setup

- **Data source:** `$WS_ESP32_S3_CAM_ROOT/projects/cry-detect-01/logs/canonical/wavs/`
  → 475 captures, 16 kHz mono, 40 s each. Discovered by filename
  pattern only — no `master.csv` or label JSON read at any point in
  training. Verified by inspection of `home_captures.py` imports.
- **Split:** 80 / 20 time-stratified by (date, hour) bucket →
  **train 380 / val 95**, deterministic at `split_seed=0`. Stratifying
  by (date, hour) avoids the time-of-day confound noted in the data
  audit (19 h skews positive, dawn skews negative).
- **Patches per clip:** 4 random 0.975 s windows during cache build
  for train; 1 centered window for val (deterministic).
- **Teacher:** YAMNet from TF Hub, frozen. Output: 521-class softmax
  averaged over per-patch scores.
- **Student:** `dscnn_student`, 80 713 params (unchanged from EXP-001).
- **Loss:** `KL(teacher_probs || softmax(student_logits))`, ε = 1e-8.
- **Optimizer:** AdamW, lr = 1e-3, weight_decay = 1e-4.
- **Training:** 50 epochs, batch 32, val every 5 epochs, best-val
  checkpoint saved to `models/exp002_dscnn.h5` (gitignored).
- **Cache trick:** Teacher outputs (mel patch + clip-level probs) are
  computed once at the start (`_build_cache`) and reused across
  epochs. Cuts wall-clock by ~50× vs recomputing teacher per step,
  but means the 1 520 patches are fixed across epochs — see
  *Analysis* below.
- **Hardware:** macOS, M-series CPU only, Python 3.13.11,
  TensorFlow 2.21.0, tensorflow-hub 0.16.1.

## Results

| metric | value |
|---|---:|
| init val KL (epoch 0, random student) | **6.624** |
| best val KL | **1.267** |
| best epoch | 35 |
| final train KL (epoch 50) | 0.920 |
| final val KL (epoch 50) | 1.295 |
| total wall-clock | ~3 min |

```
[train] epoch  0 (init): val_kl_per_clip=6.6237
[train] epoch  5: train_kl=1.8543 val_kl=3.2367   (saved best)
[train] epoch 10: train_kl=1.5876 val_kl=2.0618   (saved best)
[train] epoch 15: train_kl=1.4009 val_kl=1.4056   (saved best)
[train] epoch 25: train_kl=1.2135 val_kl=1.2691   (saved best)
[train] epoch 35: train_kl=1.0945 val_kl=1.2668   (saved best)
[train] epoch 50: train_kl=0.9204 val_kl=1.2948
```

Full per-epoch history is in
`docs/experiments/eval_home_captures_exp-002.json` (gitignored
because it derives from private captures).

## Analysis

**KL fell 5.2× (6.62 → 1.27) on a held-out time-stratified split.**
The student is genuinely matching the teacher's 521-way distribution,
not just memorizing — train and val tracked together until ~epoch 25,
then diverged (train kept descending, val plateaued at ~1.27).

**Mild overfit after epoch 30.** Train continues to drop (1.21 → 0.92)
while val sits flat at 1.27 ± 0.03. Three concurrent causes:
1. Patches are cached once, so the same 1 520 (mel, teacher_probs)
   pairs are seen every epoch — no patch-level augmentation.
2. 80 K params may be slightly under-regularized for this size.
3. Captures are dominated by ~5 unique nights, so even with hour
   stratification there's per-session acoustic correlation.

**Stretch target missed.** The plan's "KL ≤ 0.5" goal was aspirational
for a Phase-2 first-pass run with no AudioSet augmentation; 1.27 is
the honest captures-only baseline.

**Loop hygiene confirmed.** `git status` after the run showed zero
new tracked WAVs, no new files under `data/captures/` or
`data/audioset/`, and `models/exp002_dscnn.h5` is gitignored as
intended (`models/*` rule). Eval JSON is gitignored
(`docs/experiments/eval_home_captures*.json`).

## Next steps

EXP-003 will swap in held-out **AudioSet** evaluation as the headline
metric — that's the metric a stranger can reproduce from public data
alone, which is the portfolio promise. EXP-002's val_kl is a *side*
metric (useful for tracking, never the headline number).

Concrete iteration list for the next batch of experiments:

1. **EXP-003** (AudioSet only, ~1 day): build `audioset_train.csv`
   with ~1 000 segment IDs, run distillation on AudioSet audio only,
   eval on held-out AudioSet `crying_sobbing` segment IDs.
2. **EXP-004** (combined, ~1 day): 50/50 mix of captures + AudioSet,
   same eval as EXP-003 plus the EXP-002 captures-side check.
3. **EXP-005** (per-epoch patch resampling): regenerate the patch
   cache every epoch instead of once. Should narrow the train/val
   gap and probably push val KL down to ~1.0.
4. **EXP-006** (cry-head distillation): collapse the 521 teacher
   classes to the 5–10 cry-relevant ones before computing KL. With
   so few effective classes the KL target shifts (≤0.2 is realistic),
   but transferability to AudioSet eval needs verification.

## Reproducibility

```bash
# Public reproducer fails (correctly) — no captures available:
python -m yamnet_cry_distill_int8.train --config configs/exp002_captures_only.yaml
# → SystemExit: No captures found. Set WS_ESP32_S3_CAM_ROOT...

# Private reproducer (has captures):
WS_ESP32_S3_CAM_ROOT=../ws-ESP32-S3-CAM \
  python -m yamnet_cry_distill_int8.train --config configs/exp002_captures_only.yaml
```

EXP-002 is a *side* experiment by design. The headline reproducible
number lives in EXP-003 / EXP-004 once AudioSet curation lands.
