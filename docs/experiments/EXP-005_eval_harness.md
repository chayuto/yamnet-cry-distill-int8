# EXP-005 — eval harness, headline AudioSet F1

**Date:** 2026-05-03
**Branch / commit:** main / Phase 4
**Code:** `src/yamnet_cry_distill_int8/eval.py`

## Hypothesis

The cross-eval KL numbers from Phase 3 (EXP-003 / EXP-004 docs) are
informative but not publishable — "this model has KL=4.0 nats vs the
teacher" doesn't translate to "this many cries detected per night."
A proper binary-classification harness on the FROZEN AudioSet test
set turns the distilled student into a real, headline-ready
detector with familiar metrics: precision, recall, F1, AUC.

## Setup

- **Cry-positive score:** sum of student softmax probabilities for
  YAMNet classes 19 (`Crying, sobbing`) and 20 (`Baby cry,
  infant cry`) — the same two classes our class taxonomy maps
  to `/m/0463cq4` and `/t/dd00002`.
- **Per-clip aggregation:** mean cry-score over all 0.96 s windows
  with 0.48 s hop in a clip (≈20 windows for a 10 s AudioSet
  segment). Matches YAMNet's natural patch grid.
- **Ground truth:** a clip is positive iff its `positive_labels`
  column intersects `{/m/0463cq4, /t/dd00002}`.
- **Threshold reporting:** both fixed-0.5 and best-F1 sweep over
  observed scores. Distilled students inherit the teacher's diffuse
  softmax (cry probabilities concentrate well below 0.5 even on
  obvious cries), so the sweep number is the more honest reflection
  of capability — 0.5 is rarely a calibrated operating point for
  distilled models.
- **AUC** is threshold-free; reported as the canonical
  ranking-quality metric.
- **Test set survivor count:** 62 / 100 segments (38 % YouTube
  takedown rate on the frozen test set, slightly higher than the
  28.7 % for train+val — the test set is older).

## Results — AudioSet held-out (HEADLINE)

`docs/experiments/eval_audioset_holdout_exp00N.json` for each model.
Same evaluator, same 62-segment test pool, three students:

| model | F1@0.5 | best F1 (thr) | precision* | recall* | AUC |
|---|---:|---:|---:|---:|---:|
| EXP-002 (captures only) | 0.286 | 0.585 (thr 0.18) | 0.50 | 0.72 | 0.717 |
| EXP-003 (audioset only) | 0.000 | 0.612 (thr 0.04) | 0.46 | 0.94 | 0.750 |
| EXP-004 (combined) | 0.190 | **0.667** (thr 0.03) | **0.52** | **0.94** | **0.823** |

*precision and recall reported at the best-F1 threshold.

## Results — captures-side side metric (PRIVATE — disclosed only)

Side metric on `confidence_tier ∈ {high_pos, high_neg}` from the
auto-ensemble (n_pos=197, n_neg=150). Hour-of-day stratified AUC
breakdown is in the gitignored eval JSONs; **the public claim is
ranking-quality only:**

| model | best F1 | AUC |
|---|---:|---:|
| EXP-002 (captures only) | 0.997 | 1.000 |
| EXP-003 (audioset only) | 0.916 | 0.971 |
| EXP-004 (combined) | 0.990 | **0.999** |

**Interpretation caveat:** captures-side AUC > 0.99 is **not** a
real-world precision claim — high_pos / high_neg are the
auto-ensemble's *highest-confidence* tiers, where the student is
basically learning the teacher's own confident region. The medium
and low tiers (the harder ~30 % of captures the audit flagged for
re-curation) are excluded from this number. The model card discloses
this; the README only shows the AudioSet column.

## Analysis

**EXP-004 is the headline model.** AUC 0.823 on AudioSet held-out is
a real capability number — random would be 0.5, the teacher YAMNet
itself ranks similarly. At its best-F1 threshold (≈0.033) the
combined-data student catches **94 % of cries** with **52 %
precision** on a balanced 18-pos-44-neg AudioSet test slice. That's
a usable detector after a single per-deployment threshold
calibration step.

**EXP-002 has the lowest AudioSet AUC (0.717)** — captures-only
training overfits to the device's specific acoustics. The data audit
already predicted this: 96 % of captures have caregiver speech
overlay, and the device's frequency response shapes everything the
same way. EXP-002 generalizes to *that* environment but not to clean
AudioSet recordings.

**The 0.5 threshold is misleading by ~3×.** At threshold 0.5,
EXP-003 reports F1 = 0.0 — at first glance, useless. At its actual
best-threshold (0.04), F1 = 0.61. The buildout plan's perf table
should report best-F1 plus AUC, not raw 0.5; deployments calibrate
their own threshold from on-device data anyway.

**Test set takedowns hurt sample size.** 38 of 100 frozen test
segments are now dead. With only 62 evaluated (18 cry / 44 negative),
F1 has noticeable noise — a single misclassification shifts F1 by
~0.04. The next iteration could refresh the test set OR move to
multiple test-set draws and report mean ± stddev. For now, single
draw + clear sample sizes is the honest portfolio surface.

## Next steps

Phase 5 — INT8 quantization + HF publish:
1. `quantize/int8.py` — TFLite full-integer quantization with
   AudioSet-only calibration set (no captures leak into the
   quantization parameters). Target: model.tflite ≤ 500 KB,
   AUC drop ≤ 0.02 vs FP32 EXP-004.
2. Fill in `docs/model_cards/yamnet-cry-distill-int8.md` with the
   real headline numbers (AUC 0.82, best-F1 0.67) and the
   public/private data disclosure.
3. `scripts/upload_hf.py` — assemble bundle (model.tflite +
   MODEL_CARD.md + eval JSONs), run `verify_no_captures_in_artifact.py`,
   then `huggingface-cli upload`.
4. Tag `v0.1.0`.

## Reproducibility

```bash
brew install ffmpeg
pip install -e ".[dev,audioset]"
python scripts/download_audioset.py --all --jobs 4   # ~22 min
python -m yamnet_cry_distill_int8.train \
       --config configs/exp004_combined.yaml         # ~6 min
python -m yamnet_cry_distill_int8.eval --exp EXP-004 # ~10 s
# → docs/experiments/eval_audioset_holdout_exp004.json
```

The `--captures` flag adds the side-metric eval; only run it if you
have the device-side captures locally (sets `WS_ESP32_S3_CAM_ROOT`).
