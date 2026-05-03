# EXP-006 — teacher-as-filter, +0.037 AUC over EXP-004

**Date:** 2026-05-03
**Branch / commit:** main / Phase 5b
**Config:** `configs/exp006_teacher_filtered.yaml`
**Methodology:** [`docs/research/methodology-teacher-as-filter.md`](../research/methodology-teacher-as-filter.md)

## Hypothesis

EXP-002 / EXP-003 / EXP-004 randomly sampled 4 windows per 40 s
capture. The clip-level "cry-positive" labels apply to maybe 4 s of
the 40 s clip, so ~80 % of the "positive" supervision was actually
non-cry audio (silence, caregiver speech, ambient noise) sampled from
inside cry-context clips. **Filtering windows by the teacher's
own per-window cry-score before training should recover that lost
signal.**

Predicted gain: ~+0.05 AUC on AudioSet held-out vs EXP-004.

## Setup — what changed vs EXP-004

The student, optimizer, loss, epochs, batch size are all unchanged
from EXP-004. The only change is in `mixers.py`: a new
`patches_filtered_by_teacher` function replaces the random-window
sampler.

For each capture and each AudioSet segment:

1. Slide 0.4875 s hop windows across the full clip
   (~80 windows per 40 s capture, ~20 per 10 s AudioSet segment).
2. Run YAMNet on each window, compute
   `p_cry = softmax(scores)[19] + softmax(scores)[20]`
   (`Crying, sobbing` + `Baby cry, infant cry`).
3. Bin: `p_cry > 0.30` → positive pool, `p_cry < 0.05` →
   negative pool, the middle is dropped.
4. Random-subsample the larger pool down to match the smaller pool
   (1:1 class balance).

The clip-level labels are not consulted at any point — the teacher's
per-window confidence is the only selection signal. Ground truth for
the eval still uses AudioSet's `positive_labels` (since that's the
public reproducibility surface), but training is fully teacher-driven.

## Results — pool composition

The filter scan reveals striking domain differences:

| source | clips | total windows | positive (>0.30) | negative (<0.05) | positive rate |
|---|---:|---:|---:|---:|---:|
| Captures train | 380 | ~30 K | 5 301 | 23 269 | **17.6 %** |
| AudioSet train | 413 | ~8 K | 413 | 7 055 | **5.0 %** |

**Captures have 3.5× higher cry-density than AudioSet's "Crying,
sobbing" segments** — even though AudioSet segments are explicitly
labeled as cry. The 10 s AudioSet labels cover any clip with cry
*somewhere*, often less than 1 second. Captures are 40 s recordings
triggered by on-device cry detection, so cry density is naturally
higher.

After balancing 1:1, train pool = 11 428 patches (5 714 pos +
5 714 neg). **Compared to EXP-004's 3 172 patches, ~3.6× larger
training pool with ~9× higher positive density.**

## Results — held-out metrics

```
[train] init_val_kl=8.7676 best_val_kl=2.2552 @ epoch 45
```

(val_kl pool is also teacher-filtered → not directly comparable to
EXP-004's val pool, which mixed easy and hard cases; this val pool
contains *only* the hard pos/neg discriminations.)

### AudioSet test set (HEADLINE)

Same evaluator, same 62-segment frozen test pool:

| model | best F1 | best thr | precision | recall | AUC | F1@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| EXP-002 captures-only | 0.585 | 0.182 | — | — | 0.717 | 0.286 |
| EXP-003 audioset-only | 0.612 | 0.042 | — | — | 0.750 | 0.000 |
| EXP-004 combined | 0.667 | 0.033 | 0.515 | 0.944 | 0.823 | 0.190 |
| **EXP-006 combined + teacher-filter** | **0.696** | **0.096** | **0.552** | **0.944** | **0.860** | **0.286** |
| Δ vs EXP-004 | +0.029 | +0.063 | +0.037 | 0.000 | **+0.037** | +0.096 |

### INT8 quantized (the published artifact)

```
[EXP-006] F1@0.5=0.286  best_F1=0.696@thr=0.096  AUC=0.860
```

Quantization cost was **zero** at reported precision. The 80 K student
is small enough that 200 calibration patches from `audioset_val.csv`
capture activation distributions exactly. INT8 model is **110 KB**.

### Captures side metric (private)

| model | best F1 | AUC |
|---|---:|---:|
| EXP-004 | 0.990 | 0.999 |
| EXP-006 | 0.992 | 0.998 |

Effectively unchanged — the captures-side test (high_pos vs high_neg
auto-ensemble tiers) was already saturated. EXP-006 picked up the
AudioSet gain without sacrificing in-domain performance.

## Analysis

**The threshold tripled (0.033 → 0.096).** This is the most telling
single number. If the cry mass were just being shifted around (no
new signal), the AUC would change but the threshold wouldn't. The
threshold tripling means the **cry probability at correct positives
is genuinely higher in absolute terms**, not just well-ordered. EXP-006
is more *confident*, not just more accurate.

**+0.037 AUC over EXP-004, same compute budget.** Same student, same
50 epochs. The only difference is which patches the student saw.
Random sampling: 3 172 patches, ~80 % of "positives" actually non-cry.
Teacher-filter: 11 428 patches, ~100 % of positives confidently cry
per the teacher.

The data audit had hinted at this: 96 % of high-positive captures
have caregiver speech overlay, and randomly-sampled windows would
land on the speech parts more often than the cry parts. EXP-006 sidesteps
this entirely.

**Captures still help disproportionately.** Captures contributed 5 301
of 5 714 final positive patches (93 %). AudioSet's labeled cry
segments contributed only 413 confident-cry windows. **The captures-
augmentation finding from EXP-004 isn't just confirmed — it's
amplified.** Without captures, EXP-006 on AudioSet alone would have
~413 positives to train on; with captures, ~5 700. Domain bias gets
balanced out by the random subsample, but the diversity of captures'
cry moments is now properly exploited.

**INT8 = free.** Going from EXP-004's 0.002 AUC quantization cost to
EXP-006's 0.000 AUC cost is suggestive: as the model gets *better* at
the task, its activations become more concentrated, and quantization
becomes more accurate. Cleaner training data → sharper outputs →
quantization-friendlier intermediate representations.

## What surprised us

- **AudioSet's cry segments are noisier than expected.** Only 5 % of
  windows in AudioSet "Crying, sobbing" segments scored `p_cry > 0.30`
  per the teacher. A 10 s "cry-labeled" segment averages ~1 confident-
  cry window (~0.5 s of actual cry). Useful as ground truth at the
  clip level for eval, but a poor source of dense positive supervision.
- **Captures have higher per-clip cry density** (17.6 %) than
  AudioSet's "Crying" labels (5.0 %), despite weaker labels.
  Vindication of the captures-as-augmentation thesis.
- **Hard negatives from inside cry-positive captures dominate.** 23 269
  of 30 324 negatives came from captures vs only 7 055 from AudioSet.
  The deployed device's recordings of "everything that isn't cry from
  the actual nursery environment" are the cleanest negative class
  available.

## Next steps

- **Phase 5c** — re-evaluate quantization with EXP-006 as the source.
  ✓ Done above. Same numbers as FP32. Ship as `model.tflite`.
- **HF upload + tag v0.1.0** — pending. Bundle is
  `model.tflite + MODEL_CARD.md + eval_audioset_holdout_exp006_int8.json + config.json`.
- **EXP-007 ideas** (deferred):
  - Per-epoch random subsample of positive/negative pools (currently
    fixed once at filter time → mild overfit signal at epoch 30+).
  - Threshold sweep on pos_thr / neg_thr (we used 0.30 / 0.05 by
    intuition; sensitivity analysis would tighten the recommendation).
  - Cry-head-only distillation (KL on classes 19+20 only, vs full
    521) — likely faster to train, may sharpen mass further.

## Reproducibility

```bash
brew install ffmpeg
pip install -e ".[dev,audioset]"
python scripts/download_audioset.py --all --jobs 4   # ~22 min
WS_ESP32_S3_CAM_ROOT=../ws-ESP32-S3-CAM \
  python -m yamnet_cry_distill_int8.train \
        --config configs/exp006_teacher_filtered.yaml  # ~3 min
python -m yamnet_cry_distill_int8.eval --exp EXP-006   # AUC 0.860
python -m yamnet_cry_distill_int8.quantize.int8 --exp EXP-006
python -m yamnet_cry_distill_int8.eval --exp EXP-006 --tflite  # same numbers
```

Audioset-only reproducibility (no captures) is independent — that's
EXP-003 / EXP-005 and the `methodology-teacher-as-filter` doc.
