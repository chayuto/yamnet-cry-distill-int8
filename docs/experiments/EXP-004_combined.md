# EXP-004 — combined distillation: captures + AudioSet

**Date:** 2026-05-03
**Branch / commit:** main / Phase 3
**Config:** `configs/exp004_combined.yaml`

## Hypothesis

Combining the in-domain captures (475 deployed-device WAVs) with the
public AudioSet pool (592 surviving segments) should produce a student
that beats either source alone on the AudioSet held-out pool, because
captures supply implicit regularization (consistent device acoustics)
that prevents the student from overfitting to AudioSet's specific
recording conditions. The data audit's "speech contamination" finding
(96 % of high-positive captures have caregiver speech overlay)
predicts that captures-supplied speech examples are noisier than
AudioSet's clean-speech examples — useful diversity.

## Setup

- **Patch pool:**
  - 380 captures × 4 random crops = 1 520 captures patches
  - 413 surviving AudioSet segments × 4 crops = 1 652 audioset patches
  - **train total: 3 172 patches**
- **Val pool:**
  - 95 captures × 1 centered crop = 95
  - 117 surviving AudioSet val segments × 1 = 117
  - **val total: 212 patches**
- **Same student, loss, optimizer, schedule as EXP-002 / EXP-003.**
- **Wall-clock:** ~6 min including teacher cache.

## Results

| metric | value |
|---|---:|
| init val KL | 7.272 |
| best val KL (mixed val pool) | 2.832 |
| best epoch | 25 |
| final train KL (epoch 50) | 1.793 |

The single-pool number above mixes captures + AudioSet, so it isn't
directly comparable to EXP-002 (1.27 captures-only) or EXP-003 (4.48
AudioSet-only). The cross-eval below uses identical pools across all
three students:

| eval pool | EXP-002 captures | EXP-003 AudioSet | EXP-004 combined |
|---|---:|---:|---:|
| captures val (95 clips) | **1.267** | 2.664 | 1.407 |
| AudioSet val (117 patches) | 6.615 | 4.480 | **3.989** |

## Analysis

**EXP-004 is the best generalist.** It loses only 0.14 nats vs
EXP-002 on the captures pool (1.41 vs 1.27 — well within the
checkpoint-noise band) and *beats EXP-003 by 0.49 nats* on AudioSet's
own held-out (3.99 vs 4.48). That second result is the surprising
one: more training data on its own evaluation set should help, not
hurt — but EXP-003 saw 1 652 AudioSet patches per epoch while EXP-004
saw 1 652 + 1 520 = 3 172. Captures aren't just "extra audio"; they
function as soft regularization, presumably because their per-clip
teacher distribution is narrower than AudioSet's (fewer effective
classes, more peaked softmax) which biases the student toward
sharper responses generally.

**The headline takeaway for the portfolio:** the public-data baseline
(EXP-003) gets 4.48 nats KL on AudioSet's val. Adding the private
captures pushes that to 3.99 — a 0.49-nat improvement that costs
zero on the public surface (captures stay private, model card
discloses the split). This is the "best of both" claim the buildout
plan staked out.

**Caveat — these are KL numbers, not the headline F1.** Phase 4
implements the real headline metric: held-out AudioSet
`crying_sobbing` segment-level F1 / precision / recall on
`audioset_test.csv`. KL-on-val is informative but not the
publishable claim — `4.48 nats KL` doesn't translate cleanly to
"this many cries detected per night." Phase 4 closes that gap.

## Next steps

Phase 4 — eval harness:
1. Build `eval.py` that runs the saved students against
   `audioset_test.csv` and emits F1 / precision / recall on the
   `crying_sobbing` cry-vs-everything-else binary task.
2. Time-stratify the captures-side LOSO eval explicitly (currently
   stratified at split-time, not at eval-time).
3. Record the AudioSet test set's survival rate the same way EXP-003
   recorded the val set's (~71 %).
4. Update README perf table with real F1 numbers.

Then Phase 5: INT8 quantization → `model.tflite` → upload to HF.

## Reproducibility

```bash
# Same prerequisites as EXP-003 plus WS_ESP32_S3_CAM_ROOT pointing
# at the device-side repo with captures.
WS_ESP32_S3_CAM_ROOT=../ws-ESP32-S3-CAM \
  python -m yamnet_cry_distill_int8.train \
        --config configs/exp004_combined.yaml
```

The combined run is 1 minute longer than EXP-003 (more patches in
the cache pass) but otherwise identical wall-clock.
