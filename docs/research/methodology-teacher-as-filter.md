# Teacher-as-filter: window-level pseudo-labeling for clip-labeled audio

A note on a small but consequential pipeline choice in this distillation
work — and a generic technique worth knowing for any clip-labeled
classification task.

## The problem

Captures from the deployed device are 40 seconds long. The auto-ensemble
labels them at clip granularity — `high_pos`, `high_neg`, `medium_pos`,
etc. AudioSet segments are 10 seconds with binary class tags
(`Crying, sobbing` / `Baby cry, infant cry` for our cry classes).

The student operates on **0.96 second windows.** So a "cry-positive"
40-second capture contributes ~80 windows to the training pool — but
the actual cry might span only 4 seconds.

If we sample windows uniformly at random from the 40-second clip,
roughly:

- 10 % of "positive" patches are actual cry,
- 90 % of "positive" patches are speech, silence, background, or
  ambient nursery noise.

That is, **80 % of our positive supervision was teaching the student to
recognize *non-cry parts of cry-context audio.*** The student does
learn something — but it's "cry-context room tone" mixed with a small
real-cry signal, not a clean cry detector.

This was the silent flaw in EXP-002 / EXP-003 / EXP-004: the AudioSet
test AUC of 0.823 was achieved despite this, not because of clean
training data.

## The fix

Run the teacher across the whole clip with the eval-time sliding window
(0.4875 s hop, matching YAMNet's natural patch grid), get the teacher's
per-window cry-score:

```python
p_cry = softmax(teacher_logits)[19] + softmax(teacher_logits)[20]
        # YAMNet class 19 = Crying, sobbing
        # YAMNet class 20 = Baby cry, infant cry
```

…and bin every window:

| bin | rule | role |
|---|---|---|
| **positive** | `p_cry > 0.30` | real cry moments — high-quality positive supervision |
| **negative** | `p_cry < 0.05` | hard negatives — cleanly non-cry audio from the same recording conditions |
| **ambiguous** | `0.05 ≤ p_cry ≤ 0.30` | dropped — too much noise for either pool |

Train with explicit class balance (positive : negative ratio 1:1).
The clip-level labels are no longer used. The teacher's window scores
are the supervisory signal — which is exactly what we already use for
the KL distillation loss.

## Why this is essentially free

- **The teacher pass is paid.** We were already running YAMNet on every
  patch we train on. Computing the cry-score for every window of every
  clip is the same cost.
- **The clip-level labels were never load-bearing for distillation.**
  The loss is `KL(teacher_softmax || student_softmax)` on raw logits,
  not BCE on labels. Removing the clip labels removes nothing.
- **Hard negatives from inside positive clips are the cleanest
  negatives available.** Same microphone, same room acoustics, same
  gain — only the cry is missing. Synthetic negatives from random
  AudioSet samples have unfamiliar room signatures.

## When this generalizes

Any classification setup where:

- The label is at **coarser granularity** than the model's operating
  unit (clips → windows; documents → sentences; videos → frames),
- AND **a reliable pre-trained scorer** exists at the operating-unit
  granularity.

The scorer doesn't have to be the teacher. For ASR, a separate VAD
model filters speech vs silence at the frame level. For object
detection, a region-proposal scorer pre-bins regions of interest.
For document classification, BERT-NLI gives sentence-level relevance
scores. The pattern is: **pre-score, bin, balance, train.**

## When it doesn't help

- **True noise in the labels.** A mislabeled clip stays mislabeled —
  the scorer can't fix that. It only fixes *when within a correctly-
  labeled clip the label actually applies*.
- **Class imbalance at the clip level.** If you have 5 cry clips and
  500 non-cry, this technique gives you more diverse negatives but
  doesn't synthesize more positive moments.
- **Domain shift.** A captures-only model still overfits to its
  acoustics whether windows are filtered or not. Filtering improves
  data quality, not data domain.

## Used in this repo

EXP-006, see [`docs/experiments/EXP-006_teacher_filter.md`](../experiments/EXP-006_teacher_filter.md).

Implementation in
[`src/yamnet_cry_distill_int8/data/mixers.py`](../../src/yamnet_cry_distill_int8/data/mixers.py)
(`patches_filtered_by_teacher`).

## Related

- [`docs/research/methodology-distillation-pipeline.md`](methodology-distillation-pipeline.md) — the broader distillation
  methodology this fits into. *(forthcoming)*
- The auto-ensemble labels themselves were never directly used for
  distillation — only as eval-side ground truth for the captures
  side metric (Phase 4, EXP-005). Even before this teacher-filter
  insight, the training pipeline was label-free at the clip level.
