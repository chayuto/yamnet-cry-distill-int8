# Distillation pipeline methodology

A portfolio-level overview of how this repo's distillation pipeline
evolved across phases. Each phase represents a methodology shift, not
just hyperparameter tuning. The published artifact reflects the final
shape of the pipeline; this doc explains how it got there and which
choices matter.

## End-to-end shape (current — EXP-006 onward)

```
                                               (frozen, never touched
                                                  in training)
   AudioSet curated IDs    Home captures      AudioSet test (100)
   (570 train + 160 val)   (475 × 40s)
            │                   │
            ▼                   ▼
       yt-dlp + ffmpeg     filesystem path
       (~28% takedowns)    (private, gitignored)
            │                   │
            └─────────┬─────────┘
                      ▼
        ┌─────────────────────────────────┐
        │  Teacher-as-filter window scan  │
        │  ─────────────────────────────  │  ←── pipeline shift in
        │  Slide 0.4875 s hop, score      │      EXP-006: clip labels
        │  every window with YAMNet,      │      become redundant; the
        │  bin into pos / neg / drop by   │      teacher's per-window
        │  p_cry = softmax[19]+softmax[20]│      cry score is the only
        └─────────────────────────────────┘      selection signal
                      │
                      ▼
        ┌─────────────────────────────────┐
        │  Class-balanced patch pool      │
        │  pos:neg = 1:1 by random        │
        │  subsample of larger pool       │
        └─────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────┐
        │  KL distillation training       │
        │  KL(teacher || student) on the  │
        │  full 521-class softmax, 50 ep  │
        └─────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────┐
        │  Eval                           │
        │  ─────                          │
        │  Headline: AudioSet test F1/AUC │  ←── public-data
        │  Side: captures KL + AUC        │      reproducible
        └─────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────┐
        │  INT8 TFLite export             │
        │  Calibration: 200 patches from  │
        │  audioset_val (never captures)  │
        └─────────────────────────────────┘
                      │
                      ▼
              `model.tflite` ≈ 110 KB
              published to HuggingFace
```

## Phase-by-phase evolution

### Phase 0 — scaffold

Standard layout, public-private boundary fixed at this point: data
gitignored, captures path env-var-configurable, content-guard for
upload bundle. No experiments yet.

### Phase 1 — EXP-001 plumbing smoke

Synthetic 50-segment smoke set. Proved the loop closes:
CSV → loader → teacher → student → KL loss → optimizer step. Exits
in <2 s. Methodology contribution: zero — purely operational.

### Phase 2 — EXP-002 captures-only

First real distillation. **Naïve random-window sampling** — 4 random
0.96 s windows per 40 s capture per cache build, reused across epochs.
Headline: held-out captures KL fell 5.2× (6.62 → 1.27 nats).

Result on AudioSet held-out: **AUC 0.717** (barely better than random,
0.5). The captures-only model overfits hard to the deployment's
acoustics — illustrative negative result.

### Phase 3 — EXP-003 / EXP-004, AudioSet curation + combined

Curated 730 train+val + 100 frozen test segments from public AudioSet
metadata, deterministic seeds. yt-dlp + ffmpeg downloader, idempotent,
takedown-tolerant (28.7% dead).

Same naïve random-window sampling, three runs:

| | AudioSet AUC | captures KL |
|---|---:|---:|
| EXP-002 captures-only | 0.717 | 1.27 |
| EXP-003 audioset-only | 0.750 | 2.66 |
| EXP-004 combined | **0.823** | 1.41 |

EXP-004 **beats EXP-003 on AudioSet's own held-out**. Captures act as
soft regularization, not just extra training mass.

### Phase 4 — EXP-005 eval harness

Turned KL numbers into headline F1 / Precision / Recall / AUC.
Surfaced the diffuse-softmax problem: distilled students inherit
teacher's broad output distribution, so the cry-class probability mass
is tiny in absolute terms (best-F1 thresholds 0.03–0.18, not 0.5).

Methodology contribution: introduced the **threshold-sweep + AUC**
reporting pattern, since fixed-0.5 metrics misrepresent distilled-
student capability.

### Phase 5a — INT8 quantization

Full-integer TFLite export with **AudioSet-only calibration**
(200 val patches, never captures). Cost: 0.002 AUC, 0.013 best F1
— well under the ≤0.02 budget. The 80 K student is highly tolerant
of int8.

### Phase 5b — EXP-006 teacher-as-filter (current)

The methodology shift this doc was written for. Detailed in
[`methodology-teacher-as-filter.md`](methodology-teacher-as-filter.md).

In one sentence: **stop assuming clip labels apply to every sub-window;
use the teacher's per-window confidence to pick the windows you train
on.** The compute is free (we already run the teacher); the gain is
real (~80 % of EXP-002's "positive" supervision was non-cry audio).

## Why each shift mattered

| phase | shift | what it changed |
|---|---|---|
| 2 | distillation loop | label-free training via teacher logits |
| 3 | AudioSet + combined | proved soft-regularization effect |
| 4 | F1/AUC eval harness | distilled-softmax calibration insight |
| 5a | INT8 quantization | proved int8 is essentially free |
| 5b | teacher-as-filter | window-level pseudo-labeling |

The pipeline is now genuinely different from where it started — not
just better-tuned. A reader of the model card sees the final shape;
a reader of this doc + the EXP-NNN log sees how each piece earned
its place.

## What's NOT in the pipeline

- **Manual labeling** — never happened; the auto-ensemble in the
  sibling repo is the only label producer, and even those labels are
  only used for eval-side ground truth (Phase 4), never for training.
- **Active learning** — we do not loop the model's predictions back
  into the labeling pipeline. The teacher is the supervisor, full stop.
- **Feature engineering on captures** — F0, HNR, RMS, etc. all live in
  the sibling repo's audit pipeline. The student sees raw mel patches.
- **Multi-baby augmentation** — single-deployment captures is the
  acknowledged limitation. Not fixed by this pipeline.

## Anti-patterns this pipeline avoids

- **Training on auto-ensemble labels.** Tempting (we have them),
  wrong (the auto-ensemble has its own confounders, and distillation
  doesn't need labels).
- **Reporting F1 at 0.5.** For distilled students this systematically
  understates capability by 3-4×. Always sweep the threshold.
- **Calibrating quantization on captures.** Would entangle the
  published artifact with private data. AudioSet val only.
- **Letting clip labels imply window labels.** The mistake EXP-006
  fixes — random sub-window sampling spreads positive supervision
  across mostly-non-event audio.
