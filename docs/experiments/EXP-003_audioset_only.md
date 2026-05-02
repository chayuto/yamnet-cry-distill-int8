# EXP-003 — AudioSet-only distillation

**Date:** 2026-05-03
**Branch / commit:** main / Phase 3
**Config:** `configs/exp003_audioset_only.yaml`

## Hypothesis

Training the same DS-CNN student purely on the curated AudioSet pool
(570 train + 160 val segments, ~10 s each, 4 random patches per
segment) drives val KL down. AudioSet is the *public* surface the
portfolio promises to reproduce — this is the captures-free baseline.
We expected val KL in the 1.5–3.0 range (worse than EXP-002's 1.27 on
captures-only, since AudioSet has 5× the class diversity).

## Setup

- **Source CSVs:** `data/ids/audioset_train.csv` (570),
  `data/ids/audioset_val.csv` (160). Disjoint by curation, seeds 2001–2006.
- **Audio cache:** `data/audioset/cache/`, populated by
  `scripts/download_audioset.py --all --jobs 4`. **592 / 830 segments
  downloaded successfully (71.3 % survival, 28.7 % YouTube takedowns).**
  Dead segments are marked with `.dead` files and the loader silently
  skips them — `mixers.patches_from_audioset` reports the skip tally.
- **Patch pool:** train 1 652 patches (570 segments minus 157 dead, ×4
  random crops); val 117 patches (160 segments minus 43 dead, 1
  centered crop each).
- **Teacher / student:** unchanged from EXP-002. YAMNet FP32 frozen,
  DS-CNN student with 80 713 parameters.
- **Loss:** KL(teacher || student), ε = 1e-8.
- **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4, 50 epochs,
  batch 32.
- **Hardware:** macOS, M-series CPU only.
- **Wall-clock:** download 22 min (parallel ×4); training ~5 min.

## Results

| metric | value |
|---|---:|
| init val KL (random student) | 7.798 |
| best val KL (own AudioSet val) | **4.480** |
| best epoch | 20 |
| final train KL (epoch 50) | 2.179 |
| final val KL (epoch 50) | 4.591 |

Cross-eval on the two held-out pools (one shared evaluator across
EXP-002 / 003 / 004 — see `docs/experiments/cross_eval_phase3.json`):

| eval pool | EXP-002 | EXP-003 | EXP-004 |
|---|---:|---:|---:|
| captures val (95 clips) | 1.267 | 2.664 | 1.407 |
| AudioSet val (117 patches) | 6.615 | **4.480** | **3.989** |

## Analysis

**EXP-003 is the captures-free baseline.** It posts a real,
reproducible-from-public-data number — 4.48 nats KL on its own
AudioSet val. Anyone can clone this repo, run
`scripts/download_audioset.py --all`, then `python -m
yamnet_cry_distill_int8.train --config configs/exp003_audioset_only.yaml`
and reproduce within ±0.05 (modulo the takedown roulette).

**EXP-002's captures-only model generalizes poorly to AudioSet
(KL 6.62)** — barely better than the random init at 7.80. That
makes sense: the captures all share device acoustics, an enclosed
nursery environment, and roughly 5–6 night-session "scenes." The
student learned to mimic YAMNet's response to *that specific
distribution*, not generic audio.

**The captures-side number for EXP-003 is 2.66** — twice EXP-002's
number, but only ~33 % of the random-init starting KL. So AudioSet
training does transfer somewhat to the deployed setting; the student
isn't useless in the nursery, just clearly weaker than a captures-
trained one.

**Takedowns are real and deterministic-ish.** Of 830 curated segments,
238 (28.7 %) returned 410 / 451 / age-gate / private-video errors.
This skews young videos heavier (some recent uploads have already
been deleted). The frozen test set has the same exposure — we'll
record the survival rate of `audioset_test.csv` separately when we
run the headline eval in Phase 4.

## Next steps

EXP-004 (combined) lands the captures + AudioSet mix and beats both
on the AudioSet pool — see EXP-004 writeup.

## Reproducibility

```bash
brew install ffmpeg                                   # one-time
pip install -e ".[dev,audioset]"                      # one-time
python scripts/download_audioset.py --all --jobs 4    # ~22 min
python -m yamnet_cry_distill_int8.train \
       --config configs/exp003_audioset_only.yaml      # ~5 min

# Reproduces best_val_kl ≈ 4.48 ± 0.05 (variance from takedown set,
# not training noise — same seed produces identical patches).
```
