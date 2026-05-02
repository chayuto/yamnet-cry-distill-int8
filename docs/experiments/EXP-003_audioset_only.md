# EXP-003 — AudioSet-only distillation (pending)

**Status:** infrastructure shipped, waiting on AudioSet downloads.

## What's ready

- `data/ids/audioset_test.csv` — 100 segments, frozen forever, eval-derived.
- `data/ids/audioset_train.csv` — 570 segments, balanced-train derived.
- `data/ids/audioset_val.csv` — 160 segments, disjoint val slice.
- `scripts/curate_audioset_ids.py` — deterministic regenerator
  (verified bit-exact reproduction).
- `scripts/download_audioset.py` — yt-dlp + ffmpeg driver, idempotent,
  marks dead segments with `.dead` files, supports `--max N`,
  `--jobs K`, `--dry-run`.
- `src/yamnet_cry_distill_int8/data/mixers.py` — `patches_from_audioset`
  reads cached AudioSet WAVs; `build_patch_pool` switches across
  `captures` / `audioset` / `mixed` per `data.source` config.
- `configs/exp003_audioset_only.yaml`,
  `configs/exp004_combined.yaml`.
- `train.py` — refactored to consume the patch-pool abstraction.
  Same `python -m yamnet_cry_distill_int8.train --config <yaml>`
  entry point as EXP-002. Verified EXP-002 still reproduces bit-exact
  (best val KL 1.2668 @ epoch 35) after the refactor.

## What's needed before this experiment runs

1. `brew install ffmpeg` (host-side, ~50 MB).
2. `pip install -e ".[audioset]"` (yt-dlp).
3. `python scripts/download_audioset.py --all --jobs 4` — pulls ~730
   training segments. Expect 30–90 minutes wall-clock and 10–30 %
   takedown losses (those are marked `.dead` and skipped).
4. `python -m yamnet_cry_distill_int8.train --config configs/exp003_audioset_only.yaml`.

## Hypothesis (for the next session's writeup)

A captures-free student trained purely on the curated AudioSet pool
will reach a val KL in the 1.0–2.0 nat range — likely worse than
EXP-002 (val KL 1.27 on captures) because AudioSet has higher class
diversity (200 silence + 200 speech + 100 confounders dilute the cry
signal). The headline number is *test-set* AudioSet F1, not val KL —
that's the public-reproducibility metric per the buildout plan.

## Reproducibility (for the next session)

```bash
# 0. one-time host setup
brew install ffmpeg
pip install -e ".[dev,audioset]"

# 1. fetch AudioSet audio (~30-90 min)
python scripts/download_audioset.py --all --jobs 4

# 2. train
python -m yamnet_cry_distill_int8.train \
       --config configs/exp003_audioset_only.yaml

# 3. eval (Phase 4) — fills in the AudioSet F1 column of README's perf table
```
