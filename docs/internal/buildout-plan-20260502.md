# yamnet-cry-distill-int8 — buildout plan

**Date:** 2026-05-02
**Status:** plan, not yet executed.
**Target audience for this doc:** future me + collaborators executing
the build. Plain English, concrete file paths, acceptance criteria.

---

## 1. Intent

Build a slim, public-portfolio training repo that distills Google's
YAMNet (FP32, 521-class AudioSet) into a tiny INT8 student for the
ESP32-S3 cry-detect deployment. Deliverable is a HuggingFace
**model** (no dataset upload) plus a reproducible training pipeline
that runs on public data alone.

In one sentence: **a stranger should be able to clone, install,
download AudioSet IDs, train, and reproduce the headline number
within ±2% — without ever seeing our private home captures.**

## 2. Constraints

| | rule |
|---|---|
| Repo visibility | public (live portfolio) |
| Data | all audio gitignored — no WAVs, no derived caches, no embeddings derived from private captures |
| HF Hub | publish `/models/...` artifact only. Never `/datasets/`. |
| Headline metric | held-out AudioSet `crying_sobbing` precision/recall — reproducible from public data alone |
| Side metric | LOSO on private captures, time-stratified — disclosed as private in the model card, no raw numbers without context |
| Captures use | **primary** unsupervised distillation signal (teacher logits as supervision, no labels needed). AudioSet is augmentation for the public eval surface |
| Manual labeling | none, ever — the auto-ensemble in the sibling repo is the only label producer |
| Commit hygiene | `.gitignore` + lightweight pre-commit guard refuses commits touching audio paths or model binaries (other than the published one) |

## 3. Pattern reference: `au-fuel-sign-ocr-factory`

We mirror the layout and discipline of the sibling personal project
at `../au-fuel-sign-ocr-factory`. That repo has already settled the
same problem (real public-portfolio ML repo with private data) and
reads cleanly.

**What to copy verbatim:**

| element | from au-fuel | reason |
|---|---|---|
| `pyproject.toml` shape (src layout, optional `[dev]` and a per-task extras group, ruff + pytest) | yes | proven, lean |
| `src/<package>/` package import path with snake_case namespace | yes | standard packaging |
| `scripts/` directory of single-purpose runners (e.g. `scripts/run_exp003.sh`) | yes | one shell entry per experiment |
| `configs/*.yaml` for per-experiment hyperparameters | yes | swap configs without touching code |
| `docs/experiments/EXP-NNN_<short_name>.md` discipline | yes | this is the portfolio surface |
| `docs/model_cards/<model-id>.md` with HF frontmatter | yes | committed copy of what we upload |
| `docs/internal/` for working planning/state notes | yes | committed but boring |
| README ≤30 lines, pitch-first | yes | scannable in 90 seconds |
| `CLAUDE.md` with Product Goal + Core Principles + Agent Working Style | yes | sets agent + reader expectations |
| Honest perf table that shows EXP-NNN progression | yes | real numbers including the bad early ones |

**What differs:**

| element | au-fuel | this repo |
|---|---|---|
| Domain | image OCR (vision, ultralytics) | audio classification (TF Hub, librosa) |
| Training data | mostly public + scraped, hand-labeled | mostly *private captures*, no labels needed (distillation) |
| Headline data source | local labeled images, scraped + manually verified | held-out AudioSet (public) — public reproducibility is the headline |
| Public eval set | hand-curated 19-image canonical test | AudioSet held-out segment IDs (committed CSV) |
| Model registry | one HF repo per model variant | one HF repo (`chayuto/yamnet-cry-distill-int8`) for the student artifact |
| Sibling-repo coupling | none | tight — captures live in `../ws-ESP32-S3-CAM/datasets/`, env var configurable |

## 4. Target layout

```
yamnet-cry-distill-int8/
├── README.md                       # ≤30 lines pitch + perf table + reproduce
├── CLAUDE.md                       # product goal + principles + agent style
├── LICENSE                         # MIT (placeholder)
├── pyproject.toml                  # src layout, py3.11+, deps minimal
├── .gitignore                      # hardened (see §5)
├── .claude/commands/               # /train /eval /quantize /upload-hf
│   └── ml-researcher.md            # already migrated
├── configs/
│   ├── exp001_smoke.yaml           # 1-batch sanity
│   ├── exp002_captures_only.yaml
│   ├── exp003_audioset_only.yaml
│   ├── exp004_combined.yaml
│   └── student_dscnn.yaml          # arch hyperparams shared across exps
├── data/
│   └── ids/
│       ├── audioset_train.csv      # committed (segment IDs only, public info)
│       ├── audioset_val.csv
│       ├── audioset_test.csv
│       ├── audioset_smoke.csv      # 50 IDs for CI / smoke tests
│       └── README.md               # how the IDs were curated
├── scripts/
│   ├── download_audioset.py        # yt-dlp driver, idempotent, skips dead segments
│   ├── run_exp001_smoke.sh
│   ├── run_exp002_captures_only.sh
│   ├── upload_hf.py                # with content-guard pre-flight
│   └── verify_no_captures_in_artifact.py
├── src/
│   └── yamnet_cry_distill_int8/
│       ├── __init__.py
│       ├── teacher.py              # YAMNet FP32 wrapper (TF Hub)
│       ├── student/
│       │   ├── __init__.py
│       │   └── dscnn.py            # depthwise-separable CNN, ≤500 KB INT8
│       ├── data/
│       │   ├── __init__.py
│       │   ├── audioset.py         # reads ids/*.csv + fetched cache
│       │   ├── home_captures.py    # reads $WS_ESP32_S3_CAM_ROOT, optional
│       │   └── mixers.py           # weighted sampling across sources
│       ├── train.py                # distillation loop (KL on teacher logits)
│       ├── eval.py                 # AudioSet held-out (headline) + LOSO captures (side)
│       └── quantize/
│           ├── __init__.py         # already present
│           ├── repTQ.py            # already migrated
│           └── int8.py             # production INT8 export with calibration set
├── tests/
│   ├── test_teacher_shapes.py
│   ├── test_student_param_count.py
│   └── test_audioset_loader.py
├── models/
│   └── .gitkeep                    # gitignored intermediates land here
├── ml-experiments/
│   ├── .gitkeep
│   └── README.md                   # /ml-researcher discipline pointer
├── runs/                           # gitignored — TF/tensorboard logs
└── docs/
    ├── internal/
    │   └── buildout-plan-20260502.md   # this file
    ├── experiments/
    │   ├── EXP-001_smoke.md
    │   ├── EXP-002_captures_only.md
    │   ├── EXP-003_audioset_only.md
    │   └── ...                     # one per training run, no exceptions
    ├── model_cards/
    │   └── yamnet-cry-distill-int8.md  # HF-frontmatter, the published doc
    ├── architecture.md             # diagram + design decisions, public
    └── research/                   # general research notes, public-eligible
```

## 5. Publish boundary

### 5.1 What's committed

- All source under `src/`, `scripts/`, `tests/`, `configs/`.
- `data/ids/*.csv` — segment IDs only (public information).
- `data/ids/README.md` — how the lists were curated.
- All `docs/`, including `docs/internal/`.
- `pyproject.toml`, `README.md`, `CLAUDE.md`, `LICENSE`.
- `models/.gitkeep`, `ml-experiments/.gitkeep`.

### 5.2 What's gitignored (`.gitignore` additions)

```gitignore
# Audio (any form)
*.wav
*.mp3
*.flac
*.ogg

# Training-data caches
data/audioset/
data/captures/
data/cache/

# Model binaries (except .gitkeep)
*.tflite
*.h5
*.pkl
*.pt
*.onnx
models/*
!models/.gitkeep

# Training run outputs
runs/
ml-experiments/*
!ml-experiments/.gitkeep
!ml-experiments/README.md

# Eval outputs that depend on private data
docs/experiments/eval_home_captures*.json

# Local env
.env
.env.*
```

### 5.3 Pre-commit guard (`scripts/verify_no_captures_in_artifact.py`)

Single Python script. Run by `scripts/upload_hf.py` *before* upload.
Walks the upload bundle and refuses if any path matches:

- `*.wav`, `*.mp3`, `*.flac`, `*.ogg`
- substring `capture`, `home`, `session`, `night-`
- substring `cry-202` (any timestamped capture filename)
- any file > 10 MB except `*.tflite`

Prints diff and exits 1. Belt-and-braces — the `.gitignore` should
already prevent these from being in git, but the upload bundle is
constructed at runtime and could pick up local working files.

## 6. Phased build

Each phase has explicit acceptance criteria. **Don't move to the
next phase without ticking all boxes.** Each phase ends with a
single commit + a dated `docs/experiments/EXP-NNN_*.md`.

### Phase 0 — Harden the scaffold (~30 minutes, no training)

**Tasks:**
1. Rewrite `README.md` to pitch-first ≤30 lines (architecture
   block, perf-table-with-empty-rows, reproduce snippet).
2. Add the `.gitignore` rules in §5.2.
3. Add `pyproject.toml` proper (currently doesn't exist; copy au-fuel's
   layout). Deps: `tensorflow`, `tensorflow-hub`, `numpy`, `scipy`,
   `librosa`, `soundfile`, `pyyaml`. Dev: `pytest`, `ruff`. Optional
   `[audioset]`: `yt-dlp`.
4. Add `LICENSE` (MIT placeholder).
5. Write `docs/architecture.md` — single-page diagram of teacher →
   student distillation, plus the two-data-source mixer.
6. Write `docs/model_cards/yamnet-cry-distill-int8.md` — HF
   frontmatter + empty perf table + scope note ("trained on AudioSet
   + private in-domain corpus, headline eval public-data only").
7. Add `scripts/verify_no_captures_in_artifact.py` — the guard.
8. Move this plan into the new repo at
   `docs/internal/buildout-plan-20260502.md` (already lives there).
9. Single commit: "Phase 0: scaffold hardening for portfolio posture".

**Acceptance:**
- `git ls-files` shows nothing under `data/`, `models/`, `runs/`,
  `ml-experiments/`.
- `python -c "import yamnet_cry_distill_int8"` works after
  `pip install -e .`.
- README renders cleanly on GitHub.
- `verify_no_captures_in_artifact.py` smoke test against a fake
  bundle exits 1 when given a `*.wav`.

### Phase 1 — Teacher + smoke test (~half day)

**Tasks:**
1. `src/yamnet_cry_distill_int8/teacher.py` — load
   `https://tfhub.dev/google/yamnet/1` once, cache to
   `~/.cache/yamnet_teacher/`, expose `forward(waveform_16k_mono) →
   (logits, scores, embeddings)`.
2. `src/.../data/audioset.py` — read `data/ids/audioset_smoke.csv`
   (~50 IDs hand-picked for variety), call into
   `scripts/download_audioset.py` lazily.
3. `data/ids/audioset_smoke.csv` — 50 IDs: 30 `crying_sobbing`,
   10 `speech`, 5 `silence`, 5 `babbling`. Curated by hand from
   AudioSet's public segment list.
4. `src/.../student/dscnn.py` — minimal DS-CNN, ~50 KB FP32 (drop to
   ~12 KB INT8). Input: 96-mel × T spectrogram. Output: same logit
   space as teacher (521 classes), or a smaller head we map to
   teacher's cry classes — decide in EXP-001.
5. `src/.../train.py` with `--smoke` flag: load 1 batch (4 clips)
   from smoke set, forward through teacher, forward through student,
   compute KL on the cry class subset, one optimizer step. No save.
6. `tests/test_teacher_shapes.py`, `test_student_param_count.py`.
7. `scripts/run_exp001_smoke.sh` — wires it all together.
8. `docs/experiments/EXP-001_smoke.md` — hypothesis: "loop closes
   end-to-end, no NaN, student forward pass matches expected output
   shape." Result: pass/fail.

**Acceptance:**
- `bash scripts/run_exp001_smoke.sh` exits 0 in <2 minutes on this
  laptop.
- pytest passes locally.
- Student parameter count ≤ 100K (target ≤500 KB INT8 = ~125K
  parameters at int8).
- No `*.wav` files committed; the 50 smoke clips live in
  `data/audioset/cache/` (gitignored).

### Phase 2 — Real distillation, captures-primary (~1 day)

**Tasks:**
1. `src/.../data/home_captures.py` — read `$WS_ESP32_S3_CAM_ROOT/
   datasets/cry-detect-01/captures/` (default `../ws-ESP32-S3-CAM`).
   Skip silently if env var unset or path missing. Yield only WAV
   paths + timestamps (no labels).
2. `configs/exp002_captures_only.yaml` — 475 captures (or whatever
   we have at run time), 50 epochs, KL loss only, batch 32, AdamW
   1e-3.
3. Run on local (probably overnight on M-series CPU; explore
   M1-Metal/MPS later).
4. Save final checkpoint to `models/exp002_dscnn.h5` (gitignored).
5. Eval on a held-out 20% slice of captures: KL divergence between
   student and YAMNet. Write to
   `docs/experiments/eval_home_captures_exp002.json` (gitignored).
6. `docs/experiments/EXP-002_captures_only.md` — hypothesis: "student
   matches teacher's logits to within KL≤0.5 on held-out captures."
   Result with the actual number.

**Acceptance:**
- Held-out KL declines from ~init-loss to a stable value.
- No labels used anywhere in training (verify: no read of
  `master.csv` or `*.json` releases during training).
- `git status` after the run shows zero untracked WAVs, zero new
  files under `data/captures/`.

### Phase 3 — AudioSet expansion (~1-2 days)

**Tasks:**
1. Curate full AudioSet ID lists (commit to `data/ids/`):
   - `audioset_train.csv` — ~1000 IDs, mix of:
     - 400 `crying_sobbing` / `baby_cry_infant`
     - 200 hard negatives: `speech`, `child_speech`, `babbling`
     - 200 ambient negatives: `silence`, `noise`, `appliance`
     - 200 confounders: `screaming`, `whimper`, `wail_moan`
   - `audioset_val.csv` — ~200 IDs, same distribution.
   - `audioset_test.csv` — ~100 IDs, **frozen forever** — never
     touched by train. This is the headline-metric set.
2. `data/ids/README.md` — methodology: which AudioSet release,
   which class IDs, what was filtered (we drop `dog_whimper` /
   `cat_yowl` if any showed up under cry classes).
3. `scripts/download_audioset.py` — yt-dlp driver, idempotent,
   handles takedowns gracefully, caches under `data/audioset/cache/`.
4. `src/.../data/mixers.py` — weighted sampler across AudioSet +
   captures with config-driven mix ratio.
5. `configs/exp003_audioset_only.yaml`,
   `configs/exp004_combined.yaml`.
6. Run EXP-003 (AudioSet only) and EXP-004 (50/50 mix). Both
   write to `models/exp00N_dscnn.h5` (gitignored).
7. `docs/experiments/EXP-003_*.md`, `EXP-004_*.md`.

**Acceptance:**
- `data/ids/audioset_train.csv` is committed and readable.
- `git ls-files data/audioset/` is empty (only the IDs are tracked,
  not the audio).
- `download_audioset.py` is idempotent (rerunning is a no-op).
- EXP-003 has *some* number on the held-out AudioSet set.

### Phase 4 — Eval harness + portfolio numbers (~1 day)

**Tasks:**
1. `src/.../eval.py` — two evaluators:
   - `eval_audioset_holdout(model, ids_path) → {f1, precision, recall,
     auc}`. **Headline metric.** Output to
     `docs/experiments/eval_audioset_holdout_<exp>.json` (committed).
   - `eval_home_captures(model, root) → {kl, auc_vs_ensemble,
     stratified_by_hour}`. **Side metric.** Output to
     `eval_home_captures_<exp>.json` (gitignored).
2. Time-stratified split for captures (use master.csv `ts_iso`,
   bucket by hour, holdout one bucket per night).
3. Update `README.md` perf table with EXP-001..N rows. Honest
   numbers, including the early bad ones — same discipline as
   au-fuel's table.
4. Iterate on student arch / hyperparams as needed: EXP-005..N.
   Each gets its own `docs/experiments/EXP-NNN_*.md`.

**Acceptance:**
- README perf table has at least 3 rows (EXP-002, 003, 004 minimum).
- Headline number is a real measurement, not a placeholder.
- Side metric is mentioned but its private nature is disclosed.

### Phase 5 — Quantize + publish (~half day)

**Tasks:**
1. `src/.../quantize/int8.py` — TFLite INT8 export with
   representative-dataset calibration. Calibration set: a slice of
   AudioSet train (committed IDs → cached audio). **Never** include
   captures in the calibration set — they would leak into the
   published artifact's quantization parameters.
2. Update `docs/model_cards/yamnet-cry-distill-int8.md` with real
   numbers, training data description (clearly stating the public/
   private split), and the calibration-set provenance.
3. `scripts/upload_hf.py` — assembles upload bundle (only:
   `model.tflite`, `MODEL_CARD.md`, `config.json`,
   `eval_audioset_holdout_<final>.json`), runs
   `verify_no_captures_in_artifact.py`, then `huggingface-cli upload`.
4. Tag the commit `v0.1.0` and push to GitHub.
5. Update sibling repo's `tools/fetch_model.sh` to optionally pull
   the distilled model in addition to the YAMNet teacher (later;
   not in this repo).

**Acceptance:**
- `huggingface.co/chayuto/yamnet-cry-distill-int8` exists and is
  pullable.
- Inspection of the uploaded bundle shows zero capture-derived
  artifacts.
- README on GitHub renders cleanly with the perf table.

## 7. Experiment-log discipline

Every training run gets one file at `docs/experiments/EXP-NNN_<short>.md`.
Mirroring au-fuel's format. Required sections:

```markdown
# EXP-NNN — <short title>

**Date:** YYYY-MM-DD
**Branch / commit:** <sha>
**Config:** configs/expNNN_*.yaml

## Hypothesis
What we expect, why, and how we'll know.

## Setup
- Data: which IDs, which captures, mix ratio
- Architecture: which student, param count
- Training: epochs, batch, optimizer, LR schedule
- Hardware: M-series, GPU, etc.

## Results
- Headline (AudioSet held-out): F1=..., P=..., R=...
- Side (home captures held-out): KL=..., AUC vs ensemble=...
- Training curves (link or inline)

## Analysis
What surprised us. What the numbers mean.

## Next steps
What we'd change for EXP-(N+1).

## Reproducibility
`bash scripts/run_expNNN_*.sh` reproduces from a fresh clone +
AudioSet download.
```

EXP-001 through EXP-NN are sequential. **Never reused, never
deleted.** Bad results get committed alongside good ones — that's
the portfolio narrative.

## 8. HF publication checklist

Before `huggingface-cli upload`:

- [ ] `verify_no_captures_in_artifact.py` exits 0 on the bundle.
- [ ] `MODEL_CARD.md` has real numbers, not placeholders.
- [ ] `MODEL_CARD.md` discloses the public / private data split.
- [ ] No `*.wav`, `*.npy`, `*.pkl` of any kind in the bundle.
- [ ] `eval_audioset_holdout_<final>.json` is the *only* eval JSON
      in the bundle.
- [ ] Model file is `model.tflite` ≤ 1 MB.
- [ ] License header in MODEL_CARD matches `LICENSE`.
- [ ] Tag the git commit + reference it in the model card.

## 9. Decisions deferred

These get resolved during execution; documenting them now so they
don't surprise us:

| decision | resolution moment |
|---|---|
| Student architecture (DS-CNN vs CRNN vs MLP-on-mel) | EXP-001 (pick the one that smoke-tests cleanest in <2min) |
| Number of AudioSet IDs | Phase 3, may need to grow if EXP-003 underperforms |
| Whether to keep all 521 teacher classes or distill only the cry head | Phase 2 — start with the cry head (≤30 classes), expand if it limits transfer |
| Calibration set composition | Phase 5 — AudioSet only (decided), but which subset |
| Whether to publish multiple sizes (e.g. tiny vs small) | After v0.1.0 ships — defer |

## 10. Anti-goals

Things this repo deliberately does **not** do:

- No active-learning loop or human-in-the-loop labeling.
- No real-time inference / streaming code (lives in firmware in the
  sibling repo).
- No mel-feature implementation (firmware ports librosa-equivalent;
  this repo uses standard librosa for training-time mel).
- No web UI, no served inference endpoint.
- No multi-language support. English AudioSet labels only.
- No trained-model auto-deploy CI. Publishing is manual + checklist-
  gated.

## 11. Time estimate

Aggregating phases:

| phase | est. |
|---|---|
| 0 — scaffold harden | 30 min |
| 1 — teacher + smoke | 4 hr |
| 2 — captures-only run + writeup | 1 day |
| 3 — AudioSet curation + 2 runs | 1-2 days |
| 4 — eval + iteration to 3+ rows | 1-2 days |
| 5 — quantize + publish | 4 hr |
| **total** | **5-7 working days** spread over 2-3 weeks |

The two big variances are AudioSet ID curation (Phase 3) and
iteration in Phase 4. Both can stretch.

## 12. First action

Phase 0. Once approved, that's a single ~30-minute commit that
locks in the public posture before any training-related code ships.
