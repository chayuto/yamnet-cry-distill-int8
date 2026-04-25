# ML researcher mode

Activate the scientist's mindset for any ML / data-science work in this
repo: form a hypothesis, run a controlled comparison, log everything for
reproducibility, and surface only the *durable conclusions* in the
research-doc tree.

This skill is mirrored from the sibling repo `ws-ESP32-S3-CAM`. It
applies here for distillation training, student architecture sweeps,
PTQ calibration experiments, eval harness changes, etc. Cross-references
to "the device repo" or "ws-ESP32-S3-CAM" resolve to the sibling on
disk at `../ws-ESP32-S3-CAM/`.

Usage: `/ml-researcher <topic>` — explicit invocation, OR
       activate this mode whenever the work is ML/data-science (training
       a model, tuning a threshold, running an ablation, calibrating
       quantization, etc.).

---

## The principle

**Every ML experiment generates two output streams:**

1. **Internal log** (gitignored, like raw data): hypothesis statement,
   exact config, every numeric result, plots, intermediate models.
   This is the *lab notebook*. It's noisy, large, often boring.
2. **Durable conclusion** (committed): a research note that records
   what we tried, what happened, and what to do next. Short, scannable,
   future-self-friendly.

We must produce BOTH. Skipping the lab notebook means we can't verify
the conclusion. Skipping the conclusion means future-us re-treads the
same path.

---

## Workflow

### Step 1 — Pre-register the hypothesis

Before touching any tool, write down (in a fresh notebook entry):

```
Hypothesis: <1-2 sentences. What we EXPECT to happen and why.>
Falsifier:  <what result would tell us we're wrong>
Method:     <baseline + intervention + ablation, named explicitly>
Data slice: <e.g. "AudioSet held-out + cry-v0.1-ensemble high tier",
            or specific session list>
Predicted outcome: <pre-commit a number or direction>

Model / version stamp (REQUIRED — see §Step 1.5)
```

If you don't know what you expect, you're not running an experiment —
you're rummaging.

### Step 1.5 — Stamp every model version in `config.json`

Every experiment must carry a complete model/version manifest at start.
A result that holds under one model often regresses under another;
without version stamps, future-us can't reproduce or diagnose drift.

Required fields in `<experiment_dir>/config.json`:

```json
{
  "experiment_id": "YYYY-MM-DD-<slug>",
  "git_head_sha": "<short hash at experiment start>",
  "data_slice": "<release id or session list>",
  "models_used": {
    "yamnet_teacher":       "google/yamnet/1 (FP32 from TF Hub)",
    "yamnet_int8_tflite":   "chayuto/yamnet-mel-int8-tflm (sha256 …, calib synthetic|real-data)",
    "student_arch":         "<crnn|dscnn|...> v<N>",
    "ensemble_release":     "ws-ESP32-S3-CAM/datasets/.../cry-vX.Y-ensemble.json",
    "audioset_split":       "<balanced_train|eval|unbalanced>"
  },
  "host_python": "<X.Y>",
  "host_tf":     "<X.Y.Z>",
  "seed":        0
}
```

### Step 2 — Allocate an experiment dir

```
ml-experiments/YYYY-MM-DD-<topic-slug>/
├── README.md                 ← lab notebook (start with the
│                                pre-registration above)
├── config.json               ← exact config / seeds / data slice
├── artifacts/                ← models, intermediate CSVs, plots
└── results.json              ← machine-readable summary
```

This directory is **gitignored** (matches the data-vault discipline:
raw and intermediate stays out of git).

### Step 3 — Run the experiment

Iterate freely. Save EVERY intermediate result to `artifacts/`. Update
the lab notebook (`README.md`) live as you go — short bullet entries
timestamped, no need to be polished. Capture what didn't work as well
as what did.

When you change anything significant (different seed, different data
slice, different baseline), update `config.json` and note the change
in the lab notebook with the timestamp.

### Step 4 — Conclude

Decide one of:

- **(A) Significant outcome — durable.** Write a research note
  summarizing hypothesis, method, result, conclusion, next steps.
  Include numerical evidence inline. Cross-reference the experiment
  dir in case future-us wants to re-run.
  - **Where:** `docs/<topic>-YYYYMMDD.md` if it does NOT reference
    private capture timestamps. If it does, keep it local-only or
    mirror to the device repo's `docs/internal/`.

- **(B) Null / unsurprising — ephemeral.** Update the lab notebook
  with the conclusion, leave the experiment dir on disk, do NOT
  commit a separate research note. The notebook is enough.

- **(C) Negative result — durable.** Treat as (A). Negative results
  prevent re-treading. Always commit these.

The boundary is "would future-us benefit from learning this in 3 months
without reading the lab notebook?" If yes → (A) or (C). If no → (B).

---

## Conventions

- **Notebook format:** `README.md` with timestamped entries. Markdown
  + numpy + matplotlib + sklearn / TF covers everything. Persist plots
  as PNGs in `artifacts/`.
- **Seeds:** always set + log `numpy.random.default_rng(seed)`,
  `random.seed(seed)`, and any framework `random_state`.
- **Comparisons:** always include a baseline. Single-treatment
  results without a baseline are observations, not experiments.
- **Held-out for cry data:** hold out by SESSION (LOSO), not by row.
  Heavy within-session correlation; row-level splits leak.
- **Ablations:** when a positive result lands, ablate at least one
  component before believing it.
- **Naming:** `YYYY-MM-DD-<topic-slug>` for experiment dirs.

---

## Anti-patterns to avoid

- **Trying multiple seeds and reporting the best.** Set a seed,
  state the seed, run multiple seeds and report distribution if
  the result is unstable.
- **Cherry-picking favorable splits.** Use LOSO consistently for
  capture-side eval, AudioSet held-out for headline eval.
- **No pre-registered hypothesis.** Without it you can rationalize
  any outcome as success.
- **Skipping the negative-result writeup.** The only way to prevent
  re-treading.
- **Headline-eval'ing on home captures.** Public model cards must
  headline AudioSet held-out numbers.
