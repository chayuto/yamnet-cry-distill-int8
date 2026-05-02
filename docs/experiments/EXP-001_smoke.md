# EXP-001 — distillation loop smoke test

**Date:** 2026-05-02
**Branch / commit:** main / parent 7966fd6 (Phase 0)
**Config:** none — single hard-coded smoke step in `train.py --smoke`

## Hypothesis

The full plumbing — segment-ID CSV → audio loader → YAMNet teacher
forward → DS-CNN student forward → KL divergence loss → optimizer
step — closes end-to-end with no NaN/Inf and no `*.wav` artefacts
written into the repo. **This is a "does the wire light up" test, not
a quality test.** No claim about generalization, accuracy, or even
sensible loss values — only that gradients flow.

## Setup

- **Data:** `data/ids/audioset_smoke.csv`, 50 placeholder ytids
  (`_smoke_cry_*`, `_smoke_spc_*`, `_smoke_sil_*`, `_smoke_bab_*`).
  The loader recognises the `_smoke_` prefix and emits deterministic
  synthetic 0.975 s waveforms (shaped Gaussian noise plus a class-
  coloured tone) — no network, no yt-dlp, no audio committed.
- **Teacher:** YAMNet from TF Hub (`https://tfhub.dev/google/yamnet/1`),
  cached at `~/.cache/yamnet_teacher/`. Output: 521-class softmax
  scores per 0.96 s patch, mean-pooled to a single (521,) clip
  distribution.
- **Student:** `dscnn_student` — Conv2D stem + 4 depthwise-separable
  blocks (filters 16/32/64/128, strides 1/2/2/2) + GAP + Dense(521).
  **Parameters: 80,713** (315 KB FP32 → ~80 KB INT8 budget).
- **Loss:** `KL(teacher_probs || softmax(student_logits))`, ε=1e-8 floor.
- **Optimizer:** Adam, lr=1e-3, one step on a batch of 4 synthetic clips.
- **Hardware:** local M-series macOS, CPU only, Python 3.13.11,
  TensorFlow 2.21.0, tensorflow-hub 0.16.1.

## Results

```
[smoke] teacher loaded in 4.6s
[smoke] student parameter count: 80713
[smoke] batch ids: ['_smoke_cry_00', '_smoke_cry_01', '_smoke_cry_02', '_smoke_cry_03']
[smoke] step done in 0.1s, KL loss = 7.9794
[smoke] OK — loop closed end-to-end in 4.8s total.
```

- ✅ Wall-clock 4.8 s (budget: <120 s).
- ✅ KL loss is finite (7.9794) — no NaN/Inf.
- ✅ Student parameter count 80,713 ≤ 100 K budget.
- ✅ `git status` clean of `*.wav` after run; only new files are
  source/tests/CSV/runner. No `data/audioset/cache/` was created.
- ✅ `pytest -q` → 5 passed, 1 skipped (the network-gated YAMNet
  shape test, gated on `RUN_TEACHER_TEST=1`).

## Analysis

The KL value (~8 nats) is the expected order of magnitude when an
untrained student with random softmax output is compared against a
reasonably peaked teacher distribution over 521 classes — `log(521) ≈
6.26`, and the extra mass comes from the entropy gap between teacher
and uniform-ish student. Nothing to read into beyond "the loss is in
a reasonable regime and gradients can flow."

The loop closure is the result. Phase 1 makes no claim about whether
the student is *learning anything useful* — that's Phase 2 (real
captures, real epochs).

## Next steps

EXP-002 will swap the synthetic loader for the private
`home_captures` data source (475 captures from the deployed device),
run real distillation for ~50 epochs, and report held-out KL on a
time-stratified split. Captures stay gitignored; only the eval JSON
filename pattern `eval_home_captures*.json` lands in `.gitignore` so
even local outputs can never be committed.

## Reproducibility

```bash
git clone https://github.com/chayuto/yamnet-cry-distill-int8
cd yamnet-cry-distill-int8
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                          # 5 passed, 1 skipped
bash scripts/run_exp001_smoke.sh  # 0 exit, <2 min on warm cache
```

The synthetic smoke set is fully offline; no AudioSet download
required for this experiment.
