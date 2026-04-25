# CLAUDE.md — yamnet-cry-distill-int8

Training side of the cry-detect work. Owns teacher/student knowledge
distillation, public-data evaluation, INT8 quantization, and the
published `.tflite` model card.

The device side — firmware, audio harvest, on-device inference,
host-side auto-ensemble label tooling — lives in the sibling repo
`ws-ESP32-S3-CAM` (sibling on disk at `../ws-ESP32-S3-CAM`).

## Boundary

Full repo split documented at:

  https://github.com/chayuto/ws-ESP32-S3-CAM/blob/main/docs/research/repo-boundary-yamnet-cry-distill.md

In short, this repo:

- **IS for:** distillation training loop, student architectures, data
  loaders (AudioSet / ESC-50 / UrbanSound + home captures via path),
  INT8 export, MODEL_CARD.md, evaluation harness.
- **IS NOT for:** firmware, runtime mel features, on-device inference,
  the auto-ensemble label tooling, raw captures.

## Privacy invariants

- Raw audio captures live in `ws-ESP32-S3-CAM/datasets/` (gitignored
  on that side). This repo reads them via filesystem path. The env
  var `WS_ESP32_S3_CAM_ROOT` overrides the default sibling location.
  **Never copy audio into this repo.**
- Trained student weights are publishable only when the headline eval
  stands on public-data (AudioSet) on its own merits.
- Per-capture intermediates (label CSVs, release JSONs, ensemble
  pickles) stay on the device side. This repo consumes the frozen
  release JSON as a read-only contract.

## ML discipline

The `/ml-researcher` skill (`.claude/commands/ml-researcher.md`)
applies for all work in this repo: pre-register hypotheses, stamp
model versions in `config.json`, lab notebooks gitignored under
`ml-experiments/`, durable conclusions land as research notes.

## Sibling repo cross-references

- `ws-ESP32-S3-CAM/docs/research/host-side-auto-ensemble-method.md`
  — methodology for the label production that feeds eval here.
- `ws-ESP32-S3-CAM/docs/research/repo-boundary-yamnet-cry-distill.md`
  — the repo split.
- `chayuto/yamnet-mel-int8-tflm` on HuggingFace — the teacher
  artifact (already shipped from prior work).
