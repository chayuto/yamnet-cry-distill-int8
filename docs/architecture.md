# Architecture

## Goal

A ≤500 KB INT8 TFLite cry detector that runs on an ESP32-S3 (240 MHz Xtensa LX7, 8 MB PSRAM, ~120 ms inference budget per 0.96 s YAMNet frame).

The teacher is Google's pretrained YAMNet — already widely-deployed, not retrained here. This repo distills it into a smaller, faster student that mimics teacher behavior on the audio distributions our deployment encounters.

## Why distillation

Three reasons we distill rather than train a classifier from scratch:

1. **No labels needed for the loss.** The teacher's 521-class soft logits are the supervision signal. Our 475 deployment captures, which have no human labels, become first-class training data.
2. **YAMNet already encodes "this sounds like X" for 521 classes.** The student inherits that representational richness instead of relearning it from a small label set.
3. **The student can be much smaller.** YAMNet FP32 is ~17 MB; we target ≤500 KB INT8. The student doesn't need to *know* 521 classes — it only needs to match the teacher's outputs on audio close to the deployment domain.

## Data flow

```
                      ┌──────────────────────────────┐
                      │   PUBLIC                     │   reproducible from
                      │                              │   ids/*.csv alone
   AudioSet segment ──┴── yt-dlp → cache (.gitignored)
   IDs (committed CSV)         │
                               │
                               ├─► librosa.melspectrogram(96 mels, 0.96 s frames)
                               │
                      ┌────────┴────────┐
                      │                 │
   $WS_ESP32_S3_CAM_  │   home_captures │   skipped if env unset;
   ROOT/datasets/.. ──┴────►  reader    │   private path
                                        │
                                        ▼
                              src/data/mixers.py
                              (config-driven weighted sampler)
                                        │
                          ┌─────────────┴──────────────┐
                          │                            │
                          ▼                            ▼
              ┌─────────────────────┐    ┌─────────────────────┐
              │ YAMNet teacher      │    │ student (DS-CNN,    │
              │ FP32 from TF Hub    │    │ ≤100 K params)      │
              │ → 521 class logits  │    │ → cry-class logits  │
              └────────┬────────────┘    └─────────┬───────────┘
                       │                           │
                       └─────► KL divergence ◄─────┘
                                        │
                                  AdamW step

         (eval) ──► held-out AudioSet test IDs (HEADLINE, public)
                ──► held-out home captures (side, time-stratified)
```

## Key design decisions

### Captures as primary training signal

The deployment will see the audio distribution of one specific bedroom — caregiver speech, white-noise machine, occasional appliance, intermittent cries. The student's job is to behave like the teacher *on that distribution*. The 475 captures are the most direct training signal for that goal. We use them all, label-free.

AudioSet is augmentation: it broadens the distribution so the student doesn't overfit to bedroom acoustics, and it provides the headline reproducibility surface (a stranger can run the eval without our data).

### Headline metric on public data only

The model card and README must lead with a number a stranger can reproduce. That excludes anything derived from private captures. We headline `crying_sobbing` F1 on a frozen held-out AudioSet split (`data/ids/audioset_test.csv`).

The in-domain side metric (KL vs YAMNet on held-out captures, time-stratified) is reported with a disclosure note. Useful for our own iteration; not part of any external claim.

### Time-stratified eval splits for captures

The deployment data has a strong time-of-day → label correlation (cries cluster at 05h and 18-20h; non-cry concentrates at 09-15h). A naive random split inflates eval scores by letting the model learn "evening = cry." We stratify holdout by hour-of-day so the eval distribution matches the train distribution along that axis.

### INT8 budget

Target: ≤500 KB INT8 TFLite, equivalent to ~125K parameters at int8 precision (1 byte per weight) with overhead. Architectures we consider:

- **Depthwise-separable CNN** (likely choice): efficient on Xtensa, good fit for mel spectrograms.
- **CRNN**: small RNN tail over a CNN front; richer temporal modeling but more memory at runtime.
- **MLP on YAMNet embeddings**: smallest, but defeats the distillation premise (re-uses teacher's mid-stack).

Decision deferred to EXP-001 — pick the architecture that smoke-tests cleanest in <2 min on this laptop.

### Calibration set (for INT8 export)

INT8 quantization needs a representative-dataset pass. Our calibration set is **AudioSet only** — never home captures. Reason: the calibration set's statistics imprint on the published artifact's quantization parameters; mixing in private data would mean a private signal subtly shapes the public model.

## Sibling-repo contract

This repo and `ws-ESP32-S3-CAM` are coupled but never share files:

- `$WS_ESP32_S3_CAM_ROOT` (default `../ws-ESP32-S3-CAM`) is read-only from this side.
- `home_captures.py` reads WAVs by path. No copying, no symlinks under `data/`.
- The frozen release JSON (`datasets/cry-detect-01/releases/cry-vN.M-ensemble.json`) is the schema contract — version-pinned in this repo's `configs/`.
- The published HF model is consumed back by the sibling repo's `tools/fetch_model.sh`, which is pluggable across teacher (current) and student (future).

Full repo-split contract: see [`ws-ESP32-S3-CAM/docs/research/repo-boundary-yamnet-cry-distill.md`](https://github.com/chayuto/ws-ESP32-S3-CAM/blob/main/docs/research/repo-boundary-yamnet-cry-distill.md).
