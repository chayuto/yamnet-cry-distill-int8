# yamnet-cry-distill-int8

Knowledge-distillation pipeline that turns Google's pretrained YAMNet
(FP32, 521-class AudioSet) into a tiny INT8 student classifier targeted
at on-device baby-cry detection on the Espressif ESP32-S3 (Waveshare
ESP32-S3-CAM-GC2145).

**Status:** in progress — no model shipped from this repo yet.

## What this repo will do

1. Take YAMNet FP32 from TF Hub as the **teacher**.
2. Generate per-frame soft targets (logits over the 21 cry-related
   AudioSet classes plus a confounder bank).
3. Train a small **student** (CRNN or DS-CNN, target ≤ 500 KB INT8)
   on AudioSet baby-cry positives + ESC-50 / UrbanSound negatives,
   plus private in-domain captures from the device repo.
4. Headline eval: held-out AudioSet split. Side metric: leave-one-
   session-out (LOSO) on home captures.
5. Export to TFLite INT8 with representative-dataset calibration.
6. Deploy on the ESP32-S3 via the device repo's `fetch_model.sh`.

## Teacher already published

The INT8-calibrated YAMNet that ships in the device repo today (with
the documented mel-magnitude correction and double-sigmoid bug fixes)
is at:

  https://huggingface.co/chayuto/yamnet-mel-int8-tflm

This repo's eventual deliverable is a *distilled student* derived
from that teacher — a different artifact, not a replacement.

## Sibling repo

The device-side firmware, audio harvesting, on-device inference, and
the host-side auto-ensemble label pipeline live in the sibling repo
`ws-ESP32-S3-CAM`. The boundary between the two repos is documented at
`docs/research/repo-boundary-yamnet-cry-distill.md` in that repo.

## Privacy

Raw audio captures live in `ws-ESP32-S3-CAM/datasets/` (gitignored on
that side) and are read from there via filesystem path — they are
never copied into this repo. Trained student weights are publishable
only when the headline eval stands on public-data (AudioSet) on its
own merits.

## Layout (target, not yet built out)

```
yamnet-cry-distill-int8/
├── README.md
├── CLAUDE.md
├── .gitignore
├── .claude/commands/ml-researcher.md
├── src/
│   ├── teacher.py              # YAMNet FP32 from TF Hub (TBD)
│   ├── student/                # tiny architectures (TBD)
│   ├── data/                   # AudioSet / ESC-50 / UrbanSound / home captures (TBD)
│   ├── train.py                # distillation loop (TBD)
│   ├── eval.py                 # AudioSet held-out + LOSO captures (TBD)
│   └── quantize/
│       ├── __init__.py
│       └── repTQ.py            # PTQ harness migrated from ws-ESP32-S3-CAM
├── models/                     # output tflite + h5 (gitignored intermediates)
├── ml-experiments/             # lab notebooks (gitignored)
└── docs/                       # model cards, public-eligible notes
```

Most of `src/` does not exist yet — only `src/quantize/repTQ.py` is
populated at scaffold time. The rest gets built as the experiment
plan executes.
