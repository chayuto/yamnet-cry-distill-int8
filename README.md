# yamnet-cry-distill-int8

Knowledge-distillation pipeline that turns Google's pretrained YAMNet (FP32, 521-class AudioSet) into a tiny INT8 student classifier targeted at on-device baby-cry detection on the ESP32-S3.

**Status:** in progress — see [`docs/experiments/`](docs/experiments/) for the run log.

## Architecture

```
[private home captures]   [public AudioSet IDs]
        │                         │
        ▼                         ▼
   librosa mel ──► mixer ──► YAMNet teacher (FP32) ──► soft logits
                              │                              │
                              └──► tiny student (DS-CNN) ──► KL loss
                                                              │
                                              held-out AudioSet eval (HEADLINE)
                                              held-out captures eval (side, private)
                                                              │
                                                              ▼
                                                 INT8 TFLite ──► HF model hub
```

## Performance

Headline metric: held-out AudioSet `crying_sobbing` segment-level F1.

| Experiment | Train data | Params | INT8 size | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| EXP-001 (smoke) | 4 synth clips | 80,713 | ~80 KB | — | — | — |
| EXP-002 (captures-only) | 475 | TBD | TBD | TBD | TBD | TBD |
| EXP-003 (audioset-only) | TBD | TBD | TBD | TBD | TBD | TBD |
| EXP-004 (combined) | TBD | TBD | TBD | TBD | TBD | TBD |

Side metric (private in-domain): KL vs YAMNet on held-out home captures, time-stratified.

## Reproduce

```bash
git clone https://github.com/chayuto/yamnet-cry-distill-int8
cd yamnet-cry-distill-int8
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,audioset]"
pytest

# Download the public AudioSet smoke set (50 segment IDs, ~2 minutes)
python scripts/download_audioset.py --ids data/ids/audioset_smoke.csv

# Phase-1 smoke test: prove the distillation loop closes
bash scripts/run_exp001_smoke.sh
```

The full training pipeline reproduces against committed AudioSet segment IDs alone — no private data required.

## Privacy

The headline number is reproducible from public AudioSet segment IDs. A private in-domain corpus (deployed-device home captures) is used as **augmentation only** during distillation; it is never committed, uploaded to HuggingFace, or required for reproduction. See [`docs/architecture.md`](docs/architecture.md) and the [model card](docs/model_cards/yamnet-cry-distill-int8.md) for the public/private split.

## Sibling repo

Device firmware, audio harvest, on-device inference, and the host-side auto-ensemble label tooling live at [`ws-ESP32-S3-CAM`](https://github.com/chayuto/ws-ESP32-S3-CAM). The teacher artifact (`chayuto/yamnet-mel-int8-tflm`) was published from prior work in that repo.
