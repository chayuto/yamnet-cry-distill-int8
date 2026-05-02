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

Headline metric: held-out AudioSet `crying_sobbing` segment-level F1 (lands in Phase 4 — EXP-005).

Until then, KL divergence vs the YAMNet teacher on identical val pools (cross-eval, same evaluator across runs):

| Experiment | Train data | Params | Captures val KL | AudioSet val KL | Notes |
|---|---|---:|---:|---:|---|
| EXP-001 (smoke) | 4 synth clips | 80,713 | — | — | loop closes in 5 s |
| EXP-002 (captures-only) | 475 captures | 80,713 | **1.27** | 6.62 | overfits to device acoustics |
| EXP-003 (audioset-only) | 413 segments | 80,713 | 2.66 | 4.48 | public-data baseline |
| EXP-004 (combined) | 380 caps + 413 segs | 80,713 | 1.41 | **3.99** | best generalist |

EXP-004 beats EXP-003 even on AudioSet's own held-out — captures act as soft regularization. Random init lands at ~7.3 nats KL on either pool, so all three runs are real signal. AudioSet survivor counts reflect 28.7 % YouTube-takedown attrition over the 830 curated segment IDs.

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
