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

Headline metric: held-out AudioSet `crying_sobbing` segment-level F1 / AUC on the 100-segment frozen test set (62 survivors after takedowns). Best-threshold F1 reported because distilled students inherit the teacher's diffuse softmax — see `docs/experiments/EXP-005_eval_harness.md`.

| Experiment | Train data prep | Params | INT8 size | best-F1 | AUC |
|---|---|---:|---:|---:|---:|
| EXP-001 (smoke) | synthetic | 80,713 | — | — | — |
| EXP-002 (captures-only) | random 4 windows / clip | 80,713 | — | 0.585 | 0.717 |
| EXP-003 (audioset-only) | random 4 windows / clip | 80,713 | — | 0.612 | 0.750 |
| EXP-004 (combined) | random 4 windows / clip | 80,713 | — | 0.667 | 0.823 |
| **EXP-006 (combined + teacher-filter)** | **teacher-scored sliding window** | 80,713 | **110 KB** | **0.696** | **0.860** |

**EXP-006 is the published model.** The teacher-as-filter pipeline shift (running YAMNet across every clip first, keeping only windows with `p_cry > 0.30` as positives and `p_cry < 0.05` as hard negatives) added **+0.037 AUC** over the random-window baseline. INT8 quantization cost zero at reported precision. See [`docs/research/methodology-teacher-as-filter.md`](docs/research/methodology-teacher-as-filter.md).

Captures-side side metric (KL vs YAMNet on home captures, time-stratified): private. Disclosed in the model card; not published as a load-bearing number because the captures pool is the highest-confidence tier of the auto-ensemble's labels and so leans easy.

Side metric: KL vs YAMNet on held-out home captures (time-stratified). Private — disclosed only in the model card. The captures pool is the highest-confidence tier of the auto-ensemble's labels, so per-clip metrics don't translate to general nursery deployment without on-device threshold calibration.

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
