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

| Experiment | Train data prep | Params | INT8 size | best-F1 | AUC | FPR (best thr) |
|---|---|---:|---:|---:|---:|---:|
| EXP-001 (smoke) | synthetic | 80,713 | — | — | — | — |
| EXP-002 (captures-only) | random 4 windows / clip | 80,713 | — | 0.585 | 0.717 | — |
| EXP-003 (audioset-only) | random 4 windows / clip | 80,713 | — | 0.612 | 0.750 | — |
| EXP-004 (combined) | random 4 windows / clip | 80,713 | — | 0.667 | 0.823 | — |
| EXP-006 (v0.1.0) | teacher-scored sliding window, 1:1 fixed pool | 80,713 | 110 KB | 0.696 | 0.860 | 31.8 % |
| EXP-007 | + per-epoch resample, 1:3 pos:neg | 80,713 | — | 0.756 | 0.861 | 22.7 % |
| **EXP-008 (v0.2.0)** | **+ LR drop 1e-3→1e-4 @ epoch 30** | 80,713 | **110 KB** | **0.756** | **0.870** | **22.7 %** |

**EXP-008 is the published model (v0.2.0).** Two key methodology shifts:

1. **EXP-006 — teacher-as-filter** (+0.037 AUC over random-window baseline). YAMNet pre-scores every 0.5 s-hop window of every clip; only confident-positive (`p_cry > 0.30`) and confident-negative (`p_cry < 0.05`) windows enter training. See [`docs/research/methodology-teacher-as-filter.md`](docs/research/methodology-teacher-as-filter.md).
2. **EXP-007–008 — deployment-realism training** (-9 pp FPR at the operating point). 1:3 pos:neg ratio matches deployment audio better than 1:1; per-epoch random subsampling exposes the student to ~all 30 K negatives over training instead of the same 5 714 every epoch.

INT8 quantization cost is small (≤0.012 best-F1, ≤0.004 AUC). With 5-of-9 temporal voting at the operating threshold, the EXP-008 deployment math is **~5× fewer false alerts per night vs EXP-006** at the same recall.

Side metric: KL vs YAMNet on held-out home captures (time-stratified, private). The captures pool is the highest-confidence tier of the auto-ensemble's labels, so per-clip metrics don't translate to general nursery deployment without on-device threshold calibration. Aggregate per-tier behaviour comparison (no captures-derived data published) is at [`docs/research/student-on-captures-summary-20260503.md`](docs/research/student-on-captures-summary-20260503.md).

## Deployment profile (ESP32-S3)

The published model is sized for real-time inference on the Waveshare ESP32-S3-CAM-GC2145 — 240 MHz dual-core Xtensa LX7, 8 MB PSRAM, 16 MB flash. Numbers below compare the v0.2.0 student to the YAMNet teacher running INT8 on the same device (sibling repo's [`chayuto/yamnet-mel-int8-tflm`](https://huggingface.co/chayuto/yamnet-mel-int8-tflm)), since that's the reference cry detector currently deployed.

| | YAMNet teacher (INT8) | EXP-008 student (INT8) | ratio |
|---|---:|---:|---:|
| Flash footprint (`.tflite` size) | ~4.0 MB | **110 KB** | **36× smaller** |
| Parameters | ~4.0 M | **80 713** | 50× smaller |
| Tensor arena (PSRAM allocation) | ~600 KB | **~64 KB** est. | 9× smaller |
| Inference latency / patch (CPU, training host) | ~2 ms | **<1 ms** measured | ~3× faster |
| Inference latency / patch (ESP32-S3) | tens of ms | **single-digit ms** est. | ~10× faster |
| Real-time headroom @ 0.48 s patch hop | ~10× | **~100×** est. | — |

Student-on-ESP32-S3 latency is *estimated* until the on-device parallel-logging build lands (Phase A in the [firmware integration plan](https://github.com/chayuto/ws-ESP32-S3-CAM/blob/main/docs/research/student-integration-plan-20260503.md)).

**Accuracy translation to deployment.** Three different lenses on the same model:

| | what it measures | EXP-008 INT8 |
|---|---|---:|
| AudioSet test AUC | public-domain ranking quality on a frozen 62-segment slice | **0.866** |
| AudioSet test best-F1 | binary-classification quality at the F1-optimal threshold | **0.744** |
| Captures frame-correlation (high_pos) | how closely the student's per-frame cry score tracks the teacher's on the same audio | **r = 0.81** |
| Captures student max (high_neg) | how often the student wrongly fires on quiet/silent audio (lower = better) | **0.06** (was 0.15 in EXP-006) |

The student is a faithful approximator of the teacher on the deployment data — frame correlation 0.81 on high-confidence positives — and it is materially more conservative than v0.1.0 on quiet audio (high_neg max 0.15 → 0.06).

**Calibration is per-deployment.** Distilled students inherit the teacher's diffuse softmax: cry probabilities concentrate well below 0.5 even on obvious cries. The model's best-F1 threshold is ≈0.05 on the AudioSet test, but the *right* operating point depends on the specific room, mic gain, and ambient noise floor. The detector firmware (`projects/cry-detect-01/main/detector.c` in the sibling repo) exposes threshold, N-of-M consecutive-frame voting, and hold-time hysteresis as runtime-configurable parameters — so a deployment-time calibration recipe of "record 10 min of ambient, set threshold = max(score) + 0.05" is sufficient.

**Why distill at all if YAMNet already runs on-device?** Three things become possible with a 36× smaller model:

- **~3.9 MB of flash and ~540 KB of PSRAM freed** for longer audio buffers (better temporal context), parallel image features from the on-board GC2145 camera, larger OTA partitions, or just less PSRAM pressure for the rest of the firmware.
- **Lower power draw** on a battery-powered baby monitor — faster inference + smaller weight reads + fewer cache misses.
- **Portability beyond ESP32-S3.** 110 KB INT8 fits on Cortex-M4, RP2040, ESP32-S2, and most TFLite-Micro targets where YAMNet's 4 MB does not.

The deployment story is **not** "the student is more accurate than the teacher" — the teacher remains the reference. The story is "the student is *good enough* to recover most of the teacher's behaviour at a fraction of the resource cost, freeing the rest of the device for the rest of the product."

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
