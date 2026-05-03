---
license: mit
library_name: tflite
tags:
  - audio-classification
  - knowledge-distillation
  - yamnet
  - cry-detection
  - edge-ai
  - esp32
  - esp32-s3
  - int8
  - tflite-micro
  - tinyml
datasets:
  - confit/audioset
metrics:
  - f1
  - precision
  - recall
  - auc
pipeline_tag: audio-classification
---

# yamnet-cry-distill-int8

A tiny INT8 TFLite student distilled from Google's YAMNet for on-device baby-cry detection on the Espressif ESP32-S3.

**Tag:** `v0.1.0`. Source pipeline: [chayuto/yamnet-cry-distill-int8](https://github.com/chayuto/yamnet-cry-distill-int8). Sibling firmware: [chayuto/ws-ESP32-S3-CAM](https://github.com/chayuto/ws-ESP32-S3-CAM).

## Model description

| | |
|---|---|
| Architecture | DS-CNN (Conv + 4 depthwise-separable blocks + GAP + Dense-521) |
| Parameters | 80,713 |
| INT8 size | **110 KB** (`.tflite`) |
| Teacher | [google/yamnet/1](https://tfhub.dev/google/yamnet/1) (FP32, 521-class AudioSet) |
| Distillation loss | KL(teacher_softmax \|\| student_softmax) over the full 521-class space |
| Input | 16 kHz mono PCM → 0.96 s log-mel patch (96 frames × 64 mels), int8 |
| Output | 521 raw logits, int8. Cry score = softmax(logits)[19] + softmax(logits)[20] |
| Quantization | Full-integer TFLite, AudioSet-only calibration (200 patches from `audioset_val.csv`) |
| Target hardware | ESP32-S3 (Xtensa LX7 dual-core, 8 MB PSRAM); runs equally on TFLite-Micro generic |

## Performance

**Headline: 62-segment AudioSet test slice** (frozen at curation, 38 % YouTube takedown attrition since release). 18 cry / 44 negative.

Cry-positive score = student softmax probability mass on YAMNet classes 19 (`Crying, sobbing`) + 20 (`Baby cry, infant cry`), averaged over all 0.96 s patches in the 10 s segment.

| | best F1 | best threshold | precision | recall | AUC | F1 @ 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| FP32 student (EXP-006 source) | 0.696 | 0.096 | 0.552 | 0.944 | 0.860 | 0.286 |
| **INT8 quantized (this artifact)** | **0.696** | **0.096** | **0.552** | **0.944** | **0.860** | **0.286** |
| Δ from quantization | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

INT8 quantization cost was **zero** at the reported precision — the 80 K student is small enough that representative-dataset calibration captures the activation distributions exactly. The model retains 94 % cry recall at its calibrated threshold.

**Pipeline-evolution context.** Earlier baselines used naïve random sub-window sampling (4 random 0.96 s patches per 40 s capture). EXP-006 introduces *teacher-as-filter* — running YAMNet across every 0.4875 s-hop window of every clip first, then training only on patches the teacher confidently scores positive (`p_cry > 0.30`) or confidently scores negative (`p_cry < 0.05`). See [`docs/research/methodology-teacher-as-filter.md`](https://github.com/chayuto/yamnet-cry-distill-int8/blob/main/docs/research/methodology-teacher-as-filter.md). Result: +0.037 AUC and 3× sharper threshold over the random-window baseline (EXP-004 → EXP-006).

**Threshold note:** distilled students inherit the teacher's diffuse softmax — cry probabilities concentrate below 0.5 even on obvious cries — so the operating-point threshold (~0.10) is well below the naïve 0.5. Deployments should calibrate per-device with a short on-site recording session.

## Training data

**Public surface (the headline number stands on this alone):**
- AudioSet v1 segment IDs committed at [`data/ids/audioset_*.csv`](https://github.com/chayuto/yamnet-cry-distill-int8/tree/main/data/ids):
  - 570 segments train + 160 val + 100 test (frozen)
  - Classes: `Crying, sobbing` (`/m/0463cq4`), `Baby cry, infant cry` (`/t/dd00002`), plus hard-negative classes `Speech` (`/m/09x0r`), `Babbling` (`/m/0261r1`), `Silence` (`/m/028v0c`), `Inside, small room` (`/t/dd00125`), `Wail, moan` (`/m/07qw_06`), `Screaming` (`/m/03qc9zr`)
  - Sourced from `eval_segments.csv` (test) and `balanced_train_segments.csv` (train+val); reproducible bit-exactly via `scripts/curate_audioset_ids.py`
  - 28.7 % of segments are unrecoverable from YouTube as of distillation; survival rate is recorded in the eval JSONs

**Private (augmentation only, never published):**
- 475 in-domain captures from a single Australian household ESP32-S3 deployment, used unsupervised (teacher-logits matching only — no human labels). Captures stay local; nothing capture-derived enters this artifact except via the teacher's intermediate logits during training.
- Independent pipeline-trained-without-captures baseline (EXP-003): AUC 0.750, best F1 0.612.
- The headline number above (EXP-006) reflects the captures-augmented + teacher-filtered model; the captures-free baseline is ~0.11 AUC weaker.

**Calibration set provenance:** The INT8 representative dataset (200 patches) was drawn exclusively from `audioset_val.csv` — never from training data, never from captures, never from the test set. This isolates the published quantization parameters from any capture-derived information.

## Intended use

Real-time inference on the [`ws-ESP32-S3-CAM` cry-detect-01 firmware](https://github.com/chayuto/ws-ESP32-S3-CAM). Pulled into device flash via that repo's `tools/fetch_model.sh`. Should run equally on any TFLite-Micro target with sufficient flash.

## Limitations

- **Per-deployment threshold calibration is required.** The naive 0.5 threshold yields F1 = 0.19. The right operating point is in the 0.03–0.05 range and shifts with sensor gain, room acoustics, and ambient noise floor.
- **Single-household training.** Captures come from one deployment site; generalization to wildly different nursery acoustics (multiple infants, hardwood-heavy reverb, outdoor) is not characterized in this release.
- **96 % speech contamination in captures.** Caregiver speech is co-present with cry in nearly all positive captures from the augmentation pool. AudioSet supplies clean cries to mitigate, but the model has not been benchmarked in purely speech-free environments.
- **Test set decay.** AudioSet test segments are referenced by YouTube ID — 38 % were unreachable at quantization time (May 2026). Re-running the eval against today's YouTube state may produce slightly different numbers as further takedowns occur.
- **INT8 rounds extremes.** Whisper-quiet cries (RMS < 200) may register zero probability mass even when the FP32 teacher detects them.

## Citation

```bibtex
@misc{yamnet-cry-distill-int8,
  author = {Orapinpatipat, Chayut},
  title = {yamnet-cry-distill-int8: a distilled INT8 baby-cry detector for ESP32-S3},
  year = {2026},
  howpublished = {\url{https://huggingface.co/chayuto/yamnet-cry-distill-int8}},
}
```

## License

MIT. See [LICENSE](https://github.com/chayuto/yamnet-cry-distill-int8/blob/main/LICENSE) in the source repo.

## Related

- Teacher artifact: [`chayuto/yamnet-mel-int8-tflm`](https://huggingface.co/chayuto/yamnet-mel-int8-tflm) — INT8 YAMNet with documented mel-magnitude correction, also for ESP32-S3.
- Source repo: [`chayuto/yamnet-cry-distill-int8`](https://github.com/chayuto/yamnet-cry-distill-int8).
- Sibling firmware: [`chayuto/ws-ESP32-S3-CAM`](https://github.com/chayuto/ws-ESP32-S3-CAM).
