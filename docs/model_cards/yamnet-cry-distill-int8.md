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

**Tag:** `v0.2.0`. Source pipeline: [chayuto/yamnet-cry-distill-int8](https://github.com/chayuto/yamnet-cry-distill-int8). Sibling firmware: [chayuto/ws-ESP32-S3-CAM](https://github.com/chayuto/ws-ESP32-S3-CAM).

**v0.2.0 changes vs v0.1.0:** trained at 1:3 pos:neg ratio with per-epoch random negative resampling and a learning-rate drop at epoch 30. Same student architecture, same INT8 size. Best-F1 lifted from 0.696 to 0.756 and FPR at the operating threshold dropped from 31.8 % to 22.7 % — material improvement in deployment-relevant terms. See [`docs/experiments/EXP-007_per_epoch_resample.md`](https://github.com/chayuto/yamnet-cry-distill-int8/blob/main/docs/experiments/EXP-007_per_epoch_resample.md) and [`docs/experiments/EXP-008_lr_schedule.md`](https://github.com/chayuto/yamnet-cry-distill-int8/blob/main/docs/experiments/EXP-008_lr_schedule.md).

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
| FP32 student (EXP-008 source) | 0.756 | 0.047 | 0.630 | 0.944 | 0.870 | 0.286 |
| **INT8 quantized (this artifact)** | **0.744** | **0.047** | — | — | **0.866** | **0.286** |
| Δ from quantization | -0.012 | 0.000 | — | — | -0.004 | 0.000 |

INT8 quantization cost was small — 0.012 best-F1 and 0.004 AUC, well under the ≤0.02 AUC budget. The model retains 94 % cry recall at its calibrated threshold.

At the best-F1 threshold (≈0.05), the FP32 model lands **17 of 18 cries caught (94 % recall)** with **63 % precision** on a balanced 18-pos / 44-neg AudioSet test slice. False-positive rate at this operating point is **22.7 %** vs v0.1.0's 31.8 % — a 9-percentage-point reduction (-29 % relative) at the same recall.

**Pipeline-evolution context — three methodology shifts that earned this artifact:**

1. **EXP-006 — teacher-as-filter** (+0.037 AUC over random-window baseline). YAMNet pre-scores every 0.4875 s-hop window of every clip; only confident-positive (`p_cry > 0.30`) and confident-negative (`p_cry < 0.05`) windows enter training. See [`docs/research/methodology-teacher-as-filter.md`](https://github.com/chayuto/yamnet-cry-distill-int8/blob/main/docs/research/methodology-teacher-as-filter.md).
2. **EXP-007 — 1:3 pos:neg ratio + per-epoch negative resampling** (+0.060 best-F1, -9 pp FPR at operating threshold). Training at deployment-realistic class ratios pushed the operating point materially without sacrificing AUC.
3. **EXP-008 — LR drop at epoch 30** (+0.009 AUC). Fine-tuning at 10× lower learning rate after the loss plateau closed the train/val gap modestly.

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
