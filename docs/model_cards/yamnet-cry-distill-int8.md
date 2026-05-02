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
  - int8
  - tflite-micro
datasets:
  - confit/audioset
metrics:
  - f1
  - precision
  - recall
pipeline_tag: audio-classification
---

# yamnet-cry-distill-int8

A tiny INT8 TFLite student model distilled from Google's YAMNet for on-device baby-cry detection on the Espressif ESP32-S3.

> **Status:** placeholder model card — performance numbers populated upon first published release. See [GitHub repo](https://github.com/chayuto/yamnet-cry-distill-int8) for the training pipeline and [`docs/experiments/`](https://github.com/chayuto/yamnet-cry-distill-int8/tree/main/docs/experiments) for the run log.

## Model description

| | |
|---|---|
| **Architecture** | TBD (DS-CNN or CRNN, ≤500 KB INT8) |
| **Teacher** | [google/yamnet/1](https://tfhub.dev/google/yamnet/1) (FP32, 521-class AudioSet) |
| **Distillation loss** | KL divergence on cry-related class logits |
| **Input** | 16 kHz mono PCM, 0.96 s frames → 96-mel log-spectrogram |
| **Output** | Per-frame logit / probability for `crying_sobbing` |
| **Quantization** | INT8 TFLite, representative-dataset calibration on AudioSet held-out subset |
| **Target hardware** | ESP32-S3 (Xtensa LX7 dual-core, 8 MB PSRAM) |

## Performance

Evaluated on a frozen held-out AudioSet `crying_sobbing` test split.

| Version | Params | INT8 size | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| (placeholder — populated at v0.1.0 release) | — | — | — | — | — |

Side metric (private in-domain home captures, time-stratified hold-out): reported during iteration, not in the public artifact.

## Training data

**Public:**
- AudioSet segment IDs committed at [`data/ids/audioset_*.csv`](https://github.com/chayuto/yamnet-cry-distill-int8/tree/main/data/ids). Segment IDs include `crying_sobbing`, `baby_cry_infant`, plus hard-negative classes (`speech`, `child_speech`, `babbling`, `screaming`).

**Private (augmentation only):**
- ~475 in-domain captures from a single Australian household ESP32-S3 deployment, used unsupervised (teacher-logits matching only — no human labels). The captures themselves are not published in any form. The headline performance number stands on the public AudioSet eval alone.

## Intended use

Real-time inference on ESP32-S3 baby-monitor hardware. Designed for the [`ws-ESP32-S3-CAM` cry-detect-01 firmware](https://github.com/chayuto/ws-ESP32-S3-CAM) and pulled into device flash via that repo's `tools/fetch_model.sh`.

## Limitations

- Trained augmentation data comes from a single household; performance in dramatically different acoustic environments (e.g. nursery with multiple infants, outdoor) is not characterised.
- Caregiver speech overlaps with cry in ~96% of in-domain training audio. The model has been exposed to clean cries via AudioSet to mitigate, but performance in a *purely silent* environment with isolated cry audio may not match in-domain numbers.
- INT8 quantization rounds extremes; whisper-quiet cries (RMS < 200) may be missed even when YAMNet teacher detects them.

## Citation

If you reference this work:

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
