"""yamnet-cry-distill-int8 — knowledge distillation pipeline.

Distills Google's pretrained YAMNet (FP32, 521-class AudioSet) into a
tiny INT8 student classifier targeted at on-device baby-cry detection
on the ESP32-S3.

See README.md and docs/architecture.md.
"""

__version__ = "0.1.0"
