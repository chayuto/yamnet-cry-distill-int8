"""YAMNet teacher wrapper.

Loads Google's pretrained YAMNet (FP32, 521-class AudioSet) from TF Hub.
The model is cached in `~/.cache/yamnet_teacher/` (or `$TFHUB_CACHE_DIR`)
on first load and reused on subsequent loads.

Output of `forward(waveform)`:
    scores      — (n_patches, 521) per-patch softmax probabilities
    embeddings  — (n_patches, 1024) penultimate embeddings
    log_mel     — (n_total_frames, 64) full-clip log-mel spectrogram

A YAMNet "patch" is a 0.96 s analysis window with 0.48 s hop. A 0.975 s
clip yields exactly one patch (`PATCH_SAMPLES = 15600`).
"""

from __future__ import annotations

import os
from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub

YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16000
PATCH_SAMPLES = 15600  # 0.975 s — one YAMNet analysis window
NUM_CLASSES = 521
MEL_BINS = 64
PATCH_FRAMES = 96  # log-mel time frames per 0.96 s patch


class YAMNetTeacher:
    def __init__(self, cache_dir: str | None = None):
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/yamnet_teacher")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TFHUB_CACHE_DIR", cache_dir)
        self.model = hub.load(YAMNET_HUB_URL)

    def forward(self, waveform_1d: tf.Tensor):
        wav = tf.convert_to_tensor(waveform_1d, dtype=tf.float32)
        if wav.shape.rank != 1:
            raise ValueError(
                f"YAMNet expects a 1-D waveform, got shape {wav.shape.as_list()}"
            )
        return self.model(wav)

    def clip_probs(self, waveform_1d: tf.Tensor) -> tf.Tensor:
        scores, _, _ = self.forward(waveform_1d)
        return tf.reduce_mean(scores, axis=0)

    def patch_mel(self, waveform_1d: tf.Tensor) -> tf.Tensor:
        _, _, log_mel = self.forward(waveform_1d)
        if log_mel.shape[0] < PATCH_FRAMES:
            pad = PATCH_FRAMES - log_mel.shape[0]
            log_mel = tf.pad(log_mel, [[0, pad], [0, 0]])
        return log_mel[:PATCH_FRAMES, :]
