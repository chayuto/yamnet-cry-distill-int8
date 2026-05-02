"""YAMNet shape contract.

Skipped by default because loading YAMNet from TF Hub is slow on a cold
cache. The smoke runner (`scripts/run_exp001_smoke.sh`) exercises this
path end-to-end — this test exists to pin shape expectations once the
hub model is cached locally.

Run with `RUN_TEACHER_TEST=1 pytest tests/test_teacher_shapes.py`.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import tensorflow as tf

from yamnet_cry_distill_int8.teacher import (
    MEL_BINS,
    NUM_CLASSES,
    PATCH_SAMPLES,
    YAMNetTeacher,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_TEACHER_TEST") != "1",
    reason="set RUN_TEACHER_TEST=1 to load YAMNet (network on cold cache)",
)


def test_yamnet_outputs_have_expected_shapes():
    teacher = YAMNetTeacher()
    waveform = tf.constant(np.zeros(PATCH_SAMPLES, dtype=np.float32))
    scores, embeddings, log_mel = teacher.forward(waveform)
    assert scores.shape[-1] == NUM_CLASSES
    assert embeddings.shape[-1] == 1024
    assert log_mel.shape[-1] == MEL_BINS
    assert scores.shape[0] == embeddings.shape[0]
