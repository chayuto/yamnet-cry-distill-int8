"""Depthwise-separable CNN student.

Input  — log-mel patch of shape (PATCH_FRAMES=96, MEL_BINS=64, 1).
Output — 521 raw logits, matching YAMNet's class space.

Sized to land ≤100 K parameters (≤500 KB INT8 with 4× shrink budget).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from ..teacher import MEL_BINS, NUM_CLASSES, PATCH_FRAMES


def _ds_block(x, filters: int, stride: int, name: str):
    x = layers.DepthwiseConv2D(
        kernel_size=3, strides=stride, padding="same", use_bias=False,
        name=f"{name}_dw",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)
    x = layers.Conv2D(
        filters, kernel_size=1, padding="same", use_bias=False,
        name=f"{name}_pw",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = layers.ReLU(name=f"{name}_pw_relu")(x)
    return x


def build_student(
    num_classes: int = NUM_CLASSES,
    input_shape=(PATCH_FRAMES, MEL_BINS, 1),
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="log_mel_patch")
    x = layers.Conv2D(8, 3, strides=2, padding="same", use_bias=False, name="stem")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = _ds_block(x, 16, stride=1, name="ds1")
    x = _ds_block(x, 32, stride=2, name="ds2")
    x = _ds_block(x, 64, stride=2, name="ds3")
    x = _ds_block(x, 128, stride=2, name="ds4")
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    logits = layers.Dense(num_classes, name="logits")(x)
    return models.Model(inputs=inputs, outputs=logits, name="dscnn_student")
