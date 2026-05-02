"""Distillation training entry point.

Phase 1 implements only `--smoke`: a single-batch sanity step that
proves the loop closes end-to-end (CSV → loader → teacher → student
→ KL loss → optimizer step) with no NaN. Real multi-epoch training
arrives in Phase 2 (`exp002_captures_only`).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import tensorflow as tf

from .data.audioset import load_batch
from .student.dscnn import build_student
from .teacher import PATCH_FRAMES, YAMNetTeacher


def _teacher_pass(teacher: YAMNetTeacher, waveforms: tf.Tensor):
    """YAMNet expects a 1-D waveform per call. Run it per clip and
    stack the results into a batch."""
    teacher_probs, mel_patches = [], []
    for i in range(waveforms.shape[0]):
        scores, _, log_mel = teacher.forward(waveforms[i])
        teacher_probs.append(tf.reduce_mean(scores, axis=0))
        if log_mel.shape[0] < PATCH_FRAMES:
            pad = PATCH_FRAMES - log_mel.shape[0]
            log_mel = tf.pad(log_mel, [[0, pad], [0, 0]])
        mel_patches.append(log_mel[:PATCH_FRAMES, :])
    teacher_probs = tf.stack(teacher_probs, axis=0)
    mel_batch = tf.expand_dims(tf.stack(mel_patches, axis=0), axis=-1)
    return teacher_probs, mel_batch


def smoke_step(
    teacher: YAMNetTeacher,
    student: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    waveforms: tf.Tensor,
) -> float:
    teacher_probs, mel_batch = _teacher_pass(teacher, waveforms)
    eps = 1e-8
    with tf.GradientTape() as tape:
        student_logits = student(mel_batch, training=True)
        student_probs = tf.nn.softmax(student_logits)
        kl_per_sample = tf.reduce_sum(
            teacher_probs
            * (tf.math.log(teacher_probs + eps) - tf.math.log(student_probs + eps)),
            axis=-1,
        )
        loss = tf.reduce_mean(kl_per_sample)
    grads = tape.gradient(loss, student.trainable_variables)
    optimizer.apply_gradients(zip(grads, student.trainable_variables))
    return float(loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("data/ids/audioset_smoke.csv"),
        help="Segment-ID CSV to read",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.smoke:
        raise SystemExit("Phase 1 only supports --smoke. Real training lands in Phase 2.")

    tf.keras.utils.set_random_seed(args.seed)

    t0 = time.time()
    print("[smoke] loading YAMNet teacher...")
    teacher = YAMNetTeacher()
    print(f"[smoke] teacher loaded in {time.time() - t0:.1f}s")

    student = build_student()
    n_params = student.count_params()
    print(f"[smoke] student parameter count: {n_params}")
    if n_params > 100_000:
        raise SystemExit(f"Student over budget: {n_params} > 100K params")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    print(f"[smoke] loading {args.batch} clips from {args.ids}...")
    waveforms, ids = load_batch(args.ids, n=args.batch)
    print(f"[smoke] batch ids: {ids}")

    print("[smoke] running 1 distillation step...")
    t1 = time.time()
    loss = smoke_step(teacher, student, optimizer, waveforms)
    print(f"[smoke] step done in {time.time() - t1:.1f}s, KL loss = {loss:.4f}")

    if math.isnan(loss) or math.isinf(loss):
        raise SystemExit(f"KL loss was {loss} — loop did not close cleanly")
    print(f"[smoke] OK — loop closed end-to-end in {time.time() - t0:.1f}s total.")


if __name__ == "__main__":
    main()
