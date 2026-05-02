"""Distillation training entry point.

Two modes:

  python -m yamnet_cry_distill_int8.train --smoke
      EXP-001 plumbing test, 4 synthetic clips, one optimizer step.

  python -m yamnet_cry_distill_int8.train --config configs/exp002_captures_only.yaml
      Real multi-epoch distillation, captures-only, KL on YAMNet logits,
      time-stratified held-out eval. Saves best checkpoint to a
      gitignored path under models/.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml

from .data.audioset import load_batch
from .data.home_captures import (
    Capture,
    discover_captures,
    load_centered_patch,
    load_random_patch,
    time_stratified_split,
)
from .student.dscnn import build_student
from .teacher import PATCH_FRAMES, YAMNetTeacher


def _teacher_pass(teacher: YAMNetTeacher, waveforms: tf.Tensor):
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


def _kl(teacher_probs: tf.Tensor, student_logits: tf.Tensor, eps: float) -> tf.Tensor:
    student_probs = tf.nn.softmax(student_logits)
    return tf.reduce_sum(
        teacher_probs
        * (tf.math.log(teacher_probs + eps) - tf.math.log(student_probs + eps)),
        axis=-1,
    )


def smoke_step(
    teacher: YAMNetTeacher,
    student: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    waveforms: tf.Tensor,
) -> float:
    teacher_probs, mel_batch = _teacher_pass(teacher, waveforms)
    with tf.GradientTape() as tape:
        student_logits = student(mel_batch, training=True)
        loss = tf.reduce_mean(_kl(teacher_probs, student_logits, 1e-8))
    grads = tape.gradient(loss, student.trainable_variables)
    optimizer.apply_gradients(zip(grads, student.trainable_variables))
    return float(loss)


def run_smoke(args):
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


@dataclass
class CachedPatch:
    mel: np.ndarray
    teacher_probs: np.ndarray


def _build_cache(
    teacher: YAMNetTeacher,
    captures: list[Capture],
    patches_per_clip: int,
    seed: int,
    label: str,
    deterministic: bool = False,
) -> list[CachedPatch]:
    rng = np.random.default_rng(seed)
    cache: list[CachedPatch] = []
    t0 = time.time()
    for i, cap in enumerate(captures):
        if deterministic:
            wavs = [load_centered_patch(cap)]
        else:
            wavs = [load_random_patch(cap, rng) for _ in range(patches_per_clip)]
        for wav in wavs:
            scores, _, log_mel = teacher.forward(wav)
            if log_mel.shape[0] < PATCH_FRAMES:
                pad = PATCH_FRAMES - log_mel.shape[0]
                log_mel = tf.pad(log_mel, [[0, pad], [0, 0]])
            mel = log_mel[:PATCH_FRAMES, :].numpy()
            probs = tf.reduce_mean(scores, axis=0).numpy()
            cache.append(CachedPatch(mel=mel, teacher_probs=probs))
        if (i + 1) % 50 == 0 or i == len(captures) - 1:
            elapsed = time.time() - t0
            print(
                f"  [{label}] {i + 1}/{len(captures)} clips, "
                f"{len(cache)} patches, {elapsed:.1f}s"
            )
    return cache


def _eval_kl(student: tf.keras.Model, cache: list[CachedPatch], eps: float, batch: int = 64) -> float:
    losses = []
    for i in range(0, len(cache), batch):
        chunk = cache[i : i + batch]
        mel = np.stack([c.mel for c in chunk])[..., None]
        probs = np.stack([c.teacher_probs for c in chunk])
        logits = student(tf.constant(mel), training=False)
        kl = _kl(tf.constant(probs), logits, eps)
        losses.append(float(tf.reduce_sum(kl)))
    return sum(losses) / len(cache)


def _make_optimizer(name: str, lr: float, wd: float):
    name = name.lower()
    if name == "adamw" and hasattr(tf.keras.optimizers, "AdamW"):
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    if name == "adamw":
        print("[train] AdamW unavailable in this Keras build — falling back to Adam")
    return tf.keras.optimizers.Adam(learning_rate=lr)


def run_experiment(args):
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    print(f"[train] loaded config {cfg_path} ({cfg['experiment_id']})")

    tf.keras.utils.set_random_seed(cfg.get("data", {}).get("split_seed", 0))

    # ----- data -----
    captures = discover_captures()
    if not captures:
        raise SystemExit(
            "No captures found. Set WS_ESP32_S3_CAM_ROOT to point at the "
            "device-side repo, or place captures under "
            "../ws-ESP32-S3-CAM/projects/cry-detect-01/logs/canonical/wavs/."
        )
    train_caps, val_caps = time_stratified_split(
        captures,
        val_frac=cfg["data"]["val_frac"],
        seed=cfg["data"]["split_seed"],
    )
    print(f"[train] {len(captures)} captures → train {len(train_caps)} / val {len(val_caps)}")

    # ----- teacher + student -----
    print("[train] loading YAMNet teacher...")
    teacher = YAMNetTeacher()
    student = build_student()
    print(f"[train] student params: {student.count_params()}")
    optimizer = _make_optimizer(
        cfg["train"]["optimizer"],
        cfg["train"]["learning_rate"],
        cfg["train"].get("weight_decay", 0.0),
    )

    eps = float(cfg["loss"]["epsilon"])
    patches_per_clip = int(cfg["data"]["patches_per_clip"])
    epochs = int(cfg["train"]["epochs"])
    batch_size = int(cfg["train"]["batch_size"])

    # ----- precompute teacher outputs once (frozen teacher → cache is stable) -----
    print("[train] caching teacher outputs on train set...")
    train_cache = _build_cache(
        teacher,
        train_caps,
        patches_per_clip=patches_per_clip,
        seed=cfg["train"]["shuffle_seed"],
        label="train",
        deterministic=False,
    )
    print("[train] caching teacher outputs on val set (centered patches)...")
    val_cache = _build_cache(
        teacher,
        val_caps,
        patches_per_clip=1,
        seed=0,
        label="val",
        deterministic=True,
    )
    print(f"[train] cached: train={len(train_cache)} patches, val={len(val_cache)} patches")

    # ----- training loop -----
    history: list[dict] = []
    init_val = _eval_kl(student, val_cache, eps)
    print(f"[train] epoch  0 (init): val_kl_per_clip={init_val:.4f}")
    history.append({"epoch": 0, "train_kl": None, "val_kl": init_val})

    best_val = init_val
    best_epoch = 0
    out_path = Path(cfg["checkpoint"]["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["train"]["shuffle_seed"])

    eval_every = int(cfg["eval"].get("every_n_epochs", 5))

    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(train_cache))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i : i + batch_size]
            mel = np.stack([train_cache[j].mel for j in batch_idx])[..., None]
            probs = np.stack([train_cache[j].teacher_probs for j in batch_idx])
            with tf.GradientTape() as tape:
                logits = student(tf.constant(mel), training=True)
                loss = tf.reduce_mean(_kl(tf.constant(probs), logits, eps))
            grads = tape.gradient(loss, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))
            epoch_loss += float(loss)
            n_batches += 1

        train_kl = epoch_loss / max(1, n_batches)
        if epoch % eval_every == 0 or epoch == epochs:
            val_kl = _eval_kl(student, val_cache, eps)
            print(f"[train] epoch {epoch:>2}: train_kl={train_kl:.4f} val_kl={val_kl:.4f}")
            history.append({"epoch": epoch, "train_kl": train_kl, "val_kl": val_kl})
            if val_kl < best_val:
                best_val = val_kl
                best_epoch = epoch
                student.save(str(out_path))
                print(f"[train]   saved best to {out_path} (val_kl={val_kl:.4f})")
        else:
            print(f"[train] epoch {epoch:>2}: train_kl={train_kl:.4f}")
            history.append({"epoch": epoch, "train_kl": train_kl, "val_kl": None})

    # ----- summary -----
    print(
        f"[train] done. init_val_kl={init_val:.4f} best_val_kl={best_val:.4f} "
        f"@ epoch {best_epoch} → {out_path if best_epoch > 0 else '<no checkpoint saved>'}"
    )
    eval_log_path = Path("docs/experiments") / f"eval_home_captures_{cfg['experiment_id'].lower()}.json"
    eval_log_path.parent.mkdir(parents=True, exist_ok=True)
    eval_log_path.write_text(
        json.dumps(
            {
                "experiment_id": cfg["experiment_id"],
                "n_train_captures": len(train_caps),
                "n_val_captures": len(val_caps),
                "n_train_patches": len(train_cache),
                "n_val_patches": len(val_cache),
                "init_val_kl": init_val,
                "best_val_kl": best_val,
                "best_epoch": best_epoch,
                "history": history,
                "config_path": str(cfg_path),
            },
            indent=2,
        )
    )
    print(f"[train] eval log → {eval_log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("data/ids/audioset_smoke.csv"),
        help="Segment-ID CSV for --smoke",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, help="Path to experiment YAML")
    args = parser.parse_args()

    if args.smoke:
        run_smoke(args)
    elif args.config:
        run_experiment(args)
    else:
        raise SystemExit("Pass either --smoke or --config <path>")


if __name__ == "__main__":
    main()
