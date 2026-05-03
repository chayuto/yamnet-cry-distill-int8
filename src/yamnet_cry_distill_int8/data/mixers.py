"""Patch sources for distillation training.

Three providers, each returning patches ready for the teacher cache:

- `patches_from_audioset(csv_path, cache_dir, ...)` — reads cached
  AudioSet segments, samples N random 0.975 s windows per segment.
  Skips synthetic `_smoke_*` rows and segments that are missing or
  marked dead.
- `patches_from_captures(captures, ...)` — wraps
  `home_captures.load_random_patch` for the deployed-device WAVs.
- `patches_filtered_by_teacher(teacher, captures, audioset_csv, ...)`
  — runs the teacher across full clips with sliding windows, bins
  windows by `p_cry = softmax[19] + softmax[20]`, returns separate
  positive (`p_cry > pos_thr`) and negative (`p_cry < neg_thr`)
  pools. The clip-level labels are not used; the teacher's per-window
  confidence is the only signal. See
  `docs/research/methodology-teacher-as-filter.md` for the rationale.

The first two return raw waveforms (the teacher pass happens later in
`train.py:_teacher_cache`). The third returns pre-cached
`(mel, teacher_probs)` pairs because it had to run the teacher anyway
to score the windows — saving a second teacher pass.

`build_patch_pool(config, captures=None)` is the top-level entry the
training loop calls — config-driven, returns the pool ready for the
teacher pass (or pre-cached patches in the filtered case).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

from ..teacher import PATCH_FRAMES, PATCH_SAMPLES, SAMPLE_RATE
from .audioset import DEFAULT_CACHE_DIR, read_segments_csv
from .home_captures import Capture, load_centered_patch, load_random_patch

CRY_CLASS_IDXS = (19, 20)  # YAMNet: Crying-sobbing + Baby-cry, infant-cry
HOP_SAMPLES = PATCH_SAMPLES // 2  # 0.4875 s, matches eval / YAMNet patch grid


@dataclass
class CachedPatch:
    """A teacher-cached training sample (mel patch + 521-class softmax).

    Re-defined here (not imported from train.py) so the data layer can
    return cache-ready outputs without circular import.
    """
    mel: np.ndarray
    teacher_probs: np.ndarray


def _fit_or_pad(audio: np.ndarray, n: int) -> np.ndarray:
    if len(audio) >= n:
        return audio[:n].astype(np.float32, copy=False)
    out = np.zeros(n, dtype=np.float32)
    out[: len(audio)] = audio
    return out


def patches_from_audioset(
    csv_path: str | Path,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    patches_per_seg: int = 1,
    seed: int = 0,
    deterministic: bool = False,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = read_segments_csv(csv_path)
    cache_dir = Path(cache_dir)
    out: list[np.ndarray] = []
    skipped_dead = skipped_missing = skipped_smoke = 0

    for r in rows:
        if r["ytid"].startswith("_smoke_"):
            skipped_smoke += 1
            continue
        wav_path = cache_dir / f"{r['ytid']}_{r['start_s']:.1f}.wav"
        if wav_path.with_suffix(".dead").exists():
            skipped_dead += 1
            continue
        if not wav_path.exists() or wav_path.stat().st_size == 0:
            skipped_missing += 1
            continue

        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if sr != SAMPLE_RATE:
            skipped_missing += 1
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        for _ in range(patches_per_seg):
            if deterministic or len(audio) <= PATCH_SAMPLES:
                start = max(0, (len(audio) - PATCH_SAMPLES) // 2)
            else:
                start = int(rng.integers(0, len(audio) - PATCH_SAMPLES + 1))
            patch = audio[start : start + PATCH_SAMPLES]
            out.append(_fit_or_pad(patch, PATCH_SAMPLES))

    if skipped_dead or skipped_missing or skipped_smoke:
        print(
            f"  [audioset] {len(out)} patches; "
            f"skipped: smoke={skipped_smoke} missing={skipped_missing} dead={skipped_dead}"
        )
    return out


def patches_from_captures(
    captures: list[Capture],
    patches_per_clip: int = 4,
    seed: int = 0,
    deterministic: bool = False,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for cap in captures:
        if deterministic:
            out.append(load_centered_patch(cap))
        else:
            for _ in range(patches_per_clip):
                out.append(load_random_patch(cap, rng))
    return out


def _load_full_clip(path: Path) -> np.ndarray | None:
    info = sf.info(str(path))
    if info.samplerate != SAMPLE_RATE:
        return None
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False)


def _slide_windows(audio: np.ndarray) -> list[np.ndarray]:
    if len(audio) < PATCH_SAMPLES:
        out = np.zeros(PATCH_SAMPLES, dtype=np.float32)
        out[: len(audio)] = audio
        return [out]
    starts = list(range(0, len(audio) - PATCH_SAMPLES + 1, HOP_SAMPLES))
    return [audio[s : s + PATCH_SAMPLES] for s in starts]


def _teacher_pass_one(teacher, wav: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Run teacher on a single 0.975 s window. Returns (mel, probs, p_cry)."""
    scores, _, log_mel = teacher.forward(tf.constant(wav))
    if log_mel.shape[0] < PATCH_FRAMES:
        log_mel = tf.pad(log_mel, [[0, PATCH_FRAMES - log_mel.shape[0]], [0, 0]])
    mel = log_mel[:PATCH_FRAMES, :].numpy()
    probs = tf.reduce_mean(scores, axis=0).numpy()
    p_cry = float(probs[CRY_CLASS_IDXS[0]] + probs[CRY_CLASS_IDXS[1]])
    return mel, probs, p_cry


def _filter_clip(
    teacher,
    audio: np.ndarray,
    pos_thr: float,
    neg_thr: float,
) -> tuple[list[CachedPatch], list[CachedPatch]]:
    pos: list[CachedPatch] = []
    neg: list[CachedPatch] = []
    for wav in _slide_windows(audio):
        mel, probs, p_cry = _teacher_pass_one(teacher, wav)
        if p_cry > pos_thr:
            pos.append(CachedPatch(mel=mel, teacher_probs=probs))
        elif p_cry < neg_thr:
            neg.append(CachedPatch(mel=mel, teacher_probs=probs))
    return pos, neg


def patches_filtered_by_teacher(
    teacher,
    captures: list[Capture] | None,
    audioset_csv: Path | None,
    audioset_cache: Path = DEFAULT_CACHE_DIR,
    pos_thr: float = 0.30,
    neg_thr: float = 0.05,
    balance: bool = True,
    seed: int = 0,
    label: str = "train",
) -> list[CachedPatch]:
    """Run the teacher across full clips with sliding windows, bin into
    positive/negative pools by cry-score, balance, and return the
    combined pool ready for distillation training.

    Clip-level labels are not consulted — the teacher's per-window
    `p_cry = softmax(scores)[19] + softmax(scores)[20]` is the sole
    selection signal. See
    `docs/research/methodology-teacher-as-filter.md`.
    """
    pos_all: list[CachedPatch] = []
    neg_all: list[CachedPatch] = []
    n_caps_clip = n_caps_pos = n_caps_neg = 0
    n_audio_clip = n_audio_pos = n_audio_neg = 0
    t0 = time.time()

    if captures:
        for i, cap in enumerate(captures):
            audio = _load_full_clip(cap.path)
            if audio is None:
                continue
            pos, neg = _filter_clip(teacher, audio, pos_thr, neg_thr)
            pos_all.extend(pos)
            neg_all.extend(neg)
            n_caps_clip += 1
            n_caps_pos += len(pos)
            n_caps_neg += len(neg)
            if (i + 1) % 50 == 0:
                print(
                    f"  [filter caps {label}] {i + 1}/{len(captures)} "
                    f"pos={n_caps_pos} neg={n_caps_neg} ({time.time() - t0:.1f}s)"
                )
        print(
            f"  [filter caps {label}] {n_caps_clip} clips → "
            f"pos={n_caps_pos} neg={n_caps_neg}"
        )

    if audioset_csv is not None:
        rows = read_segments_csv(audioset_csv)
        for i, r in enumerate(rows):
            if r["ytid"].startswith("_smoke_"):
                continue
            wav_path = audioset_cache / f"{r['ytid']}_{r['start_s']:.1f}.wav"
            if (
                wav_path.with_suffix(".dead").exists()
                or not wav_path.exists()
                or wav_path.stat().st_size == 0
            ):
                continue
            audio = _load_full_clip(wav_path)
            if audio is None:
                continue
            pos, neg = _filter_clip(teacher, audio, pos_thr, neg_thr)
            pos_all.extend(pos)
            neg_all.extend(neg)
            n_audio_clip += 1
            n_audio_pos += len(pos)
            n_audio_neg += len(neg)
            if (i + 1) % 100 == 0:
                print(
                    f"  [filter audio {label}] {i + 1}/{len(rows)} "
                    f"pos={n_audio_pos} neg={n_audio_neg} ({time.time() - t0:.1f}s)"
                )
        print(
            f"  [filter audio {label}] {n_audio_clip} clips → "
            f"pos={n_audio_pos} neg={n_audio_neg}"
        )

    print(
        f"  [filter total {label}] pos={len(pos_all)} neg={len(neg_all)}"
    )

    if balance and pos_all and neg_all:
        rng = np.random.default_rng(seed)
        n = min(len(pos_all), len(neg_all))
        if len(pos_all) > n:
            idx = rng.permutation(len(pos_all))[:n]
            pos_all = [pos_all[j] for j in idx]
        if len(neg_all) > n:
            idx = rng.permutation(len(neg_all))[:n]
            neg_all = [neg_all[j] for j in idx]
        print(f"  [filter balanced {label}] {n} pos / {n} neg = {2 * n} total")

    return pos_all + neg_all


def build_patch_pool(
    cfg_data: dict,
    captures: list[Capture] | None = None,
    seed: int = 0,
    role: str = "train",
) -> list[np.ndarray]:
    """Read `config['data']` (and optionally a captures list) and build
    the patch pool for either the train or val role.

    Recognised `cfg_data['source']` values:
        "captures"   — home captures only (Phase 2)
        "audioset"   — committed AudioSet IDs only (Phase 3)
        "mixed"      — both, concatenated
    """
    source = cfg_data["source"]
    if source not in ("captures", "audioset", "mixed"):
        raise ValueError(f"unknown data.source: {source}")

    is_val = role == "val"
    out: list[np.ndarray] = []

    if source in ("captures", "mixed"):
        if captures is None:
            raise ValueError("captures source requested but none discovered")
        ppclip = 1 if is_val else int(cfg_data.get("patches_per_clip", 4))
        out += patches_from_captures(
            captures,
            patches_per_clip=ppclip,
            seed=seed,
            deterministic=is_val,
        )

    if source in ("audioset", "mixed"):
        ids_csv = Path(cfg_data[f"audioset_{role}_csv"])
        ppseg = 1 if is_val else int(cfg_data.get("audioset_patches_per_seg", 4))
        out += patches_from_audioset(
            ids_csv,
            patches_per_seg=ppseg,
            seed=seed,
            deterministic=is_val,
        )

    return out
