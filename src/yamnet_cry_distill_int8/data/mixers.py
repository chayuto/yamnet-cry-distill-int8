"""Patch sources for distillation training.

Two providers:

- `patches_from_audioset(csv_path, cache_dir, ...)` — reads cached
  AudioSet segments (downloaded by `scripts/download_audioset.py`),
  samples N random 0.975 s windows per segment. Skips synthetic
  `_smoke_*` rows and segments that are missing or marked dead.
- `patches_from_captures(captures, ...)` — wraps
  `home_captures.load_random_patch` for the deployed-device WAVs.

Both return plain `list[np.ndarray]`, each entry shaped (15 600,)
float32 mono at 16 kHz. The training loop runs YAMNet on each patch
once at start (`_build_teacher_cache`) and reuses the (mel, probs)
pair across epochs — no labels, no audio re-reads.

`build_patch_pool(config, captures=None)` is the top-level entry the
training loop calls — config-driven, returns the pool ready for the
teacher pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from ..teacher import PATCH_SAMPLES, SAMPLE_RATE
from .audioset import DEFAULT_CACHE_DIR, read_segments_csv
from .home_captures import Capture, load_centered_patch, load_random_patch


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
