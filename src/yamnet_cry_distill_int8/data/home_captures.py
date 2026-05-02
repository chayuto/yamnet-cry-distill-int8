"""Private home-capture loader (deployed-device WAVs).

Reads `$WS_ESP32_S3_CAM_ROOT/projects/cry-detect-01/logs/canonical/wavs/`,
the deduplicated set of 40 s clips harvested from the deployed cry
detector. **No label files are read here** — distillation supervises
on the YAMNet teacher's own logits, so the only thing this loader
returns is (waveform, ts_iso, source_path) tuples. That's the
contract for keeping the training pipeline label-free.

The default root is `../ws-ESP32-S3-CAM` relative to this repo, with
override via `$WS_ESP32_S3_CAM_ROOT`. If neither resolves, the loader
returns an empty list — Phase 2 then errors out cleanly with a usage
message rather than silently training on nothing.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from ..teacher import PATCH_SAMPLES, SAMPLE_RATE

CANONICAL_REL = Path("projects/cry-detect-01/logs/canonical/wavs")
TS_RE = re.compile(r"cry-(\d{8})T(\d{6})([+-]\d{4})")


@dataclass(frozen=True)
class Capture:
    path: Path
    ts_iso: str

    @property
    def hour(self) -> int:
        return int(self.ts_iso[11:13])

    @property
    def date(self) -> str:
        return self.ts_iso[:10]


def resolve_root() -> Path | None:
    env = os.environ.get("WS_ESP32_S3_CAM_ROOT")
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path(__file__).resolve().parents[4] / "ws-ESP32-S3-CAM")
    for c in candidates:
        if (c / CANONICAL_REL).is_dir():
            return c
    return None


def discover_captures(root: Path | None = None) -> list[Capture]:
    if root is None:
        root = resolve_root()
    if root is None:
        return []
    wavs_dir = root / CANONICAL_REL
    out: list[Capture] = []
    for wav_path in sorted(wavs_dir.glob("cry-*.wav")):
        m = TS_RE.match(wav_path.name)
        if not m:
            continue
        date_s, time_s, tz = m.groups()
        ts_iso = (
            f"{date_s[:4]}-{date_s[4:6]}-{date_s[6:8]}T"
            f"{time_s[:2]}:{time_s[2:4]}:{time_s[4:6]}{tz[:3]}:{tz[3:]}"
        )
        out.append(Capture(path=wav_path, ts_iso=ts_iso))
    return out


def time_stratified_split(
    captures: list[Capture],
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[list[Capture], list[Capture]]:
    """Hold out one capture per (date, hour) bucket until ~val_frac is met.

    This stratifies across the time-of-day confound noted in the data
    audit: 19h is over-represented for cry, dawn buckets are mostly
    silence. Random splitting would let one tier dominate val.
    """
    rng = np.random.default_rng(seed)
    by_bucket: dict[tuple[str, int], list[Capture]] = {}
    for c in captures:
        by_bucket.setdefault((c.date, c.hour), []).append(c)

    target = max(1, int(round(len(captures) * val_frac)))
    val: list[Capture] = []
    bucket_keys = list(by_bucket.keys())
    rng.shuffle(bucket_keys)

    while len(val) < target and any(by_bucket[k] for k in bucket_keys):
        for k in bucket_keys:
            if not by_bucket[k]:
                continue
            idx = int(rng.integers(0, len(by_bucket[k])))
            val.append(by_bucket[k].pop(idx))
            if len(val) >= target:
                break

    val_ids = {c.path for c in val}
    train = [c for c in captures if c.path not in val_ids]
    return train, val


def load_random_patch(capture: Capture, rng: np.random.Generator) -> np.ndarray:
    """Sample one PATCH_SAMPLES-long window uniformly from the clip."""
    info = sf.info(str(capture.path))
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(f"{capture.path}: sr={info.samplerate} != {SAMPLE_RATE}")
    if info.frames < PATCH_SAMPLES:
        raise ValueError(f"{capture.path}: only {info.frames} samples, need {PATCH_SAMPLES}")
    start = int(rng.integers(0, info.frames - PATCH_SAMPLES + 1))
    audio, _ = sf.read(
        str(capture.path),
        start=start,
        frames=PATCH_SAMPLES,
        dtype="float32",
        always_2d=False,
    )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False)


def load_centered_patch(capture: Capture) -> np.ndarray:
    """Deterministic centered patch — used for held-out eval."""
    info = sf.info(str(capture.path))
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(f"{capture.path}: sr={info.samplerate} != {SAMPLE_RATE}")
    start = max(0, (info.frames - PATCH_SAMPLES) // 2)
    audio, _ = sf.read(
        str(capture.path),
        start=start,
        frames=PATCH_SAMPLES,
        dtype="float32",
        always_2d=False,
    )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False)
