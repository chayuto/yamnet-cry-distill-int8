"""AudioSet segment loader.

Reads `data/ids/*.csv` (committed segment IDs) and yields 16-kHz mono
waveforms suitable for the YAMNet teacher. Real segments are pulled
from `data/audioset/cache/` (downloaded by `scripts/download_audioset.py`).

For the EXP-001 smoke test, ytids prefixed `_smoke_` route to a
deterministic synthetic generator instead — the test runs offline and
in CI without any network or yt-dlp dependency. Real AudioSet ytids
are exactly 11 characters and never lead with an underscore, so the
two paths cannot collide.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

from ..teacher import PATCH_SAMPLES, SAMPLE_RATE

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "audioset" / "cache"


def read_segments_csv(csv_path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        rows.append(
            {
                "ytid": row["ytid"].strip(),
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "positive_labels": row["positive_labels"].strip(),
            }
        )
    return rows


def load_segment(
    ytid: str,
    start_s: float,
    end_s: float,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    n_samples: int = PATCH_SAMPLES,
) -> np.ndarray:
    if ytid.startswith("_smoke_"):
        return _synthetic_waveform(ytid, n_samples)
    path = Path(cache_dir) / f"{ytid}_{start_s:.1f}.wav"
    if not path.exists():
        raise FileNotFoundError(
            f"Segment {ytid}@{start_s:.1f}s not cached at {path}. "
            "Run scripts/download_audioset.py first."
        )
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} sample rate {sr} != {SAMPLE_RATE}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return _fit_length(audio, n_samples)


def load_batch(
    csv_path: str | Path,
    n: int,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> tuple[tf.Tensor, list[str]]:
    rows = read_segments_csv(csv_path)[:n]
    if not rows:
        raise ValueError(f"No rows read from {csv_path}")
    waveforms = np.stack(
        [load_segment(r["ytid"], r["start_s"], r["end_s"], cache_dir) for r in rows]
    )
    return tf.constant(waveforms, dtype=tf.float32), [r["ytid"] for r in rows]


def _fit_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
    if len(audio) >= n_samples:
        return audio[:n_samples].astype(np.float32, copy=False)
    out = np.zeros(n_samples, dtype=np.float32)
    out[: len(audio)] = audio
    return out


def _synthetic_waveform(ytid: str, n_samples: int) -> np.ndarray:
    """Deterministic per-ytid waveform: shaped noise + class-coloured tone.

    Class is read from the prefix (`cry`, `spc`, `sil`, `bab`) so the
    spectral content varies across the smoke set, which gives the
    teacher non-degenerate logits to distill.
    """
    seed = int(hashlib.sha1(ytid.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    cls = ytid[7:10] if len(ytid) >= 10 else "cry"
    base_freq, noise_scale = {
        "cry": (550.0, 0.12),
        "spc": (180.0, 0.08),
        "sil": (0.0, 0.005),
        "bab": (320.0, 0.10),
    }.get(cls, (440.0, 0.1))

    t = np.arange(n_samples, dtype=np.float32) / SAMPLE_RATE
    noise = rng.standard_normal(n_samples).astype(np.float32) * noise_scale
    if base_freq > 0:
        f0 = base_freq * (1.0 + 0.05 * rng.standard_normal())
        tone = 0.15 * np.sin(2 * np.pi * f0 * t).astype(np.float32)
        wav = noise + tone
    else:
        wav = noise
    peak = float(np.max(np.abs(wav)))
    if peak > 0.99:
        wav = wav / peak * 0.95
    return wav.astype(np.float32)
