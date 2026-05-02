"""Smoke loader sanity — fully offline."""

from __future__ import annotations

from pathlib import Path

from yamnet_cry_distill_int8.data.audioset import (
    load_batch,
    load_segment,
    read_segments_csv,
)
from yamnet_cry_distill_int8.teacher import PATCH_SAMPLES

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CSV = REPO_ROOT / "data" / "ids" / "audioset_smoke.csv"


def test_smoke_csv_has_50_rows():
    rows = read_segments_csv(SMOKE_CSV)
    assert len(rows) == 50
    cls = {"cry": 0, "spc": 0, "sil": 0, "bab": 0}
    for r in rows:
        prefix = r["ytid"][7:10]
        if prefix in cls:
            cls[prefix] += 1
    assert cls == {"cry": 30, "spc": 10, "sil": 5, "bab": 5}


def test_synthetic_segment_is_deterministic_and_shaped():
    a = load_segment("_smoke_cry_00", 0.0, 0.975)
    b = load_segment("_smoke_cry_00", 0.0, 0.975)
    assert a.shape == (PATCH_SAMPLES,)
    assert (a == b).all()
    c = load_segment("_smoke_sil_00", 0.0, 0.975)
    assert (a != c).any()


def test_load_batch_returns_correct_shape():
    waveforms, ids = load_batch(SMOKE_CSV, n=4)
    assert waveforms.shape == (4, PATCH_SAMPLES)
    assert len(ids) == 4
