#!/usr/bin/env python
"""Regenerate `data/ids/audioset_{train,val,test}.csv` from the public
AudioSet metadata. Deterministic — same inputs + same seeds produce
identical CSVs.

Inputs (downloaded from public Google sources, see data/ids/README.md):
    /tmp/eval_segments.csv
    /tmp/balanced_train_segments.csv

Run:
    curl -fsSL -o /tmp/eval_segments.csv \\
      http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv
    curl -fsSL -o /tmp/balanced_train_segments.csv \\
      http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
    python scripts/curate_audioset_ids.py
"""

from __future__ import annotations

import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "data" / "ids"

CRY = {"/m/0463cq4", "/t/dd00002"}
SPEECH = {"/m/09x0r", "/m/05zppz", "/m/02zsn", "/m/0ytgt"}
BABBLING = {"/m/0261r1"}
SILENCE = {"/m/028v0c", "/t/dd00125"}
WAIL = {"/m/07qw_06"}
SCREAM = {"/m/03qc9zr"}


def _parse(path: Path) -> list[tuple]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(", ", 3)
        if len(parts) != 4:
            continue
        ytid = parts[0].strip()
        start = float(parts[1])
        end = float(parts[2])
        labels_str = parts[3].strip().strip('"')
        labels = set(labels_str.split(","))
        rows.append((ytid, start, end, labels_str, labels))
    return rows


def _filter(rows, include: set[str], exclude: set[str] = frozenset()) -> list[tuple]:
    return [r for r in rows if (r[4] & include) and not (r[4] & exclude)]


def _take(rows: list[tuple], n: int, seed: int) -> list[tuple]:
    rnd = random.Random(seed)
    pool = list(rows)
    rnd.shuffle(pool)
    return pool[:n]


def _split(pool: list[tuple], n_train: int, n_val: int, seed: int):
    rnd = random.Random(seed)
    pool = list(pool)
    rnd.shuffle(pool)
    return pool[:n_train], pool[n_train : n_train + n_val]


def _write(rows: list[tuple], out_path: Path, header: str) -> None:
    lines = [header, "ytid,start_s,end_s,positive_labels"]
    for ytid, start, end, labels_str, _ in rows:
        lines.append(f"{ytid},{start},{end},{labels_str}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}: {len(rows)} rows")


def main() -> None:
    eval_csv = Path("/tmp/eval_segments.csv")
    balanced_csv = Path("/tmp/balanced_train_segments.csv")
    if not eval_csv.exists() or not balanced_csv.exists():
        raise SystemExit(
            "Missing /tmp/eval_segments.csv or /tmp/balanced_train_segments.csv. "
            "Run the curl commands at the top of this file first."
        )

    eval_rows = _parse(eval_csv)
    balanced_rows = _parse(balanced_csv)
    print(f"sourced: eval={len(eval_rows)}  balanced_train={len(balanced_rows)}")

    # ----- TEST (eval-derived, FROZEN) -----
    test_set = (
        _take(_filter(eval_rows, CRY), 50, seed=1001)
        + _take(_filter(eval_rows, SPEECH, exclude=CRY | BABBLING | WAIL | SCREAM), 25, seed=1002)
        + _take(_filter(eval_rows, SILENCE, exclude=CRY | SPEECH | BABBLING | WAIL | SCREAM), 15, seed=1003)
        + _take(_filter(eval_rows, SCREAM, exclude=CRY), 5, seed=1004)
        + _take(_filter(eval_rows, WAIL, exclude=CRY), 5, seed=1005)
    )

    # ----- TRAIN + VAL (balanced-derived, disjoint) -----
    cry_tr, cry_va = _split(_filter(balanced_rows, CRY), 80, 25, seed=2001)
    spe_tr, spe_va = _split(_filter(balanced_rows, SPEECH, exclude=CRY | BABBLING | WAIL | SCREAM), 200, 50, seed=2002)
    sil_tr, sil_va = _split(_filter(balanced_rows, SILENCE, exclude=CRY | SPEECH | BABBLING | WAIL | SCREAM), 200, 50, seed=2003)
    bab_tr, bab_va = _split(_filter(balanced_rows, BABBLING, exclude=CRY), 30, 15, seed=2004)
    scr_tr, scr_va = _split(_filter(balanced_rows, SCREAM, exclude=CRY), 30, 10, seed=2005)
    wail_tr, wail_va = _split(_filter(balanced_rows, WAIL, exclude=CRY), 30, 10, seed=2006)

    train_set = cry_tr + spe_tr + sil_tr + bab_tr + scr_tr + wail_tr
    val_set = cry_va + spe_va + sil_va + bab_va + scr_va + wail_va

    test_ids = {r[0] for r in test_set}
    train_ids = {r[0] for r in train_set}
    val_ids = {r[0] for r in val_set}
    overlap = (
        len(test_ids & train_ids),
        len(test_ids & val_ids),
        len(train_ids & val_ids),
    )
    if any(overlap):
        raise SystemExit(f"split overlap detected: {overlap}")

    HEADER_TEST = (
        "# audioset_test.csv — FROZEN held-out evaluation set.\n"
        "# Source: AudioSet v1 eval_segments.csv (public).\n"
        "# 100 segments: 50 crying_sobbing+baby_cry, 25 speech, 15 silence, 5 screaming, 5 wail_moan.\n"
        "# DO NOT MODIFY after first publication — this is the headline-metric\n"
        "# reproducibility surface.  Curated by scripts (deterministic seeds 1001-1005).\n"
        "# See data/ids/README.md for the curation methodology."
    )
    HEADER_TRAIN = (
        "# audioset_train.csv — distillation training set, sampled from\n"
        "# AudioSet v1 balanced_train_segments.csv (public).\n"
        "# 570 segments: 80 cry, 200 speech, 200 silence, 30 babbling, 30 screaming, 30 wail.\n"
        "# Disjoint from audioset_val.csv and audioset_test.csv.\n"
        "# Curated by scripts (deterministic seeds 2001-2006).\n"
        "# See data/ids/README.md for the curation methodology."
    )
    HEADER_VAL = (
        "# audioset_val.csv — distillation validation set, sampled from\n"
        "# AudioSet v1 balanced_train_segments.csv (public).\n"
        "# 160 segments: 25 cry, 50 speech, 50 silence, 15 babbling, 10 screaming, 10 wail.\n"
        "# Disjoint from audioset_train.csv and audioset_test.csv.\n"
        "# Curated by scripts (deterministic seeds 2001-2006, val slice).\n"
        "# See data/ids/README.md for the curation methodology."
    )

    _write(test_set, OUT / "audioset_test.csv", HEADER_TEST)
    _write(train_set, OUT / "audioset_train.csv", HEADER_TRAIN)
    _write(val_set, OUT / "audioset_val.csv", HEADER_VAL)


if __name__ == "__main__":
    main()
