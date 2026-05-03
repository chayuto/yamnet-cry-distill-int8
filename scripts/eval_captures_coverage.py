#!/usr/bin/env python
"""Score every capture with the student + teacher, dump per-tier stats.

This is the eval the public AudioSet-test eval cannot tell you:
**how does the distilled student behave on the actual deployment data,
across the full confidence spectrum, not just the easy tiers?**

Step 1 — coverage. For all ~475 captures, compute per-frame teacher
and student cry-scores at the YAMNet sliding-window grid (0.4875 s hop),
aggregate to per-clip max + mean, and group by `confidence_tier`.
Dump a JSON aggregate + a markdown summary.

Step 2 — timelines. For 4 captures sampled per tier (20 total), keep
the full per-frame teacher and student score timelines. The eyes-only
question this answers: does the student's cry-score *track* the
teacher's spikes within a clip, or does it drift / smooth / lag?

Outputs are gitignored — they derive from private capture data.
Layout:
    docs/experiments/eval_home_captures_coverage_<exp>.json
    docs/experiments/eval_home_captures_timelines_<exp>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

from yamnet_cry_distill_int8.data.home_captures import resolve_root
from yamnet_cry_distill_int8.eval import _load_student
from yamnet_cry_distill_int8.teacher import (
    PATCH_FRAMES,
    PATCH_SAMPLES,
    SAMPLE_RATE,
    YAMNetTeacher,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HOP_SAMPLES = PATCH_SAMPLES // 2
CRY_IDX = (19, 20)
TIERS = ("high_pos", "medium_pos", "low", "medium_neg", "high_neg")
TIMELINE_PER_TIER = 4


def _load_clip(path: Path) -> np.ndarray | None:
    info = sf.info(str(path))
    if info.samplerate != SAMPLE_RATE:
        return None
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False)


def _windows(audio: np.ndarray) -> list[np.ndarray]:
    if len(audio) < PATCH_SAMPLES:
        out = np.zeros(PATCH_SAMPLES, dtype=np.float32)
        out[: len(audio)] = audio
        return [out]
    starts = list(range(0, len(audio) - PATCH_SAMPLES + 1, HOP_SAMPLES))
    return [audio[s : s + PATCH_SAMPLES] for s in starts]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64) - x.max()
    return np.exp(x) / np.exp(x).sum()


def _pearson(a: list[float], b: list[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _read_master(path: Path) -> dict:
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["capture_file"]] = row
    return out


def score_all(model_path: Path, master: dict, captures_dir: Path,
              timeline_targets: set[str]) -> tuple[list[dict], dict]:
    print(f"[score] model={model_path}")
    teacher = YAMNetTeacher()
    student_predict, backend = _load_student(model_path)
    print(f"[score] student backend: {backend}")

    paths = sorted(captures_dir.glob("cry-*.wav"))
    summary: list[dict] = []
    timelines: dict[str, dict] = {}
    t0 = time.time()
    for i, wav_path in enumerate(paths):
        meta = master.get(wav_path.name)
        if meta is None:
            continue
        audio = _load_clip(wav_path)
        if audio is None:
            continue
        wins = _windows(audio)
        teacher_scores: list[float] = []
        student_scores: list[float] = []
        for w in wins:
            scores, _, log_mel = teacher.forward(tf.constant(w))
            if log_mel.shape[0] < PATCH_FRAMES:
                log_mel = tf.pad(log_mel, [[0, PATCH_FRAMES - log_mel.shape[0]], [0, 0]])
            mel = log_mel[:PATCH_FRAMES, :].numpy()
            probs = tf.reduce_mean(scores, axis=0).numpy()
            t_cry = float(probs[CRY_IDX[0]] + probs[CRY_IDX[1]])
            s_logits = student_predict(mel[None, ..., None].astype(np.float32))
            s_probs = _softmax(s_logits[0])
            s_cry = float(s_probs[CRY_IDX[0]] + s_probs[CRY_IDX[1]])
            teacher_scores.append(t_cry)
            student_scores.append(s_cry)
        summary.append({
            "file": wav_path.name,
            "ts_iso": meta.get("ts_iso", ""),
            "tier": meta.get("confidence_tier", ""),
            "n_frames": len(wins),
            "teacher_max": max(teacher_scores) if teacher_scores else 0.0,
            "teacher_mean": float(np.mean(teacher_scores)) if teacher_scores else 0.0,
            "student_max": max(student_scores) if student_scores else 0.0,
            "student_mean": float(np.mean(student_scores)) if student_scores else 0.0,
            "frame_corr": _pearson(teacher_scores, student_scores),
        })
        if wav_path.name in timeline_targets:
            timelines[wav_path.name] = {
                "tier": meta.get("confidence_tier", ""),
                "ts_iso": meta.get("ts_iso", ""),
                "frame_hop_s": HOP_SAMPLES / SAMPLE_RATE,
                "teacher_cry": teacher_scores,
                "student_cry": student_scores,
            }
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(paths)}] {time.time() - t0:.1f}s")
    print(f"[score] done in {time.time() - t0:.1f}s, {len(summary)} captures scored")
    return summary, timelines


def aggregate_by_tier(summary: list[dict]) -> dict:
    by_tier: dict[str, dict] = {}
    for s in summary:
        t = s["tier"]
        b = by_tier.setdefault(t, {
            "n": 0, "tmax": [], "smax": [], "tmean": [], "smean": [], "corr": []
        })
        b["n"] += 1
        b["tmax"].append(s["teacher_max"])
        b["smax"].append(s["student_max"])
        b["tmean"].append(s["teacher_mean"])
        b["smean"].append(s["student_mean"])
        b["corr"].append(s["frame_corr"])

    out = {}
    for t in TIERS:
        if t not in by_tier:
            continue
        b = by_tier[t]
        out[t] = {
            "n": b["n"],
            "teacher_max_mean": float(np.mean(b["tmax"])),
            "teacher_max_p10": float(np.quantile(b["tmax"], 0.10)),
            "teacher_max_p90": float(np.quantile(b["tmax"], 0.90)),
            "student_max_mean": float(np.mean(b["smax"])),
            "student_max_p10": float(np.quantile(b["smax"], 0.10)),
            "student_max_p90": float(np.quantile(b["smax"], 0.90)),
            "teacher_mean_of_mean": float(np.mean(b["tmean"])),
            "student_mean_of_mean": float(np.mean(b["smean"])),
            "frame_corr_mean": float(np.mean(b["corr"])),
            "frame_corr_p10": float(np.quantile(b["corr"], 0.10)),
            "frame_corr_p90": float(np.quantile(b["corr"], 0.90)),
        }
    out["_total_captures"] = sum(by_tier[t]["n"] for t in by_tier)
    return out


def select_timeline_targets(master: dict, captures_dir: Path, seed: int = 0) -> set[str]:
    """Pick TIMELINE_PER_TIER captures per tier, deterministically."""
    rng = np.random.default_rng(seed)
    by_tier: dict[str, list[str]] = {}
    available = {p.name for p in captures_dir.glob("cry-*.wav")}
    for fname, meta in master.items():
        if fname not in available:
            continue
        by_tier.setdefault(meta.get("confidence_tier", ""), []).append(fname)
    targets: set[str] = set()
    for t in TIERS:
        files = sorted(by_tier.get(t, []))
        if not files:
            continue
        idx = rng.permutation(len(files))[:TIMELINE_PER_TIER]
        for i in idx:
            targets.add(files[int(i)])
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="EXP-008", help="EXP-NNN to score (uses INT8 by default)")
    parser.add_argument("--fp32", action="store_true", help="Use the .h5 checkpoint instead of INT8")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    suffix = args.exp.replace("EXP-", "exp")
    ext = ".h5" if args.fp32 else ".tflite"
    model_path = REPO_ROOT / "models" / f"{suffix}_dscnn{ext}"
    if not model_path.exists():
        raise SystemExit(f"no checkpoint at {model_path}")

    root = resolve_root()
    if root is None:
        raise SystemExit("no captures root resolvable")
    master_csv = root / "datasets" / "cry-detect-01" / "labels" / "master.csv"
    captures_dir = root / "projects" / "cry-detect-01" / "logs" / "canonical" / "wavs"
    print(f"[score] master: {master_csv}")
    print(f"[score] captures: {captures_dir}")
    master = _read_master(master_csv)
    print(f"[score] master rows: {len(master)}")

    timeline_targets = select_timeline_targets(master, captures_dir, args.seed)
    print(f"[score] timeline captures: {len(timeline_targets)} (4/tier)")

    summary, timelines = score_all(model_path, master, captures_dir, timeline_targets)
    by_tier = aggregate_by_tier(summary)

    backend_label = "fp32" if args.fp32 else "int8"
    out_summary = REPO_ROOT / "docs" / "experiments" / f"eval_home_captures_coverage_{suffix}_{backend_label}.json"
    out_timeline = REPO_ROOT / "docs" / "experiments" / f"eval_home_captures_timelines_{suffix}_{backend_label}.json"
    out_summary.write_text(json.dumps({
        "exp": args.exp,
        "model_path": str(model_path.relative_to(REPO_ROOT)),
        "backend": backend_label,
        "n_captures": len(summary),
        "by_tier": by_tier,
        "per_capture": summary,
    }, indent=2))
    out_timeline.write_text(json.dumps({
        "exp": args.exp,
        "model_path": str(model_path.relative_to(REPO_ROOT)),
        "backend": backend_label,
        "timelines": timelines,
    }, indent=2))
    print(f"-> {out_summary.relative_to(REPO_ROOT)}")
    print(f"-> {out_timeline.relative_to(REPO_ROOT)}")

    print("\n=== per-tier summary ===")
    print(f"{'tier':<14} {'n':>4} {'T_max_mean':>11} {'S_max_mean':>11} {'frame_r':>8}")
    for t in TIERS:
        if t not in by_tier:
            continue
        b = by_tier[t]
        print(
            f"{t:<14} {b['n']:>4} {b['teacher_max_mean']:>11.3f} "
            f"{b['student_max_mean']:>11.3f} {b['frame_corr_mean']:>8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
