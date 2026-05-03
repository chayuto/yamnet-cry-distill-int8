"""Evaluation harness — headline AudioSet F1 + side-metric captures KL.

Two evaluators:

- `eval_audioset_holdout(model_path, ids_csv, cache_dir)` — runs the
  saved student against AudioSet held-out segments, scores
  `cry_score = p[19] + p[20]` (Crying-sobbing + Baby-cry), aggregates
  per-segment via mean over 0.96 s windows with 0.48 s hop, reports
  precision / recall / F1 / AUC at threshold 0.5. **This is the
  headline metric** — its JSON output is committed.
- `eval_home_captures(model_path, root, master_csv)` — captures-side
  side metric. Computes mean-of-patches cry score per capture and
  scores against the auto-ensemble's `confidence_tier` (high_pos vs
  high_neg = ground truth) plus AUC. **Side metric** — JSON output
  is gitignored (touches private data).

Run via:
    python -m yamnet_cry_distill_int8.eval --exp EXP-004
    python -m yamnet_cry_distill_int8.eval --all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import tensorflow as tf

from .data.audioset import DEFAULT_CACHE_DIR, read_segments_csv
from .data.home_captures import resolve_root
from .teacher import (
    MEL_BINS,
    PATCH_FRAMES,
    PATCH_SAMPLES,
    SAMPLE_RATE,
    YAMNetTeacher,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CRY_CLASS_IDXS = (19, 20)  # Crying-sobbing + Baby-cry, infant-cry


def _rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)

CRY_MIDS = frozenset({"/m/0463cq4", "/t/dd00002"})
HOP_SAMPLES = PATCH_SAMPLES // 2  # 0.4875 s hop, matches YAMNet's overlap


def _load_clip(path: Path) -> np.ndarray | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False)


def _patches_from_clip(audio: np.ndarray) -> np.ndarray:
    if len(audio) < PATCH_SAMPLES:
        out = np.zeros(PATCH_SAMPLES, dtype=np.float32)
        out[: len(audio)] = audio
        return out[None, :]
    starts = list(range(0, len(audio) - PATCH_SAMPLES + 1, HOP_SAMPLES))
    if not starts:
        starts = [0]
    return np.stack([audio[s : s + PATCH_SAMPLES] for s in starts])


def _load_student(model_path: Path):
    """Returns a `predict(mel_fp32)` callable handling either Keras (.h5)
    or TFLite (.tflite) checkpoints. Float32 logits in/out — input mel
    has shape (1, 96, 64, 1)."""
    p = str(model_path)
    if p.endswith(".tflite"):
        interp = tf.lite.Interpreter(model_path=p)
        interp.allocate_tensors()
        in_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]
        in_scale, in_zp = in_det["quantization"]
        out_scale, out_zp = out_det["quantization"]

        def predict(mel_fp32: np.ndarray) -> np.ndarray:
            x = np.clip(
                np.round(mel_fp32 / in_scale + in_zp), -128, 127
            ).astype(np.int8)
            interp.set_tensor(in_det["index"], x)
            interp.invoke()
            raw = interp.get_tensor(out_det["index"])
            return (raw.astype(np.float32) - out_zp) * out_scale

        return predict, "tflite_int8"

    keras = tf.keras.models.load_model(p, compile=False)

    def predict(mel_fp32: np.ndarray) -> np.ndarray:
        return keras(mel_fp32, training=False).numpy()

    return predict, "keras_fp32"


def _student_cry_score_per_clip(
    student_predict,
    teacher: YAMNetTeacher,
    audio: np.ndarray,
) -> float:
    patches = _patches_from_clip(audio)
    cry_scores = []
    for wav in patches:
        _, _, log_mel = teacher.forward(tf.constant(wav))
        if log_mel.shape[0] < PATCH_FRAMES:
            log_mel = tf.pad(log_mel, [[0, PATCH_FRAMES - log_mel.shape[0]], [0, 0]])
        mel = log_mel[:PATCH_FRAMES, :].numpy()
        logits = student_predict(mel[None, ..., None].astype(np.float32))
        # softmax in fp64 to keep numeric stability for tiny tflite logits
        x = logits[0].astype(np.float64)
        x -= x.max()
        probs = np.exp(x) / np.exp(x).sum()
        cry_scores.append(float(probs[CRY_CLASS_IDXS[0]] + probs[CRY_CLASS_IDXS[1]]))
    return float(np.mean(cry_scores))


def _f1_at(scores: np.ndarray, y_true: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int32)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def _binary_metrics(scores: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> dict:
    """Both fixed-threshold (default 0.5) and best-threshold (sweep) F1.

    The default-threshold F1 is what a deployment without per-model
    calibration would see. The best-threshold F1 reflects the model's
    raw capability — students inherit teacher's small-cry-mass softmax
    so 0.5 is rarely the right operating point for distilled models.
    """
    fixed = _f1_at(scores, y_true, threshold)

    best = {"f1": -1.0}
    for thr in np.unique(np.concatenate([scores, [0.0, 1.0]])):
        m = _f1_at(scores, y_true, float(thr))
        if m["f1"] > best["f1"]:
            best = m

    return {
        "threshold": fixed["threshold"],
        "tp": fixed["tp"], "fp": fixed["fp"],
        "fn": fixed["fn"], "tn": fixed["tn"],
        "precision": fixed["precision"],
        "recall": fixed["recall"],
        "f1": fixed["f1"],
        "best_threshold": best["threshold"],
        "best_f1": best["f1"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "auc": _auc(scores, y_true),
        "n_pos": int(y_true.sum()),
        "n_neg": int(len(y_true) - y_true.sum()),
        "n_total": int(len(y_true)),
    }


def _auc(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Mann-Whitney U-statistic AUC."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    rank_sum_pos = ranks[: len(pos)].sum()
    u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def eval_audioset_holdout(
    model_path: Path,
    ids_csv: Path,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    threshold: float = 0.5,
) -> dict:
    rows = read_segments_csv(ids_csv)
    teacher = YAMNetTeacher()
    student_predict, backend = _load_student(model_path)

    scores: list[float] = []
    y_true: list[int] = []
    skipped = {"missing": 0, "dead": 0}
    t0 = time.time()
    for i, r in enumerate(rows):
        wav_path = cache_dir / f"{r['ytid']}_{r['start_s']:.1f}.wav"
        dead_path = wav_path.with_suffix(".dead")
        audio = _load_clip(wav_path)
        if audio is None:
            if dead_path.exists():
                skipped["dead"] += 1
            else:
                skipped["missing"] += 1
            continue
        score = _student_cry_score_per_clip(student_predict, teacher, audio)
        is_cry = bool(set(r["positive_labels"].split(",")) & CRY_MIDS)
        scores.append(score)
        y_true.append(int(is_cry))
        if (i + 1) % 25 == 0 or i == len(rows) - 1:
            print(f"  [{i + 1}/{len(rows)}] {time.time() - t0:.1f}s")

    s = np.array(scores)
    y = np.array(y_true)
    metrics = _binary_metrics(s, y, threshold=threshold)
    metrics["skipped_dead"] = skipped["dead"]
    metrics["skipped_missing"] = skipped["missing"]
    metrics["n_evaluated"] = int(len(s))
    metrics["backend"] = backend
    metrics["model_path"] = _rel(model_path)
    metrics["ids_csv"] = _rel(ids_csv)
    return metrics


def _read_master_csv(master_csv: Path) -> list[dict]:
    rows: list[dict] = []
    with open(master_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def eval_home_captures(
    model_path: Path,
    threshold: float = 0.5,
) -> dict:
    """Captures-side side metric — AUC of the student's cry-score against
    the auto-ensemble's confidence_tier (high_pos = positive, high_neg =
    negative; medium / low tiers excluded — they're the unconfident
    fringes the audit flagged for re-curation).
    """
    root = resolve_root()
    if root is None:
        return {"error": "no captures root resolvable"}
    master_csv = root / "datasets" / "cry-detect-01" / "labels" / "master.csv"
    canonical = root / "projects" / "cry-detect-01" / "logs" / "canonical" / "wavs"
    if not master_csv.exists() or not canonical.exists():
        return {"error": f"missing {master_csv} or {canonical}"}

    teacher = YAMNetTeacher()
    student_predict, backend = _load_student(model_path)
    rows = _read_master_csv(master_csv)
    by_file = {r["capture_file"]: r for r in rows}

    scores: list[float] = []
    y_true: list[int] = []
    hours: list[int] = []
    n_skipped = 0
    t0 = time.time()
    paths = sorted(canonical.glob("cry-*.wav"))
    for i, wav_path in enumerate(paths):
        meta = by_file.get(wav_path.name)
        if meta is None:
            n_skipped += 1
            continue
        tier = meta.get("confidence_tier", "")
        if tier not in ("high_pos", "high_neg"):
            continue
        audio = _load_clip(wav_path)
        if audio is None:
            n_skipped += 1
            continue
        score = _student_cry_score_per_clip(student_predict, teacher, audio)
        scores.append(score)
        y_true.append(1 if tier == "high_pos" else 0)
        ts_iso = meta.get("ts_iso", "")
        hours.append(int(ts_iso[11:13]) if len(ts_iso) >= 13 else -1)
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(paths)}] {time.time() - t0:.1f}s")

    s = np.array(scores)
    y = np.array(y_true)
    h = np.array(hours)
    metrics = _binary_metrics(s, y, threshold=threshold)

    # Hour-of-day stratification: AUC computed within each hour bucket
    by_hour = {}
    for hr in sorted(set(h.tolist())):
        if hr < 0:
            continue
        mask = h == hr
        if mask.sum() < 6 or y[mask].sum() in (0, mask.sum()):
            continue
        by_hour[int(hr)] = {
            "n": int(mask.sum()),
            "n_pos": int(y[mask].sum()),
            "auc": _auc(s[mask], y[mask]),
            "mean_score_pos": float(s[mask][y[mask] == 1].mean()) if y[mask].sum() else None,
            "mean_score_neg": float(s[mask][y[mask] == 0].mean()) if (mask.sum() - y[mask].sum()) else None,
        }

    metrics["by_hour"] = by_hour
    metrics["n_skipped"] = n_skipped
    metrics["model_path"] = _rel(model_path)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["EXP-002", "EXP-003", "EXP-004", "EXP-006"], action="append")
    parser.add_argument("--all", action="store_true",
                        help="Run EXP-002, EXP-003, EXP-004 (the original baselines).")
    parser.add_argument("--ids", type=Path, default=Path("data/ids/audioset_test.csv"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--captures", action="store_true",
                        help="Also run the captures-side side metric.")
    parser.add_argument("--tflite", action="store_true",
                        help="Use the .tflite checkpoint for the chosen --exp(s).")
    args = parser.parse_args()

    if args.all:
        exps = ["EXP-002", "EXP-003", "EXP-004"]
    elif args.exp:
        exps = args.exp
    else:
        parser.error("pass --exp <id> or --all")

    docs_dir = REPO_ROOT / "docs" / "experiments"
    for exp in exps:
        suffix = exp.replace("EXP-", "exp")  # e.g. EXP-002 -> exp002
        ext = ".tflite" if args.tflite else ".h5"
        suffix_out = f"{suffix}_int8" if args.tflite else suffix
        model_path = REPO_ROOT / "models" / f"{suffix}_dscnn{ext}"
        if not model_path.exists():
            print(f"[{exp}] no checkpoint at {model_path}, skipping")
            continue
        suffix = suffix_out
        print(f"[{exp}] AudioSet held-out eval: {args.ids}")
        m = eval_audioset_holdout(model_path, args.ids, threshold=args.threshold)
        print(
            f"[{exp}] F1@0.5={m['f1']:.3f}  best_F1={m['best_f1']:.3f}@thr={m['best_threshold']:.3f}  "
            f"AUC={m['auc']:.3f}  (n={m['n_evaluated']}, dead={m['skipped_dead']})"
        )
        out_path = docs_dir / f"eval_audioset_holdout_{suffix}.json"
        out_path.write_text(json.dumps(m, indent=2))
        print(f"  -> {out_path}")

        if args.captures:
            print(f"[{exp}] captures-side side metric:")
            cm = eval_home_captures(model_path, threshold=args.threshold)
            if "error" in cm:
                print(f"  skipped: {cm['error']}")
            else:
                print(
                    f"[{exp}] captures: F1@0.5={cm['f1']:.3f}  "
                    f"best_F1={cm['best_f1']:.3f}@thr={cm['best_threshold']:.3f}  "
                    f"AUC={cm['auc']:.3f}  (n_pos={cm['n_pos']}, n_neg={cm['n_neg']})"
                )
            cap_out = docs_dir / f"eval_home_captures_{suffix}.json"
            cap_out.write_text(json.dumps(cm, indent=2))
            print(f"  -> {cap_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
