#!/usr/bin/env python
"""Assemble + upload the published model bundle to HuggingFace.

Bundle composition (model-only, never dataset):

    model.tflite                            — 110 KB INT8 student
    README.md                               — model card with HF frontmatter
    config.json                             — input/output spec, class indices,
                                              recommended threshold
    eval_audioset_holdout_int8.json         — frozen-test eval (62 segments)

Pre-flight: runs `verify_no_captures_in_artifact.py` against the bundle
directory. Refuses to upload on any forbidden suffix or path pattern.

Auth: reads HF_TOKEN from the environment. Single-use tokens are
expected; do not write them to disk. The huggingface_hub stored token
under ~/.cache/huggingface/token is bypassed.

Repo: `chayuto/yamnet-cry-distill-int8` (created on first run if absent).

Usage:
    HF_TOKEN=<token> python scripts/upload_hf.py --exp EXP-006 --tag v0.1.0
    HF_TOKEN=<token> python scripts/upload_hf.py --exp EXP-006 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_REPO_ID = "chayuto/yamnet-cry-distill-int8"


def _build_bundle(staging: Path, exp: str, tag: str | None) -> dict:
    suffix = exp.replace("EXP-", "exp")  # EXP-006 -> exp006
    tflite_src = REPO_ROOT / "models" / f"{suffix}_dscnn.tflite"
    eval_src = REPO_ROOT / "docs" / "experiments" / f"eval_audioset_holdout_{suffix}_int8.json"
    card_src = REPO_ROOT / "docs" / "model_cards" / "yamnet-cry-distill-int8.md"

    for p in (tflite_src, eval_src, card_src):
        if not p.exists():
            raise SystemExit(f"missing required input: {p}")

    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tflite_src, staging / "model.tflite")
    shutil.copy2(eval_src, staging / "eval_audioset_holdout_int8.json")
    shutil.copy2(card_src, staging / "README.md")  # HF model card lives at README.md

    # Derive version string: prefer the --tag arg (stripped of leading 'v'),
    # fall back to pyproject.toml's [project].version. Avoids the bug where
    # config.json lied about the version because the upload script had a
    # hardcoded "0.1.0" that was never bumped for v0.2.0.
    version = (tag or "").lstrip("v")
    if not version:
        try:
            pp = (REPO_ROOT / "pyproject.toml").read_text()
            m = re.search(r'^version\s*=\s*"([^"]+)"', pp, flags=re.MULTILINE)
            version = m.group(1) if m else "unknown"
        except Exception:
            version = "unknown"

    eval_data = json.loads(eval_src.read_text())
    config = {
        "model_name": "yamnet-cry-distill-int8",
        "version": version,
        "source_experiment": exp,
        "architecture": "DS-CNN (Conv stem + 4 depthwise-separable blocks + GAP + Dense-521)",
        "params": 80713,
        "input": {
            "sample_rate_hz": 16000,
            "channels": 1,
            "patch_seconds": 0.96,
            "patch_samples": 15600,
            "feature": "log-mel patch",
            "mel_bins": 64,
            "mel_frames": 96,
            "dtype": "int8",
        },
        "output": {
            "shape": [1, 521],
            "dtype": "int8",
            "semantics": "YAMNet 521-class softmax (per-patch). Apply softmax after dequantizing.",
            "cry_class_indices": [19, 20],
            "cry_class_names": ["Crying, sobbing", "Baby cry, infant cry"],
            "cry_score_formula": "softmax(logits)[19] + softmax(logits)[20]",
        },
        "recommended_threshold": float(eval_data.get("best_threshold", 0.10)),
        "metrics_at_recommended_threshold": {
            "f1": float(eval_data.get("best_f1", 0.0)),
            "precision": float(eval_data.get("best_precision", 0.0)),
            "recall": float(eval_data.get("best_recall", 0.0)),
            "auc": float(eval_data.get("auc", 0.0)),
        },
        "test_set": {
            "source": "AudioSet eval_segments.csv, frozen at curation",
            "n_curated": 100,
            "n_evaluated": int(eval_data.get("n_evaluated", 0)),
            "n_takedowns": int(eval_data.get("skipped_dead", 0)),
        },
        "deployment_notes": (
            "Distilled students inherit the teacher's diffuse softmax — "
            "cry probabilities are well-ordered but small in absolute terms. "
            "Calibrate the operating-point threshold per device (~0.05-0.20 typical)."
        ),
        "teacher_model": "google/yamnet/1 (FP32, 521-class AudioSet)",
        "calibration_set": "data/ids/audioset_val.csv (200 patches, public AudioSet only)",
    }
    (staging / "config.json").write_text(json.dumps(config, indent=2))

    return {
        "files": sorted(p.name for p in staging.iterdir()),
        "total_bytes": sum(p.stat().st_size for p in staging.iterdir()),
    }


def _run_content_guard(staging: Path) -> None:
    guard = REPO_ROOT / "scripts" / "verify_no_captures_in_artifact.py"
    print(f"[upload] content guard: {guard}")
    result = subprocess.run(
        [sys.executable, str(guard), str(staging)],
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(f"content guard refused: exit {result.returncode}")
    print("[upload] content guard passed")


def _upload(staging: Path, repo_id: str, token: str, tag: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    print(f"[upload] ensuring repo {repo_id} exists...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"[upload] uploading folder {staging} → {repo_id}")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="model",
        commit_message="v0.1.0 initial release: EXP-006 teacher-filtered INT8 student",
    )
    if tag:
        print(f"[upload] tagging {tag} on the hub...")
        api.create_tag(repo_id=repo_id, tag=tag, repo_type="model")
    print(f"[upload] done. https://huggingface.co/{repo_id}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="EXP-006")
    parser.add_argument("--tag", default=None,
                        help="HF tag to create after upload (e.g. v0.1.0)")
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--staging", type=Path,
                        default=REPO_ROOT / "ml-experiments" / "hf_bundle")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build + content-guard the bundle but skip upload.")
    args = parser.parse_args()

    if args.staging.exists():
        shutil.rmtree(args.staging)
    info = _build_bundle(args.staging, args.exp, args.tag)
    print(f"[upload] bundle:")
    for f in info["files"]:
        size = (args.staging / f).stat().st_size
        print(f"  {f:40s} {size:>9} bytes")
    print(f"[upload] total: {info['total_bytes']:,} bytes")

    _run_content_guard(args.staging)

    if args.dry_run:
        print("[upload] --dry-run set, skipping upload.")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN env var required. Use a single-use token from huggingface.co; "
            "this script does not write it to disk."
        )
    _upload(args.staging, args.repo_id, token, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
