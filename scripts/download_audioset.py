#!/usr/bin/env python
"""Download AudioSet segments to `data/audioset/cache/`.

Reads a `data/ids/*.csv` file (or all four if `--all`), downloads each
YouTube source via yt-dlp, and writes a 16 kHz mono WAV trimmed to
`[start_s, end_s]`. Idempotent — existing outputs are skipped, and
failed downloads (takedowns, region locks, age gates, etc.) leave a
`.dead` marker so they aren't retried on subsequent runs.

Requires: `pip install -e ".[audioset]"` (yt-dlp) and ffmpeg on PATH.

Usage:
    python scripts/download_audioset.py --ids data/ids/audioset_smoke.csv
    python scripts/download_audioset.py --ids data/ids/audioset_test.csv --max 50
    python scripts/download_audioset.py --all --jobs 4
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "data" / "audioset" / "cache"
DEFAULT_IDS_DIR = REPO_ROOT / "data" / "ids"
SAMPLE_RATE = 16000


def _read_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        ytid = row["ytid"].strip()
        if ytid.startswith("_smoke_"):
            continue  # synthetic placeholders, no download
        rows.append(
            {
                "ytid": ytid,
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "positive_labels": row["positive_labels"].strip(),
            }
        )
    return rows


def _output_paths(ytid: str, start_s: float, cache_dir: Path) -> tuple[Path, Path]:
    stem = f"{ytid}_{start_s:.1f}"
    return cache_dir / f"{stem}.wav", cache_dir / f"{stem}.dead"


def _check_tools() -> None:
    for tool in ("yt-dlp", "ffmpeg"):
        if shutil.which(tool) is None:
            raise SystemExit(
                f"`{tool}` not found on PATH. Install with:\n"
                "  pip install -e \".[audioset]\"   # for yt-dlp\n"
                "  brew install ffmpeg               # for ffmpeg"
            )


def _download_one(row: dict, cache_dir: Path, timeout_s: int = 90) -> tuple[str, str]:
    """Returns (status, message). status ∈ {ok, skip, dead, error}."""
    ytid = row["ytid"]
    start_s = row["start_s"]
    end_s = row["end_s"]
    out_wav, dead_marker = _output_paths(ytid, start_s, cache_dir)

    if out_wav.exists() and out_wav.stat().st_size > 0:
        return "skip", str(out_wav)
    if dead_marker.exists():
        return "skip", str(dead_marker)

    duration = max(0.5, end_s - start_s)
    url = f"https://www.youtube.com/watch?v={ytid}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Stage 1: yt-dlp pulls the audio stream as wav.
        ytdlp_out = tmp_path / f"{ytid}.%(ext)s"
        try:
            subprocess.run(
                [
                    "yt-dlp", "-q", "--no-warnings",
                    "-f", "bestaudio",
                    "-x", "--audio-format", "wav",
                    "--no-playlist",
                    "-o", str(ytdlp_out),
                    url,
                ],
                check=True,
                timeout=timeout_s,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            dead_marker.write_text(f"yt-dlp failed: {type(e).__name__}\n")
            return "dead", str(dead_marker)

        full_wav = next(tmp_path.glob(f"{ytid}.wav"), None)
        if full_wav is None or full_wav.stat().st_size == 0:
            dead_marker.write_text("yt-dlp produced no wav\n")
            return "dead", str(dead_marker)

        # Stage 2: ffmpeg trims to [start, end] and resamples to 16 kHz mono.
        try:
            subprocess.run(
                [
                    "ffmpeg", "-loglevel", "error", "-y",
                    "-ss", f"{start_s}",
                    "-i", str(full_wav),
                    "-t", f"{duration}",
                    "-ar", str(SAMPLE_RATE),
                    "-ac", "1",
                    str(out_wav),
                ],
                check=True,
                timeout=30,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            dead_marker.write_text(f"ffmpeg failed: {type(e).__name__}\n")
            return "dead", str(dead_marker)

    if not out_wav.exists() or out_wav.stat().st_size == 0:
        dead_marker.write_text("ffmpeg produced empty wav\n")
        if out_wav.exists():
            out_wav.unlink()
        return "dead", str(dead_marker)
    return "ok", str(out_wav)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, help="One CSV under data/ids/.")
    parser.add_argument("--all", action="store_true",
                        help="Process audioset_train|val|test.csv (skips smoke).")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--max", type=int, default=0,
                        help="Stop after N attempted segments (debug).")
    parser.add_argument("--jobs", type=int, default=2,
                        help="Parallel downloads (be polite to YouTube).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be downloaded; touch nothing.")
    args = parser.parse_args()

    if not args.ids and not args.all:
        parser.error("Pass --ids <csv> or --all.")

    if args.all:
        csv_paths = [
            DEFAULT_IDS_DIR / f"audioset_{split}.csv" for split in ("train", "val", "test")
        ]
    else:
        csv_paths = [args.ids]

    rows: list[dict] = []
    for p in csv_paths:
        if not p.exists():
            print(f"[skip] {p} (not found)", file=sys.stderr)
            continue
        rows.extend(_read_csv(p))

    if args.max:
        rows = rows[: args.max]
    if not rows:
        print("Nothing to do.")
        return 0

    args.cache.mkdir(parents=True, exist_ok=True)
    print(f"[download] {len(rows)} segments → {args.cache}  (jobs={args.jobs})")

    if args.dry_run:
        for r in rows[:10]:
            print(f"  {r['ytid']} {r['start_s']:.1f}-{r['end_s']:.1f}")
        if len(rows) > 10:
            print(f"  … and {len(rows) - 10} more")
        return 0

    _check_tools()

    tally = {"ok": 0, "skip": 0, "dead": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_download_one, r, args.cache): r for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            r = futures[fut]
            try:
                status, msg = fut.result()
            except Exception as e:
                status, msg = "error", f"{type(e).__name__}: {e}"
            tally[status] += 1
            if i % 10 == 0 or i == len(rows):
                print(
                    f"  [{i}/{len(rows)}] ok={tally['ok']} "
                    f"skip={tally['skip']} dead={tally['dead']} err={tally['error']}"
                )
            if status == "error":
                print(f"  ! {r['ytid']}: {msg}", file=sys.stderr)

    print(f"[download] done. {tally}")
    return 0 if tally["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
