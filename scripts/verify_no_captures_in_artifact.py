#!/usr/bin/env python3
"""Pre-flight guard for HuggingFace upload bundles.

Refuses any bundle that contains audio, capture-derived files, or
oversized binaries. Run by `scripts/upload_hf.py` before invoking
`huggingface-cli upload`. Exits 1 on any offence with a printed diff.

Belt-and-braces: `.gitignore` should already keep these out of git,
but the upload bundle is constructed at runtime and could pick up
local working files.

Usage:
    python scripts/verify_no_captures_in_artifact.py <bundle_dir>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# Anything matching these patterns is forbidden in an upload bundle.
FORBIDDEN_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".npy", ".pkl", ".pt", ".h5"}
FORBIDDEN_SUBSTRINGS = ("capture", "home_captures", "session", "night-")
# Capture filename pattern: cry-YYYYMMDDTHHMMSS+TZ
CAPTURE_FILENAME_RE = re.compile(r"cry-\d{8}T\d{6}")
# Allowed large file (the actual model artifact)
ALLOWED_LARGE_SUFFIXES = {".tflite"}
LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10 MB


def find_offences(bundle_dir: Path) -> list[tuple[Path, str]]:
    """Walk bundle_dir, return [(path, reason), ...] for any rule violation."""
    offences: list[tuple[Path, str]] = []
    if not bundle_dir.is_dir():
        offences.append((bundle_dir, f"bundle path is not a directory: {bundle_dir}"))
        return offences

    for p in sorted(bundle_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(bundle_dir)
        rel_str = str(rel)

        if p.suffix.lower() in FORBIDDEN_SUFFIXES:
            offences.append((rel, f"forbidden suffix: {p.suffix}"))
            continue

        if CAPTURE_FILENAME_RE.search(rel_str):
            offences.append((rel, "filename matches capture timestamp pattern"))
            continue

        lower = rel_str.lower()
        for sub in FORBIDDEN_SUBSTRINGS:
            if sub in lower:
                offences.append((rel, f"path contains forbidden substring: {sub!r}"))
                break
        else:
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size > LARGE_FILE_THRESHOLD and p.suffix.lower() not in ALLOWED_LARGE_SUFFIXES:
                offences.append((rel, f"oversized non-tflite file: {size / 1e6:.1f} MB"))

    return offences


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: verify_no_captures_in_artifact.py <bundle_dir>\n")
        return 2

    bundle_dir = Path(argv[1]).resolve()
    offences = find_offences(bundle_dir)

    if not offences:
        print(f"OK: {bundle_dir} clean ({sum(1 for _ in bundle_dir.rglob('*') if _.is_file())} files inspected)")
        return 0

    print(f"REFUSED: {bundle_dir}")
    print(f"  {len(offences)} offence(s):")
    for path, reason in offences:
        print(f"    - {path}  ←  {reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
