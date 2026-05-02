#!/usr/bin/env bash
# EXP-001 smoke runner — proves the distillation loop closes end-to-end.
# Usage: bash scripts/run_exp001_smoke.sh
#
# Loads YAMNet from TF Hub (cached after first run), builds the DS-CNN
# student, takes a 4-clip synthetic batch from `data/ids/audioset_smoke.csv`,
# runs one optimizer step on the KL loss between teacher probs and
# student logits, and exits 0 on success.
#
# Acceptance: completes in <2 minutes on warm cache, no NaN, student
# parameter count ≤100K. No `*.wav` is written or committed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "[exp001] python: $(python --version 2>&1)"
echo "[exp001] cwd: $(pwd)"

python -m yamnet_cry_distill_int8.train --smoke --batch 4

echo "[exp001] done."
