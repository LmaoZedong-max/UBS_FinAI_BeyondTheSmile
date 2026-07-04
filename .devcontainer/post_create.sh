#!/usr/bin/env bash
# post_create.sh - runs once after the Codespace container is built.
# Installs Python deps, builds the model store (a few factors, ~3 min),
# and installs frontend deps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Installing Python dependencies..."
pip install --quiet --disable-pip-version-check -r "$ROOT/requirements-ci.txt"

echo "==> Building model store (--skip-sentiment, CNH_ATM_PC1 CNH_ATM_PC2 CNH_25RR_PC1)..."
cd "$ROOT"
python -W ignore -m finai.pipeline.run_pipeline \
  --skip-sentiment \
  --factors CNH_ATM_PC1 CNH_ATM_PC2 CNH_25RR_PC1

echo "==> Installing frontend dependencies..."
cd "$ROOT/frontend"
npm ci

echo "==> post_create done. Run:  bash scripts/codespace_start.sh"
