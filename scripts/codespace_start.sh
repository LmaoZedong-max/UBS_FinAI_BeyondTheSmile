#!/usr/bin/env bash
# codespace_start.sh - starts the full stack inside a GitHub Codespace.
# Run after postCreate has finished:
#   bash scripts/codespace_start.sh
#
# Backend (FastAPI) runs in the background on :8000.
# Frontend (Vite) runs in the foreground on :5173.
# Codespaces will expose both ports automatically.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Starting FastAPI backend on :8000 (background)..."
nohup python3 -m uvicorn backend.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  > "$ROOT/backend.log" 2>&1 &
BACKEND_PID=$!
echo "    backend PID $BACKEND_PID  (logs: backend.log)"

# Give the backend a moment to bind before the frontend proxy starts hitting it
sleep 2

echo ""
echo "==> Starting Vite frontend on :5173 (foreground)..."
echo "    Codespaces URL hint: the 5173 port will be auto-forwarded."
echo "    Look for the 'Open in Browser' notification, or open the PORTS panel."
echo ""

cd "$ROOT/frontend"
exec npm run dev -- --host
