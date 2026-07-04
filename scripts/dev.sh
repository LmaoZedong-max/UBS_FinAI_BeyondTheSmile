#!/usr/bin/env bash
# Beyond the Smile - UBS Fin AI Bootcamp: one-command dev environment.
# Starts the FastAPI backend (:8000) and the React frontend (:5173).
# Usage: ./scripts/dev.sh [--with-streamlit]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${BTS_PYTHON:-$HOME/.venvs/beyond-the-smile/bin/python3}"

if [[ ! -x "$PY" ]]; then
  echo "Python venv not found at $PY - set BTS_PYTHON or create the venv:" >&2
  echo "  python3 -m venv ~/.venvs/beyond-the-smile && ~/.venvs/beyond-the-smile/bin/pip install -r requirements.txt" >&2
  exit 1
fi

if [[ ! -f "$ROOT/finai/store/daily_outputs.parquet" ]]; then
  echo "Store is empty - building it first (vol models + SHAP, ~2 min)..."
  (cd "$ROOT" && "$PY" -W ignore -m finai.pipeline.run_pipeline --skip-sentiment)
fi

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "Starting backend on :8000 ..."
(cd "$ROOT" && exec "$PY" -m uvicorn backend.main:app --port 8000) &
pids+=($!)

if [[ "${1:-}" == "--with-streamlit" ]]; then
  echo "Starting Streamlit on :8501 ..."
  (cd "$ROOT" && exec "$PY" -m streamlit run finai/app/main.py --server.headless true --server.port 8501) &
  pids+=($!)
fi

echo "Starting frontend on :5173 ..."
(cd "$ROOT/frontend" && exec npm run dev) &
pids+=($!)

echo
echo "Beyond the Smile is up:"
echo "  terminal   http://localhost:5173"
echo "  API docs   http://localhost:8000/docs"
[[ "${1:-}" == "--with-streamlit" ]] && echo "  streamlit  http://localhost:8501"
echo "Ctrl-C stops everything."
wait
