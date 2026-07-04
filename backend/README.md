# Beyond the Smile — Backend (UBS Fin AI Bootcamp)

FastAPI service exposing the vol-forecast / SHAP / sentiment / alert store and the
grounded DeepSeek chat. API shapes are defined in `docs/API_CONTRACT.md`.

## Run

```bash
cd /path/to/14-01-UBS_SH
~/.venvs/beyond-the-smile/bin/python3 -m uvicorn backend.main:app --port 8000
```

Interactive docs at http://localhost:8000/docs.

## Requirements

- `finai/store/*.parquet` must exist — produce them with
  `python -m finai.pipeline.run_pipeline` first.
- `DEEPSEEK_API_KEY` in the environment (or a `.env` at repo root) for
  `POST /api/chat`; every other endpoint works without it (chat returns 503).
