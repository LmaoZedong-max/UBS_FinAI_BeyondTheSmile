# UBS_FinAI_BeyondTheSmile

**Beyond the Smile — UBS Fin AI Bootcamp**

Quant research stack for USD/CNY & USD/CNH FX implied-volatility surfaces:
PCA factor extraction → rolling HAR-X / GBM vol forecasts → SHAP driver
attribution → FinBERT news sentiment → LLM risk alerts → professional web
terminal with grounded chat.

## Components

| Piece | Path | Run |
|---|---|---|
| Research pipeline | `finai/pipeline/` | `python -m finai.pipeline.run_pipeline` |
| Output store | `finai/store/*.parquet` | produced by the pipeline |
| Streamlit research UI | `finai/app/` | `streamlit run finai/app/main.py` |
| FastAPI backend | `backend/` | `uvicorn backend.main:app --port 8000` |
| React terminal | `frontend/` | `npm run dev` (port 5173, proxies /api → :8000) |
| R&D notebooks | `finai/notebooks/` | reference only |

## Quick start

```bash
# 1. environment (Python 3.9+; venv lives outside the repo for preview compat)
~/.venvs/beyond-the-smile/bin/pip install -r requirements.txt

# 2. secrets — never commit .env
cp .env.example .env   # then fill in DEEPSEEK_API_KEY

# 3. build the store (vol models + SHAP; add news/alerts with no flags)
~/.venvs/beyond-the-smile/bin/python3 -m finai.pipeline.run_pipeline --skip-sentiment

# 4. backend + frontend (or run them individually — see backend/README.md)
./scripts/dev.sh            # add --with-streamlit for the research UI too
```

Open http://localhost:5173.

## Modeling notes

- In-sample/out-of-sample split at 2019-12-31/2020-01-01; PCA, AR(1) de-meaning,
  and all rolling model fits use strictly no-lookahead windows.
- SHAP: `TreeExplainer` on the rolling LightGBM, `LinearExplainer` on the rolling
  HAR-X OLS — one explainer per refit window, persisted long-format to the store.
- Chat answers are grounded: the LLM must fetch numbers via tool calls against
  the parquet store and cannot invent figures.
