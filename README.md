<div align="center">

# 📈 Beyond the Smile

### UBS Fin AI Bootcamp — CNY/CNH Volatility Intelligence

**Decomposing the USD/CNY & USD/CNH implied-volatility surface — beyond ATM, beyond the smile.**

[![CI](https://github.com/nl2992/UBS_FinAI_BeyondTheSmile/actions/workflows/ci.yml/badge.svg)](https://github.com/nl2992/UBS_FinAI_BeyondTheSmile/actions)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![React](https://img.shields.io/badge/react-19-61dafb)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20%2B%20SSE-009688)
![Tests](https://img.shields.io/badge/tests-36%20backend%20%2B%2019%20frontend-success)

*Two decades of vol-surface data → leakage-safe factor models → explainable forecasts → an AI research terminal that shows its work.*

</div>

---

## What it does

Ask it a question like:

> **"What drove CNH ATM vol on 2025-12-16?"**

and the terminal streams back a grounded answer — realized RV **0.488** vs HAR-X forecast **0.538**, dominant SHAP driver **prior-week volatility (+1.08)**, with China onshore repo-rate changes and CSI 300 spillovers flagged in the GBM view. Every number is fetched live from the model store via tool calls; **the LLM cannot invent figures**.

Behind that sits a full research stack:

| | |
|---|---|
| 🧠 **Factor models** | PCA decomposition of the CNY & CNH vol surfaces (level / skew / curvature / tails), fit strictly in-sample |
| 📉 **Vol forecasting** | Rolling, no-lookahead **HAR-X**, **GARCH(1,1)**, **MIDAS-GARCH**, and **LightGBM** — benchmarked OOS with QLIKE & correlation vs naive baselines |
| 🔍 **Explainability** | **SHAP** on every rolling refit window — `TreeExplainer` for the GBM, `LinearExplainer` for HAR-X — per-date driver attribution + monthly stacked view |
| 📰 **News intelligence** | **FinBERT** sentiment scoring of curated CNY/CNH news, aligned to panel dates |
| 🚨 **Risk alerts** | LLM-written, UBS-style daily reports — generated *only* from structured model output |
| 💬 **Grounded chat** | Streaming DeepSeek assistant with tool-calling against the parquet store (SSE, token-by-token) |
| 📄 **Tear sheets** | One-click branded PDF per factor/model — chart, SHAP drivers, full model table |

## Architecture

```mermaid
flowchart LR
    A[("FINAL_Data.csv<br/>5,929 days × 145 cols")] --> B["finai/pipeline<br/>PCA → AR(1) → HAR-X · GBM<br/>SHAP · FinBERT"]
    B --> C[("finai/store<br/>parquet")]
    C --> D["backend/<br/>FastAPI · 11 endpoints<br/>REST + SSE chat"]
    D --> E["frontend/<br/>React · Vite · Tailwind<br/>Terminal · Alerts · Chat"]
    D -.tool calls.-> F["DeepSeek LLM"]
    F -.grounded answers.-> D
    C --> G["finai/app<br/>Streamlit research UI"]
```

**No look-ahead, anywhere.** IS/OOS split at 2019-12-31: PCA loadings, AR(1) de-meaning, scaling constants, and every rolling model fit use only information available at the time of the forecast.

## Quick start

```bash
# 1) environment
python3 -m venv ~/.venvs/beyond-the-smile
~/.venvs/beyond-the-smile/bin/pip install -r requirements.txt

# 2) secrets (never committed)
cp .env.example .env          # add your DEEPSEEK_API_KEY

# 3) everything, one command
./scripts/dev.sh              # --with-streamlit for the research UI too
```

→ **http://localhost:5173** — the terminal. **http://localhost:8000/docs** — the API.

### 🐳 Docker

```bash
echo "DEEPSEEK_API_KEY=sk-..." > .env
docker compose up --build     # terminal at http://localhost:8080
```

The backend image bakes the entire model store at build time — containers start instantly. The nginx layer proxies `/api` with SSE-safe streaming. No keys are ever baked into images.

## The terminal

| Page | What you get |
|---|---|
| **Overview** | Branded landing — methodology walk-through + live OOS metrics |
| **Terminal** | Realized vs HAR-X vs GBM vol · OOS model league table · per-date SHAP drivers · monthly attribution stacks · **PDF tear-sheet export** |
| **Risk Alerts** | FinBERT sentiment strip · generated UBS-style daily reports |
| **Chat** | Streaming grounded Q&A — watch it consult the data store in real time |

## Engineering

- **55 automated tests** — 36 backend (every endpoint, incl. SSE frames + streaming-generator unit tests) + 19 frontend (SSE chunk-boundary parsing, typed error contract) — all gated in **GitHub Actions CI** on every push, which rebuilds the model store from raw data first.
- **API contract** — [docs/API_CONTRACT.md](docs/API_CONTRACT.md) is the single source of truth both halves build against.
- **Reproducible** — pinned CI deps ([requirements-ci.txt](requirements-ci.txt)), containerized runtime, deterministic pipeline (`python -m finai.pipeline.run_pipeline`).

## Repo map

```
finai/pipeline/   ingestion · PCA factors · vol models · SHAP · sentiment · alerts · CLI
finai/store/      parquet output store (rebuilt by the pipeline; gitignored)
finai/app/        Streamlit research UI + shared data access + chat tool layer
backend/          FastAPI service (REST + SSE chat + PDF tear sheets)
frontend/         React terminal (Vite · TypeScript · Tailwind · recharts)
tests/            backend pytest suite
docs/             API contract
scripts/dev.sh    one-command dev environment
```

## Methodology notes

- **Surface blocks**: ATM (level, 2 PCs), 25Δ risk-reversal (skew), 25Δ butterfly (curvature), 10Δ RR/BF (tails) — per market, log-diff for ATM, level-diff for wings.
- **HAR-X**: forecasts log RV(t+1) from daily/weekly/monthly RV terms + ~40 macro exogenous drivers (repo rates, yields, DXY, VIX, equity & FX proxies), refit every 21 days on a 1,260-day window.
- **SHAP**: one explainer per refit window, so attribution always reflects the coefficients actually used for that forecast — persisted long-format for the API and chat tools.
- **Grounding discipline**: both the risk-alert generator and the chat assistant operate under "write only from this data" prompts, with the chat additionally forced through typed tool calls.

---

<div align="center">
<sub>Built for the UBS Fin AI Bootcamp · Author: <a href="https://github.com/nl2992">Nigel Li</a></sub>
</div>
