# Beyond the Smile — API Contract (backend :8000 ⇄ frontend :5173)

Brand: **Beyond the Smile — UBS Fin AI Bootcamp**. All endpoints prefixed `/api`. JSON everywhere. CORS must allow http://localhost:5173.

Data source: `finai/store/*.parquet` (already produced by `python -m finai.pipeline.run_pipeline`), read via `finai/app/data_access.py` helpers where possible. Alert text files live in `NLP Outputs/risk_alert_YYYY-MM-DD.txt`.

## Endpoints

### GET /api/health
`{"status": "ok"}`

### GET /api/factors
`{"factors": ["CNH_ATM_PC1", ...]}` — from `data_access.available_factors()`.

### GET /api/forecasts?factor=CNH_ATM_PC1&start=2020-01-01&end=2025-12-31
```json
{"factor": "CNH_ATM_PC1",
 "series": [{"date": "2020-01-02", "rv_realized": 0.51, "rv_forecast_harx": 0.49, "rv_forecast_gbm": 0.47}, ...]}
```
Nulls allowed for missing values. Dates ISO `YYYY-MM-DD`. start/end optional.

### GET /api/eval?factor=CNH_ATM_PC1
```json
{"factor": "...", "rows": [{"model": "HAR-X", "n_oos": 1620, "QLIKE_daily": ..., "Corr_daily": ..., "QLIKE_week": ..., "Corr_week": ...}, ...]}
```

### GET /api/shap?factor=CNH_ATM_PC1&date=2025-12-16&model=HAR-X&k=8
`model` ∈ {"HAR-X", "GBM"}; returns `{"date", "factor", "model", "drivers": [{"feature", "shap_value"}]}` sorted by |shap| desc. 404 with `{"detail": ...}` if not found.

### GET /api/shap/dates?factor=CNH_ATM_PC1
`{"dates": ["2020-01-01", ...]}` — dates with SHAP rows for that factor (for the date picker).

### GET /api/sentiment
All rows of sentiment_daily: `{"rows": [{"date", "doc_id", "label", "sent_score", "confidence"}]}` (may be empty).

### GET /api/alerts
`{"alerts": [{"date": "2025-12-02", "filename": "risk_alert_2025-12-02.txt"}]}` from listing `NLP Outputs/`.

### GET /api/alerts/{date}
`{"date": "...", "text": "full alert text"}` or 404.

### POST /api/chat
Request: `{"messages": [{"role": "user"|"assistant", "content": "..."}]}` (full history, last item is the new user message).
Response: `{"reply": "assistant text"}`.
Implementation: reuse the tool-calling loop from `finai/app/chat.py` (`run_chat_turn`) — DeepSeek via `finai/pipeline/llm_client.py`, key from env `DEEPSEEK_API_KEY`. If key missing, return 503 with `{"detail": "DEEPSEEK_API_KEY not configured"}`.

## Frontend expectations
- Dev server: Vite on :5173, proxy `/api` → `http://localhost:8000`.
- Pages: **Terminal** (dashboard: factor picker, vol line chart realized/HAR-X/GBM, SHAP horizontal bar chart with model toggle + date picker, eval table), **Risk Alerts** (list + reader), **Chat** (persistent panel or page).
- Theme: dark (#111315 bg, #1B1E21 panels), UBS red #E60000 accents, white text; top bar shows "Beyond the Smile" with subtitle "UBS Fin AI Bootcamp".
