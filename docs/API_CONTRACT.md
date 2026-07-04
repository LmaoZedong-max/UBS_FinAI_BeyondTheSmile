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

## v2 additions

### GET /api/shap/timeseries?factor=CNH_ATM_PC1&model=HAR-X&top_k=6&freq=M
Monthly (freq=M) mean SHAP contribution per feature, macro drivers only (exclude
log_rv_d/log_rv_w/log_rv_m core terms). Keep the top_k features by mean |value|
across the window; sum the remainder into "Other".
```json
{"factor": "...", "model": "HAR-X", "freq": "M",
 "features": ["CHG_DRV_VIX_bps", ..., "Other"],
 "series": [{"period": "2020-01", "values": {"CHG_DRV_VIX_bps": 0.01, ..., "Other": -0.003}}, ...]}
```
422 on bad model; empty series if no data.

### Frontend v2
- Terminal page gains an "Attribution" panel: signed stacked monthly bar chart of
  /api/shap/timeseries (recharts stacked BarChart, positive stacks above zero,
  negative below; consistent color per feature; legend).
- Alerts page gains a "News Sentiment" strip above the alert list: one card per
  /api/sentiment row (date, doc_id, label badge — red negative / grey neutral /
  green positive — score to 2dp, confidence to 2dp).

## v3 additions

### POST /api/chat/stream
Same request body as /api/chat. Response: Server-Sent Events (`text/event-stream`).
Events, in order:
- zero or more `event: tool` — `data: {"name": "get_shap_drivers", "status": "called"}` (one per tool invocation, lets the UI show "consulting store…")
- one or more `event: delta` — `data: {"text": "..."}` incremental assistant text
- terminal `event: done` — `data: {}`
- on error: `event: error` — `data: {"detail": "..."}` then close. Missing key → single error event with detail "DEEPSEEK_API_KEY not configured".
Implementation: run the existing tool-hop loop non-streamed; stream only the final completion (stream=True on the last DeepSeek call, forwarding content deltas).

### Frontend v3
- Chat page uses /api/chat/stream via fetch + ReadableStream (POST body, parse SSE frames manually).
- Assistant bubble renders deltas incrementally; while tool events arrive show a muted "consulting data store…" status line above the bubble.
- On error event render the detail in the existing error style. Keep POST /api/chat code as fallback if stream fails to open.
