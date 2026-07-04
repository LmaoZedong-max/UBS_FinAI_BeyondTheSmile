"""Beyond the Smile — UBS Fin AI Bootcamp API

FastAPI backend implementing the API contract at docs/API_CONTRACT.md.
Run from the project root so that `finai` imports resolve:

    cd /path/to/14-01-UBS_SH
    ~/.venvs/beyond-the-smile/bin/python -m uvicorn backend.main:app --port 8000
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Beyond the Smile — UBS Fin AI Bootcamp API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str


class FactorsResponse(BaseModel):
    factors: list[str]


class ForecastRow(BaseModel):
    date: str
    rv_realized: float | None
    rv_forecast_harx: float | None
    rv_forecast_gbm: float | None


class ForecastsResponse(BaseModel):
    factor: str
    series: list[ForecastRow]


class EvalRow(BaseModel):
    model: str
    n_oos: int
    QLIKE_daily: float | None
    Corr_daily: float | None
    QLIKE_week: float | None
    Corr_week: float | None


class EvalResponse(BaseModel):
    factor: str
    rows: list[EvalRow]


class ShapDriver(BaseModel):
    feature: str
    shap_value: float


class ShapResponse(BaseModel):
    date: str
    factor: str
    model: str
    drivers: list[ShapDriver]


class ShapDatesResponse(BaseModel):
    dates: list[str]


class SentimentRow(BaseModel):
    date: str
    doc_id: Any
    label: Any
    sent_score: Any
    confidence: Any


class SentimentResponse(BaseModel):
    rows: list[SentimentRow]


class AlertItem(BaseModel):
    date: str
    filename: str


class AlertsResponse(BaseModel):
    alerts: list[AlertItem]


class AlertDetailResponse(BaseModel):
    date: str
    text: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class ChatResponse(BaseModel):
    reply: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NLP_OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "NLP Outputs"
_DATE_RE = re.compile(r"risk_alert_(\d{4}-\d{2}-\d{2})\.txt$")


def _import_da():
    """Lazy import so the module is importable even if finai isn't on sys.path yet."""
    from finai.app import data_access as da  # noqa: PLC0415
    return da


def _float_or_none(val) -> float | None:
    import math
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/api/factors", response_model=FactorsResponse)
def factors() -> FactorsResponse:
    da = _import_da()
    return FactorsResponse(factors=da.available_factors())


@app.get("/api/forecasts", response_model=ForecastsResponse)
def forecasts(
    factor: str = Query(...),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
) -> ForecastsResponse:
    import pandas as pd

    da = _import_da()
    df = da.daily_outputs()

    if df.empty:
        return ForecastsResponse(factor=factor, series=[])

    sub = df[df["factor"] == factor].copy()

    if start:
        sub = sub[sub["date"] >= pd.Timestamp(start)]
    if end:
        sub = sub[sub["date"] <= pd.Timestamp(end)]

    sub = sub.sort_values("date")

    series = [
        ForecastRow(
            date=str(row["date"].date()) if hasattr(row["date"], "date") else str(row["date"])[:10],
            rv_realized=_float_or_none(row.get("rv_realized")),
            rv_forecast_harx=_float_or_none(row.get("rv_forecast_harx")),
            rv_forecast_gbm=_float_or_none(row.get("rv_forecast_gbm")),
        )
        for _, row in sub.iterrows()
    ]
    return ForecastsResponse(factor=factor, series=series)


@app.get("/api/eval", response_model=EvalResponse)
def eval_metrics(factor: str = Query(...)) -> EvalResponse:
    da = _import_da()
    df = da.model_eval()

    if df.empty:
        return EvalResponse(factor=factor, rows=[])

    sub = df[df["factor"] == factor]

    rows = [
        EvalRow(
            model=str(row["model"]),
            n_oos=int(row["n_oos"]),
            QLIKE_daily=_float_or_none(row.get("QLIKE_daily")),
            Corr_daily=_float_or_none(row.get("Corr_daily")),
            QLIKE_week=_float_or_none(row.get("QLIKE_week")),
            Corr_week=_float_or_none(row.get("Corr_week")),
        )
        for _, row in sub.iterrows()
    ]
    return EvalResponse(factor=factor, rows=rows)


@app.get("/api/shap/dates", response_model=ShapDatesResponse)
def shap_dates(factor: str = Query(...)) -> ShapDatesResponse:
    da = _import_da()
    df = da.shap_top_drivers()

    if df.empty:
        return ShapDatesResponse(dates=[])

    sub = df[df["factor"] == factor]
    dates = sorted(
        {
            str(d.date()) if hasattr(d, "date") else str(d)[:10]
            for d in sub["date"]
        }
    )
    return ShapDatesResponse(dates=dates)


@app.get("/api/shap", response_model=ShapResponse)
def shap(
    factor: str = Query(...),
    date: str = Query(...),
    model: str = Query(...),
    k: int = Query(default=8),
) -> ShapResponse:
    if model not in {"HAR-X", "GBM"}:
        raise HTTPException(status_code=422, detail="model must be 'HAR-X' or 'GBM'")

    da = _import_da()
    result = da.get_shap_drivers(date=date, factor=factor, model=model, k=k)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return ShapResponse(
        date=result["date"],
        factor=result["factor"],
        model=result["model"],
        drivers=[ShapDriver(**d) for d in result["drivers"]],
    )


@app.get("/api/sentiment", response_model=SentimentResponse)
def sentiment() -> SentimentResponse:
    da = _import_da()
    df = da.sentiment_daily()

    if df.empty:
        return SentimentResponse(rows=[])

    import pandas as pd

    rows: list[SentimentRow] = []
    for idx, row in df.iterrows():
        # Index may be the date
        try:
            date_str = str(pd.Timestamp(idx).date())
        except Exception:
            date_str = str(idx)[:10]

        rows.append(
            SentimentRow(
                date=date_str,
                doc_id=row.get("doc_id") if "doc_id" in df.columns else None,
                label=row.get("label") if "label" in df.columns else None,
                sent_score=_float_or_none(row.get("sent_score")) if "sent_score" in df.columns else None,
                confidence=_float_or_none(row.get("confidence")) if "confidence" in df.columns else None,
            )
        )
    return SentimentResponse(rows=rows)


@app.get("/api/alerts", response_model=AlertsResponse)
def alerts() -> AlertsResponse:
    da = _import_da()
    files = da.list_alert_files()
    items: list[AlertItem] = []
    for p in files:
        m = _DATE_RE.search(p.name)
        if m:
            items.append(AlertItem(date=m.group(1), filename=p.name))
    return AlertsResponse(alerts=items)


@app.get("/api/alerts/{date}", response_model=AlertDetailResponse)
def alert_detail(date: str) -> AlertDetailResponse:
    da = _import_da()
    result = da.get_alert_text(date)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return AlertDetailResponse(date=result["date"], text=result["text"])


@app.post("/api/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY not configured")

    from finai.app.chat import run_chat_turn  # noqa: PLC0415

    history = [{"role": m.role, "content": m.content} for m in body.messages[:-1]]
    user_message = body.messages[-1].content if body.messages else ""

    try:
        reply = run_chat_turn(history=history, user_message=user_message)
    except RuntimeError as exc:
        # get_client() raises RuntimeError if key missing (race condition guard)
        raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY not configured") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(reply=reply)
