"""Thin, cached read-only access to finai/store/*.parquet.

Used by both the Streamlit pages and the chat tool-calling layer, so the two
surfaces always agree on what "the data" is.
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import pandas as pd

STORE_DIR = Path(__file__).resolve().parents[1] / "store"


def _read(name: str) -> pd.DataFrame:
    path = STORE_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def daily_outputs() -> pd.DataFrame:
    return _read("daily_outputs.parquet")


@lru_cache(maxsize=1)
def model_eval() -> pd.DataFrame:
    return _read("model_eval.parquet")


@lru_cache(maxsize=1)
def shap_values() -> pd.DataFrame:
    return _read("shap_values.parquet")


@lru_cache(maxsize=1)
def shap_top_drivers() -> pd.DataFrame:
    return _read("shap_top_drivers.parquet")


@lru_cache(maxsize=1)
def sentiment_daily() -> pd.DataFrame:
    return _read("sentiment_daily.parquet")


@lru_cache(maxsize=1)
def news_top5() -> pd.DataFrame:
    return _read("news_top5.parquet")


def available_factors() -> list[str]:
    df = daily_outputs()
    if df.empty:
        return []
    return sorted(df["factor"].unique().tolist())


def get_forecast(date: str, factor: str) -> dict:
    df = daily_outputs()
    if df.empty:
        return {"error": "no pipeline output found -- run finai.pipeline.run_pipeline first"}
    row = df[(df["factor"] == factor) & (df["date"].astype(str) == str(date))]
    if row.empty:
        return {"error": f"no forecast for factor={factor} date={date}"}
    r = row.iloc[0]
    return {
        "date": str(pd.Timestamp(r["date"]).date()),
        "factor": factor,
        "rv_realized": float(r["rv_realized"]) if pd.notna(r["rv_realized"]) else None,
        "rv_forecast_harx": float(r["rv_forecast_harx"]) if pd.notna(r["rv_forecast_harx"]) else None,
        "rv_forecast_gbm": float(r["rv_forecast_gbm"]) if pd.notna(r["rv_forecast_gbm"]) else None,
    }


def get_shap_drivers(date: str, factor: str, model: str = "HAR-X", k: int = 5) -> dict:
    df = shap_top_drivers()
    if df.empty:
        return {"error": "no SHAP output found -- run finai.pipeline.run_pipeline first"}
    sub = df[(df["factor"] == factor) & (df["model"] == model) & (df["date"].astype(str) == str(date))]
    if sub.empty:
        return {"error": f"no SHAP drivers for factor={factor} model={model} date={date}"}
    sub = sub.sort_values("shap_value", key=lambda s: s.abs(), ascending=False).head(k)
    return {"date": str(date), "factor": factor, "model": model,
            "drivers": [{"feature": row["feature"], "shap_value": float(row["shap_value"])} for _, row in sub.iterrows()]}


def get_sentiment(date: str) -> dict:
    df = sentiment_daily()
    if df.empty:
        return {"error": "no sentiment output found"}
    sub = df[df.index.astype(str) == str(date)]
    if sub.empty:
        return {"error": f"no sentiment for date={date}"}
    r = sub.iloc[0]
    return {"date": str(date), "label": r.get("label"), "sent_score": float(r.get("sent_score", float("nan"))),
            "confidence": float(r.get("confidence", float("nan"))), "doc_id": r.get("doc_id")}


def get_alert_text(date: str) -> dict:
    outputs_dir = Path(__file__).resolve().parents[2] / "NLP Outputs"
    p = outputs_dir / f"risk_alert_{date}.txt"
    if not p.exists():
        return {"error": f"no risk alert file for date={date}"}
    return {"date": date, "text": p.read_text(encoding="utf-8", errors="ignore")}


def list_alert_files() -> list[Path]:
    outputs_dir = Path(__file__).resolve().parents[2] / "NLP Outputs"
    if not outputs_dir.exists():
        return []
    return sorted(outputs_dir.glob("risk_alert_*.txt"))
