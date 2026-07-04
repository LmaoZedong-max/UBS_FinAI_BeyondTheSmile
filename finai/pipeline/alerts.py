"""LLM-generated 'UBS Professional Risk Alert' reports from HAR-X/GBM driver
attribution + FinBERT sentiment, grounded strictly in a structured payload.

Ported from SentimentReport.ipynb; the OpenAI Responses API call is replaced
with the shared DeepSeek chat-completions client.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .llm_client import DEFAULT_MODEL, get_client

ALERT_SYSTEM_PROMPT = (
    "You are a UBS FX risk analyst. Write a concise 'Professional Risk Alert' for "
    "USD/CNH using ONLY the structured data and news text provided. Do not invent "
    "numbers or facts not present in the input. Structure the report with these "
    "sections: 1. Executive Summary, 2. Cross-Asset & Volatility Narrative, "
    "3. Positioning Considerations (Illustrative). Be precise about numbers and "
    "cite the named macro drivers explicitly."
)


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)


def load_cleaned_news_text(clean_dir: Path, doc_id: str) -> str:
    clean_dir = Path(clean_dir)
    p = clean_dir / f"{doc_id}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8", errors="ignore").strip()
    hits = list(clean_dir.glob(f"{doc_id}*.txt"))
    if hits:
        return hits[0].read_text(encoding="utf-8", errors="ignore").strip()
    return "N/A"


def build_payload_from_row(row: pd.Series, news_text: str) -> dict:
    drivers = []
    for k in range(1, 6):
        d = row.get(f"driver_{k}")
        c = row.get(f"contrib_{k}")
        if pd.notna(d) and pd.notna(c):
            drivers.append({"driver": str(d), "contribution_to_pred_logRV": float(c)})

    return {
        "as_of_date": str(pd.to_datetime(row["panel_date"]).date()),
        "pair": "USD/CNH",
        "market_regime_inputs": {
            "sum_macro_contrib": float(row.get("sum_macro_contrib")) if pd.notna(row.get("sum_macro_contrib")) else "N/A",
        },
        "iv_factor": {"factor_name": str(row.get("factor", "N/A"))},
        "sentiment": {
            "label": str(row.get("label", "N/A")),
            "confidence": float(row.get("confidence")) if pd.notna(row.get("confidence")) else "N/A",
            "sent_score": float(row.get("sent_score")) if pd.notna(row.get("sent_score")) else "N/A",
            "sent_signal": int(row.get("sent_signal")) if pd.notna(row.get("sent_signal")) else "N/A",
        },
        "macro_drivers_top5": drivers,
        "news_cleaned_text": news_text if news_text else "N/A",
        "meta": {
            "doc_id": str(row.get("doc_id", "N/A")),
            "news_date": str(pd.to_datetime(row["news_date"]).date()) if pd.notna(row.get("news_date")) else "N/A",
        },
    }


def generate_risk_alert(report_payload: dict, model: str = DEFAULT_MODEL) -> str:
    client = get_client()
    user_prompt = (
        "Write the report using this data (do not add anything outside it):\n\n"
        + json.dumps(report_payload, ensure_ascii=False, indent=2)
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ALERT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=900,
    )
    return resp.choices[0].message.content.strip()


def generate_alerts_for_days(news_top5: pd.DataFrame, clean_dir: Path, outputs_dir: Path, model: str = DEFAULT_MODEL) -> list[Path]:
    """For each unique panel_date in news_top5, generate one risk-alert text file
    covering all news docs attributed to that day."""
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    df = news_top5.copy()
    df["news_date"] = pd.to_datetime(df["news_date"], errors="coerce")
    df["panel_date"] = pd.to_datetime(df["panel_date"], errors="coerce")

    written = []
    for day in sorted(df["panel_date"].dropna().dt.normalize().unique()):
        day = pd.Timestamp(day)
        day_df = df[df["panel_date"].dt.normalize() == day]
        if day_df.empty:
            continue

        reports = []
        for _, r in day_df.iterrows():
            doc_id = str(r["doc_id"])
            news_text = load_cleaned_news_text(clean_dir, doc_id)
            payload = build_payload_from_row(r, news_text)
            txt = generate_risk_alert(payload, model=model)
            header = (
                f"\n\n{'=' * 90}\n"
                f"DOC: {doc_id} | news_date={pd.to_datetime(r['news_date']).date()} | panel_date={pd.to_datetime(r['panel_date']).date()}\n"
                f"{'=' * 90}\n"
            )
            reports.append(header + txt)

        out_text = f"UBS Professional Risk Alerts - Panel Date: {day.date()}\n" + "\n".join(reports)
        out_file = outputs_dir / f"risk_alert_{safe_filename(str(day.date()))}.txt"
        out_file.write_text(out_text, encoding="utf-8")
        written.append(out_file)

    return written


def build_news_top5(contrib_macro: pd.DataFrame, sent_daily2: pd.DataFrame, factor: str, top_k: int = 5) -> pd.DataFrame:
    """For each aligned news day, pick the top-k |contribution| macro drivers
    from contrib_macro at that day's panel_date."""
    rows = []
    for news_dt, row in sent_daily2.iterrows():
        t = row["panel_date"]
        if t not in contrib_macro.index:
            continue
        c = contrib_macro.loc[t].astype(float).replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
        top_idx = c.abs().sort_values(ascending=False).head(top_k).index
        top = c.reindex(top_idx)

        out = {
            "news_date": news_dt,
            "panel_date": t,
            "doc_id": row["doc_id"],
            "label": row["label"],
            "confidence": float(row["confidence"]),
            "sent_score": float(row["sent_score"]),
            "sent_signal": int(row["sent_signal"]),
            "factor": factor,
            "sum_macro_contrib": float(c.sum()),
        }
        for i, (k, v) in enumerate(top.items(), 1):
            out[f"driver_{i}"] = k
            out[f"contrib_{i}"] = float(v)
        rows.append(out)

    return pd.DataFrame(rows).sort_values(["news_date"]).reset_index(drop=True)
