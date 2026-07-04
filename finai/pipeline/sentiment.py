"""News cleaning (DeepSeek) + FinBERT sentiment scoring + date alignment.

Ported from SentimentReport.ipynb. The cleaning step originally called the
OpenAI Responses API with a hardcoded key; it now goes through the shared
DeepSeek client (finai.pipeline.llm_client), reading the key from the
DEEPSEEK_API_KEY env var.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .llm_client import DEFAULT_MODEL, get_client

CLEAN_SYSTEM_PROMPT = (
    "You clean financial news text for downstream NLP. "
    "Remove boilerplate, navigation, ads, duplicate lines, disclaimers, copyright lines, "
    "and unrelated website UI fragments. Keep dates, numbers, and FX/macro content. "
    "Do not add facts. Return ONLY the cleaned text, no markdown, no commentary."
)


def clean_with_llm(raw_text: str, model: str = DEFAULT_MODEL, max_retries: int = 5) -> str:
    client = get_client()
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CLEAN_SYSTEM_PROMPT},
                    {"role": "user", "content": raw_text[:120000]},
                ],
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Max retries exceeded.")


def clean_news_dir(raw_dir: Path, clean_dir: Path, model: str = DEFAULT_MODEL) -> dict[str, str]:
    clean_dir.mkdir(parents=True, exist_ok=True)
    news_files = sorted(Path(raw_dir).glob("news*.txt"))
    cleaned = {}
    for f in news_files:
        raw_text = f.read_text(encoding="utf-8")
        cleaned_text = clean_with_llm(raw_text, model=model)
        (clean_dir / f"{f.stem}.txt").write_text(cleaned_text, encoding="utf-8")
        cleaned[f.stem] = cleaned_text
    return cleaned


# --------------------------------------------------------------------------- #
# FinBERT scoring
# --------------------------------------------------------------------------- #
_PIPE = None


def _get_finbert_pipe():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    _PIPE = pipeline(
        "text-classification", model=model, tokenizer=tokenizer,
        return_all_scores=True, device=-1, truncation=True,
    )
    return _PIPE


def chunk_text_by_words(text: str, max_words: int = 220):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])


def finbert_doc_scores(text: str) -> dict:
    pipe = _get_finbert_pipe()
    probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    n = 0
    for chunk in chunk_text_by_words(text, max_words=220):
        out = pipe(chunk)[0]
        d = {x["label"].lower(): float(x["score"]) for x in out}
        for k in probs:
            probs[k] += d.get(k, 0.0)
        n += 1
    if n == 0:
        return {"positive": None, "negative": None, "neutral": None, "label": None, "confidence": None, "sent_score": None}

    for k in probs:
        probs[k] /= n

    label = max(probs, key=probs.get)
    conf = probs[label]
    sent_score = probs["positive"] - probs["negative"]
    return {**probs, "label": label, "confidence": conf, "sent_score": sent_score}


def score_news_dir(clean_dir: Path) -> pd.DataFrame:
    clean_files = sorted(Path(clean_dir).glob("news*.txt"))
    rows = []
    for f in clean_files:
        doc_id = f.stem
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        sc = finbert_doc_scores(text)
        rows.append({"doc_id": doc_id, **sc, "n_chars": len(text)})
    return pd.DataFrame(rows).sort_values("doc_id").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Date inference + alignment to panel dates
# --------------------------------------------------------------------------- #
def infer_date_from_doc_id(doc_id: str, fallback_year: int = 2025) -> pd.Timestamp:
    s = str(doc_id)
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", s)
    if m:
        return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    m2 = re.search(r"(?:^|_)(\d{4})(?:$|_)", s)
    if m2:
        mmdd = m2.group(1)
        mm, dd = int(mmdd[:2]), int(mmdd[2:])
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return pd.Timestamp(f"{fallback_year}-{mm:02d}-{dd:02d}")
    return pd.NaT


def build_sent_daily(df_finbert: pd.DataFrame, clean_files: list[Path], fallback_year: int = 2025, signal_threshold: float = 0.30) -> pd.DataFrame:
    sent_daily = df_finbert.copy()
    sent_daily["news_date"] = sent_daily["doc_id"].apply(lambda x: infer_date_from_doc_id(x, fallback_year=fallback_year))

    if sent_daily["news_date"].isna().any():
        mtime = {f.stem: pd.Timestamp(f.stat().st_mtime, unit="s").normalize() for f in clean_files}
        miss = sent_daily["news_date"].isna()
        sent_daily.loc[miss, "news_date"] = sent_daily.loc[miss, "doc_id"].map(mtime)

    sent_daily["sent_signal"] = (sent_daily["sent_score"].abs() > signal_threshold).astype(int)
    return sent_daily.dropna(subset=["news_date"]).set_index("news_date").sort_index()


def align_news_to_panel_date(news_dates: pd.DatetimeIndex, panel_index: pd.DatetimeIndex) -> pd.Series:
    tmp = pd.DataFrame({"news_date": pd.to_datetime(news_dates)}).sort_values("news_date")
    pan = pd.DataFrame({"panel_date": pd.to_datetime(panel_index).sort_values()})
    out = pd.merge_asof(tmp, pan, left_on="news_date", right_on="panel_date", direction="backward")
    return out.set_index("news_date")["panel_date"]
