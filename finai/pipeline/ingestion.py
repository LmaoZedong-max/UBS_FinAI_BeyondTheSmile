"""Panel ingestion: mixed-format date parsing and column taxonomy.

Ported unchanged from FX_IV_VolBacktest.ipynb / SentimentReport.ipynb so the
rest of the pipeline behaves identically to the original notebooks.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

TENORS = ["1M", "3M", "6M", "1Y"]
SMILES = ["ATM", "25DRR", "25DBF", "10DRR", "10DBF"]

IS_END = pd.Timestamp("2019-12-31")
OOS_START = pd.Timestamp("2020-01-01")


def parse_mixed_date(x) -> pd.Timestamp:
    dt = pd.to_datetime(x, errors="coerce")
    if not pd.isna(dt):
        return dt
    try:
        xf = float(x)
        if 30000 < xf < 50000:
            return pd.to_datetime("1899-12-30") + pd.to_timedelta(xf, unit="D")
    except Exception:
        pass
    return pd.NaT


def _taxonomy(df: pd.DataFrame) -> dict:
    surface_cols = [
        c for c in df.columns
        if re.match(r"^(CNY|CNH)_(1M|3M|6M|1Y)_(ATM|25DRR|25DBF|10DRR|10DBF)$", str(c))
    ]
    spot_cols = [
        c for c in [
            "CNY_SPOT", "CNH_SPOT",
            "CNY_CNY_PBOC_FIXING", "CNH_CNH_PBOC_FIXING",
            "FEAT_CNY_Spot_Fix_Dev",
        ]
        if c in df.columns
    ]
    driver_level_cols = [c for c in df.columns if str(c).startswith(("DRV_", "POL_", "CN_", "US_", "OTH_"))]
    return_cols = [c for c in df.columns if str(c).startswith("RET_")]
    change_cols = [c for c in df.columns if str(c).startswith("CHG_")]
    return dict(
        SURFACE_COLS=surface_cols,
        SPOT_COLS=spot_cols,
        DRIVER_LEVEL_COLS=driver_level_cols,
        RETURN_COLS=return_cols,
        CHANGE_COLS=change_cols,
    )


def load_panel0(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load FINAL_Data.csv (or an .xlsx export) into a clean, numeric, date-indexed panel.

    Returns (panel0, taxonomy) where taxonomy holds the column-name buckets used
    downstream for PCA block construction and macro-exogenous feature selection.
    """
    path = Path(path)
    suf = path.suffix.lower()

    if suf in (".xlsx", ".xls"):
        xls = pd.ExcelFile(path)
        blocks = []
        for sh in xls.sheet_names:
            sheet_df = pd.read_excel(path, sheet_name=sh)
            if sheet_df.shape[1] < 2:
                continue
            date_col = None
            for c in sheet_df.columns:
                if str(c).strip().lower() in ("date", "dates", "time", "datetime"):
                    date_col = c
                    break
            if date_col is None:
                date_col = sheet_df.columns[0]

            sheet_df[date_col] = sheet_df[date_col].apply(parse_mixed_date)
            sheet_df = (
                sheet_df.dropna(subset=[date_col])
                .rename(columns={date_col: "Date"})
                .set_index("Date")
                .sort_index()
            )
            sheet_df = sheet_df.loc[~sheet_df.index.duplicated(keep="last")]
            for c in sheet_df.columns:
                sheet_df[c] = pd.to_numeric(sheet_df[c], errors="coerce")
            sheet_df = sheet_df.dropna(axis=1, how="all")
            if sheet_df.shape[1] > 0:
                blocks.append(sheet_df)

        if not blocks:
            raise ValueError(f"No usable sheets found in {path}")

        df = pd.concat(blocks, axis=1).sort_index()
        df = df.loc[~df.index.duplicated(keep="last")]

    elif suf == ".csv":
        df_raw = pd.read_csv(path, low_memory=False)
        if "Date" not in df_raw.columns:
            raise ValueError("CSV must have a 'Date' column.")
        df_raw["Date_parsed"] = df_raw["Date"].apply(parse_mixed_date)
        df = (
            df_raw.dropna(subset=["Date_parsed"])
            .drop(columns=["Date"])
            .rename(columns={"Date_parsed": "Date"})
            .sort_values("Date")
            .set_index("Date")
        )
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing

    tax = _taxonomy(df)
    keep_cols = (
        tax["SURFACE_COLS"] + tax["SPOT_COLS"] + tax["DRIVER_LEVEL_COLS"]
        + tax["RETURN_COLS"] + tax["CHANGE_COLS"]
    )
    panel = df[keep_cols].copy()
    panel = panel.dropna(subset=tax["SURFACE_COLS"], how="all")
    panel = panel.apply(pd.to_numeric, errors="coerce").sort_index()

    return panel.copy(), tax
