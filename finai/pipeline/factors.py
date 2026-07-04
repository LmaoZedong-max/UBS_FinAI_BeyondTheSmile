"""Leakage-safe PCA factor extraction on CNY/CNH implied-vol surface blocks.

Ported from FX_IV_VolBacktest.ipynb (section 2-3) / SentimentReport.ipynb
(build_factors_safe). PCA is always fit on the in-sample window only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .ingestion import TENORS

EPS = 1e-12


def dlog(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return np.log(s).replace([np.inf, -np.inf], np.nan).diff()


def dlevel(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.diff()


def surface_block_cols(df: pd.DataFrame, mkt: str, smile: str) -> list[str]:
    cols = []
    for t in TENORS:
        pat = re.compile(rf"^{mkt}_{t}_{smile}$")
        cols += [c for c in df.columns if pat.match(str(c))]
    return cols


def first_valid_date(df: pd.DataFrame, cols: list[str]) -> pd.Timestamp | None:
    if not cols:
        return None
    ok = df[cols].notna().any(axis=1)
    return ok.index[ok.values][0] if ok.any() else None


def make_pca_input(df: pd.DataFrame, cols: list[str], mode: str) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=df.index)
    out = {c: (dlog(df[c]) if mode == "dlog" else dlevel(df[c])) for c in cols}
    return pd.DataFrame(out, index=df.index)


@dataclass
class PCAModel:
    cols: list[str]
    mode: str
    scaler: StandardScaler
    pca: PCA
    explained_var: np.ndarray
    loadings: pd.DataFrame


def fit_pca_block(df_is: pd.DataFrame, cols: list[str], mode: str, n_components: int) -> PCAModel:
    if not cols:
        raise ValueError("Empty PCA block.")
    X = make_pca_input(df_is, cols, mode=mode).dropna(how="any")
    if X.shape[0] < max(80, 10 * n_components):
        raise ValueError(f"Too few rows for PCA: {X.shape[0]}")

    sc = StandardScaler()
    Xz = sc.fit_transform(X.values)
    pca = PCA(n_components=n_components, random_state=0).fit(Xz)

    pc_cols = [f"PC{i + 1}" for i in range(n_components)]
    load = pd.DataFrame(pca.components_.T, index=X.columns, columns=pc_cols)
    return PCAModel(list(X.columns), mode, sc, pca, pca.explained_variance_ratio_, load)


def apply_pca_block(df: pd.DataFrame, mdl: PCAModel, prefix: str) -> pd.DataFrame:
    X = make_pca_input(df, mdl.cols, mode=mdl.mode).sort_index().ffill()
    X = X.fillna(X.median(numeric_only=True))
    Xz = mdl.scaler.transform(X.values)
    scores = mdl.pca.transform(Xz)
    cols = [f"{prefix}_PC{i + 1}" for i in range(mdl.pca.n_components_)]
    return pd.DataFrame(scores, index=X.index, columns=cols)


def _try_join_pca(pf, panel0, df_is, cols, mode, n_components, prefix):
    if not cols:
        return pf
    try:
        mdl = fit_pca_block(df_is, cols, mode=mode, n_components=n_components)
        return pf.join(apply_pca_block(panel0, mdl, prefix))
    except Exception:
        return pf


def build_factors_safe(panel0: pd.DataFrame, is_end: pd.Timestamp) -> tuple[pd.DataFrame, list[str], pd.Timestamp | None]:
    """Build all CNY/CNH PCA factor blocks (ATM, 25RR, 25BF, 10RR, 10BF).

    Returns (panel_factors, factor_cols, cnh_start).
    """
    panel_is = panel0.loc[:is_end].copy()
    pf = panel0.copy()

    blocks_cny = {smile: surface_block_cols(panel0, "CNY", smile) for smile in ["ATM", "25DRR", "25DBF", "10DRR", "10DBF"]}
    blocks_cnh = {smile: surface_block_cols(panel0, "CNH", smile) for smile in ["ATM", "25DRR", "25DBF", "10DRR", "10DBF"]}

    cnh_all = sum(blocks_cnh.values(), [])
    cnh_start = first_valid_date(panel0, cnh_all)

    pf = _try_join_pca(pf, panel0, panel_is, blocks_cny["ATM"], "dlog", 2, "CNY_ATM")
    pf = _try_join_pca(pf, panel0, panel_is, blocks_cny["25DRR"], "dlevel", 1, "CNY_25RR")
    pf = _try_join_pca(pf, panel0, panel_is, blocks_cny["25DBF"], "dlevel", 1, "CNY_25BF")
    if len(blocks_cny["10DRR"]) >= 2:
        pf = _try_join_pca(pf, panel0, panel_is, blocks_cny["10DRR"], "dlevel", 1, "CNY_10RR")
    if len(blocks_cny["10DBF"]) >= 2:
        pf = _try_join_pca(pf, panel0, panel_is, blocks_cny["10DBF"], "dlevel", 1, "CNY_10BF")

    if (cnh_start is not None) and (cnh_start <= is_end):
        panel_is_cnh = panel_is.loc[panel_is.index >= cnh_start].copy()
        if len(blocks_cnh["ATM"]) >= 2:
            pf = _try_join_pca(pf, panel0, panel_is_cnh, blocks_cnh["ATM"], "dlog", 2, "CNH_ATM")
        if len(blocks_cnh["25DRR"]) >= 2:
            pf = _try_join_pca(pf, panel0, panel_is_cnh, blocks_cnh["25DRR"], "dlevel", 1, "CNH_25RR")
        if len(blocks_cnh["25DBF"]) >= 2:
            pf = _try_join_pca(pf, panel0, panel_is_cnh, blocks_cnh["25DBF"], "dlevel", 1, "CNH_25BF")
        if len(blocks_cnh["10DRR"]) >= 2:
            pf = _try_join_pca(pf, panel0, panel_is_cnh, blocks_cnh["10DRR"], "dlevel", 1, "CNH_10RR")
        if len(blocks_cnh["10DBF"]) >= 2:
            pf = _try_join_pca(pf, panel0, panel_is_cnh, blocks_cnh["10DBF"], "dlevel", 1, "CNH_10BF")

    factor_cols = [c for c in pf.columns if re.match(r"^(CNY|CNH)_(ATM|25RR|25BF|10RR|10BF)_PC\d+$", str(c))]
    return pf, factor_cols, cnh_start
