"""Mean-model de-meaning (AR(1)) and standardisation, plus shared vol-model utils.

Ported from FX_IV_VolBacktest.ipynb sections 5-6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

EPS = 1e-12


def as_clean_series(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s = s.sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("Series index must be DatetimeIndex.")
    return s[~s.index.duplicated(keep="last")]


def rv_proxy(r: pd.Series, window: int) -> pd.Series:
    r2 = r.astype(float) ** 2
    return r2.rolling(window, min_periods=window).mean()


def logrv_proxy(r: pd.Series, window: int) -> pd.Series:
    return np.log(rv_proxy(r, window).clip(EPS))


def qlike(y: pd.Series, yhat: pd.Series) -> float:
    s = yhat.reindex(y.index).astype(float).clip(lower=EPS)
    y = y.astype(float)
    return float(np.nanmean(np.log(s) + y / s))


def corr_series(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 5:
        return np.nan
    return float(df.corr().iloc[0, 1])


def corr_logrv(y: pd.Series, yhat: pd.Series) -> float:
    a = np.log(y.astype(float).clip(EPS))
    f = np.log(yhat.reindex(a.index).astype(float).clip(EPS))
    return corr_series(a, f)


def make_ar1_residuals(panel_factors: pd.DataFrame, factor_cols: list[str], is_end: pd.Timestamp) -> dict[str, pd.Series]:
    resid_map = {}
    for c in factor_cols:
        s = panel_factors[c].dropna().astype(float)
        s_is = s.loc[:is_end]
        if len(s_is) < 300:
            continue
        y = s_is.iloc[1:].values
        X = sm.add_constant(s_is.iloc[:-1].values)
        fit = sm.OLS(y, X).fit()
        c0, phi = float(fit.params[0]), float(fit.params[1])
        e = (s - (c0 + phi * s.shift(1))).dropna()
        resid_map[c] = e.rename(f"{c}_resid_AR1")
    return resid_map


def standardise_resid(resid: pd.Series, is_end: pd.Timestamp) -> tuple[pd.Series, float]:
    r = as_clean_series(resid).astype(float)
    sd = float(r.loc[:is_end].std(ddof=0))
    if (not np.isfinite(sd)) or sd <= 0:
        sd = 1.0
    return (r / sd).rename(f"{resid.name}_std"), sd
