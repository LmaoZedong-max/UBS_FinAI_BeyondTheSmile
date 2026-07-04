"""Rolling, no-lookahead volatility models: GARCH(1,1), MIDAS-GARCH, HAR-X.

Ported from FX_IV_VolBacktest.ipynb sections 6-7. Math is unchanged from the
notebook; only reorganised into a module and given a shared `pick_macro_exog_cols`
so gbm_model.py can reuse the exact same design matrix as HAR-X.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

from .mean_models import EPS, as_clean_series, logrv_proxy, qlike, rv_proxy


# --------------------------------------------------------------------------- #
# Macro-exogenous feature selection (shared with GBM model)
# --------------------------------------------------------------------------- #
def pick_macro_exog_cols(df: pd.DataFrame) -> list[str]:
    keep_prefix = (
        "RET_DRV_", "CHG_DRV_",
        "RET_POL_", "CHG_POL_",
        "RET_CN_", "CHG_CN_",
        "RET_US_", "CHG_US_",
        "RET_OTH_", "CHG_OTH_",
        "FEAT_",
        "RET_CNY_SPOT", "RET_CNH_SPOT",
    )
    cols = []
    for c in df.columns.astype(str):
        if c.startswith("CHG_CNH_") or (c.startswith("RET_CNH_") and c != "RET_CNH_SPOT"):
            continue
        if c.startswith("CHG_CNY_") or (c.startswith("RET_CNY_") and c != "RET_CNY_SPOT"):
            continue
        if c.startswith(keep_prefix):
            cols.append(c)
    out, seen = [], set()
    for c in cols:
        if c in df.columns and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def clean_exog(Xe: pd.DataFrame, min_non_nan_frac: float = 0.90) -> pd.DataFrame:
    Xe = Xe.replace([np.inf, -np.inf], np.nan).copy()
    for c in Xe.columns:
        Xe[c] = pd.to_numeric(Xe[c], errors="coerce")

    keep = Xe.notna().mean(axis=0) >= float(min_non_nan_frac)
    Xe = Xe.loc[:, keep]

    retchg = [c for c in Xe.columns if str(c).startswith(("RET_", "CHG_"))]
    other = [c for c in Xe.columns if c not in retchg]
    if other:
        Xe[other] = Xe[other].ffill()
    if retchg:
        Xe[retchg] = Xe[retchg].fillna(0.0)

    nunq = Xe.nunique(dropna=True)
    return Xe.loc[:, nunq > 1]


def harx_design(r: pd.Series, X_exog: pd.DataFrame | None, rv_window: int) -> tuple[pd.DataFrame, pd.Series]:
    r = as_clean_series(r).astype(float)
    r2 = r ** 2
    X_core = pd.DataFrame({
        "log_rv_d": np.log(r2.clip(EPS)),
        "log_rv_w": np.log(r2.rolling(5, min_periods=5).mean().clip(EPS)),
        "log_rv_m": np.log(r2.rolling(22, min_periods=22).mean().clip(EPS)),
    }, index=r.index)

    y_next = logrv_proxy(r, rv_window).shift(-1).rename("y_next")

    X = X_core
    if X_exog is not None:
        Xe = clean_exog(X_exog.reindex(r.index), min_non_nan_frac=0.90)
        X = pd.concat([X, Xe], axis=1)

    df = pd.concat([y_next, X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    X_aligned = df.drop(columns=["y_next"])
    nunq = X_aligned.nunique(dropna=True)
    X_aligned = X_aligned.loc[:, nunq > 1]
    return X_aligned, df["y_next"]


HAR_CORE_COLS = {"log_rv_d", "log_rv_w", "log_rv_m"}


@dataclass
class HARXWindowFit:
    refit_date: pd.Timestamp
    beta: pd.Series  # index = ["const"] + feature_names
    feature_names: list[str]
    X_fit: pd.DataFrame  # in-sample design matrix used for this window (SHAP background)
    pred_index: pd.DatetimeIndex


def rolling_harx_fit_windows(
    r: pd.Series,
    X_exog: pd.DataFrame | None,
    oos_start: pd.Timestamp,
    rv_window: int = 5,
    lookback: int = 1260,
    refit_every: int = 21,
    min_fit_rows: int = 400,
) -> tuple[pd.Series, list[HARXWindowFit]]:
    """Rolling HAR-X OLS forecast of log(RV_{t+1}), refit every `refit_every` days
    on a trailing `lookback` window using only data up to t-1. Returns the forecast
    plus per-window fitted coefficients/background data (used both for linear
    attribution and for shap.LinearExplainer)."""
    r = as_clean_series(r)
    X_all, y_all = harx_design(r, X_exog, rv_window)
    idx = r.index

    yhat_next = pd.Series(index=y_all.index, dtype=float, name=f"HARX_logRV_next_w{rv_window}")
    windows: list[HARXWindowFit] = []

    start_i = idx.get_indexer([oos_start], method="bfill")[0]
    i = start_i
    while i < len(idx):
        t_date = idx[i]
        end_fit_date = idx[i - 1] if i > 0 else t_date
        j_date = idx[min(len(idx) - 1, i + refit_every - 1)]

        df_fit = pd.concat([y_all, X_all], axis=1).loc[:end_fit_date].dropna()
        if len(df_fit) < min_fit_rows:
            i += refit_every
            continue
        df_fit = df_fit.iloc[-lookback:] if len(df_fit) > lookback else df_fit

        Y = df_fit.iloc[:, 0]
        Xmat = df_fit.iloc[:, 1:].replace([np.inf, -np.inf], np.nan).dropna()
        Y = Y.reindex(Xmat.index)
        if len(Xmat) < 250:
            i += refit_every
            continue

        Xis = sm.add_constant(Xmat, has_constant="add")
        fit = sm.OLS(Y.values, Xis.values).fit()
        beta = pd.Series(fit.params, index=["const"] + list(Xmat.columns)).fillna(0.0)

        df_pred = X_all.loc[t_date:j_date].replace([np.inf, -np.inf], np.nan).dropna()
        df_pred = df_pred[Xmat.columns]
        if len(df_pred) > 0:
            Xp = sm.add_constant(df_pred, has_constant="add")
            yhat_next.loc[df_pred.index] = fit.predict(Xp.values)
            windows.append(HARXWindowFit(
                refit_date=end_fit_date,
                beta=beta,
                feature_names=list(Xmat.columns),
                X_fit=Xmat,
                pred_index=df_pred.index,
            ))

        i += refit_every

    return yhat_next, windows


def rolling_harx_with_attrib(
    r: pd.Series,
    X_exog: pd.DataFrame | None,
    oos_start: pd.Timestamp,
    rv_window: int = 5,
    lookback: int = 1260,
    refit_every: int = 21,
    min_fit_rows: int = 400,
) -> tuple[pd.Series, pd.DataFrame]:
    """Rolling HAR-X forecast of log(RV_{t+1}) with per-feature linear attribution
    (beta_k * x_k,t at prediction time, uncentered). Thin wrapper around
    rolling_harx_fit_windows for backward compatibility with the notebook API."""
    yhat_next, windows = rolling_harx_fit_windows(
        r, X_exog, oos_start, rv_window=rv_window, lookback=lookback,
        refit_every=refit_every, min_fit_rows=min_fit_rows,
    )
    X_all, _ = harx_design(as_clean_series(r), X_exog, rv_window)
    all_cols = sorted({c for w in windows for c in w.feature_names})
    contrib = pd.DataFrame(index=yhat_next.index, columns=all_cols, dtype=float)
    for w in windows:
        df_pred = X_all.loc[w.pred_index, w.feature_names]
        b = w.beta.reindex(["const"] + w.feature_names).fillna(0.0)
        contrib.loc[df_pred.index, df_pred.columns] = df_pred.mul(b[df_pred.columns], axis=1).values
    return yhat_next, contrib


def harx_logrv_to_sig2hat(yhat_logrv_next: pd.Series) -> pd.Series:
    rv_hat_next = np.exp(yhat_logrv_next).rename("HARX_RV_hat_next")
    return rv_hat_next.shift(1).rename("HAR-X")


# --------------------------------------------------------------------------- #
# GARCH(1,1)
# --------------------------------------------------------------------------- #
@dataclass
class GarchFit:
    omega: float
    alpha: float
    beta: float


def _fit_garch11_params(r_est: pd.Series, maxiter: int = 3000) -> GarchFit:
    y = r_est.astype(float).values
    if len(y) < 400:
        raise ValueError("Too few points to fit GARCH(1,1).")

    def _unpack(z):
        zw, za, zb = z
        omega = np.exp(zw) + 1e-12
        ea, eb = np.exp(za), np.exp(zb)
        denom = 1.0 + ea + eb
        return omega, ea / denom, eb / denom

    def _nll(z):
        omega, alpha, beta = _unpack(z)
        T = len(y)
        sig2 = np.empty(T, dtype=float)
        sig2[0] = np.var(y) + 1e-12
        for t in range(1, T):
            sig2[t] = omega + alpha * (y[t - 1] ** 2) + beta * sig2[t - 1]
            if (not np.isfinite(sig2[t])) or sig2[t] <= 0:
                return 1e50
        ll = -0.5 * (np.log(2 * np.pi) + np.log(sig2) + (y ** 2) / sig2)
        if not np.all(np.isfinite(ll)):
            return 1e50
        return float(-np.sum(ll))

    z0 = np.array([np.log(np.var(y) * 0.1 + 1e-12), np.log(0.05), np.log(0.90)], dtype=float)
    opt = minimize(_nll, z0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10})
    if not opt.success:
        raise RuntimeError(f"GARCH(1,1) fit failed: {opt.message}")
    omega, alpha, beta = _unpack(opt.x)
    return GarchFit(float(omega), float(alpha), float(beta))


def _filter_garch11_one_step(r_seg: pd.Series, fit: GarchFit, init_sig2: float) -> pd.Series:
    y = r_seg.astype(float).values
    sig2_hat = np.empty(len(y), dtype=float)
    sig2_hat[0] = float(init_sig2)
    for t in range(1, len(y)):
        sig2_hat[t] = fit.omega + fit.alpha * (y[t - 1] ** 2) + fit.beta * sig2_hat[t - 1]
        if (not np.isfinite(sig2_hat[t])) or sig2_hat[t] <= 0:
            sig2_hat[t] = EPS
    return pd.Series(sig2_hat, index=r_seg.index, name="GARCH11_sigma2_hat")


def rolling_garch11_forecast(r: pd.Series, oos_start: pd.Timestamp, lookback: int = 1260, refit_every: int = 21) -> pd.Series:
    r = as_clean_series(r).astype(float)
    idx = r.index
    start_i = idx.get_indexer([oos_start], method="bfill")[0]
    sig2_out = pd.Series(index=idx, dtype=float)

    i = start_i
    while i < len(idx):
        end_fit_i = i - 1
        if end_fit_i <= 10:
            i += refit_every
            continue
        fit_start_i = max(0, end_fit_i - lookback + 1)
        r_est = r.iloc[fit_start_i:end_fit_i + 1]
        fit = _fit_garch11_params(r_est)

        j = min(len(idx), i + refit_every)
        r_seg = r.iloc[i:j]
        init_sig2 = float((r_est ** 2).mean())
        seg_sig2 = _filter_garch11_one_step(r_seg, fit, init_sig2=init_sig2)
        sig2_out.loc[seg_sig2.index] = seg_sig2
        i = j

    sig2_out.name = "GARCH(1,1)"
    return sig2_out


# --------------------------------------------------------------------------- #
# MIDAS-GARCH
# --------------------------------------------------------------------------- #
@dataclass
class MidasFit:
    alpha: float
    beta: float
    m: float
    theta: float
    omega2: float
    K: int
    omega1_fixed: float


def _beta_weights(K: int, omega1: float, omega2: float) -> np.ndarray:
    k = np.arange(1, K + 1, dtype=float)
    a, b = float(omega1), float(omega2)
    num = (k / K) ** (a - 1.0) * (1.0 - k / K) ** (b - 1.0)
    num = np.clip(num, 1e-18, np.inf)
    return num / np.sum(num)


def _monthly_rv_from_daily(r: pd.Series, how: str = "sum") -> pd.Series:
    r = as_clean_series(r).astype(float)
    sq = r ** 2
    if how == "sum":
        return sq.resample("ME").sum()
    if how == "mean":
        return sq.resample("ME").mean()
    raise ValueError("how must be 'sum' or 'mean'")


def _build_daily_midas_lags_no_lookahead(daily_idx: pd.DatetimeIndex, monthly_x: pd.Series, K: int) -> pd.DataFrame:
    mx = monthly_x.sort_index().copy()
    X_m = pd.concat({f"x_lag{k}": mx.shift(k) for k in range(1, K + 1)}, axis=1)
    days = pd.DataFrame({"Date": pd.DatetimeIndex(daily_idx).sort_values()})
    months = X_m.reset_index().rename(columns={"index": "Date"}).sort_values("Date")
    return pd.merge_asof(days, months, on="Date", direction="backward", allow_exact_matches=False).set_index("Date")


def _make_midas_design_endogenous(r: pd.Series, end_fit_date: pd.Timestamp, K: int = 12, rv_how: str = "sum") -> pd.DataFrame:
    rv_m = _monthly_rv_from_daily(r, how=rv_how)
    x_m = np.log(rv_m.clip(EPS))
    x_est = x_m.loc[x_m.index <= end_fit_date]
    mu = float(x_est.mean())
    sd = float(x_est.std(ddof=0))
    if (not np.isfinite(sd)) or sd <= 0:
        sd = 1.0
    x_z = (x_m - mu) / sd
    Xlags = _build_daily_midas_lags_no_lookahead(r.index, x_z, K=K)
    Xlags.attrs["mu"] = mu
    Xlags.attrs["sd"] = sd
    return Xlags


def _fit_midas_garch_params(r_est: pd.Series, X_est: pd.DataFrame, K: int, omega1_fixed: float = 1.0, maxiter: int = 6000, n_starts: int = 6) -> MidasFit:
    y = r_est.astype(float).values
    Xmat = X_est.astype(float).values

    def _unpack(z):
        za, zb, zm, zt, zow2 = z
        ea, eb = np.exp(za), np.exp(zb)
        denom = 1.0 + ea + eb
        return ea / denom, eb / denom, zm, zt, 1.01 + np.exp(zow2)

    def _nll(z):
        a, b, m, theta, omega2 = _unpack(z)
        w = _beta_weights(K, omega1_fixed, omega2)
        mid = Xmat @ w
        tau = np.exp(np.clip(m + theta * mid, -50, 50))
        T = len(y)
        g = np.empty(T, dtype=float)
        g[0] = 1.0
        for t in range(1, T):
            tau_tm1 = max(tau[t - 1], EPS)
            g[t] = (1.0 - a - b) + a * (y[t - 1] ** 2 / tau_tm1) + b * g[t - 1]
            if (not np.isfinite(g[t])) or g[t] <= 0:
                return 1e50
        sig2 = np.clip(tau * g, EPS, np.inf)
        ll = -0.5 * (np.log(2 * np.pi) + np.log(sig2) + (y ** 2) / sig2)
        if not np.all(np.isfinite(ll)):
            return 1e50
        return float(-np.sum(ll))

    base = np.array([np.log(0.05), np.log(0.90), np.log(np.var(y) + 1e-8), 0.0, np.log(2.0)], dtype=float)
    rng = np.random.default_rng(0)
    starts = [base] + [base + rng.normal(0.0, 0.8, size=base.size) for _ in range(max(0, n_starts - 1))]

    best = None
    for z0 in starts:
        opt = minimize(_nll, z0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10})
        if best is None or (opt.success and opt.fun < best.fun):
            best = opt

    if (best is None) or (not best.success):
        nm = minimize(_nll, base, method="Nelder-Mead", options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-6})
        if best is None or (nm.success and nm.fun < best.fun):
            best = nm

    if (best is None) or (not best.success):
        raise RuntimeError(f"MIDAS-GARCH fit failed: {best.message if best is not None else 'no result'}")

    a, b, m, theta, omega2 = _unpack(best.x)
    return MidasFit(float(a), float(b), float(m), float(theta), float(omega2), K=int(K), omega1_fixed=float(omega1_fixed))


def _filter_midas_garch_one_step(r_seg: pd.Series, X_seg: pd.DataFrame, fit: MidasFit, init_g: float = 1.0) -> pd.Series:
    y = r_seg.astype(float).values
    Xmat = X_seg.astype(float).values
    w = _beta_weights(fit.K, fit.omega1_fixed, fit.omega2)
    mid = Xmat @ w
    tau = np.exp(np.clip(fit.m + fit.theta * mid, -50, 50))

    g = np.empty(len(y), dtype=float)
    g[0] = float(init_g)
    sig2_hat = np.empty(len(y), dtype=float)
    sig2_hat[0] = max(tau[0] * g[0], EPS)

    for t in range(1, len(y)):
        tau_tm1 = max(tau[t - 1], EPS)
        g[t] = (1.0 - fit.alpha - fit.beta) + fit.alpha * (y[t - 1] ** 2 / tau_tm1) + fit.beta * g[t - 1]
        if (not np.isfinite(g[t])) or g[t] <= 0:
            g[t] = EPS
        sig2_hat[t] = max(tau[t] * g[t], EPS)

    return pd.Series(sig2_hat, index=r_seg.index, name="MIDAS-GARCH_sigma2_hat")


def rolling_midas_garch_forecast(r: pd.Series, oos_start: pd.Timestamp, lookback: int = 1800, refit_every: int = 21, K: int = 12, rv_how: str = "sum", omega1_fixed: float = 1.0) -> pd.Series:
    r = as_clean_series(r).astype(float)
    idx = r.index
    start_i = idx.get_indexer([oos_start], method="bfill")[0]
    sig2_out = pd.Series(index=idx, dtype=float)

    i = start_i
    while i < len(idx):
        end_fit_i = i - 1
        if end_fit_i <= 30:
            i += refit_every
            continue
        fit_start_i = max(0, end_fit_i - lookback + 1)
        r_est = r.iloc[fit_start_i:end_fit_i + 1]
        end_fit_date = idx[end_fit_i]

        X_all = _make_midas_design_endogenous(r, end_fit_date=end_fit_date, K=K, rv_how=rv_how)
        X_est = X_all.loc[r_est.index]
        good_est = X_est.notna().all(axis=1)
        r_est2, X_est2 = r_est.loc[good_est], X_est.loc[good_est]
        if len(r_est2) < 600:
            i += refit_every
            continue

        fit = _fit_midas_garch_params(r_est2, X_est2, K=K, omega1_fixed=omega1_fixed)

        j = min(len(idx), i + refit_every)
        r_seg = r.iloc[i:j]
        X_seg = X_all.loc[r_seg.index]
        good_seg = X_seg.notna().all(axis=1)
        r_seg2, X_seg2 = r_seg.loc[good_seg], X_seg.loc[good_seg]
        if len(r_seg2) > 0:
            seg_sig2 = _filter_midas_garch_one_step(r_seg2, X_seg2, fit, init_g=1.0)
            sig2_out.loc[seg_sig2.index] = seg_sig2
        i = j

    sig2_out.name = "MIDAS-GARCH"
    return sig2_out


# --------------------------------------------------------------------------- #
# Naive baselines + HAR-RV
# --------------------------------------------------------------------------- #
def forecast_uncond(sig2_is: float, idx: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(sig2_is, index=idx, name="UNCOND")


def forecast_ewma(r: pd.Series, lam: float = 0.94) -> pd.Series:
    r = r.astype(float)
    s2 = pd.Series(index=r.index, dtype=float)
    s2.iloc[0] = float(r.var(ddof=0))
    for t in range(1, len(r)):
        s2.iloc[t] = lam * s2.iloc[t - 1] + (1.0 - lam) * (r.iloc[t - 1] ** 2)
    s2.name = f"EWMA_{lam}"
    return s2


def forecast_rollvar(r: pd.Series, window: int = 20) -> pd.Series:
    s2 = (r.astype(float) ** 2).rolling(window, min_periods=window).mean().shift(1)
    s2.name = f"ROLLVAR_{window}"
    return s2


def forecast_har_rv(r: pd.Series, is_end: pd.Timestamp) -> pd.Series:
    r2 = r.astype(float) ** 2
    rv_d, rv_w, rv_m = r2, r2.rolling(5, min_periods=5).mean(), r2.rolling(22, min_periods=22).mean()

    X = pd.concat([np.log(rv_d.clip(1e-18)), np.log(rv_w.clip(1e-18)), np.log(rv_m.clip(1e-18))], axis=1)
    X.columns = ["log_rv_d", "log_rv_w", "log_rv_m"]
    y = np.log(rv_d.shift(-1).clip(1e-18))

    df = pd.concat([y, X], axis=1).dropna()
    df_is = df.loc[:is_end]
    if len(df_is) < 400:
        raise ValueError("Too few IS rows for HAR.")

    Y = df_is.iloc[:, 0].values
    fit = sm.OLS(Y, sm.add_constant(df_is.iloc[:, 1:].values)).fit()
    yhat_log = fit.predict(sm.add_constant(df.iloc[:, 1:].values))
    return pd.Series(np.exp(yhat_log), index=df.index, name="HAR_RV").reindex(r.index)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class VolForecastResult:
    sigma2_hat: pd.Series
    meta: dict


def _is_sd(resid: pd.Series, is_end: pd.Timestamp) -> float:
    r = resid.loc[:is_end].dropna().astype(float)
    sd = float(r.std(ddof=0))
    return sd if (np.isfinite(sd) and sd > 0) else 1.0


def fit_forecast_garch11(resid: pd.Series, is_end: pd.Timestamp, oos_start: pd.Timestamp, maxiter: int = 4000) -> VolForecastResult:
    r = as_clean_series(resid).astype(float)
    r_is = r.loc[:is_end]
    if len(r_is) < 500:
        raise ValueError(f"Too few IS observations for GARCH(1,1): {len(r_is)}")

    fit = _fit_garch11_params(r_is, maxiter=maxiter)

    y_full = r.astype(float).values
    T = len(y_full)
    sig2 = np.empty(T, dtype=float)
    sig2[0] = np.var(r_is.values) + 1e-12
    for t in range(1, T):
        sig2[t] = fit.omega + fit.alpha * (y_full[t - 1] ** 2) + fit.beta * sig2[t - 1]
        if not np.isfinite(sig2[t]) or sig2[t] <= 0:
            sig2[t] = 1e-18

    sigma2_hat = pd.Series(sig2, index=r.index, name="GARCH11_sigma2_hat")
    sigma2_oos = sigma2_hat.loc[sigma2_hat.index >= oos_start].copy()
    return VolForecastResult(sigma2_hat=sigma2_oos.astype(float), meta={"model": "GARCH(1,1)", "params": fit.__dict__})


def eval_one_forecast(r_std: pd.Series, s2_hat: pd.Series, oos_start: pd.Timestamp) -> dict:
    r_std = as_clean_series(r_std)
    y2_d = (r_std ** 2).loc[r_std.index >= oos_start]
    y2_w = rv_proxy(r_std, 5).loc[r_std.index >= oos_start]

    s2_d = s2_hat.reindex(y2_d.index).astype(float).clip(lower=1e-18)
    s2_w = s2_hat.reindex(y2_w.index).astype(float).clip(lower=1e-18)

    return {
        "n_oos": int(len(y2_d)),
        "QLIKE_daily": qlike(y2_d, s2_d),
        "Corr_daily": float(pd.concat([y2_d, s2_d], axis=1).corr().iloc[0, 1]),
        "QLIKE_week": qlike(y2_w.dropna(), s2_w.dropna()),
        "Corr_week": float(pd.concat([y2_w, s2_w], axis=1).dropna().corr().iloc[0, 1]),
    }


def eval_all_models_for_factor(
    resid_raw: pd.Series,
    is_end: pd.Timestamp,
    oos_start: pd.Timestamp,
    extra_forecasts: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Baselines + GARCH(1,1) + MIDAS-GARCH, optionally + any extra model forecasts
    (e.g. the HAR-X / GBM sigma2 series produced by run_rolling_vol_suite / gbm_model)."""
    from .mean_models import standardise_resid

    r_std, _ = standardise_resid(as_clean_series(resid_raw), is_end)

    sig2_is = float((r_std.loc[:is_end] ** 2).mean())
    preds = {
        "UNCOND": forecast_uncond(sig2_is, r_std.index),
        "EWMA_0.94": forecast_ewma(r_std, 0.94),
        "ROLLVAR_20": forecast_rollvar(r_std, 20),
        "HAR_RV": forecast_har_rv(r_std, is_end),
    }

    g = fit_forecast_garch11(r_std, is_end=is_end, oos_start=oos_start)
    preds["GARCH(1,1)"] = g.sigma2_hat.reindex(r_std.index)

    if extra_forecasts:
        for name, s2 in extra_forecasts.items():
            preds[name] = s2.reindex(r_std.index)

    rows = []
    for name, s2 in preds.items():
        met = eval_one_forecast(r_std, s2, oos_start)
        rows.append({"model": name, **met})

    return pd.DataFrame(rows).sort_values("QLIKE_daily").reset_index(drop=True)
