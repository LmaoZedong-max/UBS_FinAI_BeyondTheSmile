"""Rolling, no-lookahead LightGBM forecaster on the same design matrix as HAR-X.

This is the NEW model (not in the original notebooks) added so that
shap_explain.py has a model with clean, exact SHAP support (TreeExplainer)
to compare against the HAR-X linear attribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .mean_models import as_clean_series
from .vol_models import harx_design

DEFAULT_LGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=0,
    verbosity=-1,
)


@dataclass
class GBMWindowFit:
    refit_date: pd.Timestamp
    model: lgb.LGBMRegressor
    feature_names: list[str]
    pred_index: pd.DatetimeIndex


@dataclass
class GBMForecastResult:
    yhat_next: pd.Series
    windows: list[GBMWindowFit] = field(default_factory=list)


def rolling_gbm_forecast(
    r: pd.Series,
    X_exog: pd.DataFrame | None,
    oos_start: pd.Timestamp,
    rv_window: int = 5,
    lookback: int = 1260,
    refit_every: int = 21,
    min_fit_rows: int = 400,
    lgb_params: dict | None = None,
) -> GBMForecastResult:
    """Predict log(RV_{t+1}) with a LightGBM regressor, refit every `refit_every`
    trading days on a trailing `lookback` window using only data up to t-1 -
    same no-lookahead discipline and same feature set (harx_design) as HAR-X,
    so the two models are directly comparable in eval_all_models_for_factor."""
    params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}

    r = as_clean_series(r)
    X_all, y_all = harx_design(r, X_exog, rv_window)
    idx = r.index

    yhat_next = pd.Series(index=y_all.index, dtype=float, name=f"GBM_logRV_next_w{rv_window}")
    windows: list[GBMWindowFit] = []

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

        model = lgb.LGBMRegressor(**params)
        model.fit(Xmat.values, Y.values)

        df_pred = X_all.loc[t_date:j_date].replace([np.inf, -np.inf], np.nan).dropna()
        df_pred = df_pred[Xmat.columns]
        if len(df_pred) > 0:
            yhat_next.loc[df_pred.index] = model.predict(df_pred.values)
            windows.append(GBMWindowFit(
                refit_date=end_fit_date,
                model=model,
                feature_names=list(Xmat.columns),
                pred_index=df_pred.index,
            ))

        i += refit_every

    return GBMForecastResult(yhat_next=yhat_next, windows=windows)


def gbm_logrv_to_sig2hat(yhat_logrv_next: pd.Series) -> pd.Series:
    rv_hat_next = np.exp(yhat_logrv_next).rename("GBM_RV_hat_next")
    return rv_hat_next.shift(1).rename("GBM")
