"""SHAP explainability for the GBM (TreeExplainer) and HAR-X (LinearExplainer) models.

Produces a single long-format table: (date, factor, model, feature, shap_value),
plus a helper to derive the top-k drivers per date/model for cheap lookups from
the chat tool layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .gbm_model import GBMWindowFit
from .vol_models import HARXWindowFit


def shap_for_gbm(windows: list[GBMWindowFit], X_all: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Long-format SHAP values for the GBM model across all rolling windows.

    X_all: the same design matrix (harx_design output) used to fit/predict the GBM.
    """
    frames = []
    for w in windows:
        if len(w.pred_index) == 0:
            continue
        X_pred = X_all.loc[w.pred_index, w.feature_names]
        explainer = shap.TreeExplainer(w.model)
        sv = explainer.shap_values(X_pred.values)
        df = pd.DataFrame(sv, index=X_pred.index, columns=w.feature_names)
        long = df.reset_index(names="date").melt(id_vars="date", var_name="feature", value_name="shap_value")
        frames.append(long)
    if not frames:
        return pd.DataFrame(columns=["date", "feature", "shap_value", "model", "factor"])
    out = pd.concat(frames, ignore_index=True)
    out["model"] = "GBM"
    out["factor"] = factor
    return out


def shap_for_harx(windows: list[HARXWindowFit], X_all: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Long-format SHAP values for HAR-X using shap.LinearExplainer, one explainer
    per rolling-refit window (coefficients + in-sample background for centering)."""
    frames = []
    for w in windows:
        if len(w.pred_index) == 0:
            continue
        coef = w.beta.reindex(w.feature_names).fillna(0.0).values
        intercept = float(w.beta.get("const", 0.0))
        X_pred = X_all.loc[w.pred_index, w.feature_names]

        explainer = shap.LinearExplainer(
            (coef, intercept),
            w.X_fit[w.feature_names].values,
            feature_perturbation="interventional",
        )
        sv = explainer.shap_values(X_pred.values)
        df = pd.DataFrame(sv, index=X_pred.index, columns=w.feature_names)
        long = df.reset_index(names="date").melt(id_vars="date", var_name="feature", value_name="shap_value")
        frames.append(long)
    if not frames:
        return pd.DataFrame(columns=["date", "feature", "shap_value", "model", "factor"])
    out = pd.concat(frames, ignore_index=True)
    out["model"] = "HAR-X"
    out["factor"] = factor
    return out


def top_k_drivers(shap_long: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Collapse a long SHAP table to top-k |shap_value| drivers per (date, model, factor)."""
    if shap_long.empty:
        return shap_long
    ranked = (
        shap_long.assign(abs_shap=shap_long["shap_value"].abs())
        .sort_values(["date", "model", "factor", "abs_shap"], ascending=[True, True, True, False])
        .groupby(["date", "model", "factor"], group_keys=False)
        .head(k)
        .drop(columns="abs_shap")
    )
    return ranked.reset_index(drop=True)
