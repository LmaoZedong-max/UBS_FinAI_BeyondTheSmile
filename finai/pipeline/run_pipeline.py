"""End-to-end pipeline orchestrator.

    python -m finai.pipeline.run_pipeline

Ingests FINAL_Data.csv -> builds PCA factors -> fits HAR-X + GBM (rolling,
no-lookahead) -> computes SHAP attributions for both -> (optionally) cleans
news, scores sentiment with FinBERT, and generates LLM risk alerts -> writes
everything to finai/store/*.parquet for the Streamlit app / chat tools to read.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import alerts, gbm_model, sentiment, shap_explain
from .factors import build_factors_safe
from .ingestion import IS_END, OOS_START, load_panel0
from .mean_models import make_ar1_residuals, standardise_resid
from .gbm_model import gbm_logrv_to_sig2hat
from .vol_models import (
    eval_all_models_for_factor,
    harx_logrv_to_sig2hat,
    pick_macro_exog_cols,
    rolling_harx_fit_windows,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def process_factor(
    factor: str,
    panel_factors: pd.DataFrame,
    resid_map: dict[str, pd.Series],
    cnh_start: pd.Timestamp | None,
    is_end: pd.Timestamp,
    oos_start: pd.Timestamp,
) -> dict | None:
    if factor not in resid_map:
        print(f"[skip] {factor}: no AR(1) residual (too few IS rows).")
        return None

    resid = resid_map[factor].copy()
    exog_cols = pick_macro_exog_cols(panel_factors)
    X_exog = panel_factors[exog_cols].copy()

    if cnh_start is not None and str(factor).startswith("CNH_"):
        resid = resid.loc[resid.index >= cnh_start]
        X_exog = X_exog.loc[X_exog.index >= cnh_start]

    r_std, _ = standardise_resid(resid, is_end)

    print(f"[{factor}] fitting HAR-X (rolling, no-lookahead)...")
    yhat_harx, harx_windows = rolling_harx_fit_windows(r_std, X_exog, oos_start)
    rv_hat_harx = harx_logrv_to_sig2hat(yhat_harx)

    print(f"[{factor}] fitting GBM (rolling, no-lookahead)...")
    gbm_result = gbm_model.rolling_gbm_forecast(r_std, X_exog, oos_start)
    rv_hat_gbm = gbm_logrv_to_sig2hat(gbm_result.yhat_next)

    print(f"[{factor}] evaluating OOS (QLIKE / correlation)...")
    eval_table = eval_all_models_for_factor(
        resid, is_end, oos_start,
        extra_forecasts={"HAR-X": rv_hat_harx, "GBM": rv_hat_gbm},
    )
    eval_table["factor"] = factor

    print(f"[{factor}] computing SHAP attributions...")
    from .vol_models import harx_design
    X_all, _ = harx_design(r_std, X_exog, rv_window=5)
    shap_harx = shap_explain.shap_for_harx(harx_windows, X_all, factor)
    shap_gbm = shap_explain.shap_for_gbm(gbm_result.windows, X_all, factor)
    shap_long = pd.concat([shap_harx, shap_gbm], ignore_index=True)

    from .mean_models import rv_proxy
    rv_real = rv_proxy(r_std, window=5).reindex(rv_hat_harx.index)

    daily = pd.DataFrame({
        "date": rv_hat_harx.index,
        "factor": factor,
        "rv_realized": rv_real.values,
        "rv_forecast_harx": rv_hat_harx.values,
        "rv_forecast_gbm": rv_hat_gbm.reindex(rv_hat_harx.index).values,
    })

    return {"daily": daily, "eval": eval_table, "shap": shap_long}


def run(
    data_path: Path,
    factors: list[str] | None,
    news_dir: Path,
    clean_dir: Path,
    outputs_dir: Path,
    store_dir: Path,
    skip_sentiment: bool,
    skip_alerts: bool,
    skip_news_cleaning: bool,
) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading panel from {data_path}...")
    panel0, _ = load_panel0(data_path)
    print(f"Panel shape: {panel0.shape}, range {panel0.index.min().date()} -> {panel0.index.max().date()}")

    print("Building PCA factors (IS-only fit)...")
    panel_factors, factor_cols, cnh_start = build_factors_safe(panel0, IS_END)
    print(f"Factor cols: {len(factor_cols)}, CNH_START={cnh_start}")

    resid_map = make_ar1_residuals(panel_factors, factor_cols, IS_END)
    if not resid_map:
        raise RuntimeError("No AR(1) residuals produced -- check panel/date range.")

    if factors is None:
        factors = [f for f in resid_map if str(f).startswith("CNH_")] or list(resid_map.keys())[:1]
    print(f"Processing factors: {factors}")

    daily_frames, eval_frames, shap_frames = [], [], []
    contrib_by_factor: dict[str, pd.DataFrame] = {}

    for factor in factors:
        try:
            result = process_factor(factor, panel_factors, resid_map, cnh_start, IS_END, OOS_START)
        except Exception as e:
            print(f"[error] {factor}: {type(e).__name__}: {e}")
            continue
        if result is None:
            continue
        daily_frames.append(result["daily"])
        eval_frames.append(result["eval"])
        shap_frames.append(result["shap"])

    if not daily_frames:
        raise RuntimeError("No factors produced output -- check data/date range/factor names.")

    daily_outputs = pd.concat(daily_frames, ignore_index=True)
    eval_table = pd.concat(eval_frames, ignore_index=True)
    shap_values = pd.concat(shap_frames, ignore_index=True)

    if not skip_sentiment:
        try:
            print("Sentiment pipeline: news cleaning + FinBERT scoring...")
            if not skip_news_cleaning:
                sentiment.clean_news_dir(news_dir, clean_dir)
            clean_files = sorted(Path(clean_dir).glob("news*.txt"))
            df_finbert = sentiment.score_news_dir(clean_dir)
            sent_daily = sentiment.build_sent_daily(df_finbert, clean_files)

            primary_factor = factors[0]
            _, primary_windows = rolling_harx_fit_windows(
                *_resid_and_exog(primary_factor, panel_factors, resid_map, cnh_start, IS_END, OOS_START)
            )
            X_all, _ = _harx_design_for(primary_factor, panel_factors, resid_map, cnh_start, IS_END)
            contrib_macro = _contrib_from_windows(primary_windows, X_all)

            sent_daily2 = sent_daily.copy()
            sent_daily2["panel_date"] = sentiment.align_news_to_panel_date(sent_daily2.index, contrib_macro.index)
            sent_daily2 = sent_daily2.dropna(subset=["panel_date"])
            sent_daily2 = sent_daily2.loc[sent_daily2["panel_date"].isin(contrib_macro.index)]

            news_top5 = alerts.build_news_top5(contrib_macro, sent_daily2, primary_factor)
            news_top5.to_parquet(store_dir / "news_top5.parquet", index=False)
            sent_daily.to_parquet(store_dir / "sentiment_daily.parquet")

            if not skip_alerts and len(news_top5) > 0:
                print("Generating LLM risk alerts (DeepSeek)...")
                written = alerts.generate_alerts_for_days(news_top5, clean_dir, outputs_dir)
                print(f"Wrote {len(written)} risk-alert files to {outputs_dir}")
        except Exception as e:
            print(f"[warn] sentiment/alerts pipeline failed, continuing without it: {type(e).__name__}: {e}")

    daily_outputs.to_parquet(store_dir / "daily_outputs.parquet", index=False)
    eval_table.to_parquet(store_dir / "model_eval.parquet", index=False)
    shap_values.to_parquet(store_dir / "shap_values.parquet", index=False)

    top_drivers = shap_explain.top_k_drivers(shap_values, k=5)
    top_drivers.to_parquet(store_dir / "shap_top_drivers.parquet", index=False)

    print(f"Done. Store written to {store_dir}")
    print(eval_table)


def _resid_and_exog(factor, panel_factors, resid_map, cnh_start, is_end, oos_start):
    resid = resid_map[factor].copy()
    exog_cols = pick_macro_exog_cols(panel_factors)
    X_exog = panel_factors[exog_cols].copy()
    if cnh_start is not None and str(factor).startswith("CNH_"):
        resid = resid.loc[resid.index >= cnh_start]
        X_exog = X_exog.loc[X_exog.index >= cnh_start]
    r_std, _ = standardise_resid(resid, is_end)
    return r_std, X_exog, oos_start


def _harx_design_for(factor, panel_factors, resid_map, cnh_start, is_end):
    from .vol_models import harx_design
    r_std, X_exog, _ = _resid_and_exog(factor, panel_factors, resid_map, cnh_start, is_end, OOS_START)
    return harx_design(r_std, X_exog, rv_window=5)


def _contrib_from_windows(windows, X_all) -> pd.DataFrame:
    all_cols = sorted({c for w in windows for c in w.feature_names})
    contrib = pd.DataFrame(index=X_all.index, columns=all_cols, dtype=float)
    har_core = {"log_rv_d", "log_rv_w", "log_rv_m"}
    for w in windows:
        df_pred = X_all.loc[w.pred_index, w.feature_names]
        b = w.beta.reindex(["const"] + w.feature_names).fillna(0.0)
        contrib.loc[df_pred.index, df_pred.columns] = df_pred.mul(b[df_pred.columns], axis=1).values
    contrib = contrib.drop(columns=[c for c in har_core if c in contrib.columns], errors="ignore")
    return contrib.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0).dropna(how="all")


def main():
    ap = argparse.ArgumentParser(description="Run the FinAI vol + sentiment + SHAP pipeline end-to-end.")
    ap.add_argument("--data-path", default=str(REPO_ROOT / "FINAL_Data.csv"))
    ap.add_argument("--factors", nargs="*", default=None, help="Factor columns to process, e.g. CNH_ATM_PC1")
    ap.add_argument("--news-dir", default=str(REPO_ROOT / "NLP Data"))
    ap.add_argument("--clean-dir", default=str(REPO_ROOT / "NLP Data Cleaned"))
    ap.add_argument("--outputs-dir", default=str(REPO_ROOT / "NLP Outputs"))
    ap.add_argument("--store-dir", default=str(REPO_ROOT / "finai" / "store"))
    ap.add_argument("--skip-sentiment", action="store_true")
    ap.add_argument("--skip-alerts", action="store_true", help="Score sentiment but skip LLM alert generation")
    ap.add_argument("--skip-news-cleaning", action="store_true", help="Reuse existing 'NLP Data Cleaned' files")
    args = ap.parse_args()

    run(
        data_path=Path(args.data_path),
        factors=args.factors,
        news_dir=Path(args.news_dir),
        clean_dir=Path(args.clean_dir),
        outputs_dir=Path(args.outputs_dir),
        store_dir=Path(args.store_dir),
        skip_sentiment=args.skip_sentiment,
        skip_alerts=args.skip_alerts,
        skip_news_cleaning=args.skip_news_cleaning,
    )


if __name__ == "__main__":
    main()
