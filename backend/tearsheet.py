"""Tear-sheet PDF generation for Beyond the Smile - UBS Fin AI Bootcamp.

build_tearsheet_pdf(factor, model) -> bytes
  Renders a one-page A4-landscape PDF using matplotlib (Agg backend).
  Data is read via finai.app.data_access helpers.
"""
from __future__ import annotations

import io
from datetime import date

# IMPORTANT: set non-interactive Agg backend BEFORE importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402

# UBS palette
_RED = "#E60000"
_DARK_GREY = "#2E2E2E"
_MID_GREY = "#6A6A6A"
_GRID = "#D9D9D9"
_WHITE = "#FFFFFF"
_BLACK = "#000000"


def _import_da():
    from finai.app import data_access as da  # noqa: PLC0415
    return da


def build_tearsheet_pdf(factor: str, model: str) -> bytes:
    """Render a branded one-page tear-sheet and return raw PDF bytes."""
    import pandas as pd

    da = _import_da()

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    daily = da.daily_outputs()
    eval_df = da.model_eval()
    shap_df = da.shap_top_drivers()
    sent_df = da.sentiment_daily()

    # Filter to factor; the chart is OOS-only so trim to the OOS window
    fcast = daily[daily["factor"] == factor].copy() if not daily.empty else pd.DataFrame()
    if not fcast.empty:
        fcast = fcast.sort_values("date")
        fcast = fcast[pd.to_datetime(fcast["date"]) >= pd.Timestamp("2020-01-01")]

    eval_sub = eval_df[eval_df["factor"] == factor].copy() if not eval_df.empty else pd.DataFrame()

    shap_sub = shap_df[
        (shap_df["factor"] == factor) & (shap_df["model"] == model)
    ].copy() if not shap_df.empty else pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Figure layout - A4 landscape 297×210 mm = 11.69×8.27 in
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(11.69, 8.27), facecolor=_WHITE)
    fig.patch.set_facecolor(_WHITE)

    gs = gridspec.GridSpec(
        4, 2,
        figure=fig,
        left=0.07, right=0.97,
        top=0.92, bottom=0.08,
        hspace=0.55, wspace=0.35,
        height_ratios=[0.22, 1.0, 1.0, 0.18],
    )

    # ------------------------------------------------------------------ #
    # Row 0 - Title block (spans full width)
    # ------------------------------------------------------------------ #
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")

    gen_date = date.today().strftime("%Y-%m-%d")
    ax_title.text(
        0.0, 1.0,
        "Beyond the Smile - UBS Fin AI Bootcamp",
        transform=ax_title.transAxes,
        fontsize=14, fontweight="bold", color=_RED, va="top",
    )
    ax_title.text(
        0.0, 0.38,
        f"Factor: {factor}   |   Model: {model}   |   Generated: {gen_date}",
        transform=ax_title.transAxes,
        fontsize=9, color=_DARK_GREY, va="top",
        fontfamily="monospace",
    )
    # Horizontal rule
    ax_title.axhline(y=0.0, color=_RED, linewidth=1.5, xmin=0, xmax=1)

    # ------------------------------------------------------------------ #
    # Row 1, Col 0 - OOS Forecast Line Chart
    # ------------------------------------------------------------------ #
    ax_fc = fig.add_subplot(gs[1, 0])
    ax_fc.set_facecolor(_WHITE)
    ax_fc.set_title("OOS Realized vs Forecast RV", fontsize=9, color=_DARK_GREY, pad=4, loc="left")

    if not fcast.empty:
        fcast["date_ts"] = pd.to_datetime(fcast["date"])
        # MA-smooth for readability
        realized = fcast["rv_realized"].rolling(10, min_periods=1).mean()
        forecast_col = "rv_forecast_harx" if model == "HAR-X" else "rv_forecast_gbm"
        forecasted = fcast[forecast_col].rolling(10, min_periods=1).mean()

        dates_num = fcast["date_ts"]
        ax_fc.plot(dates_num, realized, color=_BLACK, linewidth=0.9, label="Realized")
        ax_fc.plot(dates_num, forecasted, color=_RED, linewidth=0.9,
                   linestyle="--", label=f"{model} Forecast")

        ax_fc.legend(fontsize=7, frameon=False, loc="upper left")
        ax_fc.tick_params(axis="both", labelsize=7, colors=_MID_GREY)
        ax_fc.xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter("%Y")
        )
        plt.setp(ax_fc.xaxis.get_majorticklabels(), rotation=45, ha="right")
    else:
        ax_fc.text(0.5, 0.5, "No forecast data", ha="center", va="center",
                   transform=ax_fc.transAxes, color=_MID_GREY, fontsize=8)

    for spine in ax_fc.spines.values():
        spine.set_edgecolor(_GRID)
    ax_fc.yaxis.set_tick_params(labelsize=7)
    ax_fc.grid(color=_GRID, linestyle="--", linewidth=0.5, alpha=0.7)

    # ------------------------------------------------------------------ #
    # Row 1, Col 1 - Top-8 SHAP horizontal bar chart
    # ------------------------------------------------------------------ #
    ax_shap = fig.add_subplot(gs[1, 1])
    ax_shap.set_facecolor(_WHITE)
    ax_shap.set_title("Top-8 SHAP Drivers (latest date)", fontsize=9, color=_DARK_GREY, pad=4, loc="left")

    if not shap_sub.empty:
        # Latest date
        latest_date = shap_sub["date"].max()
        latest_shap = (
            shap_sub[shap_sub["date"] == latest_date]
            .copy()
            .sort_values("shap_value", key=lambda s: s.abs(), ascending=False)
            .head(8)
        )
        # Reverse for horizontal bar so largest is at top
        latest_shap = latest_shap.iloc[::-1]

        colors = [_RED if v >= 0 else _MID_GREY for v in latest_shap["shap_value"]]
        ax_shap.barh(
            latest_shap["feature"].tolist(),
            latest_shap["shap_value"].tolist(),
            color=colors,
            height=0.65,
        )
        ax_shap.axvline(x=0, color=_DARK_GREY, linewidth=0.7)
        ax_shap.tick_params(axis="y", labelsize=7, colors=_DARK_GREY)
        ax_shap.tick_params(axis="x", labelsize=7, colors=_MID_GREY)
        ax_shap.grid(axis="x", color=_GRID, linestyle="--", linewidth=0.5, alpha=0.7)
    else:
        ax_shap.text(0.5, 0.5, "No SHAP data", ha="center", va="center",
                     transform=ax_shap.transAxes, color=_MID_GREY, fontsize=8)

    for spine in ax_shap.spines.values():
        spine.set_edgecolor(_GRID)

    # ------------------------------------------------------------------ #
    # Row 2, Col 0-1 - Metrics table (spans full width)
    # ------------------------------------------------------------------ #
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis("off")
    ax_tbl.set_title("OOS Model Evaluation", fontsize=9, color=_DARK_GREY, pad=2, loc="left")

    if not eval_sub.empty:
        col_labels = ["Model", "N (OOS)", "QLIKE (d)", "Corr (d)", "QLIKE (w)", "Corr (w)"]

        def _fmt(v):
            if v is None:
                return "-"
            try:
                import math
                f = float(v)
                return "-" if math.isnan(f) else f"{f:.3f}"
            except (TypeError, ValueError):
                return "-"

        table_data = []
        for _, row in eval_sub.iterrows():
            table_data.append([
                str(row.get("model", "")),
                str(int(row.get("n_oos", 0))) if row.get("n_oos") is not None else "-",
                _fmt(row.get("QLIKE_daily")),
                _fmt(row.get("Corr_daily")),
                _fmt(row.get("QLIKE_week")),
                _fmt(row.get("Corr_week")),
            ])

        tbl = ax_tbl.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.4)

        # Style header row
        for j in range(len(col_labels)):
            cell = tbl[0, j]
            cell.set_facecolor(_DARK_GREY)
            cell.set_text_props(color=_WHITE, fontweight="bold")

        # Style data rows
        for i in range(1, len(table_data) + 1):
            for j in range(len(col_labels)):
                cell = tbl[i, j]
                cell.set_facecolor("#F5F5F5" if i % 2 == 0 else _WHITE)
                cell.set_edgecolor(_GRID)
    else:
        ax_tbl.text(0.5, 0.5, "No evaluation data", ha="center", va="center",
                    transform=ax_tbl.transAxes, color=_MID_GREY, fontsize=8)

    # ------------------------------------------------------------------ #
    # Row 3 - Footer (sentiment + source)
    # ------------------------------------------------------------------ #
    ax_foot = fig.add_subplot(gs[3, :])
    ax_foot.axis("off")

    footer_parts = []
    if not sent_df.empty:
        try:
            # sentiment_daily is date-indexed and a date can carry several news
            # docs - summarise each one on the latest date as "label score".
            latest_sent_idx = sent_df.index.max()
            latest = sent_df.loc[[latest_sent_idx]]
            sent_date = str(latest_sent_idx)[:10]
            items = ", ".join(
                f"{row.get('label', '?')} {float(row.get('sent_score')):+.3f}"
                for _, row in latest.iterrows()
                if row.get("sent_score") is not None
            )
            if items:
                footer_parts.append(f"Latest sentiment ({sent_date}): {items}")
        except Exception:
            pass

    footer_parts.append("Source: internal analysis.")
    footer_text = "   |   ".join(footer_parts)

    ax_foot.text(
        0.0, 0.85,
        footer_text,
        transform=ax_foot.transAxes,
        fontsize=7, color=_MID_GREY, va="top",
        fontfamily="monospace",
    )
    ax_foot.axhline(y=1.0, color=_GRID, linewidth=0.8)

    # ------------------------------------------------------------------ #
    # Save to BytesIO and return
    # ------------------------------------------------------------------ #
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", bbox_inches="tight", facecolor=_WHITE)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
