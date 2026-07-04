import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finai.app import data_access as da
from finai.app import theme

st.set_page_config(page_title="FinAI - Dashboard", layout="wide")
theme.inject(st)
theme.topbar(st, "Dashboard - forecasts & SHAP driver attribution")

daily = da.daily_outputs()
if daily.empty:
    st.warning("No pipeline output found. Run `python -m finai.pipeline.run_pipeline` first.")
    st.stop()

factors = da.available_factors()
col1, col2 = st.columns([1, 3])
with col1:
    factor = st.selectbox("Factor", factors)

sub = daily[daily["factor"] == factor].sort_values("date")
sub["date"] = pd.to_datetime(sub["date"])

st.markdown('<div class="finai-card"><h4>Realized vs. forecast variance (OOS)</h4></div>', unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=sub["date"], y=sub["rv_realized"], name="Realized RV", line=dict(color="#ECECEC", width=2)))
fig.add_trace(go.Scatter(x=sub["date"], y=sub["rv_forecast_harx"], name="HAR-X forecast", line=dict(color="#E60000", width=2, dash="dash")))
fig.add_trace(go.Scatter(x=sub["date"], y=sub["rv_forecast_gbm"], name="GBM forecast", line=dict(color="#6A6A6A", width=2, dash="dot")))
fig.update_layout(
    template="plotly_dark", paper_bgcolor="#1B1E21", plot_bgcolor="#1B1E21",
    height=420, margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

eval_table = da.model_eval()
if not eval_table.empty:
    st.markdown('<div class="finai-card"><h4>OOS model comparison (QLIKE / correlation)</h4></div>', unsafe_allow_html=True)
    st.dataframe(eval_table[eval_table["factor"] == factor].drop(columns=["factor"]), use_container_width=True, hide_index=True)

st.markdown('<div class="finai-card"><h4>SHAP driver attribution</h4></div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    model_choice = st.radio("Model", ["HAR-X", "GBM"], horizontal=True)
with c2:
    dates_available = sorted(sub["date"].dt.strftime("%Y-%m-%d").unique(), reverse=True)
    date_choice = st.selectbox("Date", dates_available) if dates_available else None

if date_choice:
    top = da.get_shap_drivers(date_choice, factor, model=model_choice, k=8)
    if "error" in top:
        st.info(top["error"])
    else:
        drivers_df = pd.DataFrame(top["drivers"]).sort_values("shap_value")
        bar = go.Figure(go.Bar(
            x=drivers_df["shap_value"], y=drivers_df["feature"], orientation="h",
            marker_color=["#E60000" if v > 0 else "#6A6A6A" for v in drivers_df["shap_value"]],
        ))
        bar.update_layout(
            template="plotly_dark", paper_bgcolor="#1B1E21", plot_bgcolor="#1B1E21",
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="SHAP value (contribution to predicted log-RV)",
        )
        st.plotly_chart(bar, use_container_width=True)
