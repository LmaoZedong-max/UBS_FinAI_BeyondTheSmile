import streamlit as st

from finai.app import data_access as da
from finai.app import theme

st.set_page_config(page_title="FinAI - Risk Alerts", layout="wide")
theme.inject(st)
theme.topbar(st, "Risk alerts - generated daily UBS-style reports")

files = da.list_alert_files()
if not files:
    st.warning("No risk alert files found in 'NLP Outputs/'. Run the pipeline without --skip-alerts.")
    st.stop()

labels = [f.stem.replace("risk_alert_", "") for f in files]
choice = st.selectbox("Panel date", labels, index=len(labels) - 1)
selected = files[labels.index(choice)]

st.markdown(
    f'<div class="finai-alert-card">{selected.read_text(encoding="utf-8", errors="ignore")}</div>',
    unsafe_allow_html=True,
)
