"""FinAI research terminal -- Streamlit entrypoint.

    streamlit run finai/app/main.py
"""
import streamlit as st

from finai.app import theme

st.set_page_config(page_title="FinAI — USD/CNY-CNH Vol Terminal", layout="wide", initial_sidebar_state="expanded")
theme.inject(st)
theme.topbar(st, "USD/CNY & USD/CNH volatility research terminal")

st.markdown(
    """
Use the sidebar to open **Dashboard** (forecasts + SHAP driver attribution) or
**Risk Alerts** (generated UBS-style daily reports). Ask follow-up questions
about anything you see there in the chat panel below.
"""
)

st.divider()

from finai.app import chat as chat_mod  # noqa: E402

st.subheader("Ask FinAI")
st.caption("Answers are grounded in finai/store/*.parquet via tool calls -- the model cannot invent figures.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_msg = st.chat_input("e.g. What drove the CNH_ATM_PC1 vol forecast on 2025-12-02?")
if user_msg:
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        try:
            reply = chat_mod.run_chat_turn(st.session_state.chat_history[:-1], user_msg)
        except Exception as e:
            reply = f"Chat backend error: {type(e).__name__}: {e}. Check DEEPSEEK_API_KEY in .env."
        st.markdown(reply)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
