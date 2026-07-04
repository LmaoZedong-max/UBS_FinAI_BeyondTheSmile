"""Branded UI chrome: hides Streamlit's default header/menu/footer and injects
a dark, red-accent theme matching the notebook's UBS_RED plotting palette, so
the app reads as a bespoke research terminal rather than a stock Streamlit demo.
"""

UBS_RED = "#E60000"
DARK_GREY = "#2E2E2E"
MID_GREY = "#6A6A6A"
GRID_GREY = "#D9D9D9"
BG = "#111315"
PANEL_BG = "#1B1E21"
TEXT = "#ECECEC"

CUSTOM_CSS = f"""
<style>
#MainMenu, header, footer {{visibility: hidden;}}
[data-testid="stToolbar"] {{visibility: hidden;}}

.stApp {{
    background-color: {BG};
}}

.finai-topbar {{
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    border-bottom: 2px solid {UBS_RED};
    padding-bottom: 0.6rem;
    margin-bottom: 1.4rem;
}}
.finai-topbar .brand {{
    color: {UBS_RED};
    font-weight: 800;
    font-size: 1.55rem;
    letter-spacing: 0.02em;
}}
.finai-topbar .subtitle {{
    color: {MID_GREY};
    font-size: 0.95rem;
}}

.finai-card {{
    background-color: {PANEL_BG};
    border: 1px solid #2A2D30;
    border-left: 3px solid {UBS_RED};
    border-radius: 6px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}}
.finai-card h4 {{
    margin-top: 0;
    color: {TEXT};
    font-size: 1.02rem;
    font-weight: 700;
}}
.finai-metric-label {{
    color: {MID_GREY};
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
.finai-metric-value {{
    color: {TEXT};
    font-size: 1.4rem;
    font-weight: 700;
}}

.finai-alert-card {{
    background-color: {PANEL_BG};
    border: 1px solid #2A2D30;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.9rem;
    white-space: pre-wrap;
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.85rem;
    line-height: 1.45;
    color: #D6D6D6;
}}

section[data-testid="stSidebar"] {{
    background-color: {PANEL_BG};
    border-right: 1px solid #2A2D30;
}}
</style>
"""


def inject(st) -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def topbar(st, subtitle: str) -> None:
    st.markdown(
        f'<div class="finai-topbar"><span class="brand">FinAI</span>'
        f'<span class="subtitle">{subtitle}</span></div>',
        unsafe_allow_html=True,
    )
