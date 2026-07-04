"""Grounded follow-up chat: DeepSeek with tool-calling against the parquet store.

The model can only answer with numbers it fetched via a tool call against
finai/app/data_access.py -- it cannot see the raw store and cannot invent
figures, mirroring the "write only from this data" discipline already used
in finai/pipeline/alerts.py for the risk-alert generation prompt.
"""
from __future__ import annotations

import json

from ..pipeline.llm_client import DEFAULT_MODEL, get_client
from . import data_access as da

SYSTEM_PROMPT = (
    "You are FinAI, a USD/CNY & USD/CNH FX volatility research assistant. "
    "Answer questions about vol forecasts, SHAP driver attributions, sentiment "
    "scores, and generated risk alerts. Always call a tool to fetch numbers -- "
    "never state a figure you have not just retrieved via a tool call. If a "
    "tool returns an error, say so plainly instead of guessing. Keep answers "
    "concise and quote the exact dates/factors/numbers returned by the tools."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get realized RV and HAR-X/GBM forecasted RV for a given date and factor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "factor": {"type": "string", "description": "e.g. CNH_ATM_PC1"},
                },
                "required": ["date", "factor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_shap_drivers",
            "description": "Get the top SHAP feature attributions for a given date/factor/model (HAR-X or GBM).",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "factor": {"type": "string"},
                    "model": {"type": "string", "enum": ["HAR-X", "GBM"]},
                },
                "required": ["date", "factor"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sentiment",
            "description": "Get FinBERT news sentiment (label, score, confidence) for a given date.",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_alert_text",
            "description": "Get the full generated UBS-style risk alert text for a given panel date.",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string"}},
                "required": ["date"],
            },
        },
    },
]

TOOL_IMPL = {
    "get_forecast": lambda date, factor: da.get_forecast(date, factor),
    "get_shap_drivers": lambda date, factor, model="HAR-X": da.get_shap_drivers(date, factor, model),
    "get_sentiment": lambda date: da.get_sentiment(date),
    "get_alert_text": lambda date: da.get_alert_text(date),
}


def run_chat_turn(history: list[dict], user_message: str, max_tool_hops: int = 4) -> str:
    """history: list of {"role": "user"|"assistant", "content": str} (no tool
    messages persisted across turns -- each turn resolves its own tool calls)."""
    client = get_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    for _ in range(max_tool_hops):
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.1,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
            tc.model_dump() for tc in msg.tool_calls
        ]})

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = TOOL_IMPL[name](**args)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    return "I couldn't resolve this within the tool-call budget -- try narrowing the date/factor."


def run_chat_turn_stream(
    history: list[dict], user_message: str, max_tool_hops: int = 4
):
    """Generator variant of run_chat_turn for streaming SSE responses.

    Yields:
        ("tool", name)        — for each tool call resolved during the hop loop
        ("delta", text_chunk) — incremental text chunks from the final streamed completion
        ("done", None)        — terminal signal
    """
    client = get_client()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    # Tool-hop loop (non-streamed) — identical logic to run_chat_turn
    for _ in range(max_tool_hops):
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.1,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            # Final answer already generated by this hop — emit it directly
            # rather than paying for (and risking divergence from) a second call.
            yield ("delta", msg.content or "")
            yield ("done", None)
            return

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            yield ("tool", name)
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = TOOL_IMPL[name](**args)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    # Exhausted hops without a non-tool response
    yield ("delta", "I couldn't resolve this within the tool-call budget -- try narrowing the date/factor.")
    yield ("done", None)
