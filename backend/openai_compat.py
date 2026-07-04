"""OpenAI-compatible Chat Completions API layer.

Exposes /v1/models and /v1/chat/completions so Open WebUI (and any
OpenAI-compatible client) can use the grounded chat as a model called
"beyond-the-smile".

Mount this router in backend/main.py (no /api prefix):
    app.include_router(openai_router)
"""
from __future__ import annotations

import json as _json
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

openai_router = APIRouter()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "beyond-the-smile"
OWNED_BY = "ubs-fin-ai-bootcamp"
# Unix timestamp at module import - stable across requests for /v1/models
_MODULE_CREATED_TS = int(time.time())


# ---------------------------------------------------------------------------
# Pydantic request model
# ---------------------------------------------------------------------------

class OAIMessage(BaseModel):
    role: str
    content: str


class OAIChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[OAIMessage]
    stream: bool = False
    # All other OpenAI fields (temperature, max_tokens, etc.) are ignored


# ---------------------------------------------------------------------------
# Helper: OpenAI error response
# ---------------------------------------------------------------------------

def _oai_error(message: str, type_: str, code: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "code": code}},
    )


# ---------------------------------------------------------------------------
# Helper: build history + user_message from OAI messages list
# Strip system messages (we enforce our own grounded system prompt).
# The last non-system message is taken as the user message; everything
# before it becomes history.
# ---------------------------------------------------------------------------

def _split_messages(messages: list[OAIMessage]) -> tuple[list[dict], str]:
    """Return (history, user_message).

    Strips system messages, then uses the last message as user_message and
    everything before it as history (mirroring how /api/chat works).
    """
    filtered = [m for m in messages if m.role != "system"]
    if not filtered:
        return [], ""
    history = [{"role": m.role, "content": m.content} for m in filtered[:-1]]
    user_message = filtered[-1].content
    return history, user_message


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

@openai_router.get("/v1/models")
def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": _MODULE_CREATED_TS,
                "owned_by": OWNED_BY,
            }
        ],
    }


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

@openai_router.post("/v1/chat/completions")
async def chat_completions(body: OAIChatRequest, request: Request):
    # Authorization header is accepted but never validated (Open WebUI always
    # sends a key; ignoring it is intentional per spec).

    # Check for API key before doing any work
    if not os.environ.get("DEEPSEEK_API_KEY"):
        return _oai_error(
            message="DEEPSEEK_API_KEY not configured",
            type_="invalid_request_error",
            code="missing_api_key",
            status=503,
        )

    history, user_message = _split_messages(body.messages)

    if body.stream:
        return StreamingResponse(
            _stream_generator(history, user_message),
            media_type="text/event-stream",
        )
    else:
        return await _non_stream_response(history, user_message)


# ---------------------------------------------------------------------------
# Non-streaming path
# ---------------------------------------------------------------------------

async def _non_stream_response(history: list[dict], user_message: str) -> JSONResponse:
    from finai.app.chat import run_chat_turn  # noqa: PLC0415

    try:
        content = run_chat_turn(history=history, user_message=user_message)
    except RuntimeError:
        return _oai_error(
            message="DEEPSEEK_API_KEY not configured",
            type_="invalid_request_error",
            code="missing_api_key",
            status=503,
        )
    except Exception as exc:
        return _oai_error(
            message=str(exc),
            type_="server_error",
            code="internal_error",
            status=500,
        )

    ts = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    return JSONResponse(
        content={
            "id": completion_id,
            "object": "chat.completion",
            "created": ts,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    )


# ---------------------------------------------------------------------------
# Streaming path
# ---------------------------------------------------------------------------

def _stream_generator(history: list[dict], user_message: str):
    """Yield OpenAI SSE chunk frames, then the [DONE] sentinel."""
    from finai.app.chat import run_chat_turn_stream  # noqa: PLC0415

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    ts = int(time.time())

    def _chunk(delta: dict, finish_reason: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": ts,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {_json.dumps(payload)}\n\n"

    # First chunk always carries role
    yield _chunk({"role": "assistant"})

    first_tool = True
    try:
        for kind, payload in run_chat_turn_stream(history=history, user_message=user_message):
            if kind == "tool":
                # Emit tool progress as a visible content chunk so Open WebUI
                # shows activity. Only the first one gets the newline prefix.
                if first_tool:
                    content_text = f"\n> consulting data store: {payload}()\n"
                    first_tool = False
                else:
                    content_text = f"> consulting data store: {payload}()\n"
                yield _chunk({"content": content_text})
            elif kind == "delta":
                yield _chunk({"content": payload})
            elif kind == "done":
                # Final chunk: empty delta + finish_reason stop
                yield _chunk({}, finish_reason="stop")
                yield "data: [DONE]\n\n"
                return
    except RuntimeError:
        # Missing key surfaced late (race condition)
        err = _json.dumps({
            "error": {
                "message": "DEEPSEEK_API_KEY not configured",
                "type": "invalid_request_error",
                "code": "missing_api_key",
            }
        })
        yield f"data: {err}\n\n"
    except Exception as exc:
        err = _json.dumps({
            "error": {
                "message": str(exc),
                "type": "server_error",
                "code": "internal_error",
            }
        })
        yield f"data: {err}\n\n"
