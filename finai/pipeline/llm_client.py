"""Shared DeepSeek client (OpenAI-compatible API surface).

The API key is only ever read from the DEEPSEEK_API_KEY environment variable
(loaded from a local .env, which is git-ignored) -- never hardcode a key here.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
