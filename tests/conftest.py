"""Shared pytest fixtures for the Beyond the Smile backend test suite.

The TestClient runs the real FastAPI app in-process against the actual
parquet store in finai/store/ (integration tests over real data).
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure the project root is on sys.path so `finai` resolves correctly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Session-scoped TestClient — imports app once, reuses data_access lru_cache."""
    from backend.main import app
    return TestClient(app)
