"""Integration tests for the Beyond the Smile FastAPI backend.

All tests run against the real parquet store in finai/store/ using
FastAPI's TestClient (in-process, no network required). The
DEEPSEEK_API_KEY env var must NOT be set for the chat 503 / error-stream
tests; those tests monkeypatch the env to ensure the key is absent.
"""
from __future__ import annotations

import json
import re
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

FACTOR = "CNH_ATM_PC1"
SHAP_DATE = "2020-01-01"       # confirmed present in shap_top_drivers.parquet
ALERT_DATE = "2025-12-02"      # confirmed present in NLP Outputs/
ALERT_DATE_MISSING = "1999-01-01"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _sse_frames(body: bytes) -> list[dict[str, str]]:
    """Parse raw SSE body into list of {"event": ..., "data": ...} dicts."""
    frames: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in body.decode().splitlines():
        if line.startswith("event:"):
            current["event"] = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            current["data"] = line.split(":", 1)[1].strip()
        elif line == "" and current:
            frames.append(current)
            current = {}
    if current:
        frames.append(current)
    return frames


# ===========================================================================
# /api/health
# ===========================================================================

class TestHealth:
    def test_200_ok(self, client: TestClient) -> None:
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# ===========================================================================
# /api/factors
# ===========================================================================

class TestFactors:
    def test_200_non_empty(self, client: TestClient) -> None:
        r = client.get("/api/factors")
        assert r.status_code == 200
        data = r.json()
        assert "factors" in data
        assert isinstance(data["factors"], list)
        assert len(data["factors"]) > 0

    def test_contains_cnh_atm_pc1(self, client: TestClient) -> None:
        r = client.get("/api/factors")
        assert FACTOR in r.json()["factors"]


# ===========================================================================
# /api/forecasts
# ===========================================================================

class TestForecasts:
    def test_200_non_empty_series(self, client: TestClient) -> None:
        r = client.get(f"/api/forecasts?factor={FACTOR}")
        assert r.status_code == 200
        data = r.json()
        assert data["factor"] == FACTOR
        assert isinstance(data["series"], list)
        assert len(data["series"]) > 0

    def test_row_shape(self, client: TestClient) -> None:
        r = client.get(f"/api/forecasts?factor={FACTOR}")
        row = r.json()["series"][0]
        assert DATE_RE.match(row["date"]), f"date {row['date']!r} not YYYY-MM-DD"
        assert "rv_realized" in row
        assert "rv_forecast_harx" in row
        assert "rv_forecast_gbm" in row

    def test_start_end_filter_narrows_range(self, client: TestClient) -> None:
        # Full series
        full = client.get(f"/api/forecasts?factor={FACTOR}").json()["series"]
        # Narrow to 2020 only
        narrow = client.get(
            f"/api/forecasts?factor={FACTOR}&start=2020-01-01&end=2020-12-31"
        ).json()["series"]
        assert len(narrow) < len(full), "start/end filter did not narrow the range"
        for row in narrow:
            assert "2020" in row["date"], f"date {row['date']!r} outside filter window"

    def test_start_filter_alone(self, client: TestClient) -> None:
        narrow = client.get(
            f"/api/forecasts?factor={FACTOR}&start=2024-01-01"
        ).json()["series"]
        assert all(row["date"] >= "2024-01-01" for row in narrow)

    def test_end_filter_alone(self, client: TestClient) -> None:
        narrow = client.get(
            f"/api/forecasts?factor={FACTOR}&end=2015-12-31"
        ).json()["series"]
        assert all(row["date"] <= "2015-12-31" for row in narrow)


# ===========================================================================
# /api/eval
# ===========================================================================

class TestEval:
    def test_200_at_least_5_rows(self, client: TestClient) -> None:
        r = client.get(f"/api/eval?factor={FACTOR}")
        assert r.status_code == 200
        data = r.json()
        assert data["factor"] == FACTOR
        assert len(data["rows"]) >= 5

    def test_contains_harx_and_gbm(self, client: TestClient) -> None:
        r = client.get(f"/api/eval?factor={FACTOR}")
        models = {row["model"] for row in r.json()["rows"]}
        assert "HAR-X" in models, f"HAR-X not found, got {models}"
        assert "GBM" in models, f"GBM not found, got {models}"

    def test_row_schema(self, client: TestClient) -> None:
        r = client.get(f"/api/eval?factor={FACTOR}")
        row = r.json()["rows"][0]
        for key in ("model", "n_oos", "QLIKE_daily", "Corr_daily", "QLIKE_week", "Corr_week"):
            assert key in row, f"missing key {key!r}"


# ===========================================================================
# /api/shap
# ===========================================================================

class TestShap:
    def test_200_drivers_sorted_by_abs(self, client: TestClient) -> None:
        r = client.get(
            f"/api/shap?factor={FACTOR}&date={SHAP_DATE}&model=GBM"
        )
        assert r.status_code == 200
        data = r.json()
        drivers = data["drivers"]
        assert isinstance(drivers, list)
        abs_vals = [abs(d["shap_value"]) for d in drivers]
        assert abs_vals == sorted(abs_vals, reverse=True), (
            "drivers are not sorted by |shap_value| descending"
        )

    def test_k_parameter_respected(self, client: TestClient) -> None:
        r = client.get(
            f"/api/shap?factor={FACTOR}&date={SHAP_DATE}&model=GBM&k=3"
        )
        assert r.status_code == 200
        assert len(r.json()["drivers"]) <= 3

    def test_bad_model_422(self, client: TestClient) -> None:
        r = client.get(
            f"/api/shap?factor={FACTOR}&date={SHAP_DATE}&model=BADMODEL"
        )
        assert r.status_code == 422

    def test_bogus_date_404(self, client: TestClient) -> None:
        r = client.get(
            f"/api/shap?factor={FACTOR}&date=1800-01-01&model=GBM"
        )
        assert r.status_code == 404

    def test_driver_schema(self, client: TestClient) -> None:
        r = client.get(
            f"/api/shap?factor={FACTOR}&date={SHAP_DATE}&model=GBM"
        )
        for d in r.json()["drivers"]:
            assert "feature" in d
            assert "shap_value" in d
            assert isinstance(d["shap_value"], (int, float))


# ===========================================================================
# /api/shap/dates
# ===========================================================================

class TestShapDates:
    def test_200_sorted_unique_dates(self, client: TestClient) -> None:
        r = client.get(f"/api/shap/dates?factor={FACTOR}")
        assert r.status_code == 200
        dates = r.json()["dates"]
        assert isinstance(dates, list)
        assert len(dates) > 0
        # all match YYYY-MM-DD
        for d in dates:
            assert DATE_RE.match(d), f"date {d!r} not YYYY-MM-DD"
        # sorted and unique
        assert dates == sorted(set(dates)), "dates are not sorted unique"


# ===========================================================================
# /api/shap/timeseries  (v2)
# ===========================================================================

class TestShapTimeseries:
    def test_200_harx(self, client: TestClient) -> None:
        r = client.get(f"/api/shap/timeseries?factor={FACTOR}&model=HAR-X")
        assert r.status_code == 200
        data = r.json()
        assert data["factor"] == FACTOR
        assert data["model"] == "HAR-X"

    def test_features_end_with_other(self, client: TestClient) -> None:
        r = client.get(f"/api/shap/timeseries?factor={FACTOR}&model=HAR-X")
        features = r.json()["features"]
        assert len(features) > 0
        assert features[-1] == "Other", (
            f"last feature is {features[-1]!r}, expected 'Other'"
        )

    def test_series_periods_match_yyyy_mm(self, client: TestClient) -> None:
        r = client.get(f"/api/shap/timeseries?factor={FACTOR}&model=HAR-X")
        series = r.json()["series"]
        assert len(series) > 0
        for item in series:
            assert MONTH_RE.match(item["period"]), (
                f"period {item['period']!r} does not match YYYY-MM"
            )

    def test_values_keys_equal_features(self, client: TestClient) -> None:
        r = client.get(f"/api/shap/timeseries?factor={FACTOR}&model=HAR-X")
        data = r.json()
        features = set(data["features"])
        for item in data["series"]:
            assert set(item["values"].keys()) == features, (
                f"values keys {set(item['values'].keys())} != features {features}"
            )

    def test_bad_model_422(self, client: TestClient) -> None:
        r = client.get(f"/api/shap/timeseries?factor={FACTOR}&model=BADMODEL")
        assert r.status_code == 422


# ===========================================================================
# /api/sentiment
# ===========================================================================

class TestSentiment:
    def test_200_non_empty(self, client: TestClient) -> None:
        r = client.get("/api/sentiment")
        assert r.status_code == 200
        data = r.json()
        assert "rows" in data
        assert len(data["rows"]) > 0

    def test_row_schema(self, client: TestClient) -> None:
        r = client.get("/api/sentiment")
        row = r.json()["rows"][0]
        for key in ("date", "label", "sent_score"):
            assert key in row, f"missing key {key!r}"
        assert DATE_RE.match(row["date"]), f"date {row['date']!r} not YYYY-MM-DD"


# ===========================================================================
# /api/alerts
# ===========================================================================

class TestAlerts:
    def test_200_list_with_date_and_filename(self, client: TestClient) -> None:
        r = client.get("/api/alerts")
        assert r.status_code == 200
        data = r.json()
        assert "alerts" in data
        assert len(data["alerts"]) > 0
        for item in data["alerts"]:
            assert "date" in item
            assert "filename" in item
            assert DATE_RE.match(item["date"]), f"date {item['date']!r} not YYYY-MM-DD"

    def test_alert_detail_200(self, client: TestClient) -> None:
        r = client.get(f"/api/alerts/{ALERT_DATE}")
        assert r.status_code == 200
        data = r.json()
        assert data["date"] == ALERT_DATE
        assert "text" in data
        assert len(data["text"]) > 0

    def test_alert_detail_404_on_missing(self, client: TestClient) -> None:
        r = client.get(f"/api/alerts/{ALERT_DATE_MISSING}")
        assert r.status_code == 404


# ===========================================================================
# POST /api/chat  - no DEEPSEEK_API_KEY → 503
# ===========================================================================

class TestChat:
    def test_503_when_no_api_key(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        r = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert r.status_code == 503
        assert r.json()["detail"] == "DEEPSEEK_API_KEY not configured"


# ===========================================================================
# POST /api/chat/stream - no DEEPSEEK_API_KEY → SSE error event
# ===========================================================================

class TestChatStream:
    def test_error_event_when_no_api_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        r = client.post(
            "/api/chat/stream",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        body = r.content
        body_str = body.decode()
        assert "event: error" in body_str, f"body did not contain 'event: error': {body_str!r}"
        assert "DEEPSEEK_API_KEY not configured" in body_str, (
            f"error detail not found in: {body_str!r}"
        )

    def test_error_frame_parses(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        r = client.post(
            "/api/chat/stream",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        frames = _sse_frames(r.content)
        assert len(frames) >= 1
        error_frames = [f for f in frames if f.get("event") == "error"]
        assert error_frames, f"no 'error' event found in frames: {frames}"
        detail = json.loads(error_frames[0]["data"])["detail"]
        assert detail == "DEEPSEEK_API_KEY not configured"


# ===========================================================================
# Chat generator unit test - stub get_client with two completions:
#   1st → tool_call(get_forecast)   2nd → plain message
# ===========================================================================

class TestChatGeneratorUnit:
    """Unit-tests run_chat_turn_stream with a fully stubbed client.

    The stub client's create() is called twice:
      - call 1: returns a completion with a single tool_call (get_forecast)
      - call 2: returns a plain text completion (no tool_calls)

    Expected yield sequence:
      ("tool", "get_forecast")
      ("delta", <stub text>)
      ("done", None)
    """

    _STUB_TEXT = "The realized vol was 0.5."

    def _make_stub_client(self) -> MagicMock:
        """Build a minimal OpenAI-compatible stub with two scripted responses."""
        # ---- first response: tool call ----
        tc = MagicMock()
        tc.id = "call_abc123"
        tc.function.name = "get_forecast"
        tc.function.arguments = '{"date": "2020-01-01", "factor": "CNH_ATM_PC1"}'
        tc.model_dump.return_value = {
            "id": "call_abc123",
            "type": "function",
            "function": {"name": "get_forecast", "arguments": tc.function.arguments},
        }

        first_msg = MagicMock()
        first_msg.tool_calls = [tc]
        first_msg.content = ""

        first_resp = MagicMock()
        first_resp.choices = [MagicMock(message=first_msg)]

        # ---- second response: plain text ----
        second_msg = MagicMock()
        second_msg.tool_calls = None
        second_msg.content = self._STUB_TEXT

        second_resp = MagicMock()
        second_resp.choices = [MagicMock(message=second_msg)]

        client_stub = MagicMock()
        client_stub.chat.completions.create.side_effect = [first_resp, second_resp]
        return client_stub

    def test_yields_tool_delta_done(self) -> None:
        stub = self._make_stub_client()

        with patch("finai.app.chat.get_client", return_value=stub):
            from finai.app.chat import run_chat_turn_stream

            events = list(
                run_chat_turn_stream(history=[], user_message="What was vol on 2020-01-01?")
            )

        kinds = [e[0] for e in events]
        assert kinds[0] == "tool", f"first event should be 'tool', got {kinds[0]!r}"
        assert kinds[-1] == "done", f"last event should be 'done', got {kinds[-1]!r}"
        assert "delta" in kinds, "no 'delta' event found"

        # tool event names the correct function
        tool_events = [e for e in events if e[0] == "tool"]
        assert tool_events[0][1] == "get_forecast"

        # delta carries the stub text
        delta_events = [e for e in events if e[0] == "delta"]
        combined_text = "".join(e[1] for e in delta_events)
        assert self._STUB_TEXT in combined_text, (
            f"stub text not found in deltas: {combined_text!r}"
        )

        # done is (done, None)
        assert events[-1] == ("done", None)

    def test_exactly_two_create_calls(self) -> None:
        stub = self._make_stub_client()

        with patch("finai.app.chat.get_client", return_value=stub):
            from finai.app.chat import run_chat_turn_stream

            list(run_chat_turn_stream(history=[], user_message="What was vol on 2020-01-01?"))

        assert stub.chat.completions.create.call_count == 2, (
            f"expected 2 create() calls, got {stub.chat.completions.create.call_count}"
        )


# ===========================================================================
# GET /api/tearsheet  (v4)
# ===========================================================================

class TestTearsheet:
    def test_200_pdf_for_cnh_atm_pc1_harx(self, client: TestClient) -> None:
        r = client.get(f"/api/tearsheet?factor={FACTOR}&model=HAR-X")
        assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text}"
        assert "application/pdf" in r.headers.get("content-type", ""), (
            f"expected application/pdf content-type, got {r.headers.get('content-type')!r}"
        )
        assert r.content[:4] == b"%PDF", (
            f"body does not start with %PDF: {r.content[:20]!r}"
        )

    def test_content_disposition_contains_factor(self, client: TestClient) -> None:
        r = client.get(f"/api/tearsheet?factor={FACTOR}&model=HAR-X")
        assert r.status_code == 200
        cd = r.headers.get("content-disposition", "")
        assert FACTOR in cd, (
            f"Content-Disposition {cd!r} does not contain factor {FACTOR!r}"
        )

    def test_422_bad_model(self, client: TestClient) -> None:
        r = client.get(f"/api/tearsheet?factor={FACTOR}&model=BADMODEL")
        assert r.status_code == 422

    def test_404_bogus_factor(self, client: TestClient) -> None:
        r = client.get("/api/tearsheet?factor=BOGUS_FACTOR_XYZ&model=HAR-X")
        assert r.status_code == 404
