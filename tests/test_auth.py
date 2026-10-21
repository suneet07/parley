"""Tests for API key authentication, live calls, tracing, and cost wiring."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from parley.api.auth import require_api_key
from parley.api.live import LiveCalls, live_calls
from parley.api.server import create_app
from parley.api.store import CallStore
from parley.config import BusinessConfig, EscalationConfig, HoursConfig, Settings
from parley.core.brain.state import CallState, Turn
from parley.core.pipeline.bot import CallControl
from parley.core.telemetry import tracing


class DummyAgent:
    def __init__(self, transfer_to: str | None = "+18005559999") -> None:
        self.business = BusinessConfig(
            business="Demo Shop",
            persona="Helpful voice assistant.",
            hours=HoursConfig(tz="UTC", open="09:00", close="18:00"),
            escalation=EscalationConfig(transfer_to=transfer_to, on=["human_requested"]),
        )
        self.tools = SimpleNamespace(list_definitions=lambda: [])
        self.retriever = SimpleNamespace(chunks=[])


def test_auth_no_key_non_loopback_returns_401(tmp_path: Path) -> None:
    db_path = tmp_path / "auth1.db"
    store = CallStore(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    app = create_app(agent=DummyAgent(), store=store)  # type: ignore[arg-type]

    with TestClient(app) as client:
        app.state.settings.api_key = ""

        # Starlette TestClient default host is "testclient" (not loopback) -> 401
        for path in (
            "/api/calls",
            "/api/metrics",
            "/api/evals/latest",
            "/api/config",
            "/api/live",
        ):
            resp = client.get(path)
            assert resp.status_code == 401, f"Expected 401 for {path}, got {resp.status_code}"

        # /health stays open
        health = client.get("/health")
        assert health.status_code == 200


def test_auth_loopback_allowance_direct_dependency() -> None:
    settings = Settings(api_key="")
    fake_app = SimpleNamespace(state=SimpleNamespace(settings=settings))

    for host in ("127.0.0.1", "::1"):
        fake_req = SimpleNamespace(
            app=fake_app,
            client=SimpleNamespace(host=host),
            headers={},
        )
        # Should not raise HTTPException
        require_api_key(fake_req)  # type: ignore[arg-type]


def test_auth_configured_key_missing_wrong_right(tmp_path: Path) -> None:
    db_path = tmp_path / "auth2.db"
    store = CallStore(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    app = create_app(agent=DummyAgent(), store=store)  # type: ignore[arg-type]

    with TestClient(app) as client:
        app.state.settings.api_key = "top-secret-token"

        # Missing header -> 401
        assert client.get("/api/calls").status_code == 401

        # Wrong token -> 401
        assert (
            client.get(
                "/api/calls",
                headers={"Authorization": "Bearer wrong-token"},
            ).status_code
            == 401
        )

        # Right token -> 200
        ok_resp = client.get(
            "/api/calls",
            headers={"Authorization": "Bearer top-secret-token"},
        )
        assert ok_resp.status_code == 200
        assert ok_resp.json() == []

        # /health remains public without Authorization header
        assert client.get("/health").status_code == 200


def test_live_calls_registry_and_takeover(tmp_path: Path) -> None:
    db_path = tmp_path / "live.db"
    store = CallStore(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    agent = DummyAgent(transfer_to="+18005550199")
    app = create_app(agent=agent, store=store)  # type: ignore[arg-type]

    now = datetime.now(UTC)
    state = CallState(
        call_id="live-call-1",
        direction="inbound",
        channel="phone",
        started_at=now,
        intents=["order_status"],
        verified=True,
        summary="Customer asking about order #1234",
        transcript=[
            Turn(
                role="user",
                text="My email is alice@example.com and card is 4532 0151 1283 0366",
                at=now,
            ),
        ],
    )
    fake_session = SimpleNamespace(state=state, agent=agent)

    transferred: list[tuple[str, str]] = []

    async def _transfer(call_id: str, whisper_summary: str | None) -> bool:
        transferred.append((call_id, whisper_summary or ""))
        return True

    control = CallControl(transfer=_transfer)
    live_calls.register(fake_session, "phone", control)
    try:
        with TestClient(app) as client:
            app.state.settings.api_key = "live-secret"
            headers = {"Authorization": "Bearer live-secret"}

            resp = client.get("/api/live", headers=headers)
            assert resp.status_code == 200
            items = resp.json()
            assert len(items) == 1
            item = items[0]
            assert item["call_id"] == "live-call-1"
            assert item["intent"] == "order_status"
            assert item["verified"] is True
            # Transcript must be redacted
            assert "a***@example.com" in item["transcript"][0]["text"]
            assert "[CARD]" in item["transcript"][0]["text"]

            # Takeover succeeds when control.transfer and transfer_to are set
            takeover_ok = client.post("/api/live/live-call-1/takeover", headers=headers)
            assert takeover_ok.status_code == 200
            assert takeover_ok.json() == {"ok": True}
            assert transferred == [("live-call-1", "Customer asking about order #1234")]

            # Takeover on call without transfer control -> 409
            state2 = CallState(call_id="live-webrtc-2", channel="webrtc", started_at=now)
            live_calls.register(SimpleNamespace(state=state2, agent=agent), "webrtc", None)
            takeover_409 = client.post("/api/live/live-webrtc-2/takeover", headers=headers)
            assert takeover_409.status_code == 409
    finally:
        live_calls.unregister("live-call-1")
        live_calls.unregister("live-webrtc-2")


def test_tracing_setup_noop_when_unconfigured() -> None:
    tracing._reset_for_tests()
    settings = Settings(
        otel_exporter_otlp_endpoint="",
        langfuse_public_key="",
        langfuse_secret_key="",
    )
    assert tracing.setup(settings) is False


@pytest.mark.asyncio
async def test_live_calls_isolated_unit() -> None:
    reg = LiveCalls()
    now = datetime.now(UTC)
    turns = [Turn(role="user", text=f"turn {i}", at=now) for i in range(10)]
    st = CallState(call_id="c1", started_at=now, transcript=turns)
    reg.register(SimpleNamespace(state=st), "phone", None)
    listed: list[dict[str, Any]] = reg.list()
    assert len(listed) == 1
    assert len(listed[0]["transcript"]) == 6
    reg.unregister("c1")
    assert reg.list() == []
