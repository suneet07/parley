"""Public-demo safety: kill switch, caller blocklist, per-caller daily cap."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient

from parley.api.server import create_app
from parley.api.store import CallStore
from parley.core.brain.state import CallState
from parley.core.outbound.compliance import PRESETS, ComplianceEngine
from parley.core.outbound.queue import OutboundQueue
from parley.core.outbound.store import OutboundStore
from tests.test_outbound import FakeDialer, _make_contact, _make_job, _make_settings


class DummyAgent:
    resources: dict = {}


def _app(tmp_path: Path):
    store = CallStore(f"sqlite+aiosqlite:///{(tmp_path / 'abuse.db').as_posix()}")
    return create_app(agent=DummyAgent(), store=store), store  # type: ignore[arg-type]


def _voice(client: TestClient, caller: str = "+919800000001") -> str:
    resp = client.post("/twilio/voice", data={"From": caller, "CallSid": "CA1"})
    assert resp.status_code == 200
    return resp.text


def test_kill_switch_blocks_phone_and_webrtc(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        app.state.settings.twilio_auth_token = ""
        app.state.settings.kill_switch = True
        twiml = _voice(client)
        assert "not available" in twiml and "<Hangup" in twiml and "<Stream" not in twiml
        offer = client.post("/api/offer", json={"sdp": "v=0", "type": "offer"})
        assert offer.status_code == 503


def test_blocklisted_caller_is_hung_up(tmp_path: Path) -> None:
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        app.state.settings.twilio_auth_token = ""
        app.state.settings.caller_blocklist = "+919811111111, +14155550100"
        assert "<Stream" not in _voice(client, "+919811111111")
        assert "<Stream" in _voice(client, "+919822222222")


def test_per_caller_daily_cap(tmp_path: Path) -> None:
    app, store = _app(tmp_path)
    with TestClient(app) as client:
        app.state.settings.twilio_auth_token = ""
        app.state.settings.max_calls_per_caller_per_day = 2
        caller = "+919833333333"
        for i in range(2):
            state = CallState(call_id=f"c{i}", channel="phone", caller_number=caller)
            client.portal.call(store.save, state, None)
        twiml = _voice(client, caller)
        assert "maximum number of calls" in twiml and "<Stream" not in twiml
        # A different caller is unaffected.
        assert "<Stream" in _voice(client, "+919844444444")


async def test_outbound_queue_does_not_dial_when_kill_switch_on(tmp_path: Path) -> None:
    store = OutboundStore(f"sqlite+aiosqlite:///{(tmp_path / 'out.db').as_posix()}")
    await store.init()
    settings = _make_settings()
    settings.kill_switch = True
    engine = ComplianceEngine(
        rules=PRESETS["in"], store=store, settings=settings, business_tz="Asia/Kolkata"
    )
    dialer = FakeDialer()
    queue = OutboundQueue(
        store=store,
        compliance=engine,
        dialer=dialer,
        settings=settings,
        business_name="Kettle & Co",
    )
    now = datetime(2026, 5, 10, 11, 0, tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(UTC)
    job = await queue.enqueue(_make_job(_make_contact("+919876543210"), job_id="job_ks"), now)
    after = await queue.process(job.job_id, now)
    assert after.status == "queued"
    assert dialer.dialed == []
    await store.close()
