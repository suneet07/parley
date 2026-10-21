"""Regression tests for the outbound compliance review: concurrency, time zones, replay,
consent evidence, voicemail before AI, idempotent outcomes, SMS window, STOP and DNC input."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from parley.api.server import create_app
from parley.api.store import CallStore
from parley.core.outbound.compliance import PRESETS, ComplianceEngine, callee_timezone
from parley.core.outbound.models import to_e164
from parley.core.outbound.queue import OutboundQueue
from parley.core.outbound.store import OutboundStore
from tests.test_outbound import (
    DummyAgent,
    FakeDialer,
    _make_contact,
    _make_job,
    _make_settings,
    _sign,
)

IST = ZoneInfo("Asia/Kolkata")


async def _queue(tmp_path: Path, region: str = "in", **settings_overrides):
    store = OutboundStore(f"sqlite+aiosqlite:///{(tmp_path / 'h.db').as_posix()}")
    await store.init()
    settings = _make_settings(**settings_overrides)
    engine = ComplianceEngine(
        rules=PRESETS[region], store=store, settings=settings, business_tz="Asia/Kolkata"
    )
    dialer = FakeDialer()
    queue = OutboundQueue(
        store=store,
        compliance=engine,
        dialer=dialer,
        settings=settings,
        business_name="Kettle & Co",
    )
    return queue, store, dialer


def _ist(h: int, m: int = 0) -> datetime:
    return datetime(2026, 5, 11, h, m, tzinfo=IST).astimezone(UTC)


async def test_concurrent_process_dials_once(tmp_path: Path) -> None:
    queue, store, dialer = await _queue(tmp_path)
    job = await queue.enqueue(_make_job(_make_contact("+919876543210"), job_id="job_c"), _ist(11))
    await asyncio.gather(*(queue.process(job.job_id, _ist(11)) for _ in range(5)))
    assert len(dialer.dialed) == 1
    assert (await store.get_job("default", "job_c")).attempts == 1
    await store.close()


async def test_concurrent_jobs_respect_contact_cap(tmp_path: Path) -> None:
    queue, store, dialer = await _queue(tmp_path, outbound_daily_cap_per_contact=1)
    for i in range(4):
        await queue.enqueue(_make_job(_make_contact("+919876543210"), job_id=f"job_{i}"), _ist(11))
    await asyncio.gather(*(queue.process(f"job_{i}", _ist(11)) for i in range(4)))
    assert len(dialer.dialed) == 1
    await store.close()


def test_callee_timezone_from_number_never_guesses() -> None:
    assert callee_timezone(_make_contact("+14155550100", tz=None)) == ZoneInfo(
        "America/Los_Angeles"
    )
    assert callee_timezone(_make_contact("+919876543210", tz=None)) == ZoneInfo("Asia/Kolkata")
    # Australia has several zones: without an explicit tz we refuse to guess.
    assert callee_timezone(_make_contact("+61291234567", tz=None)) == ZoneInfo("Australia/Sydney")
    assert callee_timezone(_make_contact("+61812345678", tz=None)) is None
    assert callee_timezone(_make_contact("+61812345678", tz="Australia/Perth")) == ZoneInfo(
        "Australia/Perth"
    )


async def test_pacific_callee_not_called_at_7am_local(tmp_path: Path) -> None:
    queue, store, dialer = await _queue(tmp_path, region="us")
    # 07:00 in San Francisco is 10:00 in New York: still too early for this callee.
    now = datetime(2026, 5, 11, 7, 0, tzinfo=ZoneInfo("America/Los_Angeles")).astimezone(UTC)
    await queue.enqueue(_make_job(_make_contact("+14155550100", tz=None), job_id="job_sf"), now)
    job = await queue.process("job_sf", now)
    assert job.status == "retry_scheduled"
    assert "outside_calling_hours" in (job.last_error or "")
    assert dialer.dialed == []
    await store.close()


async def test_unknown_timezone_blocks(tmp_path: Path) -> None:
    queue, store, _ = await _queue(tmp_path)
    await queue.enqueue(
        _make_job(_make_contact("+61812345678", tz=None), job_id="job_au"), _ist(11)
    )
    job = await queue.process("job_au", _ist(11))
    assert job.status == "blocked" and "unknown_timezone" in (job.last_error or "")
    await store.close()


async def test_contact_attempt_cap_across_jobs(tmp_path: Path) -> None:
    queue, store, dialer = await _queue(
        tmp_path, outbound_daily_cap_per_contact=50, outbound_max_attempts_per_contact_7d=2
    )
    for day in range(3):
        now = _ist(11) + timedelta(days=day)
        await queue.enqueue(_make_job(_make_contact("+919876543210"), job_id=f"job_d{day}"), now)
        await queue.process(f"job_d{day}", now)
    assert len(dialer.dialed) == 2
    await store.close()


async def test_double_finish_schedules_one_retry_and_one_sms(tmp_path: Path) -> None:
    queue, store, dialer = await _queue(tmp_path, outbound_max_attempts=1)
    await queue.enqueue(_make_job(_make_contact("+919876543210"), job_id="job_f"), _ist(11))
    job = await queue.process("job_f", _ist(11))
    sid = job.call_sids[-1]
    await queue.on_call_finished("job_f", "no_answer", _ist(11, 1), call_sid=sid)
    await queue.on_call_finished("job_f", "completed", _ist(11, 2), call_sid=sid)
    final = await store.get_job("default", "job_f")
    assert final.outcome == "no_answer"
    assert len(dialer.sms_sent) == 1
    await store.close()


async def test_sms_fallback_waits_for_calling_window(tmp_path: Path) -> None:
    queue, store, dialer = await _queue(tmp_path, outbound_max_attempts=1)
    await queue.enqueue(_make_job(_make_contact("+919876543210"), job_id="job_s"), _ist(20, 50))
    job = await queue.process("job_s", _ist(20, 50))
    await queue.on_call_finished("job_s", "no_answer", _ist(21, 30), call_sid=job.call_sids[-1])
    assert dialer.sms_sent == []  # 21:30 IST is outside the window
    assert await queue.send_due_sms(_ist(23)) == 0
    next_morning = _ist(9, 1) + timedelta(days=1)
    assert await queue.send_due_sms(next_morning) == 1
    assert await queue.send_due_sms(next_morning) == 0
    assert len(dialer.sms_sent) == 1
    await store.close()


def test_to_e164() -> None:
    assert to_e164("00919876543210") == "+919876543210"
    assert to_e164("+91 98765-43210") == "+919876543210"
    assert to_e164("9876543210", "IN") == "+919876543210"
    with pytest.raises(ValueError):
        to_e164("9876543210")


# ---------------------------------------------------------------- API-level checks


def _client(tmp_path: Path, queue: OutboundQueue) -> tuple[TestClient, object]:
    store = CallStore(f"sqlite+aiosqlite:///{(tmp_path / 'calls.db').as_posix()}")
    app = create_app(agent=DummyAgent(), store=store, outbound_queue=queue)  # type: ignore[arg-type]
    return TestClient(app), app


def _trigger_body(**contact) -> bytes:
    c = {
        "phone": "+919876543210",
        "consent": True,
        "consent_source": "web_form",
        "consent_at": "2026-01-01T00:00:00+00:00",
        **contact,
    }
    return json.dumps({"contact": c, "goal": "order follow up"}).encode()


async def test_trigger_rejects_stale_signature_and_missing_consent_evidence(tmp_path: Path) -> None:
    queue, store, _ = await _queue(tmp_path, outbound_webhook_secret="s3")
    client, app = _client(tmp_path, queue)
    with client:
        app.state.settings.outbound_webhook_secret = "s3"
        body = _trigger_body()
        old = int((datetime.now(UTC) - timedelta(minutes=10)).timestamp())
        r = client.post("/api/outbound/trigger", content=body, headers=_sign(body, "s3", ts=old))
        assert r.status_code == 401

        no_evidence = _trigger_body(consent_at=None)
        r = client.post(
            "/api/outbound/trigger", content=no_evidence, headers=_sign(no_evidence, "s3")
        )
        assert r.status_code == 422
    await store.close()


async def test_trigger_dedupes_identical_requests(tmp_path: Path) -> None:
    queue, store, _ = await _queue(tmp_path, outbound_webhook_secret="s3")
    client, app = _client(tmp_path, queue)
    with client:
        app.state.settings.outbound_webhook_secret = "s3"
        body = _trigger_body()
        first = client.post("/api/outbound/trigger", content=body, headers=_sign(body, "s3"))
        second = client.post("/api/outbound/trigger", content=body, headers=_sign(body, "s3"))
        assert first.status_code == 202
        assert second.status_code == 200 and second.json()["duplicate"] is True
        assert second.json()["job_id"] == first.json()["job_id"]
    await store.close()


async def test_machine_answer_gets_voicemail_not_ai(tmp_path: Path) -> None:
    queue, store, _ = await _queue(tmp_path)
    await queue.enqueue(_make_job(_make_contact("+919876543210"), job_id="job_m"), _ist(11))
    job = await queue.process("job_m", _ist(11))
    client, app = _client(tmp_path, queue)
    with client:
        app.state.settings.twilio_insecure_dev = True
        r = client.post(
            "/twilio/outbound/job_m",
            data={"CallSid": job.call_sids[-1], "AnsweredBy": "machine_end_beep"},
        )
        assert r.status_code == 200
        assert "<Stream" not in r.text and "AI assistant" in r.text
    assert (await store.get_job("default", "job_m")).outcome == "voicemail"
    await store.close()


async def test_sms_stop_adds_dnc_and_dnc_input_validated(tmp_path: Path) -> None:
    queue, store, _ = await _queue(tmp_path, outbound_webhook_secret="s3")
    client, app = _client(tmp_path, queue)
    with client:
        app.state.settings.twilio_insecure_dev = True
        app.state.settings.outbound_webhook_secret = "s3"
        r = client.post("/twilio/sms", data={"From": "+919811100000", "Body": " Stop "})
        assert r.status_code == 200
        bad = json.dumps({"phone": "12345"}).encode()
        r = client.post("/api/outbound/dnc", content=bad, headers=_sign(bad, "s3"))
        assert r.status_code == 422
    assert await store.is_dnc("default", "+919811100000")
    await store.close()


async def test_twilio_webhooks_fail_closed_without_token(tmp_path: Path) -> None:
    queue, store, _ = await _queue(tmp_path)
    client, app = _client(tmp_path, queue)
    with client:
        app.state.settings.twilio_auth_token = ""
        app.state.settings.twilio_insecure_dev = False
        assert client.post("/twilio/voice", data={"From": "+919800000001"}).status_code == 403
    await store.close()
