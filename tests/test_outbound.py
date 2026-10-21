"""Tests for outbound calling compliance, store, queue, worker, voicemail, built-in opt_out, and API routes."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

import parley.api.twilio_routes as twilio_routes
from parley.api.server import create_app
from parley.api.store import CallStore
from parley.config import Settings, load_business
from parley.core.brain.state import CallState
from parley.core.outbound.compliance import PRESETS, ComplianceEngine, callee_timezone
from parley.core.outbound.models import CallJob, Contact
from parley.core.outbound.queue import OutboundQueue
from parley.core.outbound.store import OutboundStore
from parley.core.outbound.voicemail import voicemail_twiml
from parley.core.tools.base import ToolContext
from parley.core.tools.builtins import OptOutArgs, opt_out
from parley.core.tools.gateway import ToolGateway
from parley.core.tools.registry import ToolRegistry


class FakeDialer:
    """In-memory dialer recording outbound dial and SMS calls without network."""

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.fail_with = fail_with
        self.dialed: list[CallJob] = []
        self.sms_sent: list[tuple[str, str]] = []
        self._counter = 0

    async def dial(self, job: CallJob) -> str:
        if self.fail_with is not None:
            raise self.fail_with
        self._counter += 1
        self.dialed.append(job.model_copy(deep=True))
        return f"CA_FAKE_{self._counter:04d}"

    async def send_sms(self, phone: str, body: str) -> None:
        self.sms_sent.append((phone, body))


class DummyAgent:
    def __init__(self) -> None:
        self.resources: dict[str, Any] = {}


def _make_settings(**overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "compliance_region": "in",
        "outbound_webhook_secret": "test-secret-key",
        "outbound_max_concurrent": 3,
        "outbound_daily_cap_per_number": 200,
        "outbound_daily_cap_per_contact": 2,
        "outbound_max_attempts": 3,
        "outbound_sms_fallback": True,
        "twilio_phone_number": "+918000000000",
        "tenant_id": "default",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _make_contact(
    phone: str = "+919876543210",
    *,
    name: str | None = "Aarav",
    tz: str | None = None,
    consent: bool = True,
    consent_at: datetime | None = datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
) -> Contact:
    return Contact(
        contact_id="cnt_test",
        phone=phone,
        name=name,
        tz=tz,
        consent=consent,
        consent_source="web_form" if consent else None,
        consent_at=consent_at,
    )


def _make_job(
    contact: Contact | None = None,
    *,
    job_id: str = "job_1",
    goal: str = "confirm your kettle order",
) -> CallJob:
    c = contact or _make_contact()
    now = datetime(2026, 5, 10, 10, 0, tzinfo=UTC)
    return CallJob(
        job_id=job_id,
        tenant_id="default",
        contact=c,
        goal=goal,
        trigger="webhook",
        created_at=now,
        updated_at=now,
    )


def _sign(body: bytes, secret: str, ts: int | None = None) -> dict[str, str]:
    """Headers for a signed outbound request: HMAC over "<timestamp>.<body>"."""
    ts = int(datetime.now(UTC).timestamp()) if ts is None else ts
    digest = hmac.new(secret.encode("utf-8"), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return {"X-Parley-Timestamp": str(ts), "X-Parley-Signature": f"sha256={digest}"}


# 1. Calling hours, timezones, and DST transitions
async def test_compliance_hours_and_dst(tmp_path: Path) -> None:
    store = OutboundStore(f"sqlite+aiosqlite:///{(tmp_path / 'out1.db').as_posix()}")
    await store.init()
    settings = _make_settings()
    engine_in = ComplianceEngine(
        rules=PRESETS["in"],
        store=store,
        settings=settings,
        business_tz="Asia/Kolkata",
    )

    ist = ZoneInfo("Asia/Kolkata")
    job_in = _make_job(_make_contact("+919876543210", tz=None))
    assert callee_timezone(job_in.contact, "UTC") == ist

    # 20:59 IST -> allowed
    now_2059 = datetime(2026, 5, 10, 20, 59, tzinfo=ist).astimezone(UTC)
    d_ok = await engine_in.check(job_in, now_2059)
    assert d_ok.allowed is True
    assert d_ok.reasons == []
    assert d_ok.retry_at is None

    # 21:00 IST (exclusive window end) -> blocked, retry_at = next day 09:00 IST in UTC
    now_2100 = datetime(2026, 5, 10, 21, 0, tzinfo=ist).astimezone(UTC)
    d_2100 = await engine_in.check(job_in, now_2100)
    assert d_2100.allowed is False
    assert "outside_calling_hours" in d_2100.reasons
    expected_next_morning = datetime(2026, 5, 11, 9, 0, tzinfo=ist).astimezone(UTC)
    assert d_2100.retry_at == expected_next_morning

    # 08:59 IST -> blocked, retry_at = same day 09:00 IST in UTC
    now_0859 = datetime(2026, 5, 10, 8, 59, tzinfo=ist).astimezone(UTC)
    d_0859 = await engine_in.check(job_in, now_0859)
    assert d_0859.allowed is False
    assert "outside_calling_hours" in d_0859.reasons
    expected_same_morning = datetime(2026, 5, 10, 9, 0, tzinfo=ist).astimezone(UTC)
    assert d_0859.retry_at == expected_same_morning

    # US rules: contact tz America/Los_Angeles at 07:59 local -> blocked (US window starts at 08:00)
    engine_us = ComplianceEngine(
        rules=PRESETS["us"],
        store=store,
        settings=_make_settings(compliance_region="us"),
        business_tz="America/New_York",
    )
    la_tz = ZoneInfo("America/Los_Angeles")
    job_us = _make_job(_make_contact("+14155550199", tz="America/Los_Angeles"), job_id="job_us")
    now_la_0759 = datetime(2026, 5, 10, 7, 59, tzinfo=la_tz).astimezone(UTC)
    d_us = await engine_us.check(job_us, now_la_0759)
    assert d_us.allowed is False
    assert "outside_calling_hours" in d_us.reasons
    assert d_us.retry_at == datetime(2026, 5, 10, 8, 0, tzinfo=la_tz).astimezone(UTC)

    # DST transition: America/New_York on 2nd Sunday of March (2026-03-08) at 02:30 local -> no crash
    ny_tz = ZoneInfo("America/New_York")
    job_ny = _make_job(_make_contact("+12125550123", tz="America/New_York"), job_id="job_ny")
    dst_local = datetime(2026, 3, 8, 2, 30, tzinfo=ny_tz)
    d_dst = await engine_us.check(job_ny, dst_local.astimezone(UTC))
    assert d_dst.allowed is False
    assert "outside_calling_hours" in d_dst.reasons
    assert d_dst.retry_at == datetime(2026, 3, 8, 8, 0, tzinfo=ny_tz).astimezone(UTC)

    await store.close()


# 2. Consent missing and DNC -> blocked, retry_at None, audit row written
async def test_compliance_consent_missing_and_dnc(tmp_path: Path) -> None:
    store = OutboundStore(f"sqlite+aiosqlite:///{(tmp_path / 'out2.db').as_posix()}")
    await store.init()
    settings = _make_settings()
    engine = ComplianceEngine(
        rules=PRESETS["in"],
        store=store,
        settings=settings,
        business_tz="Asia/Kolkata",
    )
    ist = ZoneInfo("Asia/Kolkata")
    now_valid = datetime(2026, 5, 10, 14, 0, tzinfo=ist).astimezone(UTC)

    # Consent False -> blocked, retry_at None, audit written
    no_consent_job = _make_job(
        _make_contact("+919876543210", consent=False, consent_at=None),
        job_id="job_noconsent",
    )
    dec1 = await engine.check(no_consent_job, now_valid)
    assert dec1.allowed is False
    assert "no_consent" in dec1.reasons
    assert dec1.retry_at is None

    # Consent True but consent_at None -> blocked with no_consent
    missing_ts_job = _make_job(
        _make_contact("+919876543210", consent=True, consent_at=None),
        job_id="job_nots",
    )
    dec_ts = await engine.check(missing_ts_job, now_valid)
    assert dec_ts.allowed is False
    assert "no_consent" in dec_ts.reasons
    assert dec_ts.retry_at is None

    # DNC -> blocked, retry_at None
    await store.add_dnc("default", "+91 98765 43211", source="manual")
    dnc_job = _make_job(_make_contact("+919876543211"), job_id="job_dnc")
    dec2 = await engine.check(dnc_job, now_valid)
    assert dec2.allowed is False
    assert "do_not_call" in dec2.reasons
    assert dec2.retry_at is None

    # Audit log has rows and masks phone numbers to last 4 digits
    logs = await store.audit_log("default")
    assert len(logs) == 3
    for entry in logs:
        assert entry["phone"].endswith(("3210", "3211"))
        assert "+91987654" not in entry["phone"]

    await store.close()


# 3. Daily caps per contact and per number
async def test_compliance_daily_caps(tmp_path: Path) -> None:
    store = OutboundStore(f"sqlite+aiosqlite:///{(tmp_path / 'out3.db').as_posix()}")
    await store.init()
    settings = _make_settings(outbound_daily_cap_per_contact=2, outbound_daily_cap_per_number=3)
    engine = ComplianceEngine(
        rules=PRESETS["in"],
        store=store,
        settings=settings,
        business_tz="Asia/Kolkata",
    )
    ist = ZoneInfo("Asia/Kolkata")
    now = datetime(2026, 5, 10, 12, 0, tzinfo=ist).astimezone(UTC)

    phone = "+919876543210"
    job = _make_job(_make_contact(phone), job_id="job_cap")

    # Record 2 dials today for the same contact
    await store.record_dial(
        "default", phone, settings.twilio_phone_number, now - timedelta(hours=2), "j1"
    )
    await store.record_dial(
        "default", phone, settings.twilio_phone_number, now - timedelta(hours=1), "j2"
    )

    # 3rd dial same local day -> blocked with contact_daily_cap and retry_at = next window start
    dec_contact = await engine.check(job, now)
    assert dec_contact.allowed is False
    assert "contact_daily_cap" in dec_contact.reasons
    expected_next_win = datetime(2026, 5, 11, 9, 0, tzinfo=ist).astimezone(UTC)
    assert dec_contact.retry_at == expected_next_win

    # Record 3rd dial from the same Twilio number to a different contact -> hits number_daily_cap (3)
    other_phone = "+919876549999"
    await store.record_dial(
        "default", other_phone, settings.twilio_phone_number, now - timedelta(minutes=10), "j3"
    )
    job_other = _make_job(_make_contact(other_phone), job_id="job_num_cap")
    dec_num = await engine.check(job_other, now)
    assert dec_num.allowed is False
    assert "number_daily_cap" in dec_num.reasons
    # The per-number cap is a UTC-day budget, so it retries at the next UTC midnight
    # (the hours check then defers to the callee's window if needed).
    assert dec_num.retry_at == datetime(now.year, now.month, now.day, tzinfo=UTC) + timedelta(
        days=1
    )

    await store.close()


# 4. All reasons reported together (no consent + outside hours both present)
async def test_compliance_reports_all_reasons_together(tmp_path: Path) -> None:
    store = OutboundStore(f"sqlite+aiosqlite:///{(tmp_path / 'out4.db').as_posix()}")
    await store.init()
    settings = _make_settings()
    engine = ComplianceEngine(
        rules=PRESETS["in"],
        store=store,
        settings=settings,
        business_tz="Asia/Kolkata",
    )
    ist = ZoneInfo("Asia/Kolkata")
    late_night = datetime(2026, 5, 10, 23, 0, tzinfo=ist).astimezone(UTC)

    job = _make_job(
        _make_contact("+919876543210", consent=False, consent_at=None),
        job_id="job_multi",
    )
    decision = await engine.check(job, late_night)
    assert decision.allowed is False
    assert "no_consent" in decision.reasons
    assert "outside_calling_hours" in decision.reasons
    # Because no_consent is non-retryable, retry_at must be None
    assert decision.retry_at is None

    await store.close()


# 5. Queue lifecycle: retries (+3h, +1d), SMS fallback after max attempts, and DNC handling
async def test_queue_retries_sms_fallback_and_dnc(tmp_path: Path) -> None:
    store = OutboundStore(f"sqlite+aiosqlite:///{(tmp_path / 'out5.db').as_posix()}")
    await store.init()
    settings = _make_settings(outbound_daily_cap_per_contact=5, outbound_max_attempts=3)
    engine = ComplianceEngine(
        rules=PRESETS["in"],
        store=store,
        settings=settings,
        business_tz="Asia/Kolkata",
    )
    dialer = FakeDialer()
    queue = OutboundQueue(
        store=store,
        compliance=engine,
        dialer=dialer,
        settings=settings,
        business_name="Kettle & Co",
    )

    ist = ZoneInfo("Asia/Kolkata")
    t0 = datetime(2026, 5, 10, 10, 0, tzinfo=ist).astimezone(UTC)

    job = _make_job(
        _make_contact("+919876543210", name="Priya"), job_id="job_q1", goal="confirm delivery"
    )
    await queue.enqueue(job, t0)

    # Attempt 1: dials once, attempts == 1, status == in_progress
    j1 = await queue.process("job_q1", t0)
    assert j1.status == "in_progress"
    assert j1.attempts == 1
    assert len(dialer.dialed) == 1
    assert j1.call_sids == ["CA_FAKE_0001"]

    # First no_answer -> retry_scheduled at t0 + 3h
    j1_fin = await queue.on_call_finished("job_q1", "no_answer", t0)
    assert j1_fin.status == "retry_scheduled"
    assert j1_fin.next_attempt_at == t0 + timedelta(hours=3)
    assert dialer.sms_sent == []

    # Attempt 2 at t0 + 3h
    t1 = t0 + timedelta(hours=3)
    j2 = await queue.process("job_q1", t1)
    assert j2.status == "in_progress"
    assert j2.attempts == 2
    assert len(dialer.dialed) == 2

    # Second no_answer -> retry_scheduled at t1 + 1d
    j2_fin = await queue.on_call_finished("job_q1", "no_answer", t1)
    assert j2_fin.status == "retry_scheduled"
    assert j2_fin.next_attempt_at == t1 + timedelta(days=1)
    assert dialer.sms_sent == []

    # Attempt 3 at t1 + 1d
    t2 = t1 + timedelta(days=1)
    j3 = await queue.process("job_q1", t2)
    assert j3.status == "in_progress"
    assert j3.attempts == 3
    assert len(dialer.dialed) == 3

    # Third no_answer -> attempts exhausted -> completed + exactly one SMS
    j3_fin = await queue.on_call_finished("job_q1", "no_answer", t2)
    assert j3_fin.status == "completed"
    assert j3_fin.outcome == "no_answer"
    assert len(dialer.sms_sent) == 1
    sms_phone, sms_body = dialer.sms_sent[0]
    assert sms_phone == "+919876543210"
    assert "Hi Priya, this is Kettle & Co" in sms_body
    # The goal is free text and could be sensitive, so the SMS never repeats it.
    assert "confirm delivery" not in sms_body
    assert "Reply STOP to opt out." in sms_body

    # do_not_call outcome -> adds DNC, later job for same phone is blocked
    dnc_phone = "+919811122233"
    job_dnc1 = _make_job(_make_contact(dnc_phone), job_id="job_dnc_1")
    await queue.enqueue(job_dnc1, t0)
    await queue.process("job_dnc_1", t0)
    fin_dnc = await queue.on_call_finished("job_dnc_1", "do_not_call", t0)
    assert fin_dnc.status == "completed"
    assert fin_dnc.outcome == "do_not_call"
    assert await store.is_dnc("default", dnc_phone) is True

    job_dnc2 = _make_job(_make_contact(dnc_phone), job_id="job_dnc_2")
    await queue.enqueue(job_dnc2, t0)
    blocked_job = await queue.process("job_dnc_2", t0)
    assert blocked_job.status == "blocked"
    assert blocked_job.last_error is not None
    assert "do_not_call" in blocked_job.last_error

    await store.close()


# 6. Dialer exception -> retry scheduled, last_error set
async def test_queue_dialer_exception_schedules_retry(tmp_path: Path) -> None:
    store = OutboundStore(f"sqlite+aiosqlite:///{(tmp_path / 'out6.db').as_posix()}")
    await store.init()
    settings = _make_settings()
    engine = ComplianceEngine(
        rules=PRESETS["in"],
        store=store,
        settings=settings,
        business_tz="Asia/Kolkata",
    )
    failing_dialer = FakeDialer(fail_with=RuntimeError("Twilio 503 Service Unavailable"))
    queue = OutboundQueue(
        store=store,
        compliance=engine,
        dialer=failing_dialer,
        settings=settings,
    )

    ist = ZoneInfo("Asia/Kolkata")
    t0 = datetime(2026, 5, 10, 11, 0, tzinfo=ist).astimezone(UTC)
    job = _make_job(_make_contact("+919876543210"), job_id="job_err")
    await queue.enqueue(job, t0)

    processed = await queue.process("job_err", t0)
    assert processed.status == "retry_scheduled"
    assert processed.attempts == 1
    assert processed.last_error is not None
    assert "Twilio 503 Service Unavailable" in processed.last_error
    assert processed.next_attempt_at == t0 + timedelta(hours=3)

    await store.close()


# 7. API trigger: no secret -> 503; bad HMAC -> 401; good HMAC -> 202 and job exists
async def test_api_outbound_trigger_and_twiml_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_db = tmp_path / "calls.db"
    out_db = tmp_path / "outbound.db"
    call_store = CallStore(f"sqlite+aiosqlite:///{call_db.as_posix()}")
    out_store = OutboundStore(f"sqlite+aiosqlite:///{out_db.as_posix()}")
    await out_store.init()

    ist = ZoneInfo("Asia/Kolkata")
    fixed_now = datetime(2026, 5, 10, 14, 0, tzinfo=ist).astimezone(UTC)

    settings = _make_settings(outbound_webhook_secret="")
    engine = ComplianceEngine(
        rules=PRESETS["in"],
        store=out_store,
        settings=settings,
        business_tz="Asia/Kolkata",
    )
    dialer = FakeDialer()
    queue = OutboundQueue(
        store=out_store,
        compliance=engine,
        dialer=dialer,
        settings=settings,
        business_name="Kettle & Co",
    )

    app = create_app(agent=DummyAgent(), store=call_store, outbound_queue=queue)  # type: ignore[arg-type]
    app.state.settings = settings

    payload = {
        "contact": {
            "contact_id": "cnt_api_1",
            "phone": "+919876543210",
            "name": "Rohan",
            "tz": "Asia/Kolkata",
            "consent": True,
            "consent_source": "checkout",
            "consent_at": "2026-05-01T10:00:00Z",
        },
        "goal": "recover abandoned cart",
        "metadata": {"cart_id": "CART-99"},
    }
    raw_body = json.dumps(payload).encode("utf-8")

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        # 1. Missing secret -> 503
        r_503 = client.post(
            "/api/outbound/trigger",
            content=raw_body,
            headers={"Content-Type": "application/json"},
        )
        assert r_503.status_code == 503
        assert r_503.json() == {"detail": "outbound disabled: set OUTBOUND_WEBHOOK_SECRET"}

        # Configure secret
        secret = "webhook-hmac-secret-123"
        app.state.settings.outbound_webhook_secret = secret

        # 2. Bad HMAC -> 401
        r_401 = client.post(
            "/api/outbound/trigger",
            content=raw_body,
            headers={
                "Content-Type": "application/json",
                "X-Parley-Timestamp": str(int(datetime.now(UTC).timestamp())),
                "X-Parley-Signature": "sha256=0000bad0000",
            },
        )
        assert r_401.status_code == 401

        # 3. Good HMAC -> 202 and job exists
        # Ensure ComplianceEngine sees daytime hours regardless of wall clock when test runs
        orig_check = engine.check

        async def _check_at_fixed_time(job: CallJob, now: datetime) -> Any:
            return await orig_check(job, fixed_now)

        monkeypatch.setattr(engine, "check", _check_at_fixed_time)

        r_202 = client.post(
            "/api/outbound/trigger",
            content=raw_body,
            headers={
                "Content-Type": "application/json",
                **_sign(raw_body, secret),
            },
        )
        assert r_202.status_code == 202
        resp_data = r_202.json()
        job_id = resp_data["job_id"]
        assert job_id.startswith("job_")

        job_resp = client.get(f"/api/outbound/jobs/{job_id}")
        assert job_resp.status_code == 200
        job_json = job_resp.json()
        assert job_json["job_id"] == job_id
        assert job_json["goal"] == "recover abandoned cart"
        # Job view masks the phone and exposes only metadata keys.
        assert job_json["contact"]["metadata"] == ["cart_id"]
        assert job_json["contact"]["phone"].startswith("******")
        assert job_json["status"] == "in_progress"

        # 4. Trigger without consent -> accepted (202) but job ends blocked with no_consent
        no_consent_payload = {
            "contact": {
                "contact_id": "cnt_noconsent",
                "phone": "+919876543299",
                "consent": False,
            },
            "goal": "promo call",
        }
        nc_bytes = json.dumps(no_consent_payload).encode("utf-8")
        r_nc = client.post(
            "/api/outbound/trigger",
            content=nc_bytes,
            headers={
                "Content-Type": "application/json",
                **_sign(nc_bytes, secret),
            },
        )
        assert r_nc.status_code == 202
        nc_job = client.get(f"/api/outbound/jobs/{r_nc.json()['job_id']}").json()
        assert nc_job["status"] == "blocked"
        assert "no_consent" in nc_job["last_error"]

        # 5. Outbound TwiML route mints stream token and /twilio/ws passes outbound context
        app.state.settings.twilio_auth_token = ""
        app.state.settings.twilio_insecure_dev = True
        twiml_resp = client.post(
            f"/twilio/outbound/{job_id}",
            data={"CallSid": "CA_FAKE_0001", "To": "+919876543210"},
        )
        assert twiml_resp.status_code == 200
        # Job context stays server-side; the stream only carries the one-time token.
        assert 'name="direction"' not in twiml_resp.text
        assert 'name="job_id"' not in twiml_resp.text
        token_match = re.search(r'<Parameter name="token" value="([^"]+)"', twiml_resp.text)
        assert token_match is not None
        stream_token = token_match.group(1)

        captured_ws_calls: list[dict[str, Any]] = []

        async def _fake_run_call(transport: Any, **kwargs: Any) -> CallState:
            captured_ws_calls.append(kwargs)
            return CallState(
                call_id=str(kwargs.get("call_id", "CA_FAKE_0001")),
                direction="outbound",
                disposition="callback_booked",
            )

        monkeypatch.setattr(twilio_routes, "run_call", _fake_run_call)

        with client.websocket_connect("/twilio/ws") as ws:
            ws.send_text(json.dumps({"event": "connected", "protocol": "Call"}))
            ws.send_text(
                json.dumps(
                    {
                        "event": "start",
                        "start": {
                            "streamSid": "MZ_OUT_1",
                            "callSid": "CA_FAKE_0001",
                            "customParameters": {"token": stream_token},
                        },
                    }
                )
            )

        assert len(captured_ws_calls) == 1
        assert captured_ws_calls[0]["direction"] == "outbound"
        assert captured_ws_calls[0]["outbound_goal"] == "recover abandoned cart"
        assert captured_ws_calls[0]["caller_number"] == "+919876543210"

        # The ws handler records the outcome after run_call returns, which can land just after
        # the client socket closes, so poll briefly.
        for _ in range(40):
            finished_job = client.get(f"/api/outbound/jobs/{job_id}").json()
            if finished_job["status"] == "completed":
                break
            await asyncio.sleep(0.05)
        assert finished_job["status"] == "completed"
        assert finished_job["outcome"] == "booked"

    await out_store.close()
    await call_store.close()


# 8. CSV campaign: 3 rows (1 invalid phone) -> accepted 2, rejected 1; pacing spacing correct
async def test_api_csv_campaign_upload_and_pacing(tmp_path: Path) -> None:
    call_store = CallStore(f"sqlite+aiosqlite:///{(tmp_path / 'c8.db').as_posix()}")
    out_store = OutboundStore(f"sqlite+aiosqlite:///{(tmp_path / 'o8.db').as_posix()}")
    await out_store.init()

    settings = _make_settings(
        outbound_webhook_secret="csv-secret",
        outbound_max_concurrent=3,  # spacing = 60 / 3 = 20s
    )
    engine = ComplianceEngine(
        rules=PRESETS["in"],
        store=out_store,
        settings=settings,
        business_tz="Asia/Kolkata",
    )
    queue = OutboundQueue(
        store=out_store,
        compliance=engine,
        dialer=FakeDialer(),
        settings=settings,
    )
    app = create_app(agent=DummyAgent(), store=call_store, outbound_queue=queue)  # type: ignore[arg-type]
    app.state.settings = settings

    csv_content = (
        b"phone,name,tz,consent,consent_source,consent_at,order_id\n"
        b"+919876543210,Alice,Asia/Kolkata,true,csv_import,2026-01-05T10:00:00+00:00,KC-101\n"
        b"98765-INVALID,Bob,Asia/Kolkata,true,csv_import,2026-01-05T10:00:00+00:00,KC-102\n"
        b"+919876543212,Charlie,Asia/Kolkata,true,csv_import,2026-01-05T10:00:00+00:00,KC-103\n"
    )

    with TestClient(app) as client:
        resp = client.post(
            "/api/outbound/campaigns?goal=delivery+update&campaign_id=camp_42",
            content=csv_content,
            headers={
                "Content-Type": "text/csv",
                **_sign(csv_content, "csv-secret"),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["campaign_id"] == "camp_42"
        assert body["accepted"] == 2
        assert len(body["rejected"]) == 1
        assert body["rejected"][0]["row"] == 2

        job_1 = await out_store.get_job("default", body["job_ids"][0])
        job_2 = await out_store.get_job("default", body["job_ids"][1])
        assert job_1 is not None and job_2 is not None
        assert job_1.next_attempt_at is not None and job_2.next_attempt_at is not None
        delta_s = (job_2.next_attempt_at - job_1.next_attempt_at).total_seconds()
        assert delta_s == pytest.approx(20.0)
        assert job_1.contact.metadata["order_id"] == "KC-101"

    await out_store.close()
    await call_store.close()


# 9. opt_out tool adds DNC and returns end_call control
async def test_opt_out_builtin_tool(tmp_path: Path) -> None:
    out_store = OutboundStore(f"sqlite+aiosqlite:///{(tmp_path / 'o9.db').as_posix()}")
    await out_store.init()

    biz_path = Path(__file__).resolve().parents[1] / "examples" / "kettle-and-co" / "business.yaml"
    business = load_business(biz_path)
    reg = ToolRegistry.for_business(business)
    assert reg.get("opt_out") is not None

    gw = ToolGateway(reg)
    state = CallState(
        call_id="call_optout_1",
        tenant_id="default",
        direction="outbound",
        caller_number="+919876543210",
    )
    ctx = ToolContext(
        state=state,
        business=business,
        resources={"outbound_store": out_store},
    )

    res = await gw.call("opt_out", {"reason": "do not call me again"}, ctx)
    assert res.status == "ok"
    assert res.control == "end_call"
    assert res.payload == {"ok": True, "opted_out": True}
    assert state.disposition == "do_not_call"
    assert await out_store.is_dnc("default", "+919876543210") is True

    # Direct function invocation also works
    direct_res = await opt_out(ctx, OptOutArgs(reason="stop calling"))
    assert direct_res.control == "end_call"
    assert direct_res.data == {"opted_out": True}

    await out_store.close()


# 10. voicemail_twiml contains AI disclosure and no <Stream
def test_voicemail_twiml_content() -> None:
    twiml = voicemail_twiml(
        business_name="Kettle & Co",
        goal="confirm your order",
        callback_number="+91 98111 22221",
    )
    assert "AI assistant" in twiml
    assert "Kettle &amp; Co" in twiml or "Kettle & Co" in twiml
    assert "about your recent order" in twiml
    assert "confirm your order" not in twiml  # free-text goal is never spoken
    assert "9, 8, 1, 1, 1, 2, 2, 2, 2, 1" in twiml
    assert "<Hangup" in twiml
    assert "<Stream" not in twiml
