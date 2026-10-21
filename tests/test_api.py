from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from twilio.request_validator import RequestValidator

from parley.api.server import create_app
from parley.api.store import CallStore
from parley.api.twilio_routes import WHISPERS
from parley.core.brain.state import CallState, Turn


class DummyAgent:
    pass


def test_api_health_twilio_webhooks_and_whisper(tmp_path: Path) -> None:
    db_path = tmp_path / "t.db"
    store = CallStore(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    app = create_app(agent=DummyAgent(), store=store)  # type: ignore[arg-type]

    with TestClient(app) as client:
        # 1. /health ok
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        body = health_resp.json()
        assert body["ok"] is True
        assert body["business"] == "Kettle & Co"

        # 2. POST /twilio/voice with auth token unset returns XML with wss:// and /twilio/ws
        app.state.settings.twilio_auth_token = ""
        app.state.settings.public_base_url = "https://demo.example.com"
        voice_resp = client.post(
            "/twilio/voice",
            data={"From": "+919876543210", "CallSid": "CA12345"},
        )
        assert voice_resp.status_code == 200
        assert '<Stream url="wss://' in voice_resp.text
        assert "/twilio/ws" in voice_resp.text

        # 3. With a token set and a wrong signature -> 403
        token = "super_secret_twilio_token"
        app.state.settings.twilio_auth_token = token
        form_data = {"From": "+919876543210", "CallSid": "CA12345"}
        bad_resp = client.post(
            "/twilio/voice",
            data=form_data,
            headers={"X-Twilio-Signature": "invalid-signature"},
        )
        assert bad_resp.status_code == 403

        # 4. With a correct signature (compute with RequestValidator.compute_signature) -> 200
        validator = RequestValidator(token)
        expected_url = "https://demo.example.com/twilio/voice"
        valid_sig = validator.compute_signature(expected_url, form_data)
        good_resp = client.post(
            "/twilio/voice",
            data=form_data,
            headers={"X-Twilio-Signature": valid_sig},
        )
        assert good_resp.status_code == 200
        assert '<Stream url="wss://' in good_resp.text

        # 5. Whisper endpoint returns the stored summary
        app.state.settings.twilio_auth_token = ""
        WHISPERS["CA999"] = "Caller wants a refund for order KC-10482."
        whisper_resp = client.post("/twilio/whisper/CA999")
        assert whisper_resp.status_code == 200
        assert "Incoming transfer from the AI agent." in whisper_resp.text
        assert "Caller wants a refund for order KC-10482." in whisper_resp.text


async def test_store_save_list_get_roundtrip_redacts_card(tmp_path: Path) -> None:
    db_path = tmp_path / "t.db"
    store = CallStore(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    await store.init()

    state = CallState(
        call_id="call-roundtrip-1",
        tenant_id="default",
        direction="inbound",
        channel="phone",
        caller_number="+919876543210",
        disposition="resolved",
        summary="Processed refund for card 4532 0151 1283 0366.",
        intents=["return_or_refund"],
        transcript=[
            Turn(
                role="user",
                text="Please charge my card 4532 0151 1283 0366 for the replacement.",
            )
        ],
    )

    await store.save(state, metrics={"p50_e2e_ms": 420.0, "count": 1})

    calls = await store.list_calls(tenant_id="default")
    assert len(calls) == 1
    assert calls[0]["id"] == "call-roundtrip-1"
    assert calls[0]["caller_number"] == "******3210"
    assert "[CARD]" in calls[0]["summary"]

    fetched = await store.get_call(tenant_id="default", call_id="call-roundtrip-1")
    assert fetched is not None
    transcript_text = fetched["state"]["transcript"][0]["text"]
    assert "[CARD]" in transcript_text
    assert "4532" not in transcript_text
    assert fetched["metrics"]["p50_e2e_ms"] == 420.0

    # Also verify via FastAPI /api/calls and /api/calls/{call_id}
    app = create_app(agent=DummyAgent(), store=store)  # type: ignore[arg-type]
    with TestClient(app) as client:
        list_resp = client.get("/api/calls")
        assert list_resp.status_code == 200
        assert len(list_resp.json()) == 1

        detail_resp = client.get("/api/calls/call-roundtrip-1")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert "[CARD]" in detail["state"]["transcript"][0]["text"]

        missing_resp = client.get("/api/calls/nonexistent")
        assert missing_resp.status_code == 404

    await store.close()
