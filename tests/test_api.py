from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from twilio.request_validator import RequestValidator

import parley.api.twilio_routes as twilio_routes
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
        app.state.settings.twilio_insecure_dev = True
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
        app.state.settings.twilio_insecure_dev = False
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
        app.state.settings.twilio_insecure_dev = True
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
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
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


def test_twilio_ws_token_auth_and_one_time_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "t_ws.db"
    store = CallStore(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    app = create_app(agent=DummyAgent(), store=store)  # type: ignore[arg-type]

    captured_calls: list[dict[str, Any]] = []

    async def _fake_run_call(transport: Any, **kwargs: Any) -> CallState:
        captured_calls.append(kwargs)
        return CallState(call_id=str(kwargs.get("call_id", "CA55555")))

    monkeypatch.setattr(twilio_routes, "run_call", _fake_run_call)

    with TestClient(app) as client:
        app.state.settings.twilio_auth_token = ""
        app.state.settings.twilio_insecure_dev = True
        app.state.settings.public_base_url = "https://demo.example.com"

        # 1. WS without token is rejected with 1008
        with client.websocket_connect("/twilio/ws") as ws:
            ws.send_text(json.dumps({"event": "connected", "protocol": "Call"}))
            ws.send_text(
                json.dumps(
                    {
                        "event": "start",
                        "start": {
                            "streamSid": "MZ111",
                            "callSid": "CA55555",
                            "customParameters": {},
                        },
                    }
                )
            )
            with pytest.raises(WebSocketDisconnect) as exc_info:
                ws.receive_text()
            assert exc_info.value.code == 1008
        assert len(captured_calls) == 0

        # 2. Mint a token via POST /twilio/voice and accept WS with valid token
        voice_resp = client.post(
            "/twilio/voice",
            data={"From": "+919876543210", "CallSid": "CA55555"},
        )
        assert voice_resp.status_code == 200
        match = re.search(r'<Parameter name="token" value="([^"]+)"', voice_resp.text)
        assert match is not None
        minted_token = match.group(1)

        with client.websocket_connect("/twilio/ws") as ws:
            ws.send_text(json.dumps({"event": "connected", "protocol": "Call"}))
            ws.send_text(
                json.dumps(
                    {
                        "event": "start",
                        "start": {
                            "streamSid": "MZ111",
                            "callSid": "CA55555",
                            "customParameters": {
                                "token": minted_token,
                                "from": "+910000000000",
                            },
                        },
                    }
                )
            )
        assert len(captured_calls) == 1
        assert captured_calls[0]["call_id"] == "CA55555"
        # Caller number must come from the stored token record, not client customParameters
        assert captured_calls[0]["caller_number"] == "+919876543210"

        # 3. Reusing the same token is rejected with 1008
        with client.websocket_connect("/twilio/ws") as ws:
            ws.send_text(json.dumps({"event": "connected", "protocol": "Call"}))
            ws.send_text(
                json.dumps(
                    {
                        "event": "start",
                        "start": {
                            "streamSid": "MZ222",
                            "callSid": "CA55555",
                            "customParameters": {"token": minted_token},
                        },
                    }
                )
            )
            with pytest.raises(WebSocketDisconnect) as exc_info:
                ws.receive_text()
            assert exc_info.value.code == 1008
        assert len(captured_calls) == 1
