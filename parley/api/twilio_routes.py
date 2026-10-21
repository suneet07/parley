from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlparse

import structlog
from fastapi import APIRouter, HTTPException, Request, Response, WebSocket
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from twilio.request_validator import RequestValidator
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import Connect, Dial, VoiceResponse

from parley.config import BusinessConfig, Settings, get_settings, load_business
from parley.core.pipeline.bot import CallControl, run_call

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/twilio", tags=["twilio"])

# In-memory store mapping call_id -> warm-transfer whisper summary for human agents.
WHISPERS: dict[str, str] = {}
_warned_missing_auth_token = False


def _get_settings_and_business(app: Any) -> tuple[Settings, BusinessConfig]:
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    business: BusinessConfig = getattr(app.state, "business", None) or load_business(
        settings.business_config
    )
    return settings, business


def _public_ws_url(public_base_url: str) -> str:
    parsed = urlparse(public_base_url)
    host = parsed.netloc or parsed.path.strip("/") or "localhost:8000"
    return f"wss://{host}/twilio/ws"


async def _validate_twilio_signature(request: Request, url: str) -> None:
    global _warned_missing_auth_token
    settings, _ = _get_settings_and_business(request.app)
    token = settings.twilio_auth_token
    if not token:
        if not _warned_missing_auth_token:
            _warned_missing_auth_token = True
            logger.warning("twilio_signature_validation_skipped", reason="empty_twilio_auth_token")
        return

    signature = request.headers.get("X-Twilio-Signature", "")
    if request.method == "POST":
        form = await request.form()
        params: dict[str, Any] = {k: str(v) for k, v in form.items()}
    else:
        params = dict(request.query_params)

    validator = RequestValidator(token)
    if not validator.validate(url, params, signature):
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")


@router.post("/voice")
async def twilio_voice_webhook(request: Request) -> Response:
    """Twilio 'A call comes in' webhook returning <Connect><Stream> TwiML."""
    settings, _ = _get_settings_and_business(request.app)
    base_url = settings.public_base_url.rstrip("/")
    await _validate_twilio_signature(request, f"{base_url}/twilio/voice")

    form = await request.form()
    from_number = str(form.get("From", ""))
    call_sid = str(form.get("CallSid", ""))

    ws_url = _public_ws_url(settings.public_base_url)

    vr = VoiceResponse()
    connect = Connect()
    stream = connect.stream(url=ws_url)
    stream.parameter(name="from", value=from_number)
    stream.parameter(name="call_sid", value=call_sid)
    vr.append(connect)

    return Response(content=str(vr), media_type="application/xml")


async def twilio_transfer(
    call_id: str,
    summary: str | None,
    *,
    settings: Settings | None = None,
    business: BusinessConfig | None = None,
) -> bool:
    """Warm-transfer a live Twilio call by updating its TwiML with a whisper URL."""
    cfg_settings = settings or get_settings()
    cfg_business = business or load_business(cfg_settings.business_config)
    transfer_to = cfg_business.escalation.transfer_to
    if not transfer_to:
        return False

    WHISPERS[call_id] = (summary or "").strip() or "No summary available."
    base_url = cfg_settings.public_base_url.rstrip("/")
    whisper_url = f"{base_url}/twilio/whisper/{call_id}"

    vr = VoiceResponse()
    vr.say("Connecting you to a member of our team.")
    dial = Dial()
    dial.number(transfer_to, url=whisper_url)
    vr.append(dial)
    twiml_str = str(vr)

    def _update_call() -> None:
        client = TwilioClient(cfg_settings.twilio_account_sid, cfg_settings.twilio_auth_token)
        client.calls(call_id).update(twiml=twiml_str)

    try:
        await asyncio.to_thread(_update_call)
        logger.info("twilio_call_transferred", call_id=call_id, transfer_to=transfer_to)
        return True
    except Exception:
        logger.exception("twilio_transfer_failed", call_id=call_id)
        return False


@router.api_route("/whisper/{call_id}", methods=["GET", "POST"])
async def twilio_whisper(call_id: str, request: Request) -> Response:
    """Warm-transfer whisper spoken to the human agent before bridging the caller."""
    settings, _ = _get_settings_and_business(request.app)
    base_url = settings.public_base_url.rstrip("/")
    await _validate_twilio_signature(request, f"{base_url}/twilio/whisper/{call_id}")

    summary = WHISPERS.get(call_id) or "No summary available."
    vr = VoiceResponse()
    vr.say(f"Incoming transfer from the AI agent. {summary}")
    return Response(content=str(vr), media_type="application/xml")


@router.websocket("/ws")
async def twilio_media_websocket(websocket: WebSocket) -> None:
    """Twilio Media Streams WebSocket endpoint driving the Pipecat 1.13 pipeline."""
    await websocket.accept()
    settings, business = _get_settings_and_business(websocket.app)

    # In Pipecat 1.13, parse_telephony_websocket reads the initial 'connected' and 'start' frames.
    _, call_data = await parse_telephony_websocket(websocket)
    stream_sid: str = call_data.get("stream_id", "")
    body: dict[str, Any] = call_data.get("body") or {}
    call_sid: str = call_data.get("call_id") or str(body.get("call_sid", "")) or stream_sid
    caller_number: str = str(body.get("from") or call_data.get("from") or "")

    # Disable auto_hang_up if Twilio REST credentials are not set to avoid ValueError in 1.13.
    has_twilio_creds = bool(settings.twilio_account_sid and settings.twilio_auth_token)
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=settings.twilio_account_sid or None,
        auth_token=settings.twilio_auth_token or None,
        params=TwilioFrameSerializer.InputParams(auto_hang_up=has_twilio_creds),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    async def _transfer_cb(cid: str, summary: str | None) -> bool:
        return await twilio_transfer(cid, summary, settings=settings, business=business)

    await run_call(
        transport,
        agent=websocket.app.state.agent,
        call_id=call_sid,
        channel="phone",
        direction="inbound",
        caller_number=caller_number,
        audio_sample_rate=8000,
        control=CallControl(transfer=_transfer_cb),
        store=getattr(websocket.app.state, "store", None),
    )
