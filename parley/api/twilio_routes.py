from __future__ import annotations

import asyncio
import secrets
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
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
from parley.core.outbound.models import Outcome, normalize_phone
from parley.core.outbound.queue import OutboundQueue
from parley.core.outbound.voicemail import voicemail_twiml
from parley.core.outbound.worker import build_outbound_queue
from parley.core.pipeline.bot import CallControl, run_call

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/twilio", tags=["twilio"])

# In-memory store mapping call_id -> warm-transfer whisper summary for human agents.
WHISPERS: dict[str, str] = {}
# One-time stream tokens minted by POST /twilio/voice for /twilio/ws authentication.
STREAM_TOKENS: dict[str, dict[str, Any]] = {}
max_phone_calls = 10
_active_phone_calls = 0
_warned_missing_auth_token = False


def _get_settings_and_business(app: Any) -> tuple[Settings, BusinessConfig]:
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    business: BusinessConfig = getattr(app.state, "business", None) or load_business(
        settings.business_config
    )
    return settings, business


async def _get_outbound_queue(app: Any) -> OutboundQueue:
    settings, _ = _get_settings_and_business(app)
    queue: OutboundQueue | None = getattr(app.state, "outbound_queue", None)
    if queue is None:
        store = getattr(app.state, "outbound_store", None)
        queue = build_outbound_queue(settings=settings, store=store)
        app.state.outbound_queue = queue
        app.state.outbound_store = queue.store
    else:
        queue.settings = settings
        queue.compliance.settings = settings
    await queue.store.init()
    return queue


def _disposition_to_outcome(disposition: str | None) -> Outcome:
    mapping: dict[str, Outcome] = {
        "do_not_call": "do_not_call",
        "not_interested": "not_interested",
        "callback_booked": "booked",
        "voicemail": "voicemail",
        "no_answer": "no_answer",
    }
    if disposition and disposition in mapping:
        return mapping[disposition]
    return "completed"


def _public_ws_url(public_base_url: str) -> str:
    parsed = urlparse(public_base_url)
    host = parsed.netloc or parsed.path.strip("/") or "localhost:8000"
    return f"wss://{host}/twilio/ws"


async def _validate_twilio_signature(request: Request, url: str) -> None:
    global _warned_missing_auth_token
    settings, _ = _get_settings_and_business(request.app)
    token = settings.twilio_auth_token
    if not token:
        if not settings.twilio_insecure_dev:
            raise HTTPException(
                status_code=403,
                detail="Twilio webhooks need TWILIO_AUTH_TOKEN (or TWILIO_INSECURE_DEV=true locally)",
            )
        if not _warned_missing_auth_token:
            _warned_missing_auth_token = True
            logger.warning("twilio_signature_validation_skipped", reason="insecure_dev_mode")
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

    if settings.kill_switch:
        vr_kill = VoiceResponse()
        vr_kill.say("Sorry, this line is not available right now.")
        vr_kill.hangup()
        return Response(content=str(vr_kill), media_type="application/xml")

    form = await request.form()
    from_number = str(form.get("From", ""))
    call_sid = str(form.get("CallSid", ""))

    if from_number and settings.caller_blocklist:
        blocklist = {normalize_phone(p) for p in settings.caller_blocklist.split(",") if p.strip()}
        if normalize_phone(from_number) in blocklist:
            vr_block = VoiceResponse()
            vr_block.say("Sorry, we are unable to take your call right now. Goodbye.")
            vr_block.hangup()
            return Response(content=str(vr_block), media_type="application/xml")

    store = getattr(request.app.state, "store", None)
    if store is not None and from_number and settings.max_calls_per_caller_per_day > 0:
        await store.init()
        since = datetime.now(UTC) - timedelta(hours=24)
        recent_count = await store.count_calls_from(
            settings.tenant_id,
            from_number,
            since,
        )
        if recent_count >= settings.max_calls_per_caller_per_day:
            vr_cap = VoiceResponse()
            vr_cap.say(
                "Sorry, you have reached the maximum number of calls for today. Please try again tomorrow."
            )
            vr_cap.hangup()
            return Response(content=str(vr_cap), media_type="application/xml")

    now = time.time()
    expired_keys = [k for k, v in STREAM_TOKENS.items() if float(v.get("expires", 0)) <= now]
    for k in expired_keys:
        STREAM_TOKENS.pop(k, None)

    stream_token = secrets.token_urlsafe(24)
    STREAM_TOKENS[stream_token] = {
        "call_sid": call_sid,
        "from": from_number,
        "expires": now + 120.0,
    }

    ws_url = _public_ws_url(settings.public_base_url)

    vr = VoiceResponse()
    connect = Connect()
    stream = connect.stream(url=ws_url)
    stream.parameter(name="token", value=stream_token)
    stream.parameter(name="from", value=from_number)
    stream.parameter(name="call_sid", value=call_sid)
    vr.append(connect)

    return Response(content=str(vr), media_type="application/xml")


_MACHINE_ANSWERS = {
    "machine_start",
    "machine_end_beep",
    "machine_end_silence",
    "machine_end_other",
    "fax",
}


def _hangup_twiml(message: str) -> Response:
    vr = VoiceResponse()
    vr.say(message)
    vr.hangup()
    return Response(content=str(vr), media_type="application/xml")


@router.post("/outbound/{job_id}")
async def twilio_outbound_webhook(job_id: str, request: Request) -> Response:
    """Answer webhook for an outbound call. The dialer uses synchronous AMD, so `AnsweredBy` is
    already known here: machines get the voicemail drop and the AI never talks over a greeting."""
    settings, business = _get_settings_and_business(request.app)
    base_url = settings.public_base_url.rstrip("/")
    await _validate_twilio_signature(request, f"{base_url}/twilio/outbound/{job_id}")

    form = await request.form()
    call_sid = str(form.get("CallSid", ""))
    answered_by = str(form.get("AnsweredBy", "")).lower()

    queue = await _get_outbound_queue(request.app)
    job = await queue.store.get_job(settings.tenant_id, job_id)
    if job is None:
        return _hangup_twiml("Sorry, this call was placed in error. Goodbye.")
    # State may have changed since we dialed (opt-out by SMS, cancelled campaign).
    if job.status not in ("dialing", "in_progress") or await queue.store.is_dnc(
        job.tenant_id, job.contact.phone
    ):
        return _hangup_twiml("Sorry to bother you. Goodbye.")

    if answered_by in _MACHINE_ANSWERS:
        await queue.on_call_finished(job_id, "voicemail", datetime.now(UTC), call_sid=call_sid)
        twiml = voicemail_twiml(business.business, job.goal, settings.twilio_phone_number or "")
        return Response(content=twiml, media_type="application/xml")

    now = time.time()
    for k in [k for k, v in STREAM_TOKENS.items() if float(v.get("expires", 0)) <= now]:
        STREAM_TOKENS.pop(k, None)
    stream_token = secrets.token_urlsafe(24)
    STREAM_TOKENS[stream_token] = {
        "call_sid": call_sid,
        "from": job.contact.phone,
        "to": job.contact.phone,
        "direction": "outbound",
        "job_id": job_id,
        "goal": job.goal,
        "expires": now + 120.0,
    }

    vr = VoiceResponse()
    connect = Connect()
    stream = connect.stream(url=_public_ws_url(settings.public_base_url))
    # Only the one-time token goes to the client; job context stays server-side.
    stream.parameter(name="token", value=stream_token)
    vr.append(connect)
    return Response(content=str(vr), media_type="application/xml")


@router.post("/amd/{job_id}")
async def twilio_amd_webhook(job_id: str, request: Request) -> Response:
    """Kept for older deployments that still point async AMD here; detection now happens
    synchronously on the answer webhook."""
    settings, _ = _get_settings_and_business(request.app)
    await _validate_twilio_signature(
        request, f"{settings.public_base_url.rstrip('/')}/twilio/amd/{job_id}"
    )
    return Response(status_code=204)


@router.post("/status/{job_id}")
async def twilio_status_webhook(job_id: str, request: Request) -> Response:
    """Twilio CallStatus callback for unanswered, busy, failed, or canceled outbound calls."""
    settings, _ = _get_settings_and_business(request.app)
    base_url = settings.public_base_url.rstrip("/")
    await _validate_twilio_signature(request, f"{base_url}/twilio/status/{job_id}")

    form = await request.form()
    call_status = str(form.get("CallStatus", "")).lower()
    status_call_sid = str(form.get("CallSid", "")) or None

    status_map: dict[str, Outcome] = {
        "no-answer": "no_answer",
        "no_answer": "no_answer",
        "busy": "busy",
        "failed": "failed",
        "canceled": "failed",
        "cancelled": "failed",
    }
    mapped_outcome = status_map.get(call_status)
    if mapped_outcome is not None:
        queue = await _get_outbound_queue(request.app)
        try:
            await queue.on_call_finished(
                job_id, mapped_outcome, datetime.now(UTC), call_sid=status_call_sid
            )
        except KeyError:
            logger.warning("twilio_status_job_not_found", job_id=job_id)

    return Response(status_code=204)


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
    global _active_phone_calls
    await websocket.accept()

    if _active_phone_calls >= max_phone_calls:
        await websocket.close(code=1013)
        return

    settings, business = _get_settings_and_business(websocket.app)

    # In Pipecat 1.13, parse_telephony_websocket reads the initial 'connected' and 'start' frames.
    try:
        _, call_data = await parse_telephony_websocket(websocket)
    except Exception:
        await websocket.close(code=1008)
        return

    stream_sid: str = call_data.get("stream_id") or ""
    start_call_sid: str = call_data.get("call_id") or ""
    body: dict[str, Any] = call_data.get("body") or {}
    token = body.get("token")
    if not isinstance(token, str) or not token:
        await websocket.close(code=1008)
        return

    record = STREAM_TOKENS.pop(token, None)
    now = time.time()
    if (
        record is None
        or float(record.get("expires", 0)) <= now
        or not start_call_sid
        or record.get("call_sid") != start_call_sid
    ):
        await websocket.close(code=1008)
        return

    call_sid: str = str(record["call_sid"])
    direction_raw = str(record.get("direction") or "inbound")
    direction: Literal["inbound", "outbound"] = (
        "outbound" if direction_raw == "outbound" else "inbound"
    )
    job_id: str | None = str(record["job_id"]) if record.get("job_id") else None
    outbound_goal: str | None = str(record["goal"]) if record.get("goal") is not None else None
    if direction == "outbound":
        caller_number: str = str(record.get("to") or record.get("from") or "")
    else:
        caller_number = str(record.get("from") or "")

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

    _active_phone_calls += 1
    final_state = None
    try:
        final_state = await run_call(
            transport,
            agent=websocket.app.state.agent,
            call_id=call_sid,
            channel="phone",
            direction=direction,
            caller_number=caller_number,
            audio_sample_rate=8000,
            control=CallControl(transfer=_transfer_cb),
            store=getattr(websocket.app.state, "store", None),
            outbound_goal=outbound_goal,
        )
    finally:
        _active_phone_calls -= 1
        if direction == "outbound" and job_id:
            disp = final_state.disposition if final_state is not None else None
            # Detached task: the handler is usually cancelled when the socket closes (and anyio
            # re-cancels every await in this finally), but the outcome (retries, DNC) must be saved.
            task = asyncio.create_task(
                _record_outbound_outcome(
                    websocket.app, job_id, _disposition_to_outcome(disp), call_sid
                )
            )
            _BACKGROUND_TASKS.add(task)
            task.add_done_callback(_BACKGROUND_TASKS.discard)


_BACKGROUND_TASKS: set[asyncio.Task] = set()


async def _record_outbound_outcome(
    app: Any, job_id: str, outcome: str, call_sid: str | None = None
) -> None:
    try:
        queue = await _get_outbound_queue(app)
        await queue.on_call_finished(job_id, outcome, datetime.now(UTC), call_sid=call_sid)
    except Exception:
        logger.exception("outbound_ws_call_finished_error", job_id=job_id)


_STOP_WORDS = {"stop", "stopall", "unsubscribe", "cancel", "end", "quit"}


@router.post("/sms")
async def twilio_inbound_sms(request: Request) -> Response:
    """Inbound SMS webhook. Opt-out keywords put the number on the do-not-call list, so a reply
    to the fallback text also stops future calls."""
    settings, _ = _get_settings_and_business(request.app)
    await _validate_twilio_signature(request, f"{settings.public_base_url.rstrip('/')}/twilio/sms")
    form = await request.form()
    body = str(form.get("Body", "")).strip().lower()
    sender = str(form.get("From", ""))
    if body in _STOP_WORDS and sender:
        queue = await _get_outbound_queue(request.app)
        await queue.store.add_dnc(settings.tenant_id, sender, source="sms_stop")
        logger.info("sms_opt_out", phone=sender[-4:])
    return Response(content="<Response/>", media_type="application/xml")
