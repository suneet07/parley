"""In-memory live calls registry and operator takeover endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from parley.api.auth import require_api_key
from parley.config import BusinessConfig, get_settings, load_business
from parley.core.telemetry.redact import redact, redact_state

router = APIRouter(
    prefix="/api/live",
    tags=["live"],
    dependencies=[Depends(require_api_key)],
)


@dataclass
class LiveCallEntry:
    session: Any
    channel: str
    control: Any = None

    def __getitem__(self, key: str) -> Any:
        if key == "call_id":
            return self.session.state.call_id
        return getattr(self, key)


def _serialize_turn(turn: Any) -> dict[str, Any]:
    raw = turn.model_dump(mode="json") if hasattr(turn, "model_dump") else dict(turn)
    redacted_turn = redact_state(raw)
    redacted_turn["text"] = redact(str(raw.get("text", "")))
    return redacted_turn


class LiveCalls:
    """In-memory registry of currently active voice calls."""

    _entries: dict[str, LiveCallEntry] = {}

    def register(self, session: Any, channel: str, control: Any = None) -> None:
        call_id = str(session.state.call_id)
        self._entries[call_id] = LiveCallEntry(
            session=session,
            channel=channel,
            control=control,
        )

    def unregister(self, call_id: str) -> None:
        self._entries.pop(call_id, None)

    def get(self, call_id: str) -> LiveCallEntry | None:
        return self._entries.get(call_id)

    def clear(self) -> None:
        self._entries.clear()

    def list(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for call_id, entry in list(self._entries.items()):
            state = entry.session.state
            started_at = state.started_at
            started_iso = (
                started_at.isoformat() if isinstance(started_at, datetime) else str(started_at)
            )
            intent = state.intent or (
                state.intents[-1] if getattr(state, "intents", None) else None
            )
            last_turns = [_serialize_turn(t) for t in (state.transcript or [])[-6:]]
            out.append(
                {
                    "call_id": call_id,
                    "channel": entry.channel or state.channel,
                    "direction": state.direction,
                    "started_at": started_iso,
                    "intent": intent,
                    "verified": bool(state.verified),
                    "transcript": last_turns,
                    "disposition": state.disposition,
                }
            )
        return out


live_calls = LiveCalls()


def register(session: Any, channel: str, control: Any = None) -> None:
    live_calls.register(session, channel, control)


def unregister(call_id: str) -> None:
    live_calls.unregister(call_id)


def get(call_id: str) -> LiveCallEntry | None:
    return live_calls.get(call_id)


@router.get("")
async def list_live_calls() -> list[dict[str, Any]]:
    return live_calls.list()


def _conflict(reason: str) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": reason, "reason": reason})


@router.post("/{call_id}/takeover")
async def takeover_live_call(call_id: str, request: Request) -> Any:
    entry = live_calls.get(call_id)
    if entry is None:
        return _conflict("Call is no longer active")

    agent = getattr(entry.session, "agent", None)
    business: BusinessConfig | None = getattr(agent, "business", None)
    if business is None:
        business = getattr(request.app.state, "business", None)
    if business is None:
        settings = getattr(request.app.state, "settings", None) or get_settings()
        business = load_business(settings.business_config)

    transfer_to = business.escalation.transfer_to if business and business.escalation else None
    if not transfer_to:
        return _conflict("No escalation transfer_to number is configured")

    transfer_fn = getattr(entry.control, "transfer", None) if entry.control is not None else None
    if transfer_fn is None:
        return _conflict("Warm transfer is not available on this channel")

    state = entry.session.state
    summary = (getattr(state, "summary", None) or "").strip() or (
        f"Operator takeover for call {call_id}."
    )

    try:
        transferred = await transfer_fn(call_id, summary)
    except Exception as exc:
        return _conflict(f"Transfer failed: {exc}")

    if not transferred:
        return _conflict("Transfer could not be completed")

    state.disposition = "escalated"
    state.escalation_reason = state.escalation_reason or "operator_takeover"
    state.ended = True
    return {"ok": True}
