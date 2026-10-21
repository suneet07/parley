"""Per-call state. Survives interruptions and tool calls; serialized to storage after the call."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class CallerProfile(BaseModel):
    customer_id: str
    name: str | None = None
    phone: str | None = None
    email: str | None = None


class ToolCall(BaseModel):
    """A tool invocation awaiting spoken confirmation (or already executed, for the log)."""

    name: str
    args: dict[str, Any]
    confirm_prompt: str  # what the agent reads back, e.g. "start a return for order KC-10482"
    created_turn: int  # turn index in which the agent proposed it


class ToolEvent(BaseModel):
    """One executed (or blocked) tool call, kept for transcripts, evals and the dashboard."""

    name: str
    args: dict[str, Any]
    status: Literal["ok", "error", "blocked", "confirmation_required"]
    result: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    turn: int
    latency_ms: float = 0.0


class Turn(BaseModel):
    role: Literal["user", "assistant", "system"]
    text: str
    at: datetime = Field(default_factory=utcnow)
    interrupted: bool = False  # assistant turn cut off by barge-in; text = what caller heard
    tool_events: list[ToolEvent] = Field(default_factory=list)


Disposition = Literal[
    "resolved",
    "escalated",
    "callback_booked",
    "ticket_created",
    "abandoned",
    "voicemail",
    "no_answer",
    "not_interested",
    "do_not_call",
    "unknown",
]


class CallState(BaseModel):
    call_id: str
    tenant_id: str = "default"
    direction: Literal["inbound", "outbound"] = "inbound"
    channel: Literal["phone", "webrtc", "text"] = "text"
    caller_number: str | None = None
    caller: CallerProfile | None = None  # looked up from phone number or set by verification
    verified: bool = False
    intent: str | None = None
    slots: dict[str, str] = Field(default_factory=dict)  # order_id, preferred_time, etc.
    pending_action: ToolCall | None = None  # awaiting spoken confirmation
    transcript: list[Turn] = Field(default_factory=list)
    tool_events: list[ToolEvent] = Field(default_factory=list)
    escalation_reason: str | None = None
    failure_counts: dict[str, int] = Field(default_factory=dict)  # per-task failures
    turn_index: int = 0  # incremented on every user turn
    ended: bool = False
    disposition: Disposition = "unknown"
    summary: str | None = None  # 3-line summary for the human / call log
    intents: list[str] = Field(default_factory=list)  # every intent seen during the call
    follow_ups: list[str] = Field(default_factory=list)
    knowledge_gaps: list[str] = Field(default_factory=list)  # questions it couldn't answer
    csat: int | None = None  # 1-5 if the caller answered the CSAT question
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None

    def last_user_text(self) -> str:
        for t in reversed(self.transcript):
            if t.role == "user":
                return t.text
        return ""
