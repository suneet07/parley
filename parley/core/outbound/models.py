"""Pydantic models for outbound contacts, call jobs, and compliance decisions."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Literal

import phonenumbers
from pydantic import BaseModel, Field, field_validator

E164_PATTERN = r"^\+[1-9]\d{7,14}$"
E164_RE = re.compile(E164_PATTERN)


def to_e164(phone: str, default_country: str = "") -> str:
    """Canonical E.164 form, or ValueError. "00<cc>..." is treated as "+<cc>...". A number without
    a country code is only accepted when default_country (ISO alpha-2, e.g. "IN") is given."""
    raw = re.sub(r"[\s\-\(\)\.]+", "", phone.strip())
    if raw.startswith("00"):
        raw = "+" + raw[2:]
    if not raw.startswith("+") and not default_country:
        raise ValueError(f"phone number needs a country code: {phone!r}")
    try:
        parsed = phonenumbers.parse(raw, None if raw.startswith("+") else default_country.upper())
    except phonenumbers.NumberParseException as exc:
        raise ValueError(f"invalid phone number: {phone!r}") from exc
    if not phonenumbers.is_possible_number(parsed):
        raise ValueError(f"invalid phone number: {phone!r}")
    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


def normalize_phone(phone: str) -> str:
    """Best-effort canonical form used for comparisons (DNC, caps). Falls back to the
    separator-stripped string when the number can't be parsed, so lookups never crash."""
    try:
        return to_e164(phone)
    except ValueError:
        return re.sub(r"[\s\-\(\)\.]+", "", phone.strip())


def utcnow() -> datetime:
    return datetime.now(UTC)


class Contact(BaseModel):
    contact_id: str = Field(default_factory=lambda: f"cnt_{uuid.uuid4().hex[:12]}")
    phone: str
    name: str | None = None
    tz: str | None = None
    consent: bool = False
    consent_source: str | None = None
    consent_at: datetime | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("phone")
    @classmethod
    def _validate_e164(cls, value: str) -> str:
        cleaned = value.strip()
        if not E164_RE.match(cleaned):
            raise ValueError(f"Invalid E.164 phone number: {value!r}")
        return cleaned


JobStatus = Literal[
    "queued",
    "dialing",
    "in_progress",
    "completed",
    "retry_scheduled",
    "failed",
    "blocked",
    "cancelled",
]

Outcome = Literal[
    "connected",
    "voicemail",
    "no_answer",
    "busy",
    "failed",
    "booked",
    "not_interested",
    "do_not_call",
    "wrong_person",
    "completed",
]


class CallJob(BaseModel):
    job_id: str = Field(default_factory=lambda: f"job_{uuid.uuid4().hex[:12]}")
    tenant_id: str = "default"
    contact: Contact
    goal: str
    trigger: Literal["webhook", "campaign", "manual"] = "manual"
    campaign_id: str | None = None
    attempts: int = 0
    status: JobStatus = "queued"
    outcome: Outcome | None = None
    next_attempt_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    call_sids: list[str] = Field(default_factory=list)
    finished_call_sids: list[str] = Field(default_factory=list)  # makes on_call_finished idempotent
    last_error: str | None = None
    idempotency_key: str | None = None
    sms_sent: bool = False
    pending_sms_at: datetime | None = None  # SMS fallback deferred to the next calling window


class ComplianceDecision(BaseModel):
    allowed: bool
    reasons: list[str] = Field(default_factory=list)
    retry_at: datetime | None = None
