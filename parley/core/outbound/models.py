"""Pydantic models for outbound contacts, call jobs, and compliance decisions."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

E164_PATTERN = r"^\+[1-9]\d{7,14}$"
E164_RE = re.compile(E164_PATTERN)


def normalize_phone(phone: str) -> str:
    """Normalize a phone string by removing whitespace and formatting separators."""
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
    last_error: str | None = None


class ComplianceDecision(BaseModel):
    allowed: bool
    reasons: list[str] = Field(default_factory=list)
    retry_at: datetime | None = None
