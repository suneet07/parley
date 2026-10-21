"""Outbound calling compliance engine: hours, consent, DNC, daily caps, and attempt limits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import phonenumbers
from phonenumbers import timezone as pn_timezone

from parley.core.outbound.models import (
    E164_RE,
    CallJob,
    ComplianceDecision,
    Contact,
    normalize_phone,
)

if TYPE_CHECKING:
    from parley.config import Settings
    from parley.core.outbound.store import OutboundStore


@dataclass(frozen=True)
class RegionRules:
    name: str
    window_start: time
    window_end: time
    allowed_weekdays: frozenset[int]
    require_consent: bool = True
    max_attempts_per_contact: int = 3


PRESETS: dict[str, RegionRules] = {
    "in": RegionRules("in", time(9), time(21), frozenset(range(7))),  # TRAI-style 9am-9pm
    "us": RegionRules(
        "us", time(8), time(21), frozenset(range(7))
    ),  # TCPA-style 8am-9pm callee local
}


def _safe_zoneinfo(tz_name: str | None) -> ZoneInfo | None:
    if not tz_name or not tz_name.strip():
        return None
    try:
        return ZoneInfo(tz_name.strip())
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        return None


def callee_timezone(contact: Contact, business_tz: str | None = None) -> ZoneInfo | None:
    """The callee's zone: contact.tz if valid, else the single zone libphonenumber maps the
    number to. Multi-zone or unknown -> None (the caller must block: never guess from the
    business's own zone, that's how a Pacific callee gets rung at 5 am)."""
    explicit = _safe_zoneinfo(contact.tz)
    if explicit is not None:
        return explicit
    try:
        parsed = phonenumbers.parse(normalize_phone(contact.phone), None)
    except phonenumbers.NumberParseException:
        return None
    zones = [z for z in pn_timezone.time_zones_for_number(parsed) if z != "Etc/Unknown"]
    if len(zones) != 1:
        return None
    return _safe_zoneinfo(_LEGACY_ZONES.get(zones[0], zones[0]))


# libphonenumber still returns a few pre-rename IANA ids
_LEGACY_ZONES = {
    "Asia/Calcutta": "Asia/Kolkata",
    "Asia/Katmandu": "Asia/Kathmandu",
    "Asia/Saigon": "Asia/Ho_Chi_Minh",
    "Asia/Rangoon": "Asia/Yangon",
}


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _next_window_start_utc(
    base_date: date,
    rules: RegionRules,
    tz: ZoneInfo,
    *,
    start_offset_days: int = 0,
) -> datetime | None:
    if not rules.allowed_weekdays:
        return None
    for offset in range(start_offset_days, start_offset_days + 8):
        candidate_date = base_date + timedelta(days=offset)
        if candidate_date.weekday() in rules.allowed_weekdays:
            local_dt = datetime.combine(candidate_date, rules.window_start, tzinfo=tz)
            return local_dt.astimezone(UTC)
    return None


class ComplianceEngine:
    """Evaluates all compliance rules for a call job and writes an immutable audit entry."""

    def __init__(
        self,
        rules: RegionRules,
        store: OutboundStore,
        settings: Settings,
        business_tz: str,
    ) -> None:
        self.rules = rules
        self.store = store
        self.settings = settings
        self.business_tz = business_tz

    async def check(self, job: CallJob, now: datetime) -> ComplianceDecision:
        now_utc = _ensure_utc(now)
        reasons: list[str] = []
        retry_candidates: list[datetime] = []
        non_retryable = False

        contact = job.contact
        norm_phone = normalize_phone(contact.phone)

        # 1. Phone format check
        if not E164_RE.match(norm_phone):
            reasons.append("invalid_phone")
            non_retryable = True

        # 2. Consent check
        if not contact.consent or (self.rules.require_consent and contact.consent_at is None):
            reasons.append("no_consent")
            non_retryable = True

        # 3. Do-Not-Call check
        if await self.store.is_dnc(job.tenant_id, norm_phone):
            reasons.append("do_not_call")
            non_retryable = True

        # 4. Calling hours check in callee timezone
        tz = callee_timezone(contact, self.business_tz)
        if tz is None:
            reasons.append("unknown_timezone")
            non_retryable = True
        else:
            local_now = now_utc.astimezone(tz)
            local_time = local_now.timetz().replace(tzinfo=None)
            in_weekday = local_now.weekday() in self.rules.allowed_weekdays
            # Window end is strictly exclusive (e.g. 21:00:00 is outside).
            in_time_window = self.rules.window_start <= local_time < self.rules.window_end
            if not (in_weekday and in_time_window):
                reasons.append("outside_calling_hours")
                offset = 0 if in_weekday and local_time < self.rules.window_start else 1
                next_win = _next_window_start_utc(
                    local_now.date(), self.rules, tz, start_offset_days=offset
                )
                if next_win is not None:
                    retry_candidates.append(next_win)
                else:
                    non_retryable = True

            # 5. Daily cap per contact (since local midnight in callee tz)
            local_midnight_utc = datetime.combine(
                local_now.date(), time(0, 0), tzinfo=tz
            ).astimezone(UTC)
            contact_dials = await self.store.count_dials(
                job.tenant_id, phone=norm_phone, since=local_midnight_utc
            )
            if contact_dials >= self.settings.outbound_daily_cap_per_contact:
                reasons.append("contact_daily_cap")
                next_day_win = _next_window_start_utc(
                    local_now.date(), self.rules, tz, start_offset_days=1
                )
                if next_day_win is not None:
                    retry_candidates.append(next_day_win)
                else:
                    non_retryable = True

        # 6. Daily cap per Twilio number: one UTC day for every callee, so the total holds.
        utc_midnight = datetime.combine(now_utc.date(), time(0, 0), tzinfo=UTC)
        number_dials = await self.store.count_dials(
            job.tenant_id, from_number=self.settings.twilio_phone_number, since=utc_midnight
        )
        if number_dials >= self.settings.outbound_daily_cap_per_number:
            reasons.append("number_daily_cap")
            retry_candidates.append(utc_midnight + timedelta(days=1))

        # 6b. Attempts to this contact across ALL jobs in the last 7 days (stops replayed or
        # duplicated triggers from resetting the per-job attempt counter).
        week_dials = await self.store.count_dials(
            job.tenant_id, phone=norm_phone, since=now_utc - timedelta(days=7)
        )
        if week_dials >= self.settings.outbound_max_attempts_per_contact_7d:
            reasons.append("contact_attempt_cap")
            retry_candidates.append(now_utc + timedelta(days=1))

        # 7. Max attempts check
        max_attempts = min(
            self.rules.max_attempts_per_contact,
            self.settings.outbound_max_attempts,
        )
        if job.attempts >= max_attempts:
            reasons.append("max_attempts")
            non_retryable = True

        allowed = len(reasons) == 0
        if allowed or non_retryable:
            retry_at = None
        else:
            retry_at = max(retry_candidates) if retry_candidates else None

        decision = ComplianceDecision(
            allowed=allowed,
            reasons=reasons,
            retry_at=retry_at,
        )
        await self.store.audit(job, decision, now_utc)
        return decision
