"""Outbound calling compliance engine: hours, consent, DNC, daily caps, and attempt limits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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


def callee_timezone(contact: Contact, business_tz: str) -> ZoneInfo:
    """Resolve the callee's local timezone from contact.tz, E.164 country prefix, or business_tz."""
    explicit = _safe_zoneinfo(contact.tz)
    if explicit is not None:
        return explicit

    phone = normalize_phone(contact.phone)
    if phone.startswith("+91"):
        return ZoneInfo("Asia/Kolkata")
    if phone.startswith("+44"):
        return ZoneInfo("Europe/London")
    if phone.startswith("+1"):
        if business_tz.strip().startswith("America/"):
            biz_zi = _safe_zoneinfo(business_tz)
            if biz_zi is not None:
                return biz_zi
        return ZoneInfo("America/New_York")

    return _safe_zoneinfo(business_tz) or ZoneInfo("UTC")


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
        local_now = now_utc.astimezone(tz)
        local_time = local_now.timetz().replace(tzinfo=None)
        in_weekday = local_now.weekday() in self.rules.allowed_weekdays
        # Window end is strictly exclusive (e.g. 21:00:00 is outside).
        in_time_window = self.rules.window_start <= local_time < self.rules.window_end

        if not (in_weekday and in_time_window):
            reasons.append("outside_calling_hours")
            if in_weekday and local_time < self.rules.window_start:
                next_win = _next_window_start_utc(
                    local_now.date(),
                    self.rules,
                    tz,
                    start_offset_days=0,
                )
            else:
                next_win = _next_window_start_utc(
                    local_now.date(),
                    self.rules,
                    tz,
                    start_offset_days=1,
                )
            if next_win is not None:
                retry_candidates.append(next_win)
            else:
                non_retryable = True

        # 5. Daily cap per contact (since local midnight in callee tz)
        local_midnight_utc = datetime.combine(local_now.date(), time(0, 0), tzinfo=tz).astimezone(
            UTC
        )
        contact_dials = await self.store.count_dials(
            job.tenant_id,
            phone=norm_phone,
            since=local_midnight_utc,
        )
        if contact_dials >= self.settings.outbound_daily_cap_per_contact:
            reasons.append("contact_daily_cap")
            next_day_win = _next_window_start_utc(
                local_now.date(),
                self.rules,
                tz,
                start_offset_days=1,
            )
            if next_day_win is not None:
                retry_candidates.append(next_day_win)
            else:
                non_retryable = True

        # 6. Daily cap per Twilio number
        from_number = self.settings.twilio_phone_number
        number_dials = await self.store.count_dials(
            job.tenant_id,
            from_number=from_number,
            since=local_midnight_utc,
        )
        if number_dials >= self.settings.outbound_daily_cap_per_number:
            reasons.append("number_daily_cap")
            next_day_win = _next_window_start_utc(
                local_now.date(),
                self.rules,
                tz,
                start_offset_days=1,
            )
            if next_day_win is not None:
                retry_candidates.append(next_day_win)
            else:
                non_retryable = True

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
