"""Outbound dial queue and retry scheduling logic (independent of arq for unit testability)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

import structlog

from parley.config import load_business
from parley.core.outbound.models import CallJob, Outcome

if TYPE_CHECKING:
    from parley.config import Settings
    from parley.core.outbound.compliance import ComplianceEngine
    from parley.core.outbound.store import OutboundStore

logger = structlog.get_logger(__name__)

RETRY_DELAYS: list[timedelta] = [
    timedelta(minutes=0),
    timedelta(hours=3),
    timedelta(days=1),
]


class Dialer(Protocol):
    async def dial(self, job: CallJob) -> str: ...

    async def send_sms(self, phone: str, body: str) -> None: ...


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _retry_delay_for_attempts(attempts: int) -> timedelta:
    idx = min(max(attempts, 0), len(RETRY_DELAYS) - 1)
    return RETRY_DELAYS[idx]


class OutboundQueue:
    """Coordinates job persistence, compliance gating, dialing, retries, and SMS fallback."""

    def __init__(
        self,
        store: OutboundStore,
        compliance: ComplianceEngine,
        dialer: Dialer,
        settings: Settings,
        business_name: str | None = None,
    ) -> None:
        self.store = store
        self.compliance = compliance
        self.dialer = dialer
        self.settings = settings
        self._business_name = business_name

    def _get_business_name(self) -> str:
        if self._business_name:
            return self._business_name
        try:
            biz = load_business(self.settings.business_config)
            self._business_name = biz.business
            return biz.business
        except Exception:
            return "Parley Support"

    async def _load_job(self, job_id: str) -> CallJob:
        job = await self.store.get_job(self.settings.tenant_id, job_id)
        if job is None:
            job = await self.store.get_job_by_id(job_id)
        if job is None:
            raise KeyError(f"Outbound job not found: {job_id}")
        return job

    async def enqueue(self, job: CallJob, now: datetime) -> CallJob:
        now_utc = _ensure_utc(now)
        job.next_attempt_at = now_utc
        job.updated_at = now_utc
        await self.store.create_job(job)
        return job

    async def process(self, job_id: str, now: datetime) -> CallJob:
        now_utc = _ensure_utc(now)
        job = await self._load_job(job_id)
        if getattr(self.settings, "kill_switch", False):
            return job
        if job.status not in ("queued", "retry_scheduled"):
            return job

        decision = await self.compliance.check(job, now_utc)
        if not decision.allowed:
            if decision.retry_at is not None:
                job.status = "retry_scheduled"
                job.next_attempt_at = decision.retry_at
                job.last_error = ", ".join(decision.reasons)
            else:
                job.status = "blocked"
                job.outcome = None
                job.next_attempt_at = None
                job.last_error = ", ".join(decision.reasons)
            job.updated_at = now_utc
            await self.store.update_job(job)
            return job

        job.attempts += 1
        job.status = "dialing"
        job.updated_at = now_utc
        await self.store.update_job(job)
        await self.store.record_dial(
            job.tenant_id,
            job.contact.phone,
            self.settings.twilio_phone_number,
            now_utc,
            job.job_id,
        )

        try:
            call_sid = await self.dialer.dial(job)
        except Exception as exc:
            logger.warning("outbound_dial_failed", job_id=job.job_id, error=str(exc))
            job.last_error = str(exc)
            if job.attempts < self.settings.outbound_max_attempts:
                job.status = "retry_scheduled"
                job.next_attempt_at = now_utc + _retry_delay_for_attempts(job.attempts)
            else:
                job.status = "failed"
                job.outcome = "failed"
                job.next_attempt_at = None
            job.updated_at = now_utc
            await self.store.update_job(job)
            return job

        job.call_sids.append(call_sid)
        job.status = "in_progress"
        job.updated_at = now_utc
        await self.store.update_job(job)
        return job

    async def on_call_finished(self, job_id: str, outcome: Outcome, now: datetime) -> CallJob:
        now_utc = _ensure_utc(now)
        job = await self._load_job(job_id)
        if job.status in ("completed", "blocked", "cancelled"):
            return job

        if outcome == "do_not_call":
            await self.store.add_dnc(job.tenant_id, job.contact.phone, source="call_opt_out")
            job.status = "completed"
            job.outcome = "do_not_call"
            job.next_attempt_at = None
            job.updated_at = now_utc
            await self.store.update_job(job)
            return job

        if outcome in ("no_answer", "busy", "failed", "voicemail"):
            if job.attempts < self.settings.outbound_max_attempts:
                job.status = "retry_scheduled"
                job.outcome = outcome
                job.next_attempt_at = now_utc + _retry_delay_for_attempts(job.attempts)
                job.updated_at = now_utc
                await self.store.update_job(job)
                return job

            job.status = "completed"
            job.outcome = outcome
            job.next_attempt_at = None
            job.updated_at = now_utc
            await self.store.update_job(job)

            if self.settings.outbound_sms_fallback and not await self.store.is_dnc(
                job.tenant_id, job.contact.phone
            ):
                name_part = f" {job.contact.name}" if job.contact.name else ""
                business_name = self._get_business_name()
                twilio_num = self.settings.twilio_phone_number or "our support number"
                sms_body = (
                    f"Hi{name_part}, this is {business_name} — we tried to reach you about "
                    f"{job.goal}. Reply or call us back at {twilio_num}. Reply STOP to opt out."
                )
                await self.dialer.send_sms(job.contact.phone, sms_body)
            return job

        job.status = "completed"
        job.outcome = outcome
        job.next_attempt_at = None
        job.updated_at = now_utc
        await self.store.update_job(job)
        return job
