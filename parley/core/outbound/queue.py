"""Outbound dial queue and retry scheduling logic (independent of arq for unit testability)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol

import structlog

from parley.config import load_business
from parley.core.outbound.compliance import _next_window_start_utc, callee_timezone
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
        self._locks: dict[str, asyncio.Lock] = {}

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

    def _tenant_lock(self, tenant_id: str) -> asyncio.Lock:
        # Serializes compliance check + dial bookkeeping per tenant inside this process so two
        # concurrent jobs for one contact can't both pass the daily cap. Run ONE worker process
        # per tenant (the default arq setup) for the caps to hold across processes too.
        lock = self._locks.get(tenant_id)
        if lock is None:
            lock = self._locks[tenant_id] = asyncio.Lock()
        return lock

    async def process(self, job_id: str, now: datetime) -> CallJob:
        now_utc = _ensure_utc(now)
        job = await self._load_job(job_id)
        if getattr(self.settings, "kill_switch", False):
            return job
        if job.status not in ("queued", "retry_scheduled"):
            return job

        async with self._tenant_lock(job.tenant_id):
            claimed = await self.store.claim_job(job.tenant_id, job.job_id, now_utc)
            if claimed is None:  # another worker got it first
                return await self._load_job(job_id)
            job = claimed

            decision = await self.compliance.check(job, now_utc)
            if not decision.allowed:
                if decision.retry_at is not None:
                    job.status = "retry_scheduled"
                    job.next_attempt_at = decision.retry_at
                else:
                    job.status = "blocked"
                    job.outcome = None
                    job.next_attempt_at = None
                job.last_error = ", ".join(decision.reasons)
                job.updated_at = now_utc
                await self.store.update_job(job)
                return job

            job.attempts += 1
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

    async def on_call_finished(
        self, job_id: str, outcome: Outcome, now: datetime, call_sid: str | None = None
    ) -> CallJob:
        """Record a call's outcome. Idempotent per call_sid: the media stream, the status callback
        and AMD can all report the same call; only the first report for the latest call counts."""
        now_utc = _ensure_utc(now)
        job = await self._load_job(job_id)
        if call_sid is not None:
            if call_sid in job.finished_call_sids:
                return job
            if job.call_sids and call_sid != job.call_sids[-1]:
                return job  # stale report about an earlier attempt
        if job.status not in ("dialing", "in_progress"):
            return job
        if call_sid is not None:
            job.finished_call_sids.append(call_sid)

        if outcome == "do_not_call":
            await self.store.add_dnc(job.tenant_id, job.contact.phone, source="call_opt_out")
            return await self._finish(job, "do_not_call", now_utc)

        if outcome in ("no_answer", "busy", "failed", "voicemail"):
            if job.attempts < self.settings.outbound_max_attempts:
                job.status = "retry_scheduled"
                job.outcome = outcome
                job.next_attempt_at = now_utc + _retry_delay_for_attempts(job.attempts)
                job.updated_at = now_utc
                await self.store.update_job(job)
                return job
            job = await self._finish(job, outcome, now_utc)
            if self.settings.outbound_sms_fallback:
                job = await self._schedule_sms(job, now_utc)
            return job

        return await self._finish(job, outcome, now_utc)

    async def _finish(self, job: CallJob, outcome: Outcome, now_utc: datetime) -> CallJob:
        job.status = "completed"
        job.outcome = outcome
        job.next_attempt_at = None
        job.updated_at = now_utc
        await self.store.update_job(job)
        return job

    async def _schedule_sms(self, job: CallJob, now_utc: datetime) -> CallJob:
        """Send the one fallback SMS now if the callee is inside their calling window, else
        defer it to the next window start (picked up by send_due_sms)."""
        if job.sms_sent or job.pending_sms_at is not None:
            return job
        window_at = await self._sms_window(job, now_utc)
        if window_at is None:
            return job  # unknown zone or DNC: never text
        job.pending_sms_at = window_at
        await self.store.update_job(job)
        if window_at <= now_utc:
            job = await self._send_sms(job)
        return job

    async def _sms_window(self, job: CallJob, now_utc: datetime) -> datetime | None:
        if await self.store.is_dnc(job.tenant_id, job.contact.phone):
            return None
        tz = callee_timezone(job.contact)
        if tz is None:
            return None
        rules = self.compliance.rules
        local = now_utc.astimezone(tz)
        local_t = local.timetz().replace(tzinfo=None)
        workday = local.weekday() in rules.allowed_weekdays
        if workday and rules.window_start <= local_t < rules.window_end:
            return now_utc
        offset = 0 if workday and local_t < rules.window_start else 1
        return _next_window_start_utc(local.date(), rules, tz, start_offset_days=offset)

    async def _send_sms(self, job: CallJob) -> CallJob:
        if job.sms_sent or await self.store.is_dnc(job.tenant_id, job.contact.phone):
            return job
        job.sms_sent = True  # set before sending: a crash must not cause a second text
        await self.store.update_job(job)
        name_part = f" {job.contact.name}" if job.contact.name else ""
        twilio_num = self.settings.twilio_phone_number or "our support number"
        body = (
            f"Hi{name_part}, this is {self._get_business_name()}. We tried to reach you about "
            f"your recent request. Call us back at {twilio_num}. Reply STOP to opt out."
        )
        await self.dialer.send_sms(job.contact.phone, body)
        return job

    async def send_due_sms(self, now: datetime) -> int:
        """Send deferred fallback SMS whose window has opened. Called by the worker sweep."""
        sent = 0
        for job in await self.store.pending_sms_jobs(_ensure_utc(now)):
            before = job.sms_sent
            job = await self._send_sms(job)
            sent += int(job.sms_sent and not before)
        return sent
