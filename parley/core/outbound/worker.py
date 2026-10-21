"""arq worker glue and TwilioDialer for outbound calling."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any

import structlog
from twilio.rest import Client as TwilioClient

from parley.config import Settings, get_settings, load_business
from parley.core.outbound.compliance import PRESETS, ComplianceEngine, RegionRules
from parley.core.outbound.models import CallJob
from parley.core.outbound.queue import OutboundQueue
from parley.core.outbound.store import OutboundStore

logger = structlog.get_logger(__name__)


def resolve_region_rules(settings: Settings) -> RegionRules:
    return PRESETS.get(settings.compliance_region, PRESETS["in"])


class TwilioDialer:
    """Production dialer placing outbound calls and SMS via Twilio REST API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def dial(self, job: CallJob) -> str:
        base_url = self.settings.public_base_url.rstrip("/")
        job_id = job.job_id

        def _create_call() -> str:
            client = TwilioClient(
                self.settings.twilio_account_sid,
                self.settings.twilio_auth_token,
            )
            call = client.calls.create(
                to=job.contact.phone,
                from_=self.settings.twilio_phone_number,
                url=f"{base_url}/twilio/outbound/{job_id}",
                status_callback=f"{base_url}/twilio/status/{job_id}",
                status_callback_event=["completed"],
                # Synchronous AMD: the answer webhook receives AnsweredBy, so a machine gets
                # the voicemail drop before the AI pipeline ever starts.
                machine_detection="Enable",
                timeout=25,
            )
            return str(call.sid)

        return await asyncio.to_thread(_create_call)

    async def send_sms(self, phone: str, body: str) -> None:
        if not (self.settings.twilio_account_sid and self.settings.twilio_auth_token):
            logger.warning("twilio_sms_skipped", reason="missing_credentials")
            return

        def _send_message() -> None:
            client = TwilioClient(
                self.settings.twilio_account_sid,
                self.settings.twilio_auth_token,
            )
            client.messages.create(
                to=phone,
                from_=self.settings.twilio_phone_number,
                body=body,
            )

        await asyncio.to_thread(_send_message)


def build_outbound_queue(
    settings: Settings | None = None,
    store: OutboundStore | None = None,
) -> OutboundQueue:
    cfg = settings or get_settings()
    business = load_business(cfg.business_config)
    outbound_store = store or OutboundStore(cfg.database_url)
    rules = resolve_region_rules(cfg)
    compliance = ComplianceEngine(
        rules=rules,
        store=outbound_store,
        settings=cfg,
        business_tz=business.hours.tz,
    )
    dialer = TwilioDialer(cfg)
    return OutboundQueue(
        store=outbound_store,
        compliance=compliance,
        dialer=dialer,
        settings=cfg,
        business_name=business.business,
    )


async def on_startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    queue = build_outbound_queue(settings=settings)
    await queue.store.init()
    ctx["settings"] = settings
    ctx["store"] = queue.store
    ctx["compliance"] = queue.compliance
    ctx["dialer"] = queue.dialer
    ctx["queue"] = queue


async def on_shutdown(ctx: dict[str, Any]) -> None:
    store: OutboundStore | None = ctx.get("store")
    if store is not None:
        await store.close()


async def process_job(ctx: dict[str, Any], job_id: str) -> CallJob:
    queue: OutboundQueue = ctx["queue"]
    return await queue.process(job_id, datetime.now(UTC))


async def sweep_due(ctx: dict[str, Any]) -> int:
    settings: Settings | None = ctx.get("settings")
    if settings is not None and settings.kill_switch:
        return 0
    store: OutboundStore = ctx["store"]
    now = datetime.now(UTC)
    due = await store.due_jobs(now)
    pool = ctx.get("redis")
    for job in due:
        if pool is not None:
            # Deterministic arq job id: repeated sweeps don't queue the same attempt twice.
            await pool.enqueue_job(
                "process_job", job.job_id, _job_id=f"{job.job_id}:{job.attempts}"
            )
        else:
            queue: OutboundQueue = ctx["queue"]
            await queue.process(job.job_id, now)
    if ctx.get("queue") is not None:
        await ctx["queue"].send_due_sms(now)  # deferred fallback texts whose window opened
    return len(due)


def _configure_worker_class() -> tuple[list[Any], Any, int]:
    settings = get_settings()
    try:
        from arq.connections import RedisSettings
        from arq.cron import cron

        cron_jobs: list[Any] = [cron(sweep_due, second={0, 30})]
        redis_settings: Any = RedisSettings.from_dsn(settings.redis_url)
    except ImportError:  # pragma: no cover
        cron_jobs = []
        redis_settings = None
    return cron_jobs, redis_settings, settings.outbound_max_concurrent


_CRON_JOBS, _REDIS_SETTINGS, _MAX_JOBS = _configure_worker_class()


class WorkerSettings:
    functions = [process_job]
    cron_jobs = _CRON_JOBS
    redis_settings = _REDIS_SETTINGS
    max_jobs = _MAX_JOBS
    on_startup = on_startup
    on_shutdown = on_shutdown


async def enqueue_now(
    job_id: str,
    *,
    queue: OutboundQueue | None = None,
    settings: Settings | None = None,
    prefer_inline: bool = False,
) -> None:
    """Enqueue a job in arq/Redis, falling back to inline background execution if unreachable."""
    cfg = settings or (queue.settings if queue is not None else get_settings())
    if not prefer_inline:
        try:
            from arq.connections import RedisSettings, create_pool

            redis_cfg = RedisSettings.from_dsn(cfg.redis_url)
            redis_cfg.conn_retries = 0
            redis_cfg.conn_timeout = 1
            pool = await create_pool(redis_cfg)
            try:
                await pool.enqueue_job("process_job", job_id, _job_id=f"{job_id}:new")
                return
            finally:
                await pool.close(close_connection_pool=True)
        except Exception as exc:
            logger.warning(
                "outbound_redis_unreachable_fallback_inline",
                job_id=job_id,
                error=str(exc),
            )

    active_queue = queue
    owns_store = False
    if active_queue is None:
        active_queue = build_outbound_queue(settings=cfg)
        await active_queue.store.init()
        owns_store = True

    async def _run_inline() -> None:
        try:
            await active_queue.process(job_id, datetime.now(UTC))
        except Exception:
            logger.exception("outbound_inline_process_failed", job_id=job_id)
        finally:
            if owns_store:
                await active_queue.store.close()

    task = asyncio.create_task(_run_inline())
    with contextlib.suppress(Exception):
        await task


def run_worker() -> None:
    """Entry point for `parley worker` CLI command."""
    from arq.connections import RedisSettings
    from arq.cron import cron
    from arq.worker import run_worker as arq_run_worker

    settings = get_settings()
    WorkerSettings.cron_jobs = [cron(sweep_due, second={0, 30})]
    WorkerSettings.redis_settings = RedisSettings.from_dsn(settings.redis_url)
    WorkerSettings.max_jobs = settings.outbound_max_concurrent
    arq_run_worker(WorkerSettings)  # type: ignore[arg-type]
