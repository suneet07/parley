"""Async SQLAlchemy 2 storage for outbound contacts, call jobs, DNC list, dials, and audit logs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from parley.core.outbound.models import (
    CallJob,
    ComplianceDecision,
    Contact,
    normalize_phone,
)

_metadata = MetaData()

outbound_contacts_table = Table(
    "outbound_contacts",
    _metadata,
    Column("id", String, primary_key=True),
    Column("tenant_id", String, index=True, nullable=False),
    Column("contact_id", String, index=True, nullable=False),
    Column("phone", String, index=True, nullable=False),
    Column("contact", JSON, nullable=False),
)

outbound_jobs_table = Table(
    "outbound_jobs",
    _metadata,
    Column("job_id", String, primary_key=True),
    Column("tenant_id", String, index=True, nullable=False),
    Column("status", String, index=True, nullable=False),
    Column("next_attempt_at", DateTime, index=True, nullable=True),
    Column("job", JSON, nullable=False),
)

outbound_dnc_table = Table(
    "outbound_dnc",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("tenant_id", String, index=True, nullable=False),
    Column("phone", String, index=True, nullable=False),
    Column("source", String, nullable=False),
    Column("created_at", DateTime, nullable=False),
    UniqueConstraint("tenant_id", "phone", name="uq_outbound_dnc_tenant_phone"),
)

outbound_dials_table = Table(
    "outbound_dials",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("tenant_id", String, index=True, nullable=False),
    Column("phone", String, index=True, nullable=False),
    Column("from_number", String, index=True, nullable=False),
    Column("at", DateTime, index=True, nullable=False),
    Column("job_id", String, index=True, nullable=False),
)

compliance_audit_table = Table(
    "compliance_audit",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("tenant_id", String, index=True, nullable=False),
    Column("job_id", String, index=True, nullable=False),
    Column("phone", String, nullable=False),
    Column("allowed", Boolean, nullable=False),
    Column("reasons", JSON, nullable=False),
    Column("retry_at", String, nullable=True),
    Column("at", DateTime, index=True, nullable=False),
)


def _to_utc_naive(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(UTC).replace(tzinfo=None)


def _ensure_utc_dt(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _mask_phone(phone: str | None) -> str:
    if not phone:
        return ""
    digits = [ch for ch in phone if ch.isdigit()]
    if len(digits) >= 4:
        return "******" + "".join(digits[-4:])
    return "******"


def _hydrate_job(payload: Any) -> CallJob:
    job = CallJob.model_validate(payload)
    job.created_at = _ensure_utc_dt(job.created_at) or datetime.now(UTC)
    job.updated_at = _ensure_utc_dt(job.updated_at) or datetime.now(UTC)
    job.next_attempt_at = _ensure_utc_dt(job.next_attempt_at)
    if job.contact.consent_at is not None:
        job.contact.consent_at = _ensure_utc_dt(job.contact.consent_at)
    return job


class OutboundStore:
    """Async SQLAlchemy 2 store for outbound contacts, call jobs, DNC, dials, and compliance audit."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._engine: AsyncEngine = create_async_engine(database_url, future=True)

    async def init(self) -> None:
        """Create outbound tables if they do not exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(_metadata.create_all)

    async def close(self) -> None:
        """Dispose of the underlying SQLAlchemy async engine."""
        await self._engine.dispose()

    async def upsert_contact(self, tenant_id: str, contact: Contact) -> None:
        row_id = f"{tenant_id}:{contact.contact_id}"
        norm_phone = normalize_phone(contact.phone)
        data = contact.model_dump(mode="json")
        values = {
            "tenant_id": tenant_id,
            "contact_id": contact.contact_id,
            "phone": norm_phone,
            "contact": data,
        }
        async with self._engine.begin() as conn:
            existing = await conn.execute(
                select(outbound_contacts_table.c.id).where(outbound_contacts_table.c.id == row_id)
            )
            if existing.scalar_one_or_none() is not None:
                await conn.execute(
                    outbound_contacts_table.update()
                    .where(outbound_contacts_table.c.id == row_id)
                    .values(**values)
                )
            else:
                await conn.execute(outbound_contacts_table.insert().values(id=row_id, **values))

    async def create_job(self, job: CallJob) -> None:
        await self.upsert_contact(job.tenant_id, job.contact)
        next_at = _to_utc_naive(job.next_attempt_at) if job.next_attempt_at is not None else None
        payload = job.model_dump(mode="json")
        values = {
            "tenant_id": job.tenant_id,
            "status": job.status,
            "next_attempt_at": next_at,
            "job": payload,
        }
        async with self._engine.begin() as conn:
            existing = await conn.execute(
                select(outbound_jobs_table.c.job_id).where(
                    outbound_jobs_table.c.job_id == job.job_id
                )
            )
            if existing.scalar_one_or_none() is not None:
                await conn.execute(
                    outbound_jobs_table.update()
                    .where(outbound_jobs_table.c.job_id == job.job_id)
                    .values(**values)
                )
            else:
                await conn.execute(outbound_jobs_table.insert().values(job_id=job.job_id, **values))

    async def get_job(self, tenant_id: str, job_id: str) -> CallJob | None:
        stmt = select(outbound_jobs_table.c.job).where(
            outbound_jobs_table.c.tenant_id == tenant_id,
            outbound_jobs_table.c.job_id == job_id,
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).first()
        if row is None:
            return None
        return _hydrate_job(row[0])

    async def get_job_by_id(self, job_id: str) -> CallJob | None:
        stmt = select(outbound_jobs_table.c.job).where(outbound_jobs_table.c.job_id == job_id)
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).first()
        if row is None:
            return None
        return _hydrate_job(row[0])

    async def update_job(self, job: CallJob) -> None:
        next_at = _to_utc_naive(job.next_attempt_at) if job.next_attempt_at is not None else None
        payload = job.model_dump(mode="json")
        async with self._engine.begin() as conn:
            await conn.execute(
                outbound_jobs_table.update()
                .where(outbound_jobs_table.c.job_id == job.job_id)
                .values(
                    status=job.status,
                    next_attempt_at=next_at,
                    job=payload,
                )
            )

    async def claim_job(self, tenant_id: str, job_id: str, now: datetime) -> CallJob | None:
        """Atomically move a queued/retry_scheduled job to 'dialing'. Returns the job only for the
        one caller whose UPDATE changed the row, so concurrent workers can't dial it twice."""
        async with self._engine.begin() as conn:
            result = await conn.execute(
                outbound_jobs_table.update()
                .where(
                    outbound_jobs_table.c.tenant_id == tenant_id,
                    outbound_jobs_table.c.job_id == job_id,
                    outbound_jobs_table.c.status.in_(("queued", "retry_scheduled")),
                )
                .values(status="dialing")
            )
            if result.rowcount != 1:
                return None
            row = (
                await conn.execute(
                    select(outbound_jobs_table.c.job).where(outbound_jobs_table.c.job_id == job_id)
                )
            ).first()
            if row is None:
                return None
            job = _hydrate_job(row[0])
            job.status = "dialing"
            job.updated_at = _ensure_utc_dt(now) or datetime.now(UTC)
            await conn.execute(
                outbound_jobs_table.update()
                .where(outbound_jobs_table.c.job_id == job_id)
                .values(job=job.model_dump(mode="json"))
            )
        return job

    async def find_recent_job(
        self,
        tenant_id: str,
        *,
        since: datetime,
        idempotency_key: str | None = None,
        phone: str | None = None,
        goal: str | None = None,
    ) -> CallJob | None:
        """Most recent job created since `since` matching an idempotency key, or (phone, goal)."""
        stmt = select(outbound_jobs_table.c.job).where(outbound_jobs_table.c.tenant_id == tenant_id)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        since_utc = _ensure_utc_dt(since)
        norm = normalize_phone(phone) if phone else None
        best: CallJob | None = None
        for (payload,) in rows:
            job = _hydrate_job(payload)
            if since_utc is not None and job.created_at < since_utc:
                continue
            if idempotency_key is not None:
                hit = job.idempotency_key == idempotency_key
            else:
                hit = normalize_phone(job.contact.phone) == norm and job.goal == goal
            if hit and (best is None or job.created_at > best.created_at):
                best = job
        return best

    async def pending_sms_jobs(self, now: datetime, limit: int = 50) -> list[CallJob]:
        stmt = select(outbound_jobs_table.c.job).where(outbound_jobs_table.c.status == "completed")
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        now_utc = _ensure_utc_dt(now)
        jobs = []
        for (payload,) in rows:
            job = _hydrate_job(payload)
            at = _ensure_utc_dt(job.pending_sms_at)
            if at is not None and not job.sms_sent and now_utc is not None and at <= now_utc:
                jobs.append(job)
        return jobs[:limit]

    async def due_jobs(self, now: datetime, limit: int = 50) -> list[CallJob]:
        now_naive = _to_utc_naive(now)
        stmt = (
            select(outbound_jobs_table.c.job)
            .where(
                outbound_jobs_table.c.status.in_(("queued", "retry_scheduled")),
                outbound_jobs_table.c.next_attempt_at.is_not(None),
                outbound_jobs_table.c.next_attempt_at <= now_naive,
            )
            .order_by(outbound_jobs_table.c.next_attempt_at.asc())
            .limit(limit)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return [_hydrate_job(r[0]) for r in rows]

    async def add_dnc(self, tenant_id: str, phone: str, source: str) -> None:
        norm_phone = normalize_phone(phone)
        now_naive = _to_utc_naive(datetime.now(UTC))
        async with self._engine.begin() as conn:
            existing = await conn.execute(
                select(outbound_dnc_table.c.id).where(
                    outbound_dnc_table.c.tenant_id == tenant_id,
                    outbound_dnc_table.c.phone == norm_phone,
                )
            )
            if existing.scalar_one_or_none() is None:
                await conn.execute(
                    outbound_dnc_table.insert().values(
                        tenant_id=tenant_id,
                        phone=norm_phone,
                        source=source,
                        created_at=now_naive,
                    )
                )

    async def is_dnc(self, tenant_id: str, phone: str) -> bool:
        norm_phone = normalize_phone(phone)
        stmt = select(outbound_dnc_table.c.id).where(
            outbound_dnc_table.c.tenant_id == tenant_id,
            outbound_dnc_table.c.phone == norm_phone,
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).first()
        return row is not None

    async def record_dial(
        self,
        tenant_id: str,
        phone: str,
        from_number: str,
        at: datetime,
        job_id: str,
    ) -> None:
        norm_phone = normalize_phone(phone)
        async with self._engine.begin() as conn:
            await conn.execute(
                outbound_dials_table.insert().values(
                    tenant_id=tenant_id,
                    phone=norm_phone,
                    from_number=from_number,
                    at=_to_utc_naive(at),
                    job_id=job_id,
                )
            )

    async def count_dials(
        self,
        tenant_id: str,
        *,
        phone: str | None = None,
        from_number: str | None = None,
        since: datetime,
    ) -> int:
        stmt = select(func.count(outbound_dials_table.c.id)).where(
            outbound_dials_table.c.tenant_id == tenant_id,
            outbound_dials_table.c.at >= _to_utc_naive(since),
        )
        if phone is not None:
            stmt = stmt.where(outbound_dials_table.c.phone == normalize_phone(phone))
        if from_number is not None:
            stmt = stmt.where(outbound_dials_table.c.from_number == from_number)
        async with self._engine.connect() as conn:
            count_val = (await conn.execute(stmt)).scalar_one()
        return int(count_val or 0)

    async def audit(self, job: CallJob, decision: ComplianceDecision, at: datetime) -> None:
        retry_str = decision.retry_at.astimezone(UTC).isoformat() if decision.retry_at else None
        async with self._engine.begin() as conn:
            await conn.execute(
                compliance_audit_table.insert().values(
                    tenant_id=job.tenant_id,
                    job_id=job.job_id,
                    phone=job.contact.phone,
                    allowed=decision.allowed,
                    reasons=list(decision.reasons),
                    retry_at=retry_str,
                    at=_to_utc_naive(at),
                )
            )

    async def audit_log(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
        stmt = (
            select(compliance_audit_table)
            .where(compliance_audit_table.c.tenant_id == tenant_id)
            .order_by(compliance_audit_table.c.at.desc(), compliance_audit_table.c.id.desc())
            .limit(limit)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        results: list[dict[str, Any]] = []
        for row in rows:
            at_dt: datetime | None = row["at"]
            at_iso = (
                at_dt.replace(tzinfo=UTC).isoformat()
                if isinstance(at_dt, datetime)
                else str(at_dt or "")
            )
            results.append(
                {
                    "job_id": row["job_id"],
                    "phone": _mask_phone(row["phone"]),
                    "allowed": bool(row["allowed"]),
                    "reasons": list(row["reasons"] or []),
                    "retry_at": row["retry_at"],
                    "at": at_iso,
                }
            )
        return results
