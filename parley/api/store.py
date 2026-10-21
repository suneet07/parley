from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Column, Float, MetaData, String, Table, select
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from parley.core.brain.state import CallState
from parley.core.telemetry.redact import redact, redact_state

_metadata = MetaData()

calls_table = Table(
    "calls",
    _metadata,
    Column("id", String, primary_key=True),
    Column("tenant_id", String, index=True, nullable=False),
    Column("started_at", String, nullable=True),
    Column("ended_at", String, nullable=True),
    Column("direction", String, nullable=False),
    Column("channel", String, nullable=False),
    Column("caller_number", String, nullable=True),
    Column("disposition", String, nullable=True),
    Column("summary", String, nullable=True),
    Column("intents", JSON, nullable=False),
    Column("escalated", Boolean, nullable=False, default=False),
    Column("duration_s", Float, nullable=False, default=0.0),
    Column("state", JSON, nullable=False),
    Column("metrics", JSON, nullable=True),
)


def _redact_caller_number(caller_number: str | None) -> str:
    if not caller_number:
        return ""
    digits = [ch for ch in caller_number if ch.isdigit()]
    if len(digits) >= 4:
        return "******" + "".join(digits[-4:])
    return redact(caller_number)


class CallStore:
    """Async SQLAlchemy 2 store for redacted call states and latency metrics."""

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._engine: AsyncEngine = create_async_engine(database_url, future=True)

    async def init(self) -> None:
        """Create database tables if they do not exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(_metadata.create_all)

    async def save(self, state: CallState, metrics: dict[str, Any] | None = None) -> None:
        """Upsert a CallState (redacted before persistence) along with latency metrics."""
        raw_dict = state.model_dump(mode="json")
        redacted_dict = redact_state(raw_dict)
        masked_caller = _redact_caller_number(state.caller_number)
        redacted_dict["caller_number"] = masked_caller

        duration_s = 0.0
        if state.ended_at is not None and state.started_at is not None:
            duration_s = max(0.0, round((state.ended_at - state.started_at).total_seconds(), 3))

        escalated = bool(state.disposition == "escalated" or state.escalation_reason)

        values: dict[str, Any] = {
            "tenant_id": state.tenant_id,
            "started_at": redacted_dict.get("started_at"),
            "ended_at": redacted_dict.get("ended_at"),
            "direction": state.direction,
            "channel": state.channel,
            "caller_number": masked_caller,
            "disposition": redacted_dict.get("disposition"),
            "summary": redacted_dict.get("summary"),
            "intents": redacted_dict.get("intents") or [],
            "escalated": escalated,
            "duration_s": duration_s,
            "state": redacted_dict,
            "metrics": metrics or {},
        }

        async with self._engine.begin() as conn:
            existing = await conn.execute(
                select(calls_table.c.id).where(calls_table.c.id == state.call_id)
            )
            if existing.scalar_one_or_none() is not None:
                await conn.execute(
                    calls_table.update().where(calls_table.c.id == state.call_id).values(**values)
                )
            else:
                await conn.execute(calls_table.insert().values(id=state.call_id, **values))

    async def list_calls(
        self, tenant_id: str, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Return summary rows for a tenant ordered by most recent first."""
        stmt = (
            select(
                calls_table.c.id,
                calls_table.c.tenant_id,
                calls_table.c.started_at,
                calls_table.c.ended_at,
                calls_table.c.direction,
                calls_table.c.channel,
                calls_table.c.caller_number,
                calls_table.c.disposition,
                calls_table.c.summary,
                calls_table.c.intents,
                calls_table.c.escalated,
                calls_table.c.duration_s,
                calls_table.c.metrics,
            )
            .where(calls_table.c.tenant_id == tenant_id)
            .order_by(calls_table.c.started_at.desc())
            .limit(limit)
            .offset(offset)
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()
        return [
            {
                "id": row["id"],
                "call_id": row["id"],
                "tenant_id": row["tenant_id"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "direction": row["direction"],
                "channel": row["channel"],
                "caller_number": row["caller_number"],
                "disposition": row["disposition"],
                "summary": row["summary"],
                "intents": row["intents"],
                "escalated": row["escalated"],
                "duration_s": row["duration_s"],
                "metrics": row["metrics"],
            }
            for row in rows
        ]

    async def get_call(self, tenant_id: str, call_id: str) -> dict[str, Any] | None:
        """Return the full redacted call state and metrics for a single call, or None."""
        stmt = select(calls_table).where(
            calls_table.c.tenant_id == tenant_id,
            calls_table.c.id == call_id,
        )
        async with self._engine.connect() as conn:
            row = (await conn.execute(stmt)).mappings().first()
        if row is None:
            return None
        state_data = dict(row["state"]) if isinstance(row["state"], dict) else {}
        return {
            **state_data,
            "id": row["id"],
            "call_id": row["id"],
            "tenant_id": row["tenant_id"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "direction": row["direction"],
            "channel": row["channel"],
            "caller_number": row["caller_number"],
            "disposition": row["disposition"],
            "summary": row["summary"],
            "intents": row["intents"],
            "escalated": row["escalated"],
            "duration_s": row["duration_s"],
            "state": state_data,
            "metrics": row["metrics"],
        }

    async def all_calls(self, tenant_id: str, since: datetime) -> list[dict[str, Any]]:
        """Return full call records (including state and metrics) for a tenant since a datetime."""
        stmt = (
            select(calls_table)
            .where(calls_table.c.tenant_id == tenant_id)
            .order_by(calls_table.c.started_at.desc())
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(stmt)).mappings().all()

        results: list[dict[str, Any]] = []
        since_dt = since if since.tzinfo is not None else since.replace(tzinfo=UTC)

        for row in rows:
            started_at_str = row["started_at"]
            if started_at_str:
                try:
                    dt = datetime.fromisoformat(started_at_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=UTC)
                    if dt < since_dt:
                        continue
                except Exception:
                    if str(started_at_str) < since.isoformat():
                        continue
            elif since is not None:
                continue

            state_data = dict(row["state"]) if isinstance(row["state"], dict) else {}
            results.append(
                {
                    **state_data,
                    "id": row["id"],
                    "call_id": row["id"],
                    "tenant_id": row["tenant_id"],
                    "started_at": row["started_at"],
                    "ended_at": row["ended_at"],
                    "direction": row["direction"],
                    "channel": row["channel"],
                    "caller_number": row["caller_number"],
                    "disposition": row["disposition"],
                    "summary": row["summary"],
                    "intents": row["intents"] if row["intents"] is not None else [],
                    "escalated": row["escalated"],
                    "duration_s": row["duration_s"],
                    "state": state_data,
                    "metrics": row["metrics"],
                }
            )
        return results

    async def close(self) -> None:
        """Dispose of the underlying SQLAlchemy async engine."""
        await self._engine.dispose()
