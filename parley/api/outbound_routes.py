"""FastAPI routes for outbound call triggers, CSV campaigns, job status, DNC, and compliance audit."""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from parley.api.auth import require_api_key
from parley.config import Settings, get_settings
from parley.core.outbound.models import CallJob, Contact, to_e164
from parley.core.outbound.queue import OutboundQueue
from parley.core.outbound.worker import build_outbound_queue, enqueue_now

router = APIRouter(prefix="/api/outbound", tags=["outbound"])

_KNOWN_CSV_COLUMNS = {
    "contact_id",
    "phone",
    "name",
    "tz",
    "consent",
    "consent_source",
    "consent_at",
}


class TriggerPayload(BaseModel):
    contact: Contact
    goal: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, max_length=200)


DEDUPE_WINDOW = timedelta(minutes=10)
IDEMPOTENCY_WINDOW = timedelta(hours=24)


class DncPayload(BaseModel):
    phone: str
    source: str = "api"


def _get_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or get_settings()


async def _get_queue_and_settings(request: Request) -> tuple[OutboundQueue, Settings, bool]:
    settings = _get_settings(request)
    queue: OutboundQueue | None = getattr(request.app.state, "outbound_queue", None)
    is_injected = bool(getattr(request.app.state, "outbound_queue_injected", False))
    if queue is None:
        store = getattr(request.app.state, "outbound_store", None)
        queue = build_outbound_queue(settings=settings, store=store)
        request.app.state.outbound_queue = queue
        request.app.state.outbound_store = queue.store
    else:
        queue.settings = settings
        queue.compliance.settings = settings
    await queue.store.init()
    return queue, settings, is_injected


async def _verify_hmac(request: Request) -> bytes:
    settings = _get_settings(request)
    secret = settings.outbound_webhook_secret
    if not secret:
        raise HTTPException(
            status_code=503,
            detail="outbound disabled: set OUTBOUND_WEBHOOK_SECRET",
        )
    raw_body = await request.body()
    # Signature covers "<timestamp>.<body>" so a captured request can't be replayed later.
    ts_header = request.headers.get("X-Parley-Timestamp", "")
    try:
        ts = int(ts_header)
    except ValueError:
        raise HTTPException(
            status_code=401, detail="Missing or invalid X-Parley-Timestamp"
        ) from None
    if abs(datetime.now(UTC).timestamp() - ts) > settings.outbound_signature_max_age_s:
        raise HTTPException(status_code=401, detail="Signature timestamp too old")
    sig_header = request.headers.get("X-Parley-Signature", "")
    signed = f"{ts}.".encode() + raw_body
    digest = hmac.new(secret.encode("utf-8"), signed, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig_header, f"sha256={digest}"):
        raise HTTPException(status_code=401, detail="Invalid signature")
    return raw_body


def _consent_problem(contact: Contact) -> str | None:
    # Consent must be evidenced by the caller of the API; we never invent a timestamp.
    if contact.consent and (not contact.consent_source or contact.consent_at is None):
        return "consent requires consent_source and consent_at"
    return None


@router.post("/trigger", status_code=202)
async def trigger_outbound_call(request: Request) -> JSONResponse:
    raw_body = await _verify_hmac(request)
    queue, settings, is_injected = await _get_queue_and_settings(request)

    try:
        data = json.loads(raw_body.decode("utf-8"))
        payload = TriggerPayload.model_validate(data)
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    now = datetime.now(UTC)
    contact = payload.contact
    if payload.metadata:
        merged = dict(contact.metadata)
        for k, v in payload.metadata.items():
            merged[str(k)] = str(v)
        contact.metadata = merged
    problem = _consent_problem(contact)
    if problem:
        raise HTTPException(status_code=422, detail=problem)

    if payload.idempotency_key:
        existing = await queue.store.find_recent_job(
            settings.tenant_id,
            since=now - IDEMPOTENCY_WINDOW,
            idempotency_key=payload.idempotency_key,
        )
    else:
        existing = await queue.store.find_recent_job(
            settings.tenant_id,
            since=now - DEDUPE_WINDOW,
            phone=contact.phone,
            goal=payload.goal,
        )
    if existing is not None:
        return JSONResponse(
            status_code=200,
            content={"job_id": existing.job_id, "status": existing.status, "duplicate": True},
        )

    job = CallJob(
        idempotency_key=payload.idempotency_key,
        tenant_id=settings.tenant_id,
        contact=contact,
        goal=payload.goal,
        trigger="webhook",
        next_attempt_at=now,
        created_at=now,
        updated_at=now,
    )
    await queue.enqueue(job, now)
    await enqueue_now(
        job.job_id,
        queue=queue,
        settings=settings,
        prefer_inline=is_injected,
    )
    current = await queue.store.get_job(settings.tenant_id, job.job_id) or job
    return JSONResponse(
        status_code=202,
        content={"job_id": current.job_id, "status": current.status},
    )


@router.post("/campaigns")
async def upload_campaign_csv(
    request: Request,
    goal: str = Query(default="follow up"),
    campaign_id: str | None = Query(default=None),
) -> dict[str, Any]:
    raw_body = await _verify_hmac(request)
    queue, settings, _ = await _get_queue_and_settings(request)

    try:
        csv_text = raw_body.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded") from exc

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    if len(rows) > 5000:
        raise HTTPException(status_code=400, detail="Campaign CSV exceeds maximum of 5000 rows")

    cid = campaign_id or f"cmp_{uuid.uuid4().hex[:10]}"
    spacing_s = 60.0 / max(1, settings.outbound_max_concurrent)
    base_now = datetime.now(UTC)

    accepted = 0
    rejected: list[dict[str, Any]] = []
    job_ids: list[str] = []

    for row_idx, row in enumerate(rows, start=1):
        phone = (row.get("phone") or "").strip()
        name = (row.get("name") or "").strip() or None
        tz = (row.get("tz") or "").strip() or None
        consent_raw = (row.get("consent") or "").strip().lower()
        consent = consent_raw in ("true", "1", "yes", "y")
        consent_source = (row.get("consent_source") or "").strip() or None
        consent_at_raw = (row.get("consent_at") or "").strip()
        if consent_at_raw:
            try:
                consent_at = datetime.fromisoformat(consent_at_raw)
                if consent_at.tzinfo is None:
                    consent_at = consent_at.replace(tzinfo=UTC)
            except ValueError:
                consent_at = None
        else:
            consent_at = None

        extra_meta: dict[str, str] = {}
        for col, val in row.items():
            if col and col not in _KNOWN_CSV_COLUMNS and val is not None and str(val).strip():
                extra_meta[col] = str(val).strip()

        contact_kwargs: dict[str, Any] = {
            "phone": phone,
            "name": name,
            "tz": tz,
            "consent": consent,
            "consent_source": consent_source,
            "consent_at": consent_at,
            "metadata": extra_meta,
        }
        cid_col = (row.get("contact_id") or "").strip()
        if cid_col:
            contact_kwargs["contact_id"] = cid_col

        try:
            contact = Contact.model_validate(contact_kwargs)
        except ValidationError:
            rejected.append({"row": row_idx, "reason": "invalid_phone"})
            continue
        problem = _consent_problem(contact)
        if problem:
            rejected.append({"row": row_idx, "reason": problem})
            continue

        scheduled_at = base_now + timedelta(seconds=accepted * spacing_s)
        job = CallJob(
            tenant_id=settings.tenant_id,
            contact=contact,
            goal=goal,
            trigger="campaign",
            campaign_id=cid,
            next_attempt_at=scheduled_at,
            created_at=base_now,
            updated_at=base_now,
        )
        await queue.enqueue(job, scheduled_at)
        accepted += 1
        job_ids.append(job.job_id)

    return {
        "campaign_id": cid,
        "accepted": accepted,
        "rejected": rejected,
        "job_ids": job_ids,
    }


@router.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
async def get_outbound_job(job_id: str, request: Request) -> dict[str, Any]:
    queue, settings, _ = await _get_queue_and_settings(request)
    job = await queue.store.get_job(settings.tenant_id, job_id)
    if job is None:
        job = await queue.store.get_job_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    data = job.model_dump(mode="json")
    contact = data["contact"]
    contact["phone"] = _mask(contact.get("phone"))
    contact["metadata"] = sorted(contact.get("metadata", {}))  # keys only; values may be personal
    return data


def _mask(phone: str | None) -> str:
    digits = [c for c in phone or "" if c.isdigit()]
    return "******" + "".join(digits[-4:]) if len(digits) >= 4 else "******"


@router.get("/audit", dependencies=[Depends(require_api_key)])
async def get_outbound_audit(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    queue, settings, _ = await _get_queue_and_settings(request)
    return await queue.store.audit_log(settings.tenant_id, limit=limit)


@router.post("/dnc")
async def add_outbound_dnc(request: Request) -> dict[str, Any]:
    raw_body = await _verify_hmac(request)
    queue, settings, _ = await _get_queue_and_settings(request)

    try:
        data = json.loads(raw_body.decode("utf-8"))
        payload = DncPayload.model_validate(data)
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        norm_phone = to_e164(payload.phone, settings.default_country)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await queue.store.add_dnc(settings.tenant_id, norm_phone, source=payload.source)
    return {"ok": True, "phone": norm_phone}
