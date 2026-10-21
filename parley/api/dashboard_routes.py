from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from parley.api.auth import require_api_key
from parley.api.store import CallStore
from parley.config import get_settings, load_business
from parley.core.tools.registry import ToolRegistry

router = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])

RESOLVED_DISPOSITIONS = frozenset({"resolved", "ticket_created", "callback_booked"})
_SUITE_PATTERN = re.compile(r"^[a-z0-9-]+$")


def _nearest_rank_percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    rank = math.ceil((p / 100.0) * len(sorted_vals))
    idx = max(0, min(rank - 1, len(sorted_vals) - 1))
    return sorted_vals[idx]


@router.get("/metrics")
async def get_metrics(
    request: Request,
    days: int = Query(default=7, ge=1),
) -> dict[str, Any]:
    settings = getattr(request.app.state, "settings", None) or get_settings()
    call_store: CallStore = request.app.state.store
    await call_store.init()

    since = datetime.now(UTC) - timedelta(days=days)
    calls = await call_store.all_calls(tenant_id=settings.tenant_id, since=since)

    total_calls = len(calls)
    resolved_count = sum(1 for c in calls if c.get("disposition") in RESOLVED_DISPOSITIONS)
    escalated_count = sum(1 for c in calls if c.get("escalated"))
    resolution_rate = round(resolved_count / total_calls, 4) if total_calls > 0 else None
    escalation_rate = round(escalated_count / total_calls, 4) if total_calls > 0 else None

    durations = [c.get("duration_s") or 0.0 for c in calls]
    avg_duration_s = round(sum(durations) / total_calls, 1) if total_calls > 0 else None

    slowest_candidates: list[dict[str, Any]] = []
    all_e2e: list[float] = []
    stt_vals: list[float] = []
    llm_vals: list[float] = []
    tts_vals: list[float] = []

    for call in calls:
        metrics = call.get("metrics") or {}
        cid = call.get("id") or call.get("call_id") or ""
        turns = metrics.get("turns") or []
        for t in turns:
            stt = t.get("stt_ms")
            llm = t.get("llm_ttft_ms")
            tts = t.get("tts_ttfb_ms")
            e2e = t.get("e2e_ms")
            if stt is not None:
                stt_vals.append(float(stt))
            if llm is not None:
                llm_vals.append(float(llm))
            if tts is not None:
                tts_vals.append(float(tts))
            if e2e is not None:
                e2e_float = float(e2e)
                all_e2e.append(e2e_float)
                slowest_candidates.append(
                    {
                        "call_id": cid,
                        "turn": t.get("turn", 0),
                        "stt_ms": float(stt) if stt is not None else None,
                        "llm_ttft_ms": float(llm) if llm is not None else None,
                        "tts_ttfb_ms": float(tts) if tts is not None else None,
                        "e2e_ms": e2e_float,
                        "interrupted": bool(t.get("interrupted", False)),
                    }
                )

    slowest_candidates.sort(key=lambda t: t["e2e_ms"], reverse=True)
    slowest_turns = slowest_candidates[:10]

    all_e2e.sort()
    p50_e2e_ms = _nearest_rank_percentile(all_e2e, 50)
    p95_e2e_ms = _nearest_rank_percentile(all_e2e, 95)

    if not all_e2e:
        call_p50s = [
            float(c["metrics"]["p50_e2e_ms"])
            for c in calls
            if c.get("metrics") and c["metrics"].get("p50_e2e_ms") is not None
        ]
        if call_p50s:
            call_p50s.sort()
            p50_e2e_ms = _nearest_rank_percentile(call_p50s, 50)
            p95_e2e_ms = _nearest_rank_percentile(call_p50s, 95)

    stage_avg_ms = {
        "stt": round(sum(stt_vals) / len(stt_vals), 1) if stt_vals else None,
        "llm": round(sum(llm_vals) / len(llm_vals), 1) if llm_vals else None,
        "tts": round(sum(tts_vals) / len(tts_vals), 1) if tts_vals else None,
    }

    intent_counts: dict[str, int] = {}
    for call in calls:
        for it in call.get("intents") or []:
            if it:
                intent_counts[it] = intent_counts.get(it, 0) + 1
    intents = [
        {"name": k, "count": v}
        for k, v in sorted(intent_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    disp_counts: dict[str, int] = {}
    for call in calls:
        disp = call.get("disposition") or "unknown"
        disp_counts[disp] = disp_counts.get(disp, 0) + 1
    dispositions = [
        {"name": k, "count": v}
        for k, v in sorted(disp_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    gap_counts: dict[str, dict[str, Any]] = {}
    for call in calls:
        state = call.get("state") or {}
        gaps = state.get("knowledge_gaps") or []
        seen_in_call: set[str] = set()
        for gap in gaps:
            if not isinstance(gap, str):
                continue
            cleaned = gap.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen_in_call:
                continue
            seen_in_call.add(key)
            if key not in gap_counts:
                gap_counts[key] = {"question": cleaned, "count": 1}
            else:
                gap_counts[key]["count"] += 1

    knowledge_gaps = sorted(
        gap_counts.values(),
        key=lambda item: (-item["count"], item["question"].lower()),
    )

    by_day_map: dict[str, dict[str, Any]] = {}
    for call in calls:
        started_at = call.get("started_at")
        date_str = ""
        if started_at:
            try:
                dt = datetime.fromisoformat(started_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                else:
                    dt = dt.astimezone(UTC)
                date_str = dt.strftime("%Y-%m-%d")
            except Exception:
                date_str = str(started_at)[:10]
        if not date_str:
            date_str = datetime.now(UTC).strftime("%Y-%m-%d")

        if date_str not in by_day_map:
            by_day_map[date_str] = {
                "date": date_str,
                "calls": 0,
                "resolved": 0,
                "escalated": 0,
            }
        by_day_map[date_str]["calls"] += 1
        if call.get("disposition") in RESOLVED_DISPOSITIONS:
            by_day_map[date_str]["resolved"] += 1
        if call.get("escalated"):
            by_day_map[date_str]["escalated"] += 1

    by_day = sorted(by_day_map.values(), key=lambda item: item["date"])

    cost_totals: list[float] = []
    cost_minutes: list[float] = []
    for call in calls:
        metrics = call.get("metrics") or {}
        cost_obj = metrics.get("cost")
        if isinstance(cost_obj, dict) and cost_obj.get("total_usd") is not None:
            cost_totals.append(float(cost_obj["total_usd"]))
            mins = cost_obj.get("minutes")
            if mins is not None:
                cost_minutes.append(float(mins))
            else:
                cost_minutes.append(float(call.get("duration_s") or 0.0) / 60.0)

    if cost_totals:
        total_usd: float | None = round(sum(cost_totals), 5)
        per_call_avg_usd: float | None = round(total_usd / len(cost_totals), 5)
        total_mins = sum(cost_minutes)
        per_5min_usd: float | None = (
            round((total_usd / total_mins) * 5.0, 5) if total_mins > 0 else None
        )
    else:
        total_usd = None
        per_call_avg_usd = None
        per_5min_usd = None

    return {
        "days": days,
        "calls": total_calls,
        "resolved": resolved_count,
        "escalated": escalated_count,
        "resolution_rate": resolution_rate,
        "escalation_rate": escalation_rate,
        "avg_duration_s": avg_duration_s,
        "p50_e2e_ms": p50_e2e_ms,
        "p95_e2e_ms": p95_e2e_ms,
        "stage_avg_ms": stage_avg_ms,
        "cost": {
            "total_usd": total_usd,
            "per_call_avg_usd": per_call_avg_usd,
            "per_5min_usd": per_5min_usd,
        },
        "intents": intents,
        "dispositions": dispositions,
        "knowledge_gaps": knowledge_gaps,
        "by_day": by_day,
        "slowest_turns": slowest_turns,
    }


@router.get("/evals/latest")
async def get_latest_eval(suite: str = Query(default="smoke")) -> Any:
    if not _SUITE_PATTERN.match(suite):
        raise HTTPException(status_code=400, detail="Invalid suite name")

    report_path = Path.cwd() / "evals" / "reports" / f"latest-{suite}.json"
    if not report_path.is_file():
        raise HTTPException(status_code=404, detail="no eval report yet")

    try:
        content = report_path.read_text(encoding="utf-8")
        return json.loads(content)
    except Exception as err:
        raise HTTPException(status_code=500, detail="failed to read eval report") from err


@router.get("/config")
async def get_config(request: Request) -> dict[str, Any]:
    business = getattr(request.app.state, "business", None)
    if business is None:
        settings = getattr(request.app.state, "settings", None) or get_settings()
        business = load_business(settings.business_config)

    registry = ToolRegistry.for_business(business)
    tools = [
        {
            "name": t.name,
            "kind": t.kind.value if hasattr(t.kind, "value") else str(t.kind),
            "description": t.description,
            "requires_verification": bool(t.requires_verification),
            "requires_confirmation": bool(t.requires_confirmation),
        }
        for t in registry.all()
    ]
    knowledge = [Path(k).name for k in business.knowledge]
    return {
        "business": business.model_dump(mode="json"),
        "tools": tools,
        "knowledge": knowledge,
    }
