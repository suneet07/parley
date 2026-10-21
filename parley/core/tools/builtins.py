"""Built-in tools available to all businesses."""

from __future__ import annotations

import asyncio
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

from parley.core.tools.base import (
    _COLLECTED_TOOLS,
    ToolContext,
    ToolKind,
    ToolResult,
    ToolSpec,
    tool,
)

logger = structlog.get_logger(__name__)


def _append_jsonl_sync(file_path: Path, line: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as fh:
        fh.write(line)


class JsonlOpsStore:
    """Fallback JSONL store for tickets and callbacks."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    async def append(self, kind: str, record: dict[str, Any]) -> None:
        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            **record,
        }
        line = json.dumps(payload) + "\n"
        target = self.base_dir / f"{kind}.jsonl"
        await asyncio.to_thread(_append_jsonl_sync, target, line)


def _get_ops_store(ctx: ToolContext) -> Any:
    store = ctx.resources.get("ops_store")
    if store is not None and hasattr(store, "append"):
        return store
    return JsonlOpsStore(base_dir=Path("data") / ctx.state.tenant_id)


class SearchKbArgs(BaseModel):
    query: str = Field(description="Search query for the business knowledge base")


class CreateTicketArgs(BaseModel):
    summary: str = Field(description="Brief summary of the customer issue")
    priority: Literal["low", "normal", "high", "urgent"] = Field(
        default="normal", description="Ticket priority level"
    )


class BookCallbackArgs(BaseModel):
    time: str = Field(description="Requested callback time window")
    phone: str | None = Field(default=None, description="Callback phone number")


class TransferToHumanArgs(BaseModel):
    summary: str = Field(description="Three-line summary of the conversation for the agent")
    reason: str = Field(description="Reason for transferring to a human")


class EndCallArgs(BaseModel):
    reason: str = Field(description="Reason for ending the call")


class OptOutArgs(BaseModel):
    reason: str = Field(
        default="caller_requested",
        description="Reason the person asked not to be called again",
    )


@tool(kind=ToolKind.READ)
async def search_kb(ctx: ToolContext, args: SearchKbArgs) -> ToolResult:
    """Search the business knowledge base for policies, FAQs, product info, and shipping details."""
    if ctx.retriever is None:
        return ToolResult(ok=False, error="no_knowledge_base")
    passages = await ctx.retriever.search(args.query, k=3)
    max_score = max((p.score for p in passages), default=0.0)
    return ToolResult(
        ok=True,
        data={
            "passages": [
                {
                    "source": p.source,
                    "heading": p.heading,
                    "text": p.text,
                    "score": p.score,
                }
                for p in passages
            ],
            "low_confidence": max_score < 0.35,
        },
    )


@tool(kind=ToolKind.WRITE, requires_confirmation=False)
async def create_ticket(ctx: ToolContext, args: CreateTicketArgs) -> ToolResult:
    """Create a support ticket for follow-up by the support team."""
    ticket_id = f"T-{secrets.token_hex(3)}"
    store = _get_ops_store(ctx)
    await store.append(
        "tickets",
        {
            "ticket_id": ticket_id,
            "summary": args.summary,
            "priority": args.priority,
            "call_id": ctx.state.call_id,
            "tenant_id": ctx.state.tenant_id,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return ToolResult(ok=True, data={"ticket_id": ticket_id})


@tool(kind=ToolKind.WRITE, confirm="book a callback for {time}")
async def book_callback(ctx: ToolContext, args: BookCallbackArgs) -> ToolResult:
    """Schedule a callback from a support representative at a requested time."""
    callback_id = f"CB-{secrets.token_hex(3)}"
    phone = (
        args.phone
        or ctx.state.caller_number
        or (ctx.state.caller.phone if ctx.state.caller else None)
    )
    store = _get_ops_store(ctx)
    await store.append(
        "callbacks",
        {
            "callback_id": callback_id,
            "time": args.time,
            "phone": phone,
            "call_id": ctx.state.call_id,
            "tenant_id": ctx.state.tenant_id,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return ToolResult(ok=True, data={"callback_id": callback_id, "time": args.time})


@tool(kind=ToolKind.ACTION)
async def transfer_to_human(ctx: ToolContext, args: TransferToHumanArgs) -> ToolResult:
    """Transfer the caller to a human support agent, or create a ticket if unavailable."""
    ctx.state.escalation_reason = args.reason
    ctx.state.summary = args.summary
    transfer_to = ctx.business.escalation.transfer_to
    if transfer_to is not None:
        return ToolResult(
            ok=True,
            control="transfer",
            data={"transferring": True, "to": transfer_to},
        )
    ticket_id = f"T-{secrets.token_hex(3)}"
    store = _get_ops_store(ctx)
    await store.append(
        "tickets",
        {
            "ticket_id": ticket_id,
            "summary": args.summary,
            "priority": "high",
            "reason": args.reason,
            "call_id": ctx.state.call_id,
            "tenant_id": ctx.state.tenant_id,
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    return ToolResult(
        ok=True,
        control=None,
        data={"transferring": False, "fallback": "ticket"},
    )


@tool(kind=ToolKind.ACTION)
async def end_call(ctx: ToolContext, args: EndCallArgs) -> ToolResult:
    """End the phone call politely once the caller's request is resolved."""
    return ToolResult(ok=True, control="end_call", data={"ending": True})


@tool(kind=ToolKind.ACTION, requires_confirmation=False)
async def opt_out(ctx: ToolContext, args: OptOutArgs) -> ToolResult:
    """Use when the person asks not to be called again."""
    outbound_store = ctx.resources.get("outbound_store")
    caller_number = ctx.state.caller_number or (
        ctx.state.caller.phone if ctx.state.caller else None
    )
    if outbound_store is not None and caller_number:
        await outbound_store.add_dnc(
            ctx.state.tenant_id,
            caller_number,
            source=f"opt_out:{args.reason}",
        )
    else:
        logger.info(
            "opt_out_without_outbound_store",
            caller_number=caller_number,
            reason=args.reason,
            call_id=ctx.state.call_id,
        )
    ctx.state.disposition = "do_not_call"
    return ToolResult(ok=True, control="end_call", data={"opted_out": True})


_BUILTIN_NAMES = {
    "search_kb",
    "create_ticket",
    "book_callback",
    "transfer_to_human",
    "end_call",
    "opt_out",
}
BUILTIN_TOOLS: list[ToolSpec] = [spec for spec in _COLLECTED_TOOLS if spec.name in _BUILTIN_NAMES]
