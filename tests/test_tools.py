"""Tests for tool registry, policy engine, tool gateway, built-ins, webhook, and Kettle & Co."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from parley.config import BusinessConfig, load_business
from parley.core.brain.state import CallState, Turn
from parley.core.knowledge.base import Passage
from parley.core.tools.base import ToolContext, ToolKind, ToolResult, ToolSpec
from parley.core.tools.gateway import ToolGateway
from parley.core.tools.registry import ToolRegistry
from parley.core.tools.webhook import webhook_tool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KETTLE_SRC = PROJECT_ROOT / "examples" / "kettle-and-co"


@pytest.fixture
def kettle_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext]:
    monkeypatch.chdir(tmp_path)
    dest = tmp_path / "kettle-and-co"
    shutil.copytree(
        KETTLE_SRC,
        dest,
        ignore=shutil.ignore_patterns("data", "__pycache__", "*.pyc"),
    )
    cfg = load_business(dest / "business.yaml")
    reg = ToolRegistry.for_business(cfg)
    gw = ToolGateway(reg)
    state = CallState(call_id="call-123", tenant_id="kettle")
    resources = reg.resources_factory(cfg) if reg.resources_factory else {}
    ctx = ToolContext(state=state, business=cfg, resources=resources)
    return cfg, reg, gw, ctx


async def test_registry_loads_tools_and_handles_duplicates_and_unknown(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
) -> None:
    cfg, reg1, gw, _ = kettle_env
    expected = [
        "book_callback",
        "create_ticket",
        "end_call",
        "issue_refund",
        "lookup_order",
        "offer_store_credit",
        "opt_out",
        "search_kb",
        "start_return",
        "track_shipment",
        "transfer_to_human",
        "verify_identity",
    ]
    assert sorted(s.name for s in reg1.all()) == expected

    # Loading twice on the same business path reuses specs and does not duplicate tools
    reg2 = ToolRegistry.for_business(cfg)
    assert sorted(s.name for s in reg2.all()) == expected
    assert len(reg2.policies_for("start_return")) == 1
    assert reg2.resources_factory is not None
    assert reg2.caller_lookup is not None

    # Schemas include all 12 tools + confirm_action
    schema_names = [s["function"]["name"] for s in gw.schemas()]
    assert len(schema_names) == 13
    assert "confirm_action" in schema_names

    # Unknown tool in business.tools raises ValueError
    bad_cfg = cfg.model_copy(deep=True)
    bad_cfg.tools = [*cfg.tools, "non_existent_tool_xyz"]
    with pytest.raises(ValueError, match="non_existent_tool_xyz"):
        ToolRegistry.for_business(bad_cfg)


async def test_lookup_order_before_verification_is_blocked(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
) -> None:
    _, _, gw, ctx = kettle_env
    res = await gw.call("lookup_order", {"order_id": "KC-10482"}, ctx)
    assert res.status == "blocked"
    assert res.payload["error"] == "policy_blocked"
    assert res.payload["reason"] == "identity_not_verified"
    assert "verify_identity" in res.payload["instruction"]
    assert ctx.state.tool_events[-1].status == "blocked"
    assert ctx.state.tool_events[-1].reason == "identity_not_verified"


async def test_verify_identity_variants_and_failure(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
) -> None:
    _, reg, gw, ctx = kettle_env

    # Failure: wrong phone/email -> no_match, no customer data leaked
    fail_res = await gw.call(
        "verify_identity",
        {"order_id": "KC-10482", "phone_or_email": "+919999900000"},
        ctx,
    )
    assert fail_res.status == "error"
    assert fail_res.payload == {"ok": False, "error": "no_match"}
    assert not ctx.state.verified
    assert ctx.state.caller is None
    assert ctx.state.failure_counts.get("verify_identity") == 1

    # Success with phone with spaces and lowercase order id without dash
    ok_res = await gw.call(
        "verify_identity",
        {"order_id": "kc 10482", "phone_or_email": "+91 98111 22221"},
        ctx,
    )
    assert ok_res.status == "ok"
    assert ok_res.payload == {"ok": True, "verified": True, "name": "Priya"}
    assert ctx.state.verified is True
    assert ctx.state.caller is not None
    assert ctx.state.caller.customer_id == "CUST-001"

    # Bare 10 digits and bare order number
    ctx.state.verified = False
    ctx.state.caller = None
    ok_res2 = await gw.call(
        "verify_identity",
        {"order_id": "10482", "phone_or_email": "9811122221"},
        ctx,
    )
    assert ok_res2.status == "ok"
    assert ctx.state.verified is True

    # Caller lookup hook by phone
    assert reg.caller_lookup is not None
    profile = await reg.caller_lookup(ctx, "98111 22221")
    assert profile is not None
    assert profile.customer_id == "CUST-001"


async def test_lookup_order_ownership_isolation(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
) -> None:
    _, _, gw, ctx = kettle_env

    # Verify as Priya (CUST-001)
    await gw.call(
        "verify_identity",
        {"order_id": "KC-10482", "phone_or_email": "priya.sharma@example.com"},
        ctx,
    )
    assert ctx.state.verified

    # Priya's own orders succeed
    own_res = await gw.call("lookup_order", {"order_id": "KC-10399"}, ctx)
    assert own_res.status == "ok"
    assert own_res.payload["status"] == "delivered"
    assert own_res.payload["return_eligible"] is True

    # Another customer's order (Rahul's KC-10501) -> not_found (identical to non-existent)
    other_res = await gw.call("lookup_order", {"order_id": "KC-10501"}, ctx)
    assert other_res.status == "error"
    assert other_res.payload == {"ok": False, "error": "not_found"}

    missing_res = await gw.call("lookup_order", {"order_id": "KC-99999"}, ctx)
    assert missing_res.status == "error"
    assert missing_res.payload == {"ok": False, "error": "not_found"}


async def test_start_return_confirmation_flow_and_same_turn_guard(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
) -> None:
    _, _, gw, ctx = kettle_env

    # Verify as Rahul (KC-10501)
    await gw.call(
        "verify_identity",
        {"order_id": "KC-10501", "phone_or_email": "9822233332"},
        ctx,
    )
    ctx.state.turn_index = 1

    # 1st call -> confirmation_required, pending set
    prop = await gw.call(
        "start_return",
        {"order_id": "KC-10501", "reason": "damaged handle"},
        ctx,
    )
    assert prop.status == "confirmation_required"
    assert ctx.state.pending_action is not None
    assert ctx.state.pending_action.name == "start_return"
    assert ctx.state.pending_action.created_turn == 1
    assert (
        prop.payload["confirm_prompt"]
        == "start a return for order KC-10501, reason: damaged handle"
    )

    # Confirm in the SAME turn -> blocked confirmation_must_come_from_caller
    same_turn = await gw.call("confirm_action", {"confirmed": True}, ctx)
    assert same_turn.status == "blocked"
    assert same_turn.payload["error"] == "confirmation_must_come_from_caller"
    assert ctx.state.pending_action is not None

    # Bump turn_index and add user Turn "yes please" -> ok, return_id present, pending cleared
    ctx.state.turn_index = 2
    ctx.state.transcript.append(Turn(role="user", text="yes please"))
    confirmed = await gw.call("confirm_action", {"confirmed": True}, ctx)
    assert confirmed.status == "ok"
    assert confirmed.payload["ok"] is True
    assert confirmed.payload["return_id"].startswith("RT-")
    assert confirmed.payload["pickup"] == "within 2 business days"
    assert ctx.state.pending_action is None
    assert confirmed.event.name == "start_return"

    # Duplicate return for the same order -> return_already_exists
    ctx.state.turn_index = 3
    await gw.call(
        "start_return",
        {"order_id": "KC-10501", "reason": "damaged handle"},
        ctx,
    )
    ctx.state.turn_index = 4
    ctx.state.transcript.append(Turn(role="user", text="yes, confirm"))
    dup = await gw.call("confirm_action", {"confirmed": True}, ctx)
    assert dup.status == "error"
    assert dup.payload["error"] == "return_already_exists"


async def test_confirm_action_negative_utterance_blocked(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
) -> None:
    _, _, gw, ctx = kettle_env

    await gw.call(
        "verify_identity",
        {"order_id": "KC-10501", "phone_or_email": "rahul.v@example.com"},
        ctx,
    )
    ctx.state.turn_index = 1
    await gw.call(
        "start_return",
        {"order_id": "KC-10501", "reason": "wrong item"},
        ctx,
    )

    # Caller says "no, wait" but LLM passes confirmed=True -> blocked caller_did_not_confirm
    ctx.state.turn_index = 2
    ctx.state.transcript.append(Turn(role="user", text="no, wait"))
    res = await gw.call("confirm_action", {"confirmed": True}, ctx)
    assert res.status == "blocked"
    assert res.payload["error"] == "caller_did_not_confirm"
    assert ctx.state.pending_action is not None

    # Confirmed=False cancels pending action cleanly
    cancel_res = await gw.call("confirm_action", {"confirmed": False}, ctx)
    assert cancel_res.status == "ok"
    assert cancel_res.payload == {"ok": True, "cancelled": True}
    assert ctx.state.pending_action is None


async def test_start_return_outside_window_blocked_with_alternatives(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
) -> None:
    _, _, gw, ctx = kettle_env

    # Verify as Priya (KC-10482 delivered 45 days ago, outside 30-day window)
    await gw.call(
        "verify_identity",
        {"order_id": "KC-10482", "phone_or_email": "+919811122221"},
        ctx,
    )
    res = await gw.call(
        "start_return",
        {"order_id": "KC-10482", "reason": "changed mind"},
        ctx,
    )
    assert res.status == "blocked"
    assert res.payload["error"] == "policy_blocked"
    assert res.payload["reason"] == "outside_return_window"
    assert res.payload["escalate"] is False
    assert "store_credit" in res.payload["alternatives"]


async def test_issue_refund_exceeds_limit_blocked_with_escalate(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
) -> None:
    _, _, gw, ctx = kettle_env

    # Verify as Neha (KC-10477, total 7800, delivered 20 days ago)
    await gw.call(
        "verify_identity",
        {"order_id": "KC-10477", "phone_or_email": "neha.g@example.com"},
        ctx,
    )
    res = await gw.call(
        "issue_refund",
        {"order_id": "KC-10477", "amount": 7800},
        ctx,
    )
    assert res.status == "blocked"
    assert res.payload["error"] == "policy_blocked"
    assert res.payload["reason"] == "exceeds_refund_limit"
    assert res.payload["escalate"] is True
    assert "transfer_to_human" in res.payload["alternatives"]


async def test_policy_recheck_on_confirm(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
) -> None:
    cfg, _, gw, ctx = kettle_env

    await gw.call(
        "verify_identity",
        {"order_id": "KC-10501", "phone_or_email": "rahul.v@example.com"},
        ctx,
    )
    ctx.state.turn_index = 1
    prop = await gw.call(
        "issue_refund",
        {"order_id": "KC-10501", "amount": 1500},
        ctx,
    )
    assert prop.status == "confirmation_required"
    assert ctx.state.pending_action is not None

    # Lower refund_max to 100 before confirming
    cfg.policies.refund_max = 100
    ctx.state.turn_index = 2
    ctx.state.transcript.append(Turn(role="user", text="yes, go ahead"))

    confirm_res = await gw.call("confirm_action", {"confirmed": True}, ctx)
    assert confirm_res.status == "blocked"
    assert confirm_res.payload["error"] == "policy_blocked"
    assert confirm_res.payload["reason"] == "exceeds_refund_limit"
    assert confirm_res.event.name == "issue_refund"
    assert ctx.state.pending_action is None


async def test_invalid_arguments_and_tool_exception_handling(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
) -> None:
    _, reg, gw, ctx = kettle_env

    # Invalid JSON string
    bad_json = await gw.call("verify_identity", "{not valid json", ctx)
    assert bad_json.status == "error"
    assert bad_json.payload["error"] == "invalid_arguments"
    assert ctx.state.failure_counts["verify_identity"] == 1

    # Missing required field in dict
    missing_field = await gw.call("verify_identity", {"order_id": "KC-10482"}, ctx)
    assert missing_field.status == "error"
    assert missing_field.payload["error"] == "invalid_arguments"
    assert ctx.state.failure_counts["verify_identity"] == 2

    # Tool raising an unhandled exception -> error tool_failed, does not raise out of call()
    class BoomArgs(BaseModel):
        msg: str = "boom"

    async def _boom_fn(c: ToolContext, a: BoomArgs) -> ToolResult:
        raise RuntimeError(f"database exploded: {a.msg}")

    boom_spec = ToolSpec(
        name="boom_tool",
        description="Fails",
        kind=ToolKind.READ,
        args_model=BoomArgs,
        fn=_boom_fn,
    )
    custom_reg = ToolRegistry(tools=[*reg.all(), boom_spec], policies={})
    custom_gw = ToolGateway(custom_reg)

    boom_res = await custom_gw.call("boom_tool", "", ctx)
    assert boom_res.status == "error"
    assert boom_res.payload["error"] == "tool_failed"
    assert "database exploded" in boom_res.payload["detail"]
    assert ctx.state.failure_counts["boom_tool"] == 1


async def test_webhook_tool_with_mock_transport(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
) -> None:
    _, _, _, ctx = kettle_env

    class CheckStockArgs(BaseModel):
        sku: str

    captured_bodies: list[dict[str, Any]] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_bodies.append(body)
        if body.get("sku") == "ERR-503":
            return httpx.Response(503, json={"detail": "unavailable"})
        if body.get("sku") == "TIMEOUT":
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"in_stock": True, "qty": 12})

    transport = httpx.MockTransport(_handler)
    spec = webhook_tool(
        name="check_stock",
        description="Check warehouse stock for a SKU",
        kind=ToolKind.READ,
        url="https://api.kettle.example.com/stock",
        args_model=CheckStockArgs,
        transport=transport,
    )
    gw = ToolGateway(ToolRegistry(tools=[spec], policies={}))

    # 200 OK response
    ok_res = await gw.call("check_stock", {"sku": "KTL-STL-15"}, ctx)
    assert ok_res.status == "ok"
    assert ok_res.payload == {"ok": True, "in_stock": True, "qty": 12}
    assert captured_bodies[0]["sku"] == "KTL-STL-15"
    assert captured_bodies[0]["call_id"] == "call-123"
    assert captured_bodies[0]["tenant_id"] == "kettle"

    # Non-2xx response -> http_503
    err_res = await gw.call("check_stock", {"sku": "ERR-503"}, ctx)
    assert err_res.status == "error"
    assert err_res.payload["error"] == "http_503"

    # Timeout -> timeout
    to_res = await gw.call("check_stock", {"sku": "TIMEOUT"}, ctx)
    assert to_res.status == "error"
    assert to_res.payload["error"] == "timeout"


async def test_builtins_and_remaining_business_tools(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
    tmp_path: Path,
) -> None:
    _, _, gw, ctx = kettle_env

    # search_kb without retriever -> no_knowledge_base
    no_kb = await gw.call("search_kb", {"query": "return policy"}, ctx)
    assert no_kb.status == "error"
    assert no_kb.payload["error"] == "no_knowledge_base"

    # search_kb with mock retriever
    class _DummyRetriever:
        async def search(self, query: str, k: int = 4) -> list[Passage]:
            return [
                Passage(
                    source="returns-policy.md",
                    chunk_id="returns-policy.md#0",
                    heading="30-Day Return Window",
                    text="30 days from delivery.",
                    score=0.82,
                )
            ]

    ctx.retriever = _DummyRetriever()
    kb_res = await gw.call("search_kb", {"query": "return policy"}, ctx)
    assert kb_res.status == "ok"
    assert kb_res.payload["low_confidence"] is False
    assert len(kb_res.payload["passages"]) == 1

    # create_ticket writes to JsonlOpsStore under tmp_path/data/<tenant_id>/tickets.jsonl
    ticket_res = await gw.call(
        "create_ticket",
        {"summary": "Customer wants bulk quote", "priority": "high"},
        ctx,
    )
    assert ticket_res.status == "ok"
    assert ticket_res.payload["ticket_id"].startswith("T-")
    tickets_file = tmp_path / "data" / "kettle" / "tickets.jsonl"
    assert tickets_file.exists()

    # transfer_to_human when transfer_to is null -> fallback ticket
    xfer_res = await gw.call(
        "transfer_to_human",
        {"summary": "Needs manager approval", "reason": "policy_exception"},
        ctx,
    )
    assert xfer_res.status == "ok"
    assert xfer_res.control is None
    assert xfer_res.payload == {"ok": True, "transferring": False, "fallback": "ticket"}
    assert ctx.state.escalation_reason == "policy_exception"

    # end_call sets control="end_call"
    end_res = await gw.call("end_call", {"reason": "resolved"}, ctx)
    assert end_res.status == "ok"
    assert end_res.control == "end_call"
    assert end_res.payload == {"ok": True, "ending": True}


async def test_cumulative_refunds_and_store_credits_blocked(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
) -> None:
    _, _, gw, ctx = kettle_env

    # Verify as Rahul (KC-10501, total ₹3999)
    await gw.call(
        "verify_identity",
        {"order_id": "KC-10501", "phone_or_email": "9822233332"},
        ctx,
    )

    # First ₹2000 refund succeeds
    ctx.state.turn_index = 1
    prop1 = await gw.call("issue_refund", {"order_id": "KC-10501", "amount": 2000}, ctx)
    assert prop1.status == "confirmation_required"
    ctx.state.turn_index = 2
    ctx.state.transcript.append(Turn(role="user", text="yes please"))
    conf1 = await gw.call("confirm_action", {"confirmed": True}, ctx)
    assert conf1.status == "ok"

    # Second ₹2000 refund on KC-10501 (2000 + 2000 > 3999) -> blocked exceeds_order_total
    ctx.state.turn_index = 3
    prop2 = await gw.call("issue_refund", {"order_id": "KC-10501", "amount": 2000}, ctx)
    assert prop2.status == "blocked"
    assert prop2.payload["reason"] == "exceeds_order_total"
    assert prop2.payload["alternatives"] == ["transfer_to_human"]

    # Store credit of ₹2000 after ₹2000 refund (2000 + 2000 > 3999) -> blocked exceeds_order_total
    credit_prop = await gw.call(
        "offer_store_credit",
        {"order_id": "KC-10501", "amount": 2000},
        ctx,
    )
    assert credit_prop.status == "blocked"
    assert credit_prop.payload["reason"] == "exceeds_order_total"
    assert credit_prop.payload["alternatives"] == ["transfer_to_human"]

    # Cumulative refund_max check on KC-10477 (total ₹7800, refund_max=5000)
    await gw.call(
        "verify_identity",
        {"order_id": "KC-10477", "phone_or_email": "neha.g@example.com"},
        ctx,
    )
    ctx.state.turn_index = 4
    r1 = await gw.call("issue_refund", {"order_id": "KC-10477", "amount": 3000}, ctx)
    assert r1.status == "confirmation_required"
    ctx.state.turn_index = 5
    ctx.state.transcript.append(Turn(role="user", text="yes"))
    r1_conf = await gw.call("confirm_action", {"confirmed": True}, ctx)
    assert r1_conf.status == "ok"

    ctx.state.turn_index = 6
    r2 = await gw.call("issue_refund", {"order_id": "KC-10477", "amount": 2500}, ctx)
    assert r2.status == "blocked"
    assert r2.payload["reason"] == "exceeds_refund_limit"
    assert r2.payload["escalate"] is True


async def test_stale_confirmation_expired_and_buried_affirmative_blocked(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
) -> None:
    _, _, gw, ctx = kettle_env

    await gw.call(
        "verify_identity",
        {"order_id": "KC-10501", "phone_or_email": "9822233332"},
        ctx,
    )
    ctx.state.turn_index = 1
    prop = await gw.call(
        "start_return",
        {"order_id": "KC-10501", "reason": "damaged"},
        ctx,
    )
    assert prop.status == "confirmation_required"
    assert ctx.state.pending_action is not None
    assert ctx.state.pending_action.created_turn == 1

    # Confirm two turns later -> confirmation_expired and pending cleared
    ctx.state.turn_index = 3
    ctx.state.transcript.append(Turn(role="user", text="yes"))
    expired = await gw.call("confirm_action", {"confirmed": True}, ctx)
    assert expired.status == "blocked"
    assert expired.payload == {
        "error": "confirmation_expired",
        "instruction": "Read the action back again and ask the caller to confirm.",
    }
    assert ctx.state.pending_action is None

    # Re-proposing the same action resets created_turn to current turn
    ctx.state.turn_index = 4
    await gw.call("start_return", {"order_id": "KC-10501", "reason": "damaged"}, ctx)
    assert ctx.state.pending_action is not None
    assert ctx.state.pending_action.created_turn == 4

    ctx.state.turn_index = 5
    await gw.call("start_return", {"order_id": "KC-10501", "reason": "damaged"}, ctx)
    assert ctx.state.pending_action.created_turn == 5

    # Long utterance with buried affirmative -> caller_did_not_confirm
    ctx.state.turn_index = 6
    ctx.state.transcript.append(
        Turn(role="user", text="ok so what about my other order, is it shipped yet right")
    )
    buried = await gw.call("confirm_action", {"confirmed": True}, ctx)
    assert buried.status == "blocked"
    assert buried.payload["error"] == "caller_did_not_confirm"
    assert ctx.state.pending_action is not None


async def test_verify_identity_brute_force_locked_after_three_failures(
    kettle_env: tuple[BusinessConfig, ToolRegistry, ToolGateway, ToolContext],
) -> None:
    _, _, gw, ctx = kettle_env

    for _ in range(3):
        res = await gw.call(
            "verify_identity",
            {"order_id": "KC-10482", "phone_or_email": "wrong@example.com"},
            ctx,
        )
        assert res.status == "error"
        assert res.payload["error"] == "no_match"

    # 4th attempt is locked even with valid credentials
    locked = await gw.call(
        "verify_identity",
        {"order_id": "KC-10482", "phone_or_email": "priya.sharma@example.com"},
        ctx,
    )
    assert locked.status == "blocked"
    assert locked.payload == {
        "error": "verification_locked",
        "escalate": True,
        "instruction": (
            "Apologize, say you can't verify the account over the phone right now, "
            "and offer a human or a ticket."
        ),
    }
    assert not ctx.state.verified
