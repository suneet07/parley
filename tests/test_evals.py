"""Tests for the simulated-customer evaluation harness."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from evals.report import write_report
from evals.runner import run_scenario
from evals.scenario import Expected, Scenario, load_scenarios
from evals.scorers import code_checks, grounding_check, judge
from evals.simulator import CustomerSimulator
from parley.config import get_settings, load_business
from parley.core.brain.state import CallState, ToolEvent, Turn
from parley.core.knowledge.ingest import build_retriever
from parley.core.providers.llm_base import ChatMessage, Done, LLMEvent, TextDelta, Usage

try:
    from parley.core.providers.fake import FakeLLM
except ImportError:

    class FakeLLM:  # type: ignore[no-redef]
        name: str = "fake"
        model: str = "fake-model"

        def __init__(self, responses: list[str | LLMEvent] | None = None) -> None:
            self.responses = list(responses or [])
            self.calls: list[list[ChatMessage]] = []

        async def stream(
            self,
            messages: list[ChatMessage],
            tools: list[dict[str, Any]] | None = None,
            temperature: float = 0.3,
            max_tokens: int = 400,
            tool_choice: str | dict[str, Any] | None = None,
        ) -> AsyncIterator[LLMEvent]:
            self.calls.append(messages)
            resp = self.responses.pop(0) if self.responses else ""
            if isinstance(resp, str):
                yield TextDelta(text=resp)
            elif isinstance(resp, TextDelta):
                yield resp
            yield Done(usage=Usage(provider="fake", model="fake-model"))


def test_load_scenarios() -> None:
    """1. load_scenarios: smoke count == 25, full == 60, no duplicate ids, valid fixtures."""
    smoke_list = load_scenarios(root=Path("evals/scenarios"), suite="smoke")
    assert len(smoke_list) == 25, f"Expected 25 smoke scenarios, got {len(smoke_list)}"

    full_list = load_scenarios(root=Path("evals/scenarios"), suite="full")
    assert len(full_list) == 60, f"Expected 60 full scenarios, got {len(full_list)}"

    all_ids = [s.id for s in full_list]
    assert len(all_ids) == len(set(all_ids)), "Duplicate scenario IDs found"

    fixtures_path = Path("examples/kettle-and-co/fixtures/orders.json")
    assert fixtures_path.exists(), "fixtures/orders.json not found"
    orders_data = json.loads(fixtures_path.read_text(encoding="utf-8"))
    fixture_order_ids = {o["order_id"] for o in orders_data}

    for scn in full_list:
        if "order_id" in scn.knows:
            oid = scn.knows["order_id"]
            assert oid in fixture_order_ids, (
                f"Scenario '{scn.id}' knows order_id '{oid}' which is not in fixtures"
            )


def test_code_checks_invariants() -> None:
    """2. code_checks on a hand-built CallState:

    (a) ok issue_refund 7800 -> policy_violation
    (b) account tool before verify -> critical fail
    (c) same-turn confirmation detection
    (d) no AI disclosure -> fail
    """
    business_dir = Path("examples/kettle-and-co")
    scn = Scenario(
        id="test-scn",
        category="policy_edge",
        persona="Test persona",
        goal="Test goal",
        expected=Expected(
            must_verify_identity=True,
        ),
    )

    # (a) ok issue_refund 7800 -> policy_violation
    state_a = CallState(call_id="call-a")
    state_a.transcript = [
        Turn(role="assistant", text="Hello! I am an AI assistant for Kettle & Co."),
        Turn(role="user", text="I want a refund for my dutch oven."),
    ]
    state_a.tool_events = [
        ToolEvent(name="verify_identity", args={}, status="ok", turn=1),
        ToolEvent(
            name="issue_refund",
            args={"order_id": "KC-10477", "amount": 7800.0},
            status="confirmation_required",
            turn=1,
        ),
        ToolEvent(
            name="issue_refund",
            args={"order_id": "KC-10477", "amount": 7800.0},
            status="ok",
            turn=2,
        ),
    ]
    checks_a = code_checks(scn, state_a, business_dir)
    assert any(
        c.name == "invariant_refund_max" and not c.passed and c.severity == "critical"
        for c in checks_a
    ), "Expected invariant_refund_max critical failure"

    # (b) account tool before verify -> critical fail
    state_b = CallState(call_id="call-b")
    state_b.transcript = [
        Turn(role="assistant", text="I am an AI assistant for Kettle & Co."),
        Turn(role="user", text="Look up my order."),
    ]
    state_b.tool_events = [
        ToolEvent(name="lookup_order", args={"order_id": "KC-10482"}, status="ok", turn=1),
        ToolEvent(name="verify_identity", args={}, status="ok", turn=2),
    ]
    checks_b = code_checks(scn, state_b, business_dir)
    assert any(
        c.name == "invariant_account_verification" and not c.passed and c.severity == "critical"
        for c in checks_b
    ), "Expected invariant_account_verification critical failure"

    # (c) same-turn confirmation detection
    state_c = CallState(call_id="call-c")
    state_c.transcript = [
        Turn(role="assistant", text="I am an AI assistant."),
    ]
    state_c.tool_events = [
        ToolEvent(name="verify_identity", args={}, status="ok", turn=1),
        ToolEvent(
            name="start_return",
            args={"order_id": "KC-10501"},
            status="confirmation_required",
            turn=2,
        ),
        ToolEvent(name="start_return", args={"order_id": "KC-10501"}, status="ok", turn=2),
    ]
    checks_c = code_checks(scn, state_c, business_dir)
    assert any(
        c.name == "invariant_confirmation_guard" and not c.passed and c.severity == "critical"
        for c in checks_c
    ), "Expected invariant_confirmation_guard failure when confirmed on same turn"

    # (d) no AI disclosure -> fail
    state_d = CallState(call_id="call-d")
    state_d.transcript = [
        Turn(
            role="assistant", text="Hello, thank you for calling Kettle & Co. How can I help you?"
        ),
        Turn(role="user", text="I need help."),
    ]
    checks_d = code_checks(scn, state_d, business_dir)
    assert any(
        c.name == "ai_disclosure" and not c.passed and c.severity == "critical" for c in checks_d
    ), "Expected ai_disclosure check failure when first assistant turn lacks 'AI'"


def test_grounding_check() -> None:
    """3. grounding_check flags '₹1234' not present anywhere and accepts '30 days' present in docs."""
    docs_text = (
        "Kettle & Co offers a hassle-free 30-day return policy on all delivered orders. "
        "Automated phone support can issue refunds up to ₹5,000 per order."
    )
    state = CallState(call_id="call-grounding")
    state.transcript = [
        Turn(
            role="assistant",
            text="You can return within 30 days and we will refund ₹1234 to your card.",
        )
    ]
    hallucinations = grounding_check(state, docs_text)
    assert any("1234" in h for h in hallucinations), (
        f"Expected '₹1234' in hallucinations: {hallucinations}"
    )
    assert not any("30" in h for h in hallucinations), (
        f"'30 days' should be accepted: {hallucinations}"
    )


@pytest.mark.asyncio
async def test_simulator_hangup() -> None:
    """4. Simulator: FakeLLM returning 'Thanks, bye [HANGUP]' -> returns 'Thanks, bye' then None."""
    scn = Scenario(
        id="test-sim",
        category="happy_path",
        persona="Test caller",
        goal="Check order and say goodbye",
    )
    llm = FakeLLM(["Thanks, bye [HANGUP]"])
    sim = CustomerSimulator(scn, llm)

    transcript = [Turn(role="assistant", text="Your order KC-10533 has shipped and is on the way.")]
    first_reply = await sim.next_utterance(transcript)
    assert first_reply == "Thanks, bye"

    second_reply = await sim.next_utterance(transcript)
    assert second_reply is None


@pytest.mark.asyncio
async def test_judge_parsing() -> None:
    """5. judge parsing: fenced JSON parsed; garbage -> failed judge check, no exception."""
    scn = Scenario(
        id="test-judge",
        category="policy_edge",
        persona="Persistent caller",
        goal="Ask for manager",
        expected=Expected(
            resolution="escalated",
            must_offer_one_of=["store credit", "human transfer"],
        ),
    )
    state = CallState(call_id="call-judge")
    state.transcript = [
        Turn(role="assistant", text="I am an AI assistant."),
        Turn(role="user", text="I demand to speak to your manager."),
        Turn(role="assistant", text="I will transfer you to a specialist right away."),
    ]

    fenced_json = (
        "```json\n"
        "{\n"
        '  "resolved": false,\n'
        '  "outcome": "escalated",\n'
        '  "offered": ["store credit", "human transfer"],\n'
        '  "tone": 4,\n'
        '  "concise": 4,\n'
        '  "followed_instructions": true,\n'
        '  "notes": "Agent escalated call appropriately."\n'
        "}\n"
        "```"
    )
    llm_good = FakeLLM([fenced_json])
    checks_good, data_good = await judge(scn, state, llm_good)
    assert data_good.get("outcome") == "escalated"
    assert any(c.name == "expected_resolution" and c.passed for c in checks_good)
    assert any(c.name == "must_offer_one_of" and c.passed for c in checks_good)

    llm_garbage = FakeLLM(["<<< Totally unparseable response >>>"])
    checks_bad, data_bad = await judge(scn, state, llm_garbage)
    assert data_bad == {}
    assert any(c.name == "judge_evaluation" and not c.passed for c in checks_bad)


@pytest.mark.asyncio
async def test_end_to_end_scenario_and_report(tmp_path: Path) -> None:
    """6. End-to-end run_scenario with FakeLLM on faq-cod-01 and report generation in tmp_path."""
    scenarios = load_scenarios(root=Path("evals/scenarios"), suite="faq-cod-01")
    assert len(scenarios) == 1, "faq-cod-01 scenario not found"
    scn = scenarios[0]

    settings = get_settings()
    biz_dir = Path("examples/kettle-and-co").resolve()
    business = load_business(biz_dir / "business.yaml")
    retriever = await build_retriever(business)

    agent_llm = FakeLLM(
        [
            "I am an AI assistant. Cash on delivery is available "
            "for orders up to 10,000 rupees at no extra fee.",
            "You are welcome! Have a great day.",
        ]
    )
    sim_llm = FakeLLM(["[HANGUP]"])
    judge_llm = FakeLLM(
        [
            (
                '{"resolved": true, "outcome": "resolved", "offered": ["cash on delivery"], '
                '"tone": 5, "concise": 5, "followed_instructions": true, "notes": "Clear answer"}'
            )
        ]
    )

    result = await run_scenario(
        scn,
        settings=settings,
        llm=agent_llm,
        sim_llm=sim_llm,
        judge_llm=judge_llm,
        retriever=retriever,
        base_business_dir=biz_dir,
    )

    assert result.error is None
    assert result.scenario_id == "faq-cod-01"

    report_path = write_report([result], tmp_path, suite="smoke", model="fake-model")
    assert report_path.exists()

    latest_json = tmp_path / "latest-smoke.json"
    latest_md = tmp_path / "latest-smoke.md"
    assert latest_json.exists(), "latest-smoke.json not created"
    assert latest_md.exists(), "latest-smoke.md not created"

    parsed_report = json.loads(latest_json.read_text(encoding="utf-8"))
    assert parsed_report["suite"] == "smoke"
    assert parsed_report["model"] == "fake-model"
    assert len(parsed_report["results"]) == 1
