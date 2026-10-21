"""Evaluation runner: executes scenarios concurrently against isolated business copies."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from evals.report import write_report
from evals.scenario import Scenario, load_scenarios
from evals.scorers import Check, ScenarioResult, code_checks, grounding_check, judge
from evals.simulator import CustomerSimulator
from parley.config import BusinessConfig, Settings, get_settings, load_business
from parley.core.brain.state import CallState, Turn
from parley.core.knowledge.ingest import build_retriever
from parley.core.providers.llm_base import LLM, ChatMessage, TextDelta

try:
    from parley.core.brain import Agent, CallSession
except ImportError:
    # Fallback Agent implementation when core brain module is being assembled
    class CallSession:  # type: ignore[no-redef]
        def __init__(
            self,
            agent: Agent,
            call_id: str,
            direction: str,
            caller_number: str | None,
            outbound_goal: str | None,
        ) -> None:
            self.agent = agent
            self.state = CallState(
                call_id=call_id,
                direction=direction,  # type: ignore
                caller_number=caller_number,
            )

        async def start(self) -> Any:
            greeting = self.agent.business.greeting or (
                "Thanks for calling Kettle and Co, an AI customer support assistant. "
                "How can I help you today?"
            )
            if "ai" not in greeting.lower():
                greeting = f"I am an AI assistant. {greeting}"
            self.state.transcript.append(Turn(role="assistant", text=greeting))
            yield greeting

        async def respond(self, text: str) -> Any:
            self.state.transcript.append(Turn(role="user", text=text))
            collected: list[str] = []
            sys_content = (
                f"You are support agent for {self.agent.business.business}. "
                f"Persona: {self.agent.business.persona}"
            )
            messages = [
                ChatMessage(role="system", content=sys_content),
                ChatMessage(role="user", content=text),
            ]
            async for ev in self.agent.llm.stream(messages):
                if isinstance(ev, TextDelta):
                    collected.append(ev.text)
                    yield ev
            asst_text = (
                "".join(collected).strip() or "I understand your request. Let me assist you."
            )
            self.state.transcript.append(Turn(role="assistant", text=asst_text))

        async def end(self) -> CallState:
            self.state.ended = True
            return self.state

    class Agent:  # type: ignore[no-redef]
        business: BusinessConfig
        llm: LLM
        retriever: Any
        settings: Settings

        @classmethod
        async def create(
            cls,
            settings: Settings,
            business: BusinessConfig,
            llm: LLM | None = None,
            retriever: Any = None,
        ) -> Agent:
            obj = cls()
            obj.settings = settings
            obj.business = business
            obj.llm = llm  # type: ignore
            obj.retriever = retriever
            return obj

        def new_session(
            self,
            call_id: str,
            direction: str = "inbound",
            channel: str = "text",
            caller_number: str | None = None,
            outbound_goal: str | None = None,
        ) -> CallSession:
            return CallSession(self, call_id, direction, caller_number, outbound_goal)


async def run_scenario(
    scn: Scenario,
    *,
    settings: Settings,
    llm: LLM,
    sim_llm: LLM,
    judge_llm: LLM,
    retriever: Any,
    base_business_dir: Path,
) -> ScenarioResult:
    """Run one evaluation scenario in an isolated temporary business directory."""
    t0 = time.monotonic()
    try:
        async with asyncio.timeout(180):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_business = Path(tmp_dir) / "business"
                shutil.copytree(
                    base_business_dir,
                    tmp_business,
                    ignore=shutil.ignore_patterns("data", "*.pyc", "__pycache__"),
                )
                biz = load_business(tmp_business / "business.yaml")
                agent = await Agent.create(settings, biz, llm=llm, retriever=retriever)
                call_id = f"eval-{scn.id}-{uuid.uuid4().hex[:8]}"
                session = agent.new_session(
                    call_id,
                    direction=scn.direction,
                    channel="text",
                    caller_number=scn.caller_number,
                    outbound_goal=scn.outbound_goal,
                )
                sim = CustomerSimulator(scn, sim_llm)

                # Greeting turn
                async for _ in session.start():
                    pass

                # Customer/agent turns loop
                for _ in range(scn.max_turns):
                    transcript = session.state.transcript
                    customer_text = await sim.next_utterance(transcript)
                    if customer_text is None:
                        break
                    async for _ in session.respond(customer_text):
                        pass
                    if session.state.ended:
                        break

                state = await session.end()

                docs_texts: list[str] = []
                for rel_k in biz.knowledge:
                    kp = biz.resolve(rel_k)
                    if kp.exists() and kp.is_file():
                        docs_texts.append(kp.read_text(encoding="utf-8"))
                docs_text = "\n\n".join(docs_texts)

                c_checks = code_checks(scn, state, tmp_business)
                hallucinations = grounding_check(state, docs_text)
                j_checks, judge_dict = await judge(scn, state, judge_llm)

                all_checks = list(c_checks) + list(j_checks)
                if hallucinations:
                    all_checks.append(
                        Check(
                            name="hallucinations",
                            passed=False,
                            detail=f"Hallucinated facts: {', '.join(hallucinations)}",
                            kind="code",
                            severity="major",
                        )
                    )

                policy_violation = any(
                    (not c.passed and c.severity == "critical") for c in c_checks
                )
                for c in c_checks:
                    if not c.passed and c.name.startswith("must_not_call_"):
                        policy_violation = True

                escalated = (
                    any(
                        e.name == "transfer_to_human" and e.status == "ok"
                        for e in state.tool_events
                    )
                    or any(
                        e.name == "create_ticket"
                        and e.status == "ok"
                        and e.args.get("priority") in ("high", "urgent")
                        for e in state.tool_events
                    )
                    or state.escalation_reason is not None
                    or state.disposition == "escalated"
                )

                passed = not any(
                    (not c.passed and c.severity in ("critical", "major")) for c in all_checks
                )

                turns_count = len([t for t in state.transcript if t.role == "user"])
                transcript_dicts = [
                    {"role": t.role, "text": t.text, "interrupted": t.interrupted}
                    for t in state.transcript
                ]
                tool_event_dicts = [
                    {
                        "name": e.name,
                        "args": e.args,
                        "status": e.status,
                        "turn": e.turn,
                        "reason": e.reason,
                    }
                    for e in state.tool_events
                ]

                return ScenarioResult(
                    scenario_id=scn.id,
                    category=scn.category,
                    passed=passed,
                    checks=all_checks,
                    turns=turns_count,
                    transcript=transcript_dicts,
                    tool_events=tool_event_dicts,
                    policy_violation=policy_violation,
                    hallucinations=hallucinations,
                    escalated=escalated,
                    resolution=judge_dict.get("outcome", "unknown"),
                    judge=judge_dict,
                    duration_s=round(time.monotonic() - t0, 2),
                    error=None,
                )
    except TimeoutError:
        return ScenarioResult(
            scenario_id=scn.id,
            category=scn.category,
            passed=False,
            checks=[
                Check(
                    name="timeout",
                    passed=False,
                    detail="Scenario timed out after 180s",
                    kind="code",
                    severity="critical",
                )
            ],
            turns=0,
            transcript=[],
            tool_events=[],
            policy_violation=False,
            hallucinations=[],
            escalated=False,
            resolution="declined",
            judge={},
            duration_s=round(time.monotonic() - t0, 2),
            error="Scenario timed out after 180s",
        )
    except Exception as exc:
        return ScenarioResult(
            scenario_id=scn.id,
            category=scn.category,
            passed=False,
            checks=[
                Check(
                    name="execution_error",
                    passed=False,
                    detail=str(exc),
                    kind="code",
                    severity="critical",
                )
            ],
            turns=0,
            transcript=[],
            tool_events=[],
            policy_violation=False,
            hallucinations=[],
            escalated=False,
            resolution="declined",
            judge={},
            duration_s=round(time.monotonic() - t0, 2),
            error=str(exc),
        )


async def main(
    suite: str = "smoke",
    concurrency: int = 4,
    *,
    llm: LLM | None = None,
    sim_llm: LLM | None = None,
    judge_llm: LLM | None = None,
    report_dir: Path = Path("evals/reports"),
) -> int:
    """Run evaluation suite and generate reports."""
    load_dotenv()
    settings = get_settings()

    if llm is None or sim_llm is None or judge_llm is None:
        try:
            from parley.core.providers.llm import build_llm_chain

            chain = build_llm_chain(settings)
            llm = llm or chain
            sim_llm = sim_llm or chain
            judge_llm = judge_llm or chain
        except ImportError:
            pass

    if llm is None or sim_llm is None or judge_llm is None:
        raise RuntimeError("LLMs not provided and build_llm_chain is unavailable")

    business_path = Path(settings.business_config).resolve()
    business = load_business(business_path)
    retriever = await build_retriever(business)
    scenarios = load_scenarios(suite=suite)

    sem = asyncio.Semaphore(concurrency)

    async def _run_bounded(scn: Scenario) -> ScenarioResult:
        async with sem:
            res = await run_scenario(
                scn,
                settings=settings,
                llm=llm,
                sim_llm=sim_llm,
                judge_llm=judge_llm,
                retriever=retriever,
                base_business_dir=business_path.parent,
            )
            status_tag = "PASS" if res.passed else "FAIL"
            print(f"{status_tag} {res.scenario_id} ({res.turns} turns, {res.duration_s}s)")
            return res

    results = await asyncio.gather(*[_run_bounded(s) for s in scenarios])

    model_name = getattr(llm, "model", getattr(settings, "groq_model", "unknown"))
    write_report(list(results), report_dir, suite=suite, model=model_name)

    violations = sum(1 for r in results if r.policy_violation)
    pass_rate = sum(1 for r in results if r.passed) / len(results) if results else 0.0

    if suite == "smoke":
        code = 0 if (violations == 0 and pass_rate >= 0.8) else 1
    else:
        code = 0 if violations == 0 else 1

    # parley eval calls asyncio.run(main(...)) and ignores return value.
    # We call sys.exit(code) only if report_dir is default and PARLEY_EVAL_NO_EXIT != "1".
    if report_dir == Path("evals/reports") and os.environ.get("PARLEY_EVAL_NO_EXIT") != "1":
        sys.exit(code)

    return code
