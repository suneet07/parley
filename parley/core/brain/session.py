"""Agent and CallSession: transport-agnostic phone support brain."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal, cast

import structlog

from parley.core.brain.events import AgentEvent, Control, SpeakDelta, TurnDone
from parley.core.brain.graph import build_turn_graph, run_turn_graph
from parley.core.brain.guard import check_turn
from parley.core.brain.prompts import build_turn_context
from parley.core.brain.state import CallState, Disposition, Turn, utcnow
from parley.core.knowledge.ingest import build_retriever
from parley.core.providers.llm import build_llm_chain
from parley.core.providers.llm_base import LLM, ChatMessage, TextDelta, Usage
from parley.core.tools.base import ToolContext
from parley.core.tools.gateway import ToolGateway
from parley.core.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from parley.config import BusinessConfig, Settings
    from parley.core.knowledge.base import Passage, Retriever

logger = structlog.get_logger(__name__)

MAX_HISTORY_MESSAGES = 30
VALID_DISPOSITIONS: frozenset[str] = frozenset(
    {
        "resolved",
        "escalated",
        "callback_booked",
        "ticket_created",
        "abandoned",
        "voicemail",
        "no_answer",
        "not_interested",
        "do_not_call",
        "unknown",
    }
)
LOW_CONFIDENCE_INTENTS: frozenset[str] = frozenset({"faq", "product", "shipping", "other"})


def _extract_json_dict(raw: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class Agent:
    """Holds business configuration, LLM chain, tool gateway, retriever, and compiled turn graph."""

    def __init__(
        self,
        business: BusinessConfig,
        llm: LLM,
        registry: ToolRegistry,
        retriever: Retriever | None = None,
        resources: dict[str, Any] | None = None,
        tenant_id: str = "default",
    ) -> None:
        self.business = business
        self.llm = llm
        self.registry = registry
        self.retriever = retriever
        if resources is not None:
            self.resources = resources
        elif registry.resources_factory is not None:
            self.resources = registry.resources_factory(business)
        else:
            self.resources = {}
        self.tenant_id = tenant_id
        self.gateway = ToolGateway(registry=registry)
        self.graph = build_turn_graph()

    @classmethod
    async def create(
        cls,
        settings: Settings,
        business: BusinessConfig,
        *,
        llm: LLM | None = None,
        retriever: Retriever | None = None,
    ) -> Agent:
        """Construct an Agent from Settings and BusinessConfig."""
        active_llm = llm if llm is not None else build_llm_chain(settings)
        active_retriever = retriever
        if active_retriever is None:
            try:
                active_retriever = await build_retriever(
                    business,
                    settings.database_url,
                    settings.tenant_id,
                )
            except Exception as exc:
                logger.warning("retriever_build_failed", error=str(exc))
                active_retriever = None

        registry = ToolRegistry.for_business(business)
        resources = (
            registry.resources_factory(business) if registry.resources_factory is not None else {}
        )
        return cls(
            business=business,
            llm=active_llm,
            registry=registry,
            retriever=active_retriever,
            resources=resources,
            tenant_id=settings.tenant_id,
        )

    def new_session(
        self,
        call_id: str | None = None,
        *,
        direction: Literal["inbound", "outbound"] = "inbound",
        channel: Literal["phone", "webrtc", "text"] = "text",
        caller_number: str | None = None,
        outbound_goal: str | None = None,
    ) -> CallSession:
        """Create a new CallSession for an incoming or outgoing call."""
        state = CallState(
            call_id=call_id or f"call_{uuid.uuid4().hex[:12]}",
            tenant_id=self.tenant_id,
            direction=direction,
            channel=channel,
            caller_number=caller_number,
        )
        return CallSession(
            agent=self,
            state=state,
            outbound_goal=outbound_goal,
        )


class CallSession:
    """Stateful per-call session driven by voice pipeline, eval harness, or REPL."""

    def __init__(
        self,
        agent: Agent,
        state: CallState,
        outbound_goal: str | None = None,
    ) -> None:
        self.agent = agent
        self.state = state
        self.history: list[ChatMessage] = []
        self.outbound_goal = outbound_goal
        self._filler_index = 0
        self._turn_text_parts: list[str] = []
        self._turn_usages: list[Usage] = []
        self._turn_control: Control | None = None
        self._knowledge_gaps: list[str] = []
        self._wrapup_done = False

    def tool_context(self) -> ToolContext:
        """Build a ToolContext bound to this session."""
        return ToolContext(
            state=self.state,
            business=self.agent.business,
            retriever=self.agent.retriever,
            resources=self.agent.resources,
        )

    async def start(self) -> AsyncIterator[AgentEvent]:
        """Emit the opening greeting with mandatory AI and recording disclosure."""
        start_t = time.perf_counter()

        if self.state.caller_number and self.agent.registry.caller_lookup is not None:
            try:
                profile = await self.agent.registry.caller_lookup(
                    self.tool_context(),
                    self.state.caller_number,
                )
                if profile is not None:
                    self.state.caller = profile
            except Exception as exc:
                logger.warning("caller_lookup_error", error=str(exc))

        biz_name = self.agent.business.business
        if self.state.direction == "inbound":
            greeting = (
                f"Hi, you've reached {biz_name}. I'm an AI assistant and this call may be recorded."
            )
            if self.state.caller and self.state.caller.name:
                first_name = self.state.caller.name.split()[0]
                greeting += f" Hi {first_name}, good to hear from you."
            tail = self.agent.business.greeting or "How can I help?"
            greeting = f"{greeting} {tail}"
        else:
            opener = self.outbound_goal or "I'm calling about your recent order."
            greeting = (
                f"Hi, this is the AI assistant from {biz_name}. This call may be recorded. {opener}"
            )

        self.state.transcript.append(Turn(role="assistant", text=greeting))
        self.history.append(ChatMessage(role="assistant", content=greeting))

        yield SpeakDelta(text=greeting)
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        yield TurnDone(text=greeting, usage=[], latency_ms=latency_ms)

    def _repair_history(self) -> None:
        """Ensure every assistant tool_calls entry has a matching tool reply message."""
        answered_ids = {
            msg.tool_call_id
            for msg in self.history
            if msg.role == "tool" and msg.tool_call_id is not None
        }
        repaired: list[ChatMessage] = []
        for idx, msg in enumerate(self.history):
            repaired.append(msg)
            if msg.role == "assistant" and msg.tool_calls:
                # Check which tool calls are answered immediately or later in history
                next_idx = idx + 1
                while next_idx < len(self.history) and self.history[next_idx].role == "tool":
                    next_idx += 1
                for tc in msg.tool_calls:
                    if tc.id not in answered_ids:
                        repaired.append(
                            ChatMessage(
                                role="tool",
                                content=json.dumps({"error": "cancelled"}),
                                tool_call_id=tc.id,
                            )
                        )
                        answered_ids.add(tc.id)
        self.history = repaired

    def _trim_history(self) -> None:
        """Keep at most MAX_HISTORY_MESSAGES without splitting assistant tool_calls from tool replies."""
        if len(self.history) <= MAX_HISTORY_MESSAGES:
            return
        cut = len(self.history) - MAX_HISTORY_MESSAGES
        while cut < len(self.history) and self.history[cut].role == "tool":
            cut += 1
        self.history = self.history[cut:]

    def _infer_disposition(self) -> Disposition:
        for ev in reversed(self.state.tool_events):
            if ev.status != "ok":
                continue
            if ev.name == "transfer_to_human" and ev.result.get("transferring") is True:
                return "escalated"
            if ev.name == "book_callback":
                return "callback_booked"
            if ev.name in ("create_ticket", "transfer_to_human"):
                return "ticket_created"
        if any(ev.status == "ok" for ev in self.state.tool_events):
            return "resolved"
        if not any(t.role == "user" for t in self.state.transcript):
            return "abandoned"
        return "unknown"

    async def respond(self, user_text: str) -> AsyncIterator[AgentEvent]:
        """Process one caller turn and stream AgentEvents."""
        text = user_text.strip()
        if not text:
            return

        start_t = time.perf_counter()
        self.state.turn_index += 1
        self.state.transcript.append(Turn(role="user", text=text))
        self.history.append(ChatMessage(role="user", content=text))

        guard = check_turn(text, self.state, self.agent.business)

        passages: list[Passage] = []
        if self.agent.retriever is not None:
            try:
                passages = await self.agent.retriever.search(text, k=3)
            except Exception as exc:
                logger.warning("retriever_search_failed", error=str(exc))
                passages = []

        max_score = max((p.score for p in passages), default=0.0)
        if max_score < 0.35 and guard.intent in LOW_CONFIDENCE_INTENTS:
            if text not in self._knowledge_gaps:
                self._knowledge_gaps.append(text)
            if text not in self.state.knowledge_gaps:
                self.state.knowledge_gaps.append(text)

        turn_context = build_turn_context(passages, guard, self.state)

        self._turn_text_parts = []
        self._turn_usages = []
        self._turn_control = None
        turn_events_start = len(self.state.tool_events)

        try:
            async for event in run_turn_graph(self, guard, turn_context):
                yield event
        finally:
            self._repair_history()
            full_text = "".join(self._turn_text_parts)
            turn_tool_events = list(self.state.tool_events[turn_events_start:])
            self.state.transcript.append(
                Turn(
                    role="assistant",
                    text=full_text,
                    tool_events=turn_tool_events,
                )
            )
            self._trim_history()

        if self._turn_control is not None:
            if self._turn_control.action == "transfer":
                self.state.ended = True
                self.state.disposition = "escalated"
            elif self._turn_control.action == "end_call":
                self.state.ended = True
                if self.state.disposition == "unknown":
                    self.state.disposition = self._infer_disposition()
            yield self._turn_control

        latency_ms = (time.perf_counter() - start_t) * 1000.0
        yield TurnDone(
            text="".join(self._turn_text_parts),
            usage=list(self._turn_usages),
            latency_ms=latency_ms,
        )

    def on_interrupted(self, heard_text: str) -> None:
        """Handle barge-in by truncating the last assistant message and turn to what was heard."""
        for idx in range(len(self.history) - 1, -1, -1):
            msg = self.history[idx]
            if msg.role == "assistant" and (msg.content is not None or msg.tool_calls):
                if msg.content is not None or not any(
                    m.role == "assistant" and m.content is not None for m in self.history
                ):
                    if heard_text:
                        msg.content = f"{heard_text} —"
                    else:
                        msg.content = None
                        if not msg.tool_calls:
                            self.history.pop(idx)
                    break

        for turn in reversed(self.state.transcript):
            if turn.role == "assistant":
                turn.text = heard_text
                turn.interrupted = True
                break

    async def reprompt(self, silence_count: int) -> AsyncIterator[AgentEvent]:
        """Handle caller silence without calling the LLM."""
        if silence_count == 1:
            text = "Are you still there?"
        elif silence_count == 2:
            text = "I'm still here if you need anything."
        else:
            text = "I'll let you go for now. Feel free to call back anytime. Goodbye!"

        self.state.transcript.append(Turn(role="assistant", text=text))
        self.history.append(ChatMessage(role="assistant", content=text))

        yield SpeakDelta(text=text)
        if silence_count >= 3:
            self.state.ended = True
            if self.state.disposition == "unknown":
                self.state.disposition = "abandoned"
            yield Control(action="end_call")
        yield TurnDone(text=text, usage=[], latency_ms=0.0)

    def _heuristic_summary(self, reason: str) -> str:
        user_lines = [t.text for t in self.state.transcript if t.role == "user" and t.text]
        asst_lines = [t.text for t in self.state.transcript if t.role == "assistant" and t.text]
        parts: list[str] = []
        if user_lines:
            parts.append(user_lines[0])
        if asst_lines:
            parts.append(asst_lines[-1])
        return "\n".join(parts) if parts else f"Call ended ({reason})."

    async def end(self, reason: str = "hangup") -> CallState:
        """Idempotent end-of-call wrap-up: summarize transcript and finalize CallState."""
        if self._wrapup_done:
            return self.state
        self._wrapup_done = True

        transcript_lines = [f"{t.role}: {t.text}" for t in self.state.transcript if t.text]
        transcript_str = "\n".join(transcript_lines) or "(empty call)"

        prompt_messages = [
            ChatMessage(
                role="system",
                content=(
                    "Summarize the phone support call transcript as STRICT JSON with keys: "
                    '"summary" (3 short lines joined by \\n), '
                    '"disposition" (one of resolved, escalated, callback_booked, ticket_created, '
                    "abandoned, voicemail, no_answer, not_interested, do_not_call, unknown), "
                    '"intents" (list of strings), '
                    '"follow_ups" (list of strings), '
                    '"knowledge_gaps" (list of strings). '
                    "Output JSON only."
                ),
            ),
            ChatMessage(role="user", content=transcript_str),
        ]

        raw_chunks: list[str] = []
        try:
            agen = self.agent.llm.stream(
                prompt_messages,
                tools=None,
                temperature=0.0,
                max_tokens=300,
            )
            try:
                async for ev in agen:
                    if isinstance(ev, TextDelta):
                        raw_chunks.append(ev.text)
            finally:
                await agen.aclose()

            parsed = _extract_json_dict("".join(raw_chunks))
            if parsed is not None:
                raw_summary = parsed.get("summary")
                if isinstance(raw_summary, list):
                    raw_summary = "\n".join(str(x) for x in raw_summary if x)
                if isinstance(raw_summary, str) and raw_summary.strip():
                    self.state.summary = raw_summary.strip()

                raw_disp = parsed.get("disposition")
                if (
                    isinstance(raw_disp, str)
                    and raw_disp in VALID_DISPOSITIONS
                    and self.state.disposition != "escalated"
                ):
                    self.state.disposition = cast(Disposition, raw_disp)

                raw_intents = parsed.get("intents")
                if isinstance(raw_intents, list):
                    for item in raw_intents:
                        if isinstance(item, str) and item and item not in self.state.intents:
                            self.state.intents.append(item)

                raw_follow_ups = parsed.get("follow_ups")
                if isinstance(raw_follow_ups, list):
                    for item in raw_follow_ups:
                        if isinstance(item, str) and item and item not in self.state.follow_ups:
                            self.state.follow_ups.append(item)

                raw_gaps = parsed.get("knowledge_gaps")
                if isinstance(raw_gaps, list):
                    for item in raw_gaps:
                        if isinstance(item, str) and item and item not in self.state.knowledge_gaps:
                            self.state.knowledge_gaps.append(item)
        except Exception as exc:
            logger.warning("call_end_summary_failed", error=str(exc))

        if not self.state.summary:
            self.state.summary = self._heuristic_summary(reason)

        inferred = self._infer_disposition()
        if self.state.disposition == "unknown" and inferred != "unknown":
            self.state.disposition = inferred

        for gap in self._knowledge_gaps:
            if gap not in self.state.knowledge_gaps:
                self.state.knowledge_gaps.append(gap)

        self.state.ended = True
        self.state.ended_at = utcnow()
        return self.state
