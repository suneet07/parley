"""LangGraph per-turn execution graph (LLM streaming node + tool gateway node)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from parley.core.brain.events import (
    AgentEvent,
    Control,
    Filler,
    SpeakDelta,
    ToolFinished,
    ToolStarted,
)
from parley.core.brain.guard import GuardResult
from parley.core.brain.prompts import build_system_prompt
from parley.core.providers.llm_base import (
    ChatMessage,
    Done,
    LLMToolCall,
    TextDelta,
    ToolCallsReady,
)
from parley.core.tools.base import ToolKind

if TYPE_CHECKING:
    from parley.core.brain.session import CallSession

MAX_TOOL_ROUNDS = 4
FALLBACK_EXHAUSTED_TEXT = (
    "Sorry, I'm having trouble with that. Let me create a ticket so someone follows up."
)
FILLER_PHRASES: tuple[str, ...] = (
    "One sec, let me check that.",
    "Just a moment while I look that up.",
    "Let me pull that up for you.",
)


class TurnGraphState(TypedDict):
    """State passed between nodes in a single caller turn."""

    session: Any  # CallSession; LangGraph resolves these hints at runtime, so no forward ref
    round: int
    pending_tool_calls: list[LLMToolCall]
    turn_text: str
    control: Control | None
    guard: GuardResult
    turn_context: str
    filler_emitted: bool


def _build_turn_messages(session: CallSession, turn_context: str) -> list[ChatMessage]:
    sys_msg = ChatMessage(
        role="system",
        content=build_system_prompt(session.agent.business, session.state, datetime.now(UTC)),
    )
    ctx_msg = ChatMessage(role="system", content=turn_context)

    last_user_idx: int | None = None
    for idx in range(len(session.history) - 1, -1, -1):
        if session.history[idx].role == "user":
            last_user_idx = idx
            break

    if last_user_idx is None:
        return [sys_msg, *session.history, ctx_msg]

    return [
        sys_msg,
        *session.history[:last_user_idx],
        ctx_msg,
        *session.history[last_user_idx:],
    ]


async def _llm_node(state: TurnGraphState) -> dict[str, Any]:
    session = state["session"]
    current_round = state["round"]
    guard = state["guard"]
    turn_context = state["turn_context"]
    writer = get_stream_writer()

    messages = _build_turn_messages(session, turn_context)
    tools = session.agent.gateway.schemas()

    if (
        guard.human_requested
        and "human_requested" in session.agent.business.escalation.on
        and current_round == 0
    ):
        tool_choice: str | dict[str, Any] = {
            "type": "function",
            "function": {"name": "transfer_to_human"},
        }
    else:
        tool_choice = "auto"

    node_text_parts: list[str] = []
    node_tool_calls: list[LLMToolCall] = []
    appended = False

    agen = session.agent.llm.stream(
        messages,
        tools=tools,
        tool_choice=tool_choice,
    )
    try:
        async for ev in agen:
            if isinstance(ev, TextDelta):
                if ev.text:
                    node_text_parts.append(ev.text)
                    session._turn_text_parts.append(ev.text)
                    writer(SpeakDelta(text=ev.text))
            elif isinstance(ev, ToolCallsReady):
                node_tool_calls = list(ev.calls)
            elif isinstance(ev, Done):
                session._turn_usages.append(ev.usage)

        node_text = "".join(node_text_parts)
        if node_text or node_tool_calls:
            session.history.append(
                ChatMessage(
                    role="assistant",
                    content=node_text if node_text else None,
                    tool_calls=node_tool_calls if node_tool_calls else None,
                )
            )
        appended = True
    finally:
        await agen.aclose()
        if not appended and (node_text_parts or node_tool_calls):
            partial_text = "".join(node_text_parts)
            session.history.append(
                ChatMessage(
                    role="assistant",
                    content=partial_text if partial_text else None,
                    tool_calls=node_tool_calls if node_tool_calls else None,
                )
            )

    if node_tool_calls and current_round >= MAX_TOOL_ROUNDS:
        for tc in node_tool_calls:
            session.history.append(
                ChatMessage(
                    role="tool",
                    content=json.dumps({"error": "too_many_tool_calls"}),
                    tool_call_id=tc.id,
                )
            )
        session._turn_text_parts.append(FALLBACK_EXHAUSTED_TEXT)
        session.history.append(
            ChatMessage(
                role="assistant",
                content=FALLBACK_EXHAUSTED_TEXT,
            )
        )
        writer(SpeakDelta(text=FALLBACK_EXHAUSTED_TEXT))
        return {
            "pending_tool_calls": [],
            "turn_text": "".join(session._turn_text_parts),
        }

    return {
        "pending_tool_calls": node_tool_calls,
        "turn_text": "".join(session._turn_text_parts),
    }


async def _tools_node(state: TurnGraphState) -> dict[str, Any]:
    session = state["session"]
    pending_calls = state["pending_tool_calls"]
    filler_emitted = state["filler_emitted"]
    control = state["control"]
    writer = get_stream_writer()

    ctx = session.tool_context()

    for tc in pending_calls:
        spec = session.agent.registry.get(tc.name)
        if (
            not filler_emitted
            and not session._turn_text_parts
            and spec is not None
            and spec.kind in (ToolKind.READ, ToolKind.VERIFY)
        ):
            phrase = FILLER_PHRASES[session._filler_index % len(FILLER_PHRASES)]
            session._filler_index += 1
            filler_emitted = True
            session._turn_filler = phrase
            writer(Filler(text=phrase))

        writer(ToolStarted(name=tc.name))
        result = await session.agent.gateway.call(tc.name, tc.arguments, ctx)
        writer(ToolFinished(event=result.event))

        session._recorded_tool_event_ids.add(id(result.event))
        session.history.append(
            ChatMessage(
                role="tool",
                content=json.dumps(result.payload),
                tool_call_id=tc.id,
            )
        )

        if result.control in ("transfer", "end_call"):
            control = Control(
                action=cast(Literal["transfer", "end_call"], result.control),
                summary=session.state.summary,
            )
            session._turn_control = control

    return {
        "round": state["round"] + 1,
        "pending_tool_calls": [],
        "filler_emitted": filler_emitted,
        "control": control,
    }


def _route_after_llm(state: TurnGraphState) -> str:
    if state["pending_tool_calls"] and state["round"] < MAX_TOOL_ROUNDS:
        return "tools"
    return END


def build_turn_graph() -> Any:
    """Compile the per-turn LangGraph StateGraph."""
    builder = StateGraph(TurnGraphState)
    builder.add_node("llm", _llm_node)
    builder.add_node("tools", _tools_node)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", _route_after_llm, {"tools": "tools", END: END})
    builder.add_edge("tools", "llm")
    return builder.compile()


async def run_turn_graph(
    session: CallSession,
    guard: GuardResult,
    turn_context: str,
) -> AsyncIterator[AgentEvent]:
    """Execute the compiled turn graph and stream AgentEvents."""
    initial_state: TurnGraphState = {
        "session": session,
        "round": 0,
        "pending_tool_calls": [],
        "turn_text": "",
        "control": None,
        "guard": guard,
        "turn_context": turn_context,
        "filler_emitted": False,
    }
    agen = session.agent.graph.astream(initial_state, stream_mode="custom")
    try:
        async for event in agen:
            yield cast(AgentEvent, event)
    finally:
        await agen.aclose()
