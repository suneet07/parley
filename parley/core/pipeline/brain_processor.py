from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    from parley.core.brain.events import (
        Control,
        Filler,
        SpeakDelta,
        ToolFinished,
        ToolStarted,
        TurnDone,
    )
except ImportError:  # pragma: no cover

    @dataclass
    class SpeakDelta:  # type: ignore[no-redef]
        text: str

    @dataclass
    class Filler:  # type: ignore[no-redef]
        text: str

    @dataclass
    class ToolStarted:  # type: ignore[no-redef]
        name: str

    @dataclass
    class ToolFinished:  # type: ignore[no-redef]
        event: Any

    @dataclass
    class Control:  # type: ignore[no-redef]
        action: str
        summary: str | None = None

    @dataclass
    class TurnDone:  # type: ignore[no-redef]
        text: str = ""
        usage: list[Any] | None = None
        latency_ms: float = 0.0


if TYPE_CHECKING:
    from parley.core.brain import CallSession
    from parley.core.pipeline.latency import LatencyTracker

logger = structlog.get_logger(__name__)

_APOLOGY_TEXT = "Sorry, I didn't catch that. Could you say it again?"


def _extract_last_user_text(context: Any) -> str:
    """Extract text from the last message with role == 'user' in an LLMContext."""
    messages = (
        context.get_messages()
        if hasattr(context, "get_messages")
        else getattr(context, "messages", [])
    )
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return " ".join(p for p in parts if p)
    return ""


class BrainProcessor(FrameProcessor):
    """Bridges Pipecat's frame pipeline with the transport-agnostic Parley CallSession."""

    def __init__(
        self,
        session: CallSession,
        *,
        on_control: Callable[[Control], Awaitable[None]],
        latency: LatencyTracker | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._session = session
        self._on_control = on_control
        self._latency = latency

    async def _consume_brain_stream(
        self,
        stream: AsyncIterator[Any],
        *,
        track_latency: bool = False,
    ) -> None:
        started_response = False
        ended_response = False
        pushed_first_text = False
        pending_controls: list[Control] = []

        try:
            async for ev in stream:
                if isinstance(ev, SpeakDelta):
                    if not ev.text:
                        continue
                    if not started_response:
                        started_response = True
                        await self.push_frame(LLMFullResponseStartFrame())
                    if track_latency and not pushed_first_text:
                        pushed_first_text = True
                        if self._latency is not None:
                            self._latency.mark_first_text()
                    await self.push_frame(LLMTextFrame(ev.text))
                elif isinstance(ev, Filler):
                    if not ev.text:
                        continue
                    if not started_response:
                        started_response = True
                        await self.push_frame(LLMFullResponseStartFrame())
                    if track_latency and not pushed_first_text:
                        pushed_first_text = True
                        if self._latency is not None:
                            self._latency.mark_first_text()
                    # In Pipecat 1.13, TTSService aggregates LLMTextFrame by sentence;
                    # sentence-final punctuation + trailing space flushes the filler
                    # immediately while keeping subsequent SpeakDelta tokens in the turn.
                    filler_text = ev.text.strip()
                    if filler_text and filler_text[-1] not in ".!?":
                        filler_text += "."
                    filler_text += " "
                    await self.push_frame(LLMTextFrame(filler_text))
                elif isinstance(ev, ToolStarted):
                    logger.info("brain_tool_started", tool=ev.name)
                elif isinstance(ev, ToolFinished):
                    logger.info(
                        "brain_tool_finished",
                        tool=getattr(ev.event, "name", None),
                        status=getattr(ev.event, "status", None),
                    )
                elif isinstance(ev, TurnDone):
                    if started_response and not ended_response:
                        ended_response = True
                        await self.push_frame(LLMFullResponseEndFrame())
                elif isinstance(ev, Control):
                    pending_controls.append(ev)

            if started_response and not ended_response:
                ended_response = True
                await self.push_frame(LLMFullResponseEndFrame())

        except Exception:
            # Note: asyncio.CancelledError inherits from BaseException (not Exception),
            # so Pipecat barge-in task cancellation propagates cleanly without triggering apology.
            logger.exception("brain_turn_exception")
            if not started_response:
                await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(_APOLOGY_TEXT))
            await self.push_frame(LLMFullResponseEndFrame())
            return

        for ctrl in pending_controls:
            await self._on_control(ctrl)

    async def greet(self) -> None:
        """Run session.start() and push greeting speech frames."""
        await self._consume_brain_stream(self._session.start(), track_latency=False)

    async def handle_idle(self, count: int) -> None:
        """Run session.reprompt(count) and push reprompt speech frames."""
        await self._consume_brain_stream(self._session.reprompt(count), track_latency=False)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Pipecat 1.13 requires calling super().process_frame first for lifecycle handling.
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            if self._latency is not None:
                self._latency.mark_transcript()
            user_text = _extract_last_user_text(frame.context)
            # Do not push LLMContextFrame downstream; BrainProcessor owns the conversation turn.
            await self._consume_brain_stream(
                self._session.respond(user_text),
                track_latency=True,
            )
            return

        await self.push_frame(frame, direction)
