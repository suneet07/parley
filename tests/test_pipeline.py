from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.utils import run_test

try:
    from parley.core.brain.events import Control, Filler, SpeakDelta, TurnDone
except ImportError:  # pragma: no cover
    from parley.core.pipeline.brain_processor import (  # type: ignore[assignment]
        Control,
        Filler,
        SpeakDelta,
        TurnDone,
    )

from parley.core.pipeline.brain_processor import BrainProcessor
from parley.core.pipeline.latency import LatencyTracker, TurnLatency


class FakeSession:
    def __init__(self, events_for_respond: list[Any] | Exception) -> None:
        self._events = events_for_respond
        self.received_texts: list[str] = []

    async def start(self) -> AsyncIterator[Any]:
        yield SpeakDelta(text="Hello! How can I help?")
        yield TurnDone(text="Hello! How can I help?", usage=[], latency_ms=10.0)

    async def respond(self, user_text: str) -> AsyncIterator[Any]:
        self.received_texts.append(user_text)
        if isinstance(self._events, Exception):
            raise self._events
        for ev in self._events:
            yield ev

    async def reprompt(self, silence_count: int) -> AsyncIterator[Any]:
        yield SpeakDelta(text="Are you still there?")
        yield TurnDone(text="Are you still there?", usage=[], latency_ms=5.0)


async def test_brain_processor_translates_context_frame_to_speech() -> None:
    session = FakeSession(
        [
            Filler(text="One moment"),
            SpeakDelta(text="Hello there! "),
            SpeakDelta(text="How can I help you today?"),
            TurnDone(
                text="Hello there! How can I help you today?",
                usage=[],
                latency_ms=42.0,
            ),
        ]
    )
    controls: list[Control] = []

    async def on_control(ctrl: Control) -> None:
        controls.append(ctrl)

    latency = LatencyTracker()
    processor = BrainProcessor(session, on_control=on_control, latency=latency)  # type: ignore[arg-type]

    ctx = LLMContext(messages=[{"role": "user", "content": "hello"}])
    received_down, _ = await run_test(
        processor,
        frames_to_send=[LLMContextFrame(context=ctx)],
    )

    assert session.received_texts == ["hello"]
    assert not any(isinstance(f, LLMContextFrame) for f in received_down)
    assert isinstance(received_down[0], LLMFullResponseStartFrame)
    assert isinstance(received_down[-1], LLMFullResponseEndFrame)

    text_frames = [f for f in received_down if isinstance(f, LLMTextFrame)]
    assert len(text_frames) >= 1
    combined_text = "".join(f.text for f in text_frames)
    assert "One moment." in combined_text
    assert "Hello there!" in combined_text
    assert controls == []
    assert len(latency.turns) == 1
    assert latency.turns[0].transcript_at is not None
    assert latency.turns[0].first_text_at is not None


async def test_brain_processor_control_triggers_on_control() -> None:
    session = FakeSession(
        [
            SpeakDelta(text="Goodbye!"),
            TurnDone(text="Goodbye!", usage=[], latency_ms=12.0),
            Control(action="end_call", summary="Resolved caller question."),
        ]
    )
    controls: list[Control] = []

    async def on_control(ctrl: Control) -> None:
        controls.append(ctrl)

    processor = BrainProcessor(session, on_control=on_control)  # type: ignore[arg-type]
    ctx = LLMContext(messages=[{"role": "user", "content": "bye"}])

    received_down, _ = await run_test(
        processor,
        frames_to_send=[LLMContextFrame(context=ctx)],
    )

    assert isinstance(received_down[0], LLMFullResponseStartFrame)
    assert isinstance(received_down[-1], LLMFullResponseEndFrame)
    assert len(controls) == 1
    assert controls[0].action == "end_call"
    assert controls[0].summary == "Resolved caller question."


async def test_brain_processor_exception_pushes_apology_without_crash() -> None:
    session = FakeSession(RuntimeError("LLM provider timeout"))
    controls: list[Control] = []

    async def on_control(ctrl: Control) -> None:
        controls.append(ctrl)

    processor = BrainProcessor(session, on_control=on_control)  # type: ignore[arg-type]
    ctx = LLMContext(messages=[{"role": "user", "content": "hello"}])

    received_down, _ = await run_test(
        processor,
        frames_to_send=[LLMContextFrame(context=ctx)],
    )

    assert isinstance(received_down[0], LLMFullResponseStartFrame)
    assert isinstance(received_down[-1], LLMFullResponseEndFrame)
    text_frames = [f for f in received_down if isinstance(f, LLMTextFrame)]
    assert len(text_frames) == 1
    assert "Sorry, I didn't catch that. Could you say it again?" in text_frames[0].text


def test_turn_latency_math_and_summary() -> None:
    t1 = TurnLatency(
        turn=1,
        user_stopped_at=10.000,
        transcript_at=10.120,
        first_text_at=10.370,
        first_audio_at=10.550,
        interrupted=False,
    )
    d1 = t1.as_dict()
    assert d1["turn"] == 1
    assert d1["stt_ms"] == pytest.approx(120.0)
    assert d1["llm_ttft_ms"] == pytest.approx(250.0)
    assert d1["tts_ttfb_ms"] == pytest.approx(180.0)
    assert d1["e2e_ms"] == pytest.approx(550.0)
    assert d1["interrupted"] is False

    t2 = TurnLatency(turn=2, user_stopped_at=20.0, transcript_at=20.1, interrupted=True)
    d2 = t2.as_dict()
    assert d2["stt_ms"] == pytest.approx(100.0)
    assert d2["llm_ttft_ms"] is None
    assert d2["tts_ttfb_ms"] is None
    assert d2["e2e_ms"] is None
    assert d2["interrupted"] is True

    tracker = LatencyTracker()
    tracker._turns.extend([t1, t2])
    summary = tracker.summary()
    assert summary["count"] == 2
    assert summary["p50_e2e_ms"] == pytest.approx(550.0)
    assert summary["p95_e2e_ms"] == pytest.approx(550.0)


async def test_brain_processor_aclosing_runs_generator_finally_on_cancel() -> None:
    import asyncio

    entered = asyncio.Event()
    finally_ran = False

    class _CancelableSession:
        async def start(self) -> AsyncIterator[Any]:
            yield SpeakDelta(text="hi")

        async def reprompt(self, count: int) -> AsyncIterator[Any]:
            yield SpeakDelta(text="still there?")

        async def respond(self, user_text: str) -> AsyncIterator[Any]:
            nonlocal finally_ran
            try:
                entered.set()
                await asyncio.Event().wait()
                yield SpeakDelta(text="never reached")
            finally:
                finally_ran = True

    async def on_control(ctrl: Control) -> None:
        return None

    processor = BrainProcessor(_CancelableSession(), on_control=on_control)  # type: ignore[arg-type]
    task = asyncio.create_task(
        processor._consume_brain_stream(processor._session.respond("hello"), track_latency=True)
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finally_ran is True


async def test_latency_tracker_ignores_greeting_and_reprompt_audio() -> None:
    from types import SimpleNamespace

    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        UserStoppedSpeakingFrame,
    )

    tracker = LatencyTracker()

    def _push(frame: Any) -> Any:
        return SimpleNamespace(frame=frame)

    # 1. Greeting audio before any user turn -> ignored
    await tracker.on_push_frame(_push(BotStartedSpeakingFrame()))  # type: ignore[arg-type]
    assert tracker.turns == []

    # 2. User stops speaking (creates turn with transcript_at=None); audio before transcript -> ignored
    await tracker.on_push_frame(_push(UserStoppedSpeakingFrame()))  # type: ignore[arg-type]
    assert len(tracker.turns) == 1
    assert tracker.turns[0].transcript_at is None
    await tracker.on_push_frame(_push(BotStartedSpeakingFrame()))  # type: ignore[arg-type]
    assert tracker.turns[0].first_audio_at is None

    # 3. Respond turn starts and marks transcript -> BotStartedSpeakingFrame is recorded
    tracker.mark_turn_start()
    tracker.mark_transcript()
    tracker.mark_first_text()
    assert not tracker.bot_stopped_speaking.is_set()
    await tracker.on_push_frame(_push(BotStartedSpeakingFrame()))  # type: ignore[arg-type]
    assert tracker.turns[0].first_audio_at is not None
    await tracker.on_push_frame(_push(BotStoppedSpeakingFrame()))  # type: ignore[arg-type]
    assert tracker.bot_stopped_speaking.is_set()
    tracker.mark_turn_end()

    # 4. Subsequent reprompt audio on a turn after mark_turn_end -> ignored
    await tracker.on_push_frame(_push(UserStoppedSpeakingFrame()))  # type: ignore[arg-type]
    tracker.mark_transcript()
    tracker.mark_turn_end()
    await tracker.on_push_frame(_push(BotStartedSpeakingFrame()))  # type: ignore[arg-type]
    assert tracker.turns[1].first_audio_at is None
