from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

import structlog
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

logger = structlog.get_logger(__name__)


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return round(sorted_values[low], 3)
    weight = rank - low
    return round(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight, 3)


@dataclass
class TurnLatency:
    turn: int
    user_stopped_at: float | None = None
    transcript_at: float | None = None
    first_text_at: float | None = None
    first_audio_at: float | None = None
    interrupted: bool = False

    def as_dict(self) -> dict[str, Any]:
        stt_ms = (
            round((self.transcript_at - self.user_stopped_at) * 1000.0, 3)
            if self.transcript_at is not None and self.user_stopped_at is not None
            else None
        )
        llm_ttft_ms = (
            round((self.first_text_at - self.transcript_at) * 1000.0, 3)
            if self.first_text_at is not None and self.transcript_at is not None
            else None
        )
        tts_ttfb_ms = (
            round((self.first_audio_at - self.first_text_at) * 1000.0, 3)
            if self.first_audio_at is not None and self.first_text_at is not None
            else None
        )
        e2e_ms = (
            round((self.first_audio_at - self.user_stopped_at) * 1000.0, 3)
            if self.first_audio_at is not None and self.user_stopped_at is not None
            else None
        )
        return {
            "turn": self.turn,
            "stt_ms": stt_ms,
            "llm_ttft_ms": llm_ttft_ms,
            "tts_ttfb_ms": tts_ttfb_ms,
            "e2e_ms": e2e_ms,
            "interrupted": self.interrupted,
        }


class LatencyTracker(BaseObserver):
    """Pipecat observer tracking per-turn STT, LLM TTFT, TTS TTFB, and E2E latency."""

    def __init__(self, **kwargs: Any) -> None:
        # In Pipecat 1.13 BaseObserver, observe_every_push=False deduplicates across pipeline hops,
        # and we also track frame IDs + broadcast sibling IDs for broadcast_frame pairs.
        super().__init__(observe_every_push=False, **kwargs)
        self._turns: list[TurnLatency] = []
        self._seen_frame_ids: set[int] = set()
        self._in_respond_turn: bool = False
        self.bot_stopped_speaking: asyncio.Event = asyncio.Event()

    @property
    def turns(self) -> list[TurnLatency]:
        return list(self._turns)

    def _current_or_new_turn_for_user_stop(self, now: float) -> TurnLatency:
        if (
            self._turns
            and self._turns[-1].transcript_at is None
            and self._turns[-1].first_text_at is None
            and self._turns[-1].first_audio_at is None
        ):
            self._turns[-1].user_stopped_at = now
            return self._turns[-1]
        turn = TurnLatency(turn=len(self._turns) + 1, user_stopped_at=now)
        self._turns.append(turn)
        return turn

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if frame.id in self._seen_frame_ids:
            return
        self._seen_frame_ids.add(frame.id)
        sibling_id = getattr(frame, "broadcast_sibling_id", None)
        if sibling_id is not None:
            self._seen_frame_ids.add(sibling_id)

        now = time.perf_counter()
        # In 1.13, LLMUserAggregator._on_user_turn_stopped broadcasts UserStoppedSpeakingFrame
        # after Smart Turn v3 confirms turn end (and VADUserStoppedSpeakingFrame on raw VAD).
        if isinstance(frame, UserStoppedSpeakingFrame) or isinstance(
            frame, VADUserStoppedSpeakingFrame
        ):
            self._current_or_new_turn_for_user_stop(now)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self.bot_stopped_speaking.clear()
            if (
                self._in_respond_turn
                and self._turns
                and self._turns[-1].transcript_at is not None
                and self._turns[-1].first_audio_at is None
            ):
                self._turns[-1].first_audio_at = now
                logger.info("turn_latency", **self._turns[-1].as_dict())
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self.bot_stopped_speaking.set()

    def mark_turn_start(self) -> None:
        self._in_respond_turn = True
        self.bot_stopped_speaking.clear()

    def mark_turn_end(self) -> None:
        self._in_respond_turn = False

    start_turn = mark_turn_start
    end_turn = mark_turn_end

    def mark_transcript(self) -> None:
        now = time.perf_counter()
        self._in_respond_turn = True
        if (
            self._turns
            and self._turns[-1].transcript_at is None
            and self._turns[-1].first_text_at is None
        ):
            self._turns[-1].transcript_at = now
        else:
            self._turns.append(TurnLatency(turn=len(self._turns) + 1, transcript_at=now))

    def mark_first_text(self) -> None:
        now = time.perf_counter()
        if not self._turns:
            self._turns.append(TurnLatency(turn=1, first_text_at=now))
            return
        last = self._turns[-1]
        if last.first_text_at is None:
            last.first_text_at = now

    def mark_interrupted(self) -> None:
        self._in_respond_turn = False
        if self._turns:
            self._turns[-1].interrupted = True

    def summary(self) -> dict[str, Any]:
        turns_dicts = [t.as_dict() for t in self._turns]
        e2e_values = sorted(float(d["e2e_ms"]) for d in turns_dicts if d["e2e_ms"] is not None)
        return {
            "turns": turns_dicts,
            "p50_e2e_ms": _percentile(e2e_values, 50.0),
            "p95_e2e_ms": _percentile(e2e_values, 95.0),
            "count": len(self._turns),
        }
