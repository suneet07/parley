"""Events yielded by the agent brain during a call session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from parley.core.brain.state import ToolEvent
from parley.core.providers.llm_base import Usage


@dataclass
class SpeakDelta:
    """Chunk of assistant speech to stream to TTS as it arrives."""

    text: str


@dataclass
class Filler:
    """Short phrase spoken while a slow read/verify tool executes."""

    text: str


@dataclass
class ToolStarted:
    """Notification that a tool invocation has started."""

    name: str


@dataclass
class ToolFinished:
    """Notification that a tool invocation has completed (or was blocked)."""

    event: ToolEvent


@dataclass
class Control:
    """Call-control signal (warm transfer or hangup)."""

    action: Literal["transfer", "end_call"]
    summary: str | None = None


@dataclass
class TurnDone:
    """Final event for an assistant turn with full text, token usage, and latency."""

    text: str
    usage: list[Usage] = field(default_factory=list)
    latency_ms: float = 0.0


AgentEvent = SpeakDelta | Filler | ToolStarted | ToolFinished | Control | TurnDone
