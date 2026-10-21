"""LLM provider contract. Every provider speaks the OpenAI-compatible chat completions API
(Groq, Cerebras, Gemini's OpenAI endpoint, OpenAI), so one implementation covers all."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel


class LLMToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[LLMToolCall] | None = None  # assistant only
    tool_call_id: str | None = None  # tool only

    def to_openai(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.arguments},
                }
                for c in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class Usage:
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float | None = None  # time to first token (text or tool call)


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallsReady:
    calls: list[LLMToolCall]


@dataclass
class Done:
    usage: Usage
    finish_reason: str | None = None


LLMEvent = TextDelta | ToolCallsReady | Done


class LLMError(Exception):
    pass


class LLM(Protocol):
    name: str  # provider name, e.g. "groq"
    model: str

    def stream(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 400,
        tool_choice: str
        | dict[str, Any]
        | None = None,  # "auto" | "none" | {"type":"function",...}
    ) -> AsyncIterator[LLMEvent]:
        """Yield TextDelta* then optionally one ToolCallsReady, then exactly one Done.
        Raise LLMError on failure."""
        ...


@dataclass
class ProviderHealth:
    calls: int = 0
    errors: int = 0
    fallbacks: int = 0
    last_error: str | None = None
    per_provider: dict[str, dict[str, int]] = field(default_factory=dict)
