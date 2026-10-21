"""Scripted FakeLLM for unit tests and simulated-customer evaluations."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import AsyncIterator
from typing import Any, NamedTuple

from parley.core.providers.llm_base import (
    ChatMessage,
    Done,
    LLMError,
    LLMEvent,
    LLMToolCall,
    TextDelta,
    ToolCallsReady,
    Usage,
)


class RecordedCall(NamedTuple):
    """One recorded invocation of FakeLLM.stream()."""

    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None
    tool_choice: str | dict[str, Any] | None


def _split_words(text: str) -> list[str]:
    """Split text into word-sized chunks whose concatenation equals text."""
    if not text:
        return []
    return re.findall(r"\s*\S+\s*", text)


def _coerce_tool_calls(raw_calls: list[Any], call_index: int) -> list[LLMToolCall]:
    calls: list[LLMToolCall] = []
    for idx, item in enumerate(raw_calls):
        if isinstance(item, LLMToolCall):
            calls.append(item)
        elif isinstance(item, tuple) and len(item) == 2:
            name, args = item
            args_str = args if isinstance(args, str) else json.dumps(args)
            calls.append(
                LLMToolCall(
                    id=f"call_{call_index}_{idx}",
                    name=str(name),
                    arguments=args_str,
                )
            )
        elif isinstance(item, dict):
            name = item["name"]
            args = item.get("arguments", item.get("args", {}))
            args_str = args if isinstance(args, str) else json.dumps(args)
            calls.append(
                LLMToolCall(
                    id=item.get("id", f"call_{call_index}_{idx}"),
                    name=str(name),
                    arguments=args_str,
                )
            )
        else:
            raise TypeError(f"Unsupported FakeLLM tool call entry: {item!r}")
    return calls


class FakeLLM:
    """Scripted LLM. Each call to stream() pops the next script entry.

    A script entry is either:
    - a `str` (streamed as word-sized TextDeltas),
    - a `list` of tool calls given as `(name, args_dict)` tuples (-> ToolCallsReady),
    - a `(str, list[tuple])` tuple for both text and tool calls in one response,
    - a `callable(messages, tools)` -> `str | list[tuple]` for dynamic behaviour, or
    - an `Exception` instance/subclass to raise `LLMError`.

    Records every call's `(messages, tools, tool_choice)` in `self.calls`.
    """

    def __init__(
        self,
        script: list[Any],
        name: str = "fake",
        model: str = "fake-1",
        delay_s: float = 0.0,
    ) -> None:
        self._script: list[Any] = list(script)
        self.name = name
        self.model = model
        self.delay_s = delay_s
        self.calls: list[RecordedCall] = []
        self._call_counter = 0

    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 400,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """Yield scripted TextDelta, ToolCallsReady, and Done events."""
        self.calls.append(RecordedCall(list(messages), tools, tool_choice))
        self._call_counter += 1
        call_idx = self._call_counter

        entry: Any = self._script.pop(0) if self._script else "Okay."

        if callable(entry) and not (isinstance(entry, type) and issubclass(entry, BaseException)):
            sig = inspect.signature(entry)
            if len(sig.parameters) >= 3:
                entry = entry(messages, tools, tool_choice)
            else:
                entry = entry(messages, tools)

        if isinstance(entry, BaseException):
            if isinstance(entry, LLMError):
                raise entry
            raise LLMError(f"{self.name}: {entry}") from entry
        if isinstance(entry, type) and issubclass(entry, BaseException):
            raise LLMError(f"{self.name}: {entry.__name__}")

        text = ""
        tool_calls: list[LLMToolCall] = []

        if isinstance(entry, str):
            text = entry
        elif isinstance(entry, list):
            tool_calls = _coerce_tool_calls(entry, call_idx)
        elif isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str):
            text = entry[0]
            tool_calls = _coerce_tool_calls(list(entry[1]), call_idx)
        elif isinstance(entry, dict):
            text = str(entry.get("text", ""))
            tool_calls = _coerce_tool_calls(list(entry.get("tool_calls", [])), call_idx)
        else:
            text = str(entry)

        if self.delay_s > 0:
            await asyncio.sleep(self.delay_s)

        words = _split_words(text)
        if words:
            yield TextDelta(text=words[0])
            if tool_calls:
                yield ToolCallsReady(calls=tool_calls)
                tool_calls = []
            for word in words[1:]:
                if self.delay_s > 0:
                    await asyncio.sleep(self.delay_s)
                yield TextDelta(text=word)

        if tool_calls:
            yield ToolCallsReady(calls=tool_calls)

        prompt_tokens = sum(len((m.content or "").split()) for m in messages)
        completion_tokens = len(text.split()) + len(tool_calls)
        yield Done(
            usage=Usage(
                provider=self.name,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_ms=self.delay_s * 1000.0,
            ),
            finish_reason="tool_calls" if not text and tool_calls else "stop",
        )
