"""Tests for OpenAICompatLLM, FallbackLLM, FakeLLM, and build_llm_chain."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from parley.config import Settings
from parley.core.providers.fake import FakeLLM
from parley.core.providers.llm import FallbackLLM, OpenAICompatLLM, build_llm_chain
from parley.core.providers.llm_base import (
    ChatMessage,
    Done,
    LLMError,
    LLMEvent,
    TextDelta,
    ToolCallsReady,
)


def _make_fake_openai_client(chunks: list[Any]) -> tuple[Any, list[dict[str, Any]]]:
    recorded_kwargs: list[dict[str, Any]] = []

    async def _create(**kwargs: Any) -> AsyncIterator[Any]:
        recorded_kwargs.append(kwargs)

        async def _gen() -> AsyncIterator[Any]:
            for ch in chunks:
                yield ch

        return _gen()

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_create),
        )
    )
    return client, recorded_kwargs


async def test_openai_compat_llm_streams_text_tool_calls_usage_and_ttft() -> None:
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Hello ", tool_calls=None),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="world!", tool_calls=None),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id="call_abc123",
                                function=SimpleNamespace(
                                    name="lookup_order",
                                    arguments='{"order_',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(
                                    name=None,
                                    arguments='id": "KC-',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=None,
                                function=SimpleNamespace(
                                    name=None,
                                    arguments='10501"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=42, completion_tokens=17),
        ),
    ]

    client, recorded = _make_fake_openai_client(chunks)
    llm = OpenAICompatLLM(
        name="groq",
        model="llama-3.3-70b-versatile",
        api_key="test-key",
        base_url="https://api.groq.com/openai/v1",
        stream_usage=True,
        client=client,
    )

    events = [
        ev
        async for ev in llm.stream(
            [ChatMessage(role="user", content="check my order")],
            tools=[{"type": "function", "function": {"name": "lookup_order"}}],
            tool_choice="auto",
        )
    ]

    assert len(recorded) == 1
    assert recorded[0]["stream_options"] == {"include_usage": True}

    text_deltas = [ev.text for ev in events if isinstance(ev, TextDelta)]
    assert text_deltas == ["Hello ", "world!"]

    tc_events = [ev for ev in events if isinstance(ev, ToolCallsReady)]
    assert len(tc_events) == 1
    assert len(tc_events[0].calls) == 1
    call = tc_events[0].calls[0]
    assert call.id == "call_abc123"
    assert call.name == "lookup_order"
    assert call.arguments == '{"order_id": "KC-10501"}'

    done_events = [ev for ev in events if isinstance(ev, Done)]
    assert len(done_events) == 1
    done = done_events[0]
    assert done.finish_reason == "tool_calls"
    assert done.usage.provider == "groq"
    assert done.usage.model == "llama-3.3-70b-versatile"
    assert done.usage.prompt_tokens == 42
    assert done.usage.completion_tokens == 17
    assert done.usage.ttft_ms is not None
    assert done.usage.ttft_ms >= 0.0


async def test_fallback_llm_raises_timeout_midstream_and_circuit_breaker() -> None:
    msg = [ChatMessage(role="user", content="hi")]

    # 1. First provider raises -> second used, health.fallbacks == 1
    p1 = FakeLLM([LLMError("p1 down")], name="p1", model="m1")
    p2 = FakeLLM(["Hello from p2"], name="p2", model="m2")
    fb1 = FallbackLLM([p1, p2], first_token_timeout_s=0.5)
    ev1 = [ev async for ev in fb1.stream(msg)]
    assert "".join(e.text for e in ev1 if isinstance(e, TextDelta)) == "Hello from p2"
    assert fb1.health.fallbacks == 1
    assert fb1.health.per_provider["p1"]["errors"] == 1
    assert fb1.health.per_provider["p2"]["calls"] == 1

    # 2. First provider hangs > first_token_timeout -> second used
    slow = FakeLLM(["too slow"], name="slow", model="mslow", delay_s=0.3)
    fast = FakeLLM(["Quick reply"], name="fast", model="mfast")
    fb2 = FallbackLLM([slow, fast], first_token_timeout_s=0.05)
    ev2 = [ev async for ev in fb2.stream(msg)]
    assert "".join(e.text for e in ev2 if isinstance(e, TextDelta)) == "Quick reply"
    assert fb2.health.fallbacks == 1

    # 3. Provider fails AFTER first token -> raises LLMError without falling back
    class _MidStreamFailLLM:
        name = "flaky"
        model = "mflaky"

        async def stream(
            self,
            messages: list[ChatMessage],
            tools: list[dict[str, Any]] | None = None,
            temperature: float = 0.3,
            max_tokens: int = 400,
            tool_choice: str | dict[str, Any] | None = None,
        ) -> AsyncIterator[LLMEvent]:
            yield TextDelta(text="Already spoken ")
            await asyncio.sleep(0)
            raise LLMError("flaky: stream cut mid-sentence")

    backup = FakeLLM(["Backup should not run"], name="backup", model="mbackup")
    fb3 = FallbackLLM([_MidStreamFailLLM(), backup], first_token_timeout_s=0.5)
    received: list[LLMEvent] = []
    with pytest.raises(LLMError, match="stream cut mid-sentence"):
        async for ev in fb3.stream(msg):
            received.append(ev)
    assert len(received) == 1
    assert isinstance(received[0], TextDelta)
    assert len(backup.calls) == 0

    # 4. Circuit breaker opens after 3 consecutive failures and skips the provider
    bad = FakeLLM(
        [LLMError("fail 1"), LLMError("fail 2"), LLMError("fail 3"), "should be skipped"],
        name="bad",
        model="mbad",
    )
    good = FakeLLM(["ok 1", "ok 2", "ok 3", "ok 4"], name="good", model="mgood")
    fb4 = FallbackLLM([bad, good], breaker_threshold=3, breaker_cooldown_s=60.0)
    assert fb4.model == "mbad"

    for _ in range(3):
        _ = [ev async for ev in fb4.stream(msg)]
    assert len(bad.calls) == 3
    assert fb4.model == "mgood"

    # 4th call: bad's circuit breaker is open, so bad.stream() is not called at all
    out4 = [ev async for ev in fb4.stream(msg)]
    assert "".join(e.text for e in out4 if isinstance(e, TextDelta)) == "ok 4"
    assert len(bad.calls) == 3
    assert len(good.calls) == 4


def test_build_llm_chain_skips_missing_keys_and_raises_when_empty() -> None:
    s_empty = Settings(
        llm_chain="groq,cerebras,gemini,openai",
        groq_api_key="",
        cerebras_api_key="",
        gemini_api_key="",
        openai_api_key="",
    )
    with pytest.raises(LLMError, match="no LLM provider configured"):
        build_llm_chain(s_empty)

    s_unknown = Settings(
        llm_chain="unknown_provider",
        groq_api_key="g-key",
    )
    with pytest.raises(ValueError, match="unknown_provider"):
        build_llm_chain(s_unknown)

    s_two = Settings(
        llm_chain="groq,cerebras,gemini,openai",
        groq_api_key="",
        cerebras_api_key="c-key",
        gemini_api_key="gem-key",
        openai_api_key="",
    )
    chain = build_llm_chain(s_two)
    assert [p.name for p in chain.providers] == ["cerebras", "gemini"]
    assert isinstance(chain.providers[0], OpenAICompatLLM)
    assert chain.providers[0].stream_usage is True
    assert isinstance(chain.providers[1], OpenAICompatLLM)
    assert chain.providers[1].stream_usage is False
