"""OpenAI-compatible streaming LLM client and circuit-breaking fallback chain."""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx
import openai
import structlog

from parley.core.providers.llm_base import (
    LLM,
    ChatMessage,
    Done,
    LLMError,
    LLMEvent,
    LLMToolCall,
    ProviderHealth,
    TextDelta,
    ToolCallsReady,
    Usage,
)

if TYPE_CHECKING:
    from parley.config import Settings

logger = structlog.get_logger(__name__)

# name -> (base_url, settings key attr, settings model attr, send_stream_usage)
PROVIDERS: dict[str, tuple[str, str, str, bool]] = {
    "groq": ("https://api.groq.com/openai/v1", "groq_api_key", "groq_model", True),
    "cerebras": ("https://api.cerebras.ai/v1", "cerebras_api_key", "cerebras_model", True),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini_api_key",
        "gemini_model",
        False,
    ),
    "openai": ("https://api.openai.com/v1", "openai_api_key", "openai_model", True),
}


class OpenAICompatLLM:
    """Streaming chat completion client for any OpenAI-compatible provider."""

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        base_url: str,
        stream_usage: bool = True,
        timeout_s: float = 20.0,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.stream_usage = stream_usage
        self.timeout_s = timeout_s
        self._client = client or openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=0,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 400,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """Stream text deltas, optional tool calls, and a final Done event."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if self.stream_usage:
            kwargs["stream_options"] = {"include_usage": True}

        start_t = time.perf_counter()
        ttft_ms: float | None = None
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason: str | None = None
        tool_acc: dict[int, dict[str, str]] = {}

        try:
            response = await self._client.chat.completions.create(**kwargs)
            async for chunk in response:
                choices = getattr(chunk, "choices", None)
                if choices:
                    choice0 = choices[0]
                    fr = getattr(choice0, "finish_reason", None)
                    if fr:
                        finish_reason = fr
                    delta = getattr(choice0, "delta", None)
                    if delta is not None:
                        content = getattr(delta, "content", None)
                        if content:
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - start_t) * 1000.0
                            yield TextDelta(text=content)

                        tc_deltas = getattr(delta, "tool_calls", None)
                        if tc_deltas:
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - start_t) * 1000.0
                            for tc in tc_deltas:
                                idx = int(getattr(tc, "index", 0) or 0)
                                entry = tool_acc.setdefault(
                                    idx, {"id": "", "name": "", "arguments": ""}
                                )
                                tc_id = getattr(tc, "id", None)
                                if tc_id:
                                    entry["id"] = tc_id
                                fn = getattr(tc, "function", None)
                                if fn is not None:
                                    fn_name = getattr(fn, "name", None)
                                    if fn_name:
                                        entry["name"] = (
                                            entry["name"] + fn_name
                                            if not entry["name"]
                                            else entry["name"]
                                        )
                                    fn_args = getattr(fn, "arguments", None)
                                    if fn_args:
                                        entry["arguments"] += fn_args

                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        except LLMError:
            raise
        except (openai.APIError, httpx.HTTPError, TimeoutError) as exc:
            raise LLMError(f"{self.name}: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"{self.name}: {exc}") from exc

        if tool_acc:
            calls = [
                LLMToolCall(
                    id=entry["id"] or f"call_{secrets.token_hex(4)}",
                    name=entry["name"],
                    arguments=entry["arguments"],
                )
                for _, entry in sorted(tool_acc.items())
            ]
            yield ToolCallsReady(calls=calls)

        yield Done(
            usage=Usage(
                provider=self.name,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_ms=ttft_ms,
            ),
            finish_reason=finish_reason,
        )


class FallbackLLM:
    """Tries LLM providers in order with first-token timeout and per-provider circuit breaker."""

    name: str = "fallback"

    def __init__(
        self,
        providers: list[LLM],
        first_token_timeout_s: float = 4.0,
        breaker_threshold: int = 3,
        breaker_cooldown_s: float = 30.0,
    ) -> None:
        if not providers:
            raise LLMError("no LLM provider configured")
        self.providers = list(providers)
        self.first_token_timeout_s = first_token_timeout_s
        self.breaker_threshold = breaker_threshold
        self.breaker_cooldown_s = breaker_cooldown_s
        self.health = ProviderHealth()
        self._consecutive_failures: dict[int, int] = {}
        self._opened_until: dict[int, float] = {}

    def _is_breaker_open(self, idx: int, now: float) -> bool:
        until = self._opened_until.get(idx)
        if until is None:
            return False
        return now < until

    @property
    def model(self) -> str:
        """Model of the current primary (first non-tripped) provider."""
        now = time.monotonic()
        for idx, provider in enumerate(self.providers):
            if not self._is_breaker_open(idx, now):
                return provider.model
        return self.providers[0].model

    def _record_call(self, provider_name: str) -> None:
        self.health.calls += 1
        stats = self.health.per_provider.setdefault(provider_name, {"calls": 0, "errors": 0})
        stats["calls"] += 1

    def _record_failure(self, idx: int, provider_name: str, error_msg: str) -> None:
        self.health.errors += 1
        self.health.last_error = error_msg
        stats = self.health.per_provider.setdefault(provider_name, {"calls": 0, "errors": 0})
        stats["errors"] += 1
        count = self._consecutive_failures.get(idx, 0) + 1
        self._consecutive_failures[idx] = count
        if count >= self.breaker_threshold:
            self._opened_until[idx] = time.monotonic() + self.breaker_cooldown_s

    def _record_success(self, idx: int) -> None:
        self._consecutive_failures[idx] = 0
        self._opened_until.pop(idx, None)

    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 400,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """Stream from the first healthy provider; fall back only before the first event."""
        errors: list[str] = []
        now = time.monotonic()

        for idx, provider in enumerate(self.providers):
            if self._is_breaker_open(idx, now):
                errors.append(f"{provider.name}: circuit breaker open")
                continue

            self._record_call(provider.name)
            agen = provider.stream(
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
            )
            try:
                try:
                    async with asyncio.timeout(self.first_token_timeout_s):
                        first_event = await anext(agen)
                except StopAsyncIteration:
                    self._record_success(idx)
                    return
                except (LLMError, TimeoutError, Exception) as exc:
                    err_msg = (
                        f"{provider.name}: first token timeout"
                        if isinstance(exc, TimeoutError)
                        else str(exc)
                    )
                    self._record_failure(idx, provider.name, err_msg)
                    self.health.fallbacks += 1
                    errors.append(err_msg)
                    logger.warning(
                        "llm_provider_fallback",
                        provider=provider.name,
                        error=err_msg,
                    )
                    continue

                # First event received — commit to this provider.
                yield first_event
                try:
                    async for event in agen:
                        yield event
                except LLMError as exc:
                    self._record_failure(idx, provider.name, str(exc))
                    raise
                except Exception as exc:
                    err_msg = f"{provider.name}: {exc}"
                    self._record_failure(idx, provider.name, err_msg)
                    raise LLMError(err_msg) from exc

                self._record_success(idx)
                return
            finally:
                await agen.aclose()

        raise LLMError(f"all providers failed: {'; '.join(errors)}")


def build_llm_chain(settings: Settings) -> FallbackLLM:
    """Build a FallbackLLM from settings.llm_chain, skipping providers without API keys."""
    raw_names = [n.strip().lower() for n in settings.llm_chain.split(",") if n.strip()]
    providers: list[LLM] = []
    for name in raw_names:
        if name not in PROVIDERS:
            raise ValueError(f"Unknown LLM provider in llm_chain: {name}")
        base_url, key_attr, model_attr, stream_usage = PROVIDERS[name]
        api_key: str = getattr(settings, key_attr, "") or ""
        if not api_key.strip():
            continue
        model: str = getattr(settings, model_attr)
        providers.append(
            OpenAICompatLLM(
                name=name,
                model=model,
                api_key=api_key.strip(),
                base_url=base_url,
                stream_usage=stream_usage,
            )
        )

    if not providers:
        raise LLMError("no LLM provider configured: set GROQ_API_KEY ...")
    return FallbackLLM(providers=providers)
