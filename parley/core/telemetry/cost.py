"""Per-call cost estimate: STT minutes, LLM tokens, TTS characters, telephony minutes.

Prices are list prices in USD and drift over time. Override any of them with a JSON file at
PARLEY_PRICES (same keys) so a deployment can publish numbers that match its own contracts.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from parley.core.brain.state import CallState

DEFAULT_PRICES: dict[str, float] = {
    # STT, per audio minute
    "stt.deepgram": 0.0077,
    "stt.groq": 0.00067,
    # LLM, per 1M tokens (input, output)
    "llm.groq.in": 0.59,
    "llm.groq.out": 0.79,
    "llm.cerebras.in": 0.85,
    "llm.cerebras.out": 1.20,
    "llm.gemini.in": 0.30,
    "llm.gemini.out": 2.50,
    "llm.openai.in": 0.40,
    "llm.openai.out": 1.60,
    "llm.fake.in": 0.0,
    "llm.fake.out": 0.0,
    # TTS, per 1K characters
    "tts.elevenlabs": 0.05,
    "tts.cartesia": 0.03,
    "tts.deepgram": 0.03,
    "tts.piper": 0.0,
    # Telephony, per minute (inbound / outbound)
    "phone.twilio.inbound": 0.0085,
    "phone.twilio.outbound": 0.014,
}


def load_prices() -> dict[str, float]:
    prices = dict(DEFAULT_PRICES)
    path = os.environ.get("PARLEY_PRICES")
    if path and Path(path).is_file():
        prices.update({k: float(v) for k, v in json.loads(Path(path).read_text()).items()})
    return prices


@dataclass
class CallCost:
    stt_usd: float = 0.0
    llm_usd: float = 0.0
    tts_usd: float = 0.0
    phone_usd: float = 0.0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    tts_chars: int = 0
    minutes: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def total_usd(self) -> float:
        return round(self.stt_usd + self.llm_usd + self.tts_usd + self.phone_usd, 5)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["total_usd"] = self.total_usd
        return d


def estimate_call_cost(
    state: CallState,
    *,
    llm_usage: list[tuple[str, int, int]],  # (provider, prompt_tokens, completion_tokens) per call
    stt_provider: str = "deepgram",
    tts_provider: str = "elevenlabs",
    prices: dict[str, float] | None = None,
) -> CallCost:
    p = prices or load_prices()
    cost = CallCost()
    if state.ended_at:
        cost.minutes = max((state.ended_at - state.started_at).total_seconds(), 0.0) / 60.0

    if state.channel != "text":
        cost.stt_usd = cost.minutes * p.get(f"stt.{stt_provider}", 0.0)
        cost.tts_chars = sum(len(t.text) for t in state.transcript if t.role == "assistant")
        cost.tts_usd = cost.tts_chars / 1000.0 * p.get(f"tts.{tts_provider}", 0.0)
    if state.channel == "phone":
        cost.phone_usd = cost.minutes * p.get(f"phone.twilio.{state.direction}", 0.0)

    for provider, tokens_in, tokens_out in llm_usage:
        cost.llm_tokens_in += tokens_in
        cost.llm_tokens_out += tokens_out
        usd = (
            tokens_in * p.get(f"llm.{provider}.in", 0.0)
            + tokens_out * p.get(f"llm.{provider}.out", 0.0)
        ) / 1_000_000
        cost.llm_usd += usd
        cost.breakdown[f"llm.{provider}"] = cost.breakdown.get(f"llm.{provider}", 0.0) + usd

    for attr in ("stt_usd", "llm_usd", "tts_usd", "phone_usd"):
        setattr(cost, attr, round(getattr(cost, attr), 5))
    return cost
