from datetime import timedelta

from parley.core.brain.state import CallState, Turn
from parley.core.telemetry.cost import estimate_call_cost


def _state(channel: str, minutes: float) -> CallState:
    s = CallState(call_id="c1", channel=channel, direction="inbound")
    s.ended_at = s.started_at + timedelta(minutes=minutes)
    s.transcript = [Turn(role="assistant", text="x" * 1000), Turn(role="user", text="hi")]
    return s


def test_phone_call_cost_components():
    prices = {
        "stt.deepgram": 0.01,
        "tts.elevenlabs": 0.05,
        "phone.twilio.inbound": 0.01,
        "llm.groq.in": 1.0,
        "llm.groq.out": 2.0,
    }
    cost = estimate_call_cost(
        _state("phone", 5), llm_usage=[("groq", 1_000_000, 500_000)], prices=prices
    )
    assert cost.stt_usd == 0.05
    assert cost.tts_chars == 1000 and cost.tts_usd == 0.05
    assert cost.phone_usd == 0.05
    assert cost.llm_usd == 2.0
    assert cost.total_usd == 2.15


def test_text_channel_has_no_audio_or_phone_cost():
    cost = estimate_call_cost(_state("text", 5), llm_usage=[("fake", 10, 10)])
    assert cost.stt_usd == cost.tts_usd == cost.phone_usd == 0.0
