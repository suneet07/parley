"""Scripted voicemail drop for outbound calls that reach an answering machine."""

from __future__ import annotations

from twilio.twiml.voice_response import VoiceResponse

# The goal is free text from the triggering system (it could name a debt or a medical topic),
# and a voicemail can be heard by anyone, so only fixed phrases are ever spoken.
_GOAL_PHRASES = {
    "appointment": "about your upcoming appointment",
    "order": "about your recent order",
    "delivery": "about your delivery",
    "cart": "about items you left in your cart",
}


def _goal_phrase(goal: str) -> str:
    g = goal.lower()
    for key, phrase in _GOAL_PHRASES.items():
        if key in g:
            return phrase
    return "about your recent request"


def voicemail_twiml(business_name: str, goal: str, callback_number: str) -> str:
    digits = ", ".join(ch for ch in callback_number if ch.isdigit())
    vr = VoiceResponse()
    message = (
        f"Hi, this is the AI assistant from {business_name}, calling {_goal_phrase(goal)}. "
        "Sorry we missed you."
    )
    if digits:
        message += f" Please call us back at {digits}."
    vr.say(message + " Thank you, goodbye.")
    vr.hangup()
    return str(vr)
