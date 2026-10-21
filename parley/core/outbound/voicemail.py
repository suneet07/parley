"""Voicemail TwiML builder for answering-machine detection callbacks."""

from __future__ import annotations

from twilio.twiml.voice_response import VoiceResponse


def _spell_digits(phone: str) -> str:
    digits = [ch for ch in phone if ch.isdigit()]
    if digits:
        return ", ".join(digits)
    return phone


def voicemail_twiml(business_name: str, goal: str, callback_number: str) -> str:
    """Return a <= 20-second <Say> + <Hangup/> TwiML with AI disclosure and no personal data."""
    spelled_number = _spell_digits(callback_number)
    script = (
        f"Hi, this is the AI assistant from {business_name}. "
        f"We called regarding {goal}. "
        f"Please call us back at {spelled_number}. Thank you, goodbye."
    )
    vr = VoiceResponse()
    vr.say(script)
    vr.hangup()
    return str(vr)
