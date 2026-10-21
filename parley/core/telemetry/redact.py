from __future__ import annotations

import copy
import re
from typing import Any

# CVV after cue words: cvv, cvc, security code
_CVV_RE = re.compile(
    r"\b((?:cvv|cvc|security\s+code)(?:\s+(?:number|code))?(?:\s*(?:is|:|=)\s*|\s+))(\d{3,4})\b",
    re.IGNORECASE,
)

# Card candidates: 13-19 digits, optionally separated by spaces or hyphens
_CARD_RE = re.compile(r"(?<![A-Za-z0-9₹])(?:\d[ -]?){12,18}\d(?![A-Za-z0-9])")

# Indian Aadhaar: 12 digits in 3 groups of 4 separated by spaces or hyphens
_AADHAAR_RE = re.compile(r"(?<![A-Za-z0-9+₹-])\b\d{4}(?: \d{4} \d{4}|-\d{4}-\d{4})\b(?![0-9-])")

# Indian PAN: 5 uppercase letters + 4 digits + 1 uppercase letter
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")

# Email address: keep first character of local part + ***@ + domain
_EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")

# Postal address: cue word within 40 chars before a 6-digit Indian PIN code
_ADDRESS_RE = re.compile(
    r"\b(?:road|street|nagar|sector|flat|house|colony|lane|apartment)\b.{0,40}?\b\d{6}\b",
    re.IGNORECASE | re.DOTALL,
)

# Phone numbers: 10-15 digits (with optional +country code, parens, spaces, dots, hyphens)
_PHONE_RE = re.compile(
    r"(?<![A-Za-z0-9₹-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,5}\)[\s.-]?|\d[\s.-]?){9,14}\d(?![A-Za-z0-9])"
)


def _luhn_valid(digits: list[int]) -> bool:
    if not (13 <= len(digits) <= 19) or sum(digits) == 0:
        return False
    total = 0
    for idx, d in enumerate(reversed(digits)):
        if idx % 2 == 1:
            doubled = d * 2
            total += doubled - 9 if doubled > 9 else doubled
        else:
            total += d
    return total % 10 == 0


def _replace_card(match: re.Match[str]) -> str:
    raw = match.group(0)
    digits = [int(ch) for ch in raw if ch.isdigit()]
    if _luhn_valid(digits):
        return "[CARD]"
    return raw


def _replace_phone(match: re.Match[str]) -> str:
    raw = match.group(0)
    digits = [ch for ch in raw if ch.isdigit()]
    if 10 <= len(digits) <= 15:
        return "******" + "".join(digits[-4:])
    return raw


def redact(text: str) -> str:
    """Redact PII (cards, CVV, Aadhaar, PAN, emails, addresses with PIN, phones) from text."""
    if not text:
        return text

    out = _CVV_RE.sub(r"\1[CVV]", text)
    out = _CARD_RE.sub(_replace_card, out)
    out = _AADHAAR_RE.sub("[AADHAAR]", out)
    out = _PAN_RE.sub("[PAN]", out)
    out = _EMAIL_RE.sub(r"\1***@\2", out)
    out = _ADDRESS_RE.sub("[ADDRESS]", out)
    out = _PHONE_RE.sub(_replace_phone, out)
    return out


def _redact_obj(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _redact_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_obj(item) for item in value]
    return value


def redact_state(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a CallState dict and redact strings in transcript text and tool event args/result."""
    data = copy.deepcopy(state_dict)

    if isinstance(data.get("caller_number"), str):
        data["caller_number"] = redact(data["caller_number"])

    if isinstance(data.get("summary"), str):
        data["summary"] = redact(data["summary"])

    if isinstance(data.get("caller"), dict):
        data["caller"] = _redact_obj(data["caller"])

    if isinstance(data.get("slots"), dict):
        data["slots"] = _redact_obj(data["slots"])

    transcript = data.get("transcript")
    if isinstance(transcript, list):
        for turn in transcript:
            if isinstance(turn, dict):
                if isinstance(turn.get("text"), str):
                    turn["text"] = redact(turn["text"])
                turn_events = turn.get("tool_events")
                if isinstance(turn_events, list):
                    for ev in turn_events:
                        if isinstance(ev, dict):
                            if "args" in ev:
                                ev["args"] = _redact_obj(ev["args"])
                            if "result" in ev:
                                ev["result"] = _redact_obj(ev["result"])

    tool_events = data.get("tool_events")
    if isinstance(tool_events, list):
        for ev in tool_events:
            if isinstance(ev, dict):
                if "args" in ev:
                    ev["args"] = _redact_obj(ev["args"])
                if "result" in ev:
                    ev["result"] = _redact_obj(ev["result"])

    return data
