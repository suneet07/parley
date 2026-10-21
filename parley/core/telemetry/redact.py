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

_SKIP_KEYS: frozenset[str] = frozenset(
    {
        "call_id",
        "tenant_id",
        "direction",
        "channel",
        "disposition",
        "role",
        "status",
        "at",
        "started_at",
        "ended_at",
    }
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


def _redact_obj(value: Any, *, parent_key: str | None = None) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        is_tool_obj = "args" in value and (
            "status" in value or "confirm_prompt" in value or "result" in value
        )
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k in _SKIP_KEYS:
                out[k] = v
            elif k == "name" and is_tool_obj:
                out[k] = v
            elif k == "name" and parent_key == "caller" and isinstance(v, str):
                first = v.strip().split()[0] if v.strip() else ""
                out[k] = redact(first)
            else:
                out[k] = _redact_obj(v, parent_key=k)
        return out
    if isinstance(value, list):
        return [_redact_obj(item, parent_key=parent_key) for item in value]
    return value


def redact_state(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a CallState dict and recursively redact every non-identifier string value."""
    data = copy.deepcopy(state_dict)
    return _redact_obj(data)
