from __future__ import annotations

from parley.core.telemetry.redact import redact, redact_state


def test_redact_card_luhn_valid() -> None:
    text = "My card is 4532 0151 1283 0366 and another is 4111-1111-1111-1111."
    out = redact(text)
    assert out == "My card is [CARD] and another is [CARD]."


def test_redact_cvv() -> None:
    assert redact("My CVV is 123 and security code: 4567") == (
        "My CVV is [CVV] and security code: [CVV]"
    )
    assert redact("cvc 890") == "cvc [CVV]"


def test_redact_aadhaar() -> None:
    assert redact("Aadhaar: 1234 5678 9012") == "Aadhaar: [AADHAAR]"
    assert redact("Aadhaar: 1234-5678-9012") == "Aadhaar: [AADHAAR]"


def test_redact_pan() -> None:
    assert redact("PAN is ABCDE1234F for tax") == "PAN is [PAN] for tax"


def test_redact_email() -> None:
    assert redact("Email me at priya.sharma@example.co.in please") == (
        "Email me at p***@example.co.in please"
    )


def test_redact_phone() -> None:
    assert redact("Call me on +91 98765 43210 or 9123456789") == (
        "Call me on ******3210 or ******6789"
    )


def test_redact_address_with_cue_word() -> None:
    text = "Ship to Flat 402, MG Road, Bengaluru 560001 tomorrow"
    out = redact(text)
    assert "[ADDRESS]" in out
    assert "560001" not in out


def test_redact_must_not_mask_order_ids_amounts_days() -> None:
    text = "Order KC-10482 for ₹2499 or 2499 is eligible within 30 days."
    assert redact(text) == text


def test_redact_state_deep_copy_and_nested_fields() -> None:
    original = {
        "call_id": "call-1",
        "tenant_id": "default",
        "caller_number": "+919876543210",
        "transcript": [
            {
                "role": "user",
                "text": "My card is 4111 1111 1111 1111 for order KC-10482",
                "tool_events": [
                    {
                        "name": "lookup_order",
                        "args": {"phone": "9876543210", "order_id": "KC-10482"},
                        "result": {"email": "rahul@kettle.co", "amount": "₹2499"},
                    }
                ],
            }
        ],
        "tool_events": [
            {
                "name": "refund",
                "args": {"pan": "ABCDE1234F"},
                "result": {"card": "4532015112830366"},
            }
        ],
    }

    redacted = redact_state(original)
    # Original must not be mutated
    assert "4111 1111 1111 1111" in original["transcript"][0]["text"]

    assert redacted["caller_number"] == "******3210"
    assert redacted["transcript"][0]["text"] == "My card is [CARD] for order KC-10482"
    turn_ev = redacted["transcript"][0]["tool_events"][0]
    assert turn_ev["args"]["phone"] == "******3210"
    assert turn_ev["args"]["order_id"] == "KC-10482"
    assert turn_ev["result"]["email"] == "r***@kettle.co"
    assert turn_ev["result"]["amount"] == "₹2499"
    assert redacted["tool_events"][0]["args"]["pan"] == "[PAN]"
    assert redacted["tool_events"][0]["result"]["card"] == "[CARD]"
