"""Fill the local call store with fictional demo calls so the dashboard has something to show.

    uv run python scripts/seed_demo_calls.py [--db sqlite+aiosqlite:///./parley.db] [--n 40]

Everything here is made up (Kettle & Co demo business). Safe to delete parley.db afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
import random
from datetime import UTC, datetime, timedelta

from parley.api.store import CallStore
from parley.core.brain.state import CallState, ToolEvent, Turn

SCRIPTS = [
    (
        "resolved",
        ["order_status"],
        [
            ("user", "Hi, where is my order KC-10533?"),
            ("assistant", "I can check that. Could you tell me the phone number on the order?"),
            ("user", "It's 98333 44443."),
            ("assistant", "Thanks, Arjun. It shipped with BlueDart and should arrive in two days."),
        ],
        [("verify_identity", "ok", 2), ("track_shipment", "ok", 2)],
        "Caller asked for order status of KC-10533.\nVerified by phone.\nShared BlueDart ETA.",
    ),
    (
        "resolved",
        ["return"],
        [
            ("user", "The spout on my copper kettle is leaking, I want to return it."),
            ("assistant", "Sorry about that. What's the order number and phone on the order?"),
            ("user", "KC-10501, 9822233332."),
            (
                "assistant",
                "Thanks. Shall I start a return for order KC-10501 because it's leaking?",
            ),
            ("user", "Yes please."),
            ("assistant", "Done. A pickup will be arranged within two business days."),
        ],
        [
            ("verify_identity", "ok", 2),
            ("start_return", "confirmation_required", 2),
            ("start_return", "ok", 3),
        ],
        "Return requested for leaking kettle KC-10501.\nVerified.\nReturn RT created, pickup in 2 days.",
    ),
    (
        "escalated",
        ["refund", "human"],
        [
            ("user", "I need a full refund for my Dutch oven, 7800 rupees."),
            ("assistant", "I can help. Could you give me the order number and your email?"),
            ("user", "KC-10477, neha.g@example.com"),
            (
                "assistant",
                "Refunds above five thousand rupees need a team member. Shall I connect you?",
            ),
            ("user", "Yes, connect me."),
        ],
        [
            ("verify_identity", "ok", 2),
            ("issue_refund", "blocked", 2),
            ("transfer_to_human", "ok", 3),
        ],
        "Refund of ₹7800 requested for KC-10477.\nAbove automated limit.\nTransferred to a person.",
    ),
    (
        "ticket_created",
        ["faq", "product"],
        [
            ("user", "Do your kettles work on induction stoves?"),
            ("assistant", "I don't have that detail. I can create a ticket so someone emails you."),
            ("user", "Sure."),
        ],
        [("search_kb", "ok", 1), ("create_ticket", "ok", 2)],
        "Asked about induction compatibility.\nNot in knowledge base.\nTicket created.",
    ),
    (
        "resolved",
        ["faq"],
        [
            ("user", "Do you have cash on delivery?"),
            ("assistant", "Yes, cash on delivery is available on most orders."),
        ],
        [],
        "Asked about cash on delivery.\nAnswered from FAQ.\nNo follow-up.",
    ),
    (
        "abandoned",
        ["other"],
        [("user", "Hello?"), ("assistant", "Are you still there?")],
        [],
        "Caller went silent.\nRe-prompted three times.\nCall ended.",
    ),
]

GAPS = [
    "Do your kettles work on induction stoves?",
    "Can I change the delivery address after ordering?",
    "Do you ship to Sri Lanka?",
]


def _turn_metrics(n: int, rng: random.Random) -> list[dict]:
    turns = []
    for i in range(1, n + 1):
        stt = rng.uniform(90, 260)
        llm = rng.uniform(150, 520)
        tts = rng.uniform(70, 220)
        e2e = stt + llm + tts + rng.uniform(250, 420)
        turns.append(
            {
                "turn": i,
                "stt_ms": round(stt, 1),
                "llm_ttft_ms": round(llm, 1),
                "tts_ttfb_ms": round(tts, 1),
                "e2e_ms": round(e2e, 1),
                "interrupted": rng.random() < 0.12,
            }
        )
    return turns


async def seed(db: str, n: int) -> None:
    rng = random.Random(7)
    store = CallStore(db)
    await store.init()
    now = datetime.now(UTC)
    for i in range(n):
        disposition, intents, lines, tools, summary = rng.choice(SCRIPTS)
        started = now - timedelta(hours=rng.uniform(0, 24 * 14))
        state = CallState(
            call_id=f"demo-{i:04d}",
            channel=rng.choice(["phone", "phone", "webrtc"]),
            direction=rng.choice(["inbound"] * 5 + ["outbound"]),
            caller_number=f"+9198{rng.randint(10000000, 99999999)}",
            started_at=started,
        )
        state.transcript = [
            Turn(
                role="assistant",
                text="Hi, you've reached Kettle and Co. I'm an AI assistant and this call may be recorded. How can I help?",
                at=started,
            )
        ]
        for k, (role, text) in enumerate(lines):
            state.transcript.append(
                Turn(role=role, text=text, at=started + timedelta(seconds=8 * (k + 1)))
            )
        state.tool_events = [
            ToolEvent(name=name, args={}, status=status, turn=turn, latency_ms=rng.uniform(20, 180))
            for name, status, turn in tools
        ]
        state.verified = any(t[0] == "verify_identity" for t in tools)
        state.disposition = disposition
        state.intents = intents
        state.summary = summary
        state.escalation_reason = "policy_exception" if disposition == "escalated" else None
        if disposition == "ticket_created":
            state.knowledge_gaps = [rng.choice(GAPS)]
        state.ended = True
        state.ended_at = started + timedelta(seconds=rng.uniform(40, 260))
        turns = _turn_metrics(sum(1 for r, _ in lines if r == "user"), rng)
        e2es = sorted(t["e2e_ms"] for t in turns)
        metrics = {
            "turns": turns,
            "count": len(turns),
            "p50_e2e_ms": e2es[len(e2es) // 2] if e2es else None,
            "p95_e2e_ms": e2es[-1] if e2es else None,
        }
        await store.save(state, metrics)
    await store.close()
    print(f"seeded {n} demo calls into {db}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="sqlite+aiosqlite:///./parley.db")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    asyncio.run(seed(args.db, args.n))


if __name__ == "__main__":
    main()
