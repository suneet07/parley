"""Kettle & Co business tools and policy rules backed by SQLite (aiosqlite)."""

from __future__ import annotations

import asyncio
import json
import re
import secrets
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
import structlog
from pydantic import BaseModel, Field

from parley.core.brain.state import CallerProfile
from parley.core.tools.base import (
    PolicyDecision,
    ToolContext,
    ToolKind,
    ToolResult,
    policy,
    tool,
)

if TYPE_CHECKING:
    from parley.config import BusinessConfig

logger = structlog.get_logger(__name__)

_DB_LOCK = asyncio.Lock()

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    phone TEXT NOT NULL,
    email TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    items TEXT NOT NULL,
    total REAL NOT NULL,
    status TEXT NOT NULL,
    placed_date TEXT NOT NULL,
    delivered_date TEXT,
    carrier TEXT,
    tracking TEXT,
    eta_date TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE TABLE IF NOT EXISTS returns (
    return_id TEXT PRIMARY KEY,
    order_id TEXT UNIQUE NOT NULL,
    customer_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS refunds (
    refund_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    amount REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS store_credits (
    credit_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    amount REAL NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _resolve_db_path(base_dir: Path) -> Path:
    db_path = base_dir / "data" / "kettle.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _load_fixture_orders(base_dir: Path) -> list[dict[str, Any]]:
    fixture_path = base_dir / "fixtures" / "orders.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


async def ensure_db(base_dir: Path) -> Path:
    """Create and seed the Kettle & Co SQLite database if not already initialized."""
    async with _DB_LOCK:
        db_path = await asyncio.to_thread(_resolve_db_path, base_dir)
        async with aiosqlite.connect(db_path) as db:
            await db.executescript(_SCHEMA_SQL)
            async with db.execute("SELECT COUNT(*) FROM orders") as cursor:
                row = await cursor.fetchone()
                count = row[0] if row else 0

            if count == 0:
                orders_data = await asyncio.to_thread(_load_fixture_orders, base_dir)
                today = date.today()
                for entry in orders_data:
                    cust = entry["customer"]
                    await db.execute(
                        """
                        INSERT OR REPLACE INTO customers (customer_id, name, phone, email)
                        VALUES (?, ?, ?, ?)
                        """,
                        (cust["customer_id"], cust["name"], cust["phone"], cust["email"]),
                    )
                    placed_date = (today - timedelta(days=entry["placed_days_ago"])).isoformat()
                    deliv_ago = entry.get("delivered_days_ago")
                    delivered_date = (
                        (today - timedelta(days=deliv_ago)).isoformat()
                        if deliv_ago is not None
                        else None
                    )
                    eta_days = entry.get("eta_days")
                    eta_date = (
                        (today + timedelta(days=eta_days)).isoformat()
                        if eta_days is not None
                        else None
                    )
                    await db.execute(
                        """
                        INSERT INTO orders (
                            order_id, customer_id, items, total, status,
                            placed_date, delivered_date, carrier, tracking, eta_date
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            entry["order_id"],
                            cust["customer_id"],
                            json.dumps(entry["items"]),
                            float(entry["total"]),
                            entry["status"],
                            placed_date,
                            delivered_date,
                            entry.get("carrier"),
                            entry.get("tracking"),
                            eta_date,
                        ),
                    )
                await db.commit()
        return db_path


def resources(business: BusinessConfig) -> dict[str, Any]:
    """Return business-specific resource paths."""
    return {"db_path": business.base_dir / "data" / "kettle.db"}


def _normalize_phone(phone: str) -> str:
    digits = "".join(ch for ch in phone if ch.isdigit())
    return digits[-10:] if len(digits) >= 10 else ""


def _normalize_order_id(order_id: str) -> str:
    cleaned = re.sub(r"[\s\-_]+", "", order_id.strip().upper())
    if cleaned.startswith("KC"):
        suffix = cleaned[2:]
        return f"KC-{suffix}" if suffix else "KC"
    if cleaned.isdigit():
        return f"KC-{cleaned}"
    return order_id.strip().upper()


async def lookup_caller(ctx: ToolContext, phone: str) -> CallerProfile | None:
    """Look up a caller profile by phone number (last 10 digits)."""
    norm = _normalize_phone(phone)
    if not norm:
        return None
    db_path = await ensure_db(ctx.business_dir)
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT customer_id, name, phone, email FROM customers") as cursor:
            rows = await cursor.fetchall()
    for row in rows:
        if _normalize_phone(row["phone"]) == norm:
            return CallerProfile(
                customer_id=row["customer_id"],
                name=row["name"],
                phone=row["phone"],
                email=row["email"],
            )
    return None


async def _fetch_order_with_customer(
    ctx: ToolContext,
    order_id: str,
) -> dict[str, Any] | None:
    norm_id = _normalize_order_id(order_id)
    db_path = await ensure_db(ctx.business_dir)
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT o.order_id, o.customer_id, o.items, o.total, o.status,
                   o.placed_date, o.delivered_date, o.carrier, o.tracking, o.eta_date,
                   c.name AS customer_name, c.phone AS customer_phone, c.email AS customer_email
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE o.order_id = ?
            """,
            (norm_id,),
        ) as cursor:
            row = await cursor.fetchone()
    if row is None:
        return None
    result = dict(row)
    delivered_date = result.get("delivered_date")
    if delivered_date:
        result["days_since_delivery"] = (date.today() - date.fromisoformat(delivered_date)).days
    else:
        result["days_since_delivery"] = None
    return result


class VerifyIdentityArgs(BaseModel):
    order_id: str = Field(description="Order ID, e.g. KC-10482")
    phone_or_email: str = Field(description="Phone number or email address on the order")


class OrderIdArgs(BaseModel):
    order_id: str = Field(description="Order ID, e.g. KC-10482")


class StartReturnArgs(BaseModel):
    order_id: str = Field(description="Order ID to return, e.g. KC-10501")
    reason: str = Field(description="Reason for returning the order")


class IssueRefundArgs(BaseModel):
    order_id: str = Field(description="Order ID to refund, e.g. KC-10501")
    amount: float = Field(description="Refund amount in INR")


class OfferStoreCreditArgs(BaseModel):
    order_id: str = Field(description="Order ID for store credit, e.g. KC-10482")
    amount: float = Field(description="Store credit amount in INR")


@tool(kind=ToolKind.VERIFY)
async def verify_identity(ctx: ToolContext, args: VerifyIdentityArgs) -> ToolResult:
    """Verify the caller's identity using their order ID and phone number or email."""
    order = await _fetch_order_with_customer(ctx, args.order_id)
    if order is None:
        return ToolResult(ok=False, error="no_match")

    candidate = args.phone_or_email.strip()
    phone_match = bool(_normalize_phone(candidate)) and _normalize_phone(
        candidate
    ) == _normalize_phone(order["customer_phone"])
    email_match = candidate.lower() == order["customer_email"].strip().lower()

    if not (phone_match or email_match):
        return ToolResult(ok=False, error="no_match")

    full_name: str = order["customer_name"]
    first_name = full_name.split()[0] if full_name else ""
    profile = CallerProfile(
        customer_id=order["customer_id"],
        name=full_name,
        phone=order["customer_phone"],
        email=order["customer_email"],
    )
    return ToolResult(
        ok=True,
        data={"verified": True, "name": first_name},
        verified_profile=profile,
    )


@tool(kind=ToolKind.READ, requires_verification=True)
async def lookup_order(ctx: ToolContext, args: OrderIdArgs) -> ToolResult:
    """Look up an order's status, items, totals, delivery dates, and return eligibility."""
    order = await _fetch_order_with_customer(ctx, args.order_id)
    if (
        order is None
        or ctx.state.caller is None
        or order["customer_id"] != ctx.state.caller.customer_id
    ):
        return ToolResult(ok=False, error="not_found")

    raw_items: list[dict[str, Any]] = json.loads(order["items"])
    items = [{"name": it["name"], "qty": it["qty"], "price": it["price"]} for it in raw_items]
    window = (
        ctx.business.policies.return_window_days
        if ctx.business.policies.return_window_days is not None
        else 30
    )
    days_since = order["days_since_delivery"]
    return_eligible = (
        order["status"] == "delivered" and days_since is not None and days_since <= window
    )
    return ToolResult(
        ok=True,
        data={
            "order_id": order["order_id"],
            "status": order["status"],
            "items": items,
            "total": order["total"],
            "placed_date": order["placed_date"],
            "delivered_date": order["delivered_date"],
            "days_since_delivery": days_since,
            "return_eligible": return_eligible,
            "return_window_days": window,
        },
    )


@tool(kind=ToolKind.READ, requires_verification=True)
async def track_shipment(ctx: ToolContext, args: OrderIdArgs) -> ToolResult:
    """Look up shipment carrier, tracking number, and estimated delivery date for an order."""
    order = await _fetch_order_with_customer(ctx, args.order_id)
    if (
        order is None
        or ctx.state.caller is None
        or order["customer_id"] != ctx.state.caller.customer_id
    ):
        return ToolResult(ok=False, error="not_found")

    return ToolResult(
        ok=True,
        data={
            "order_id": order["order_id"],
            "status": order["status"],
            "carrier": order["carrier"],
            "tracking": order["tracking"],
            "eta_date": order["eta_date"],
        },
    )


@policy("start_return")
async def _policy_start_return(ctx: ToolContext, args: StartReturnArgs) -> PolicyDecision:
    order = await _fetch_order_with_customer(ctx, args.order_id)
    if (
        order is None
        or ctx.state.caller is None
        or order["customer_id"] != ctx.state.caller.customer_id
    ):
        return PolicyDecision(False, "not_found")

    if order["status"] != "delivered" or order["days_since_delivery"] is None:
        return PolicyDecision(False, "order_not_delivered")

    window = (
        ctx.business.policies.return_window_days
        if ctx.business.policies.return_window_days is not None
        else 30
    )
    if order["days_since_delivery"] > window:
        return PolicyDecision(
            False,
            "outside_return_window",
            escalate=False,
            alternatives=["store_credit", "repair_under_warranty", "transfer_to_human"],
        )
    return PolicyDecision(True)


@tool(
    kind=ToolKind.WRITE,
    requires_verification=True,
    confirm="start a return for order {order_id}, reason: {reason}",
)
async def start_return(ctx: ToolContext, args: StartReturnArgs) -> ToolResult:
    """Start a return and schedule a free pickup for an eligible delivered order."""
    norm_id = _normalize_order_id(args.order_id)
    db_path = await ensure_db(ctx.business_dir)
    return_id = f"RT-{secrets.token_hex(3)}"
    customer_id = ctx.state.caller.customer_id if ctx.state.caller else ""

    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT return_id FROM returns WHERE order_id = ?", (norm_id,)
        ) as cursor:
            existing = await cursor.fetchone()
        if existing is not None:
            return ToolResult(ok=False, error="return_already_exists")

        await db.execute(
            """
            INSERT INTO returns (return_id, order_id, customer_id, reason, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                return_id,
                norm_id,
                customer_id,
                args.reason,
                datetime.now(UTC).isoformat(),
            ),
        )
        await db.commit()

    return ToolResult(
        ok=True,
        data={"return_id": return_id, "pickup": "within 2 business days"},
    )


@policy("issue_refund")
async def _policy_issue_refund(ctx: ToolContext, args: IssueRefundArgs) -> PolicyDecision:
    order = await _fetch_order_with_customer(ctx, args.order_id)
    if (
        order is None
        or ctx.state.caller is None
        or order["customer_id"] != ctx.state.caller.customer_id
    ):
        return PolicyDecision(False, "not_found")

    if order["status"] != "delivered" or order["days_since_delivery"] is None:
        return PolicyDecision(False, "order_not_delivered")

    window = (
        ctx.business.policies.return_window_days
        if ctx.business.policies.return_window_days is not None
        else 30
    )
    if order["days_since_delivery"] > window:
        return PolicyDecision(
            False,
            "outside_return_window",
            escalate=False,
            alternatives=["store_credit", "repair_under_warranty", "transfer_to_human"],
        )

    if args.amount <= 0:
        return PolicyDecision(False, "invalid_amount")

    if args.amount > float(order["total"]):
        return PolicyDecision(False, "amount_exceeds_order_total")

    refund_max = ctx.business.policies.refund_max
    if refund_max is not None and args.amount > refund_max:
        return PolicyDecision(
            False,
            "exceeds_refund_limit",
            escalate=True,
            alternatives=["transfer_to_human"],
        )

    return PolicyDecision(True)


@tool(
    kind=ToolKind.WRITE,
    requires_verification=True,
    confirm="refund ₹{amount} for order {order_id} to the original payment method",
)
async def issue_refund(ctx: ToolContext, args: IssueRefundArgs) -> ToolResult:
    """Issue a refund to the customer's original payment method."""
    norm_id = _normalize_order_id(args.order_id)
    db_path = await ensure_db(ctx.business_dir)
    refund_id = f"RF-{secrets.token_hex(3)}"
    customer_id = ctx.state.caller.customer_id if ctx.state.caller else ""

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO refunds (refund_id, order_id, customer_id, amount, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                refund_id,
                norm_id,
                customer_id,
                args.amount,
                datetime.now(UTC).isoformat(),
            ),
        )
        await db.commit()

    return ToolResult(
        ok=True,
        data={"refund_id": refund_id, "order_id": norm_id, "amount": args.amount},
    )


@policy("offer_store_credit")
async def _policy_offer_store_credit(
    ctx: ToolContext,
    args: OfferStoreCreditArgs,
) -> PolicyDecision:
    order = await _fetch_order_with_customer(ctx, args.order_id)
    if (
        order is None
        or ctx.state.caller is None
        or order["customer_id"] != ctx.state.caller.customer_id
    ):
        return PolicyDecision(False, "not_found")

    if order["status"] != "delivered" or order["days_since_delivery"] is None:
        return PolicyDecision(False, "order_not_delivered")

    if order["days_since_delivery"] > 90:
        return PolicyDecision(
            False,
            "outside_store_credit_window",
            escalate=True,
            alternatives=["transfer_to_human"],
        )

    if args.amount <= 0:
        return PolicyDecision(False, "invalid_amount")

    if args.amount > float(order["total"]):
        return PolicyDecision(False, "amount_exceeds_order_total")

    return PolicyDecision(True)


@tool(
    kind=ToolKind.WRITE,
    requires_verification=True,
    confirm="add ₹{amount} store credit for order {order_id}",
)
async def offer_store_credit(ctx: ToolContext, args: OfferStoreCreditArgs) -> ToolResult:
    """Add store credit to the customer's Kettle & Co account for a delivered order."""
    norm_id = _normalize_order_id(args.order_id)
    db_path = await ensure_db(ctx.business_dir)
    credit_id = f"SC-{secrets.token_hex(3)}"
    customer_id = ctx.state.caller.customer_id if ctx.state.caller else ""

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO store_credits (credit_id, order_id, customer_id, amount, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                credit_id,
                norm_id,
                customer_id,
                args.amount,
                datetime.now(UTC).isoformat(),
            ),
        )
        await db.commit()

    return ToolResult(
        ok=True,
        data={"credit_id": credit_id, "order_id": norm_id, "amount": args.amount},
    )
