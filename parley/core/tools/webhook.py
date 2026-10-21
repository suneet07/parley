"""HTTP webhook tool adapter."""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from pydantic import BaseModel

from parley.core.tools.base import ToolContext, ToolKind, ToolResult, ToolSpec

logger = structlog.get_logger(__name__)


def webhook_tool(
    *,
    name: str,
    description: str,
    kind: ToolKind,
    url: str,
    args_model: type[BaseModel],
    method: str = "POST",
    headers: dict[str, str] | None = None,
    timeout_s: float = 8.0,
    requires_verification: bool = False,
    requires_confirmation: bool | None = None,
    confirm: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ToolSpec:
    """Create a ToolSpec backed by an HTTP endpoint without registering in global collection."""

    async def _call_webhook(ctx: ToolContext, args: BaseModel) -> ToolResult:
        body: dict[str, Any] = {
            **args.model_dump(mode="json"),
            "call_id": ctx.state.call_id,
            "tenant_id": ctx.state.tenant_id,
            "caller": ctx.state.caller.model_dump(mode="json") if ctx.state.caller else None,
        }
        try:
            async with httpx.AsyncClient(
                timeout=timeout_s,
                headers=headers,
                transport=transport,
            ) as client:
                response = await client.request(method.upper(), url, json=body)
        except httpx.TimeoutException:
            return ToolResult(ok=False, error="timeout")
        except httpx.HTTPError as exc:
            return ToolResult(ok=False, error=f"http_error:{exc}")

        if 200 <= response.status_code < 300:
            try:
                data = response.json()
            except ValueError:
                data = {}
            if not isinstance(data, dict):
                data = {"result": data}
            return ToolResult(ok=True, data=data)

        return ToolResult(ok=False, error=f"http_{response.status_code}")

    effective_confirm = (
        kind == ToolKind.WRITE if requires_confirmation is None else requires_confirmation
    )
    return ToolSpec(
        name=name,
        description=description,
        kind=kind,
        args_model=args_model,
        fn=_call_webhook,
        requires_verification=requires_verification,
        requires_confirmation=effective_confirm,
        confirm_template=confirm,
    )
