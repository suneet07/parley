"""Tool contracts. Tools are plain async Python functions with a Pydantic args model.

Example (a business tools module):

    class LookupOrderArgs(BaseModel):
        order_id: str = Field(description="Order number, e.g. KC-10482")

    @tool(kind=ToolKind.READ, requires_verification=True)
    async def lookup_order(ctx: ToolContext, args: LookupOrderArgs) -> ToolResult:
        \"\"\"Look up an order's status, items and dates.\"\"\"
        ...
        return ToolResult(ok=True, data={"status": "shipped"})

The function docstring is the description shown to the LLM.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import BaseModel, Field

from parley.core.brain.state import CallerProfile

if TYPE_CHECKING:
    from parley.config import BusinessConfig
    from parley.core.brain.state import CallState
    from parley.core.knowledge.base import Retriever


class ToolKind(StrEnum):
    READ = "read"  # runs directly
    WRITE = "write"  # changes something; confirmation required by default
    ACTION = "action"  # call-control (transfer, end call); no confirmation
    VERIFY = "verify"  # identity check; on success sets state.verified + state.caller


class ToolResult(BaseModel):
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    # VERIFY tools set this on success; the gateway copies it to state.caller.
    verified_profile: CallerProfile | None = None
    # ACTION tools may request call control; the session/pipeline acts on it.
    control: str | None = None  # "transfer" | "end_call" | None


@dataclass
class ToolContext:
    state: CallState
    business: BusinessConfig
    retriever: Retriever | None = None
    resources: dict[str, Any] = field(default_factory=dict)  # business-specific (db path, clients)

    @property
    def business_dir(self) -> Path:
        return self.business.base_dir


ToolFn = Callable[[ToolContext, Any], Awaitable[ToolResult]]


@dataclass
class ToolSpec:
    name: str
    description: str
    kind: ToolKind
    args_model: type[BaseModel]
    fn: ToolFn
    requires_verification: bool = False
    requires_confirmation: bool = False
    # str.format template over args, read back before a write, e.g.
    # "start a return for order {order_id} because {reason}"
    confirm_template: str | None = None

    def openai_schema(self) -> dict[str, Any]:
        """OpenAI-compatible function tool schema."""
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": schema},
        }


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str = ""
    escalate: bool = False  # policy exception -> offer human
    alternatives: list[str] = field(default_factory=list)  # e.g. ["store_credit", "repair"]


PolicyFn = Callable[[ToolContext, Any], PolicyDecision | Awaitable[PolicyDecision]]


# Module-level collection used by the decorators; the registry imports a module and reads these.
_COLLECTED_TOOLS: list[ToolSpec] = []
_COLLECTED_POLICIES: dict[str, list[PolicyFn]] = {}


def tool(
    *,
    kind: ToolKind,
    name: str | None = None,
    requires_verification: bool = False,
    requires_confirmation: bool | None = None,  # default: True for WRITE, else False
    confirm: str | None = None,
) -> Callable[[ToolFn], ToolFn]:
    """Register an async tool function. The args model is the 2nd parameter's annotation."""

    def deco(fn: ToolFn) -> ToolFn:
        hints = get_type_hints(fn)
        params = list(inspect.signature(fn).parameters)
        if len(params) != 2:
            raise TypeError(f"tool {fn.__name__} must take (ctx, args)")
        args_model = hints.get(params[1])
        if not (isinstance(args_model, type) and issubclass(args_model, BaseModel)):
            raise TypeError(f"tool {fn.__name__}: 2nd parameter must be annotated with a BaseModel")
        if requires_confirmation is None:
            needs_confirm = kind == ToolKind.WRITE
        else:
            needs_confirm = requires_confirmation
        _COLLECTED_TOOLS.append(
            ToolSpec(
                name=name or fn.__name__,
                description=inspect.cleandoc(fn.__doc__ or fn.__name__),
                kind=kind,
                args_model=args_model,
                fn=fn,
                requires_verification=requires_verification,
                requires_confirmation=needs_confirm,
                confirm_template=confirm,
            )
        )
        return fn

    return deco


def policy(tool_name: str) -> Callable[[PolicyFn], PolicyFn]:
    """Register a code-enforced policy check for a tool. Runs before the tool, never by the LLM."""

    def deco(fn: PolicyFn) -> PolicyFn:
        _COLLECTED_POLICIES.setdefault(tool_name, []).append(fn)
        return fn

    return deco


def drain_collected() -> tuple[list[ToolSpec], dict[str, list[PolicyFn]]]:
    """Return and clear everything registered by decorators since the last drain."""
    tools, pols = list(_COLLECTED_TOOLS), {k: list(v) for k, v in _COLLECTED_POLICIES.items()}
    _COLLECTED_TOOLS.clear()
    _COLLECTED_POLICIES.clear()
    return tools, pols
