"""Deterministic policy enforcement engine."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from parley.core.tools.base import PolicyDecision, ToolContext, ToolSpec

if TYPE_CHECKING:
    from parley.core.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


class PolicyEngine:
    """Evaluates verification requirements and registered code policies for a tool call."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def check(self, spec: ToolSpec, args: BaseModel, ctx: ToolContext) -> PolicyDecision:
        if spec.requires_verification and not ctx.state.verified:
            return PolicyDecision(False, "identity_not_verified")

        for policy_fn in self.registry.policies_for(spec.name):
            outcome = policy_fn(ctx, args)
            decision = await outcome if inspect.isawaitable(outcome) else outcome
            if not decision.allowed:
                return decision

        return PolicyDecision(True)
