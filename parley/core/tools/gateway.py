"""Tool gateway: schema generation, argument validation, policy checks, and confirmation guards."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import structlog
from pydantic import BaseModel, Field, ValidationError

from parley.core.brain.state import ToolCall, ToolEvent
from parley.core.tools.base import ToolContext, ToolKind, ToolSpec
from parley.core.tools.policy_engine import PolicyEngine

if TYPE_CHECKING:
    from parley.core.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)

GatewayStatus = Literal["ok", "error", "blocked", "confirmation_required"]

_AFFIRMATIVE_RE = re.compile(
    r"\b(?:yes|yeah|yep|yup|sure|correct|right|ok|okay|confirm|confirmed|"
    r"go\s+ahead|please\s+do|do\s+it|haan|haa|ji|theek\s+hai)\b",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"\b(?:no|nope|not|don't|dont|do\s+not|cancel|stop|wait|hold\s+on|wrong|never|nahi|mat)\b",
    re.IGNORECASE,
)


class _ConfirmActionArgs(BaseModel):
    confirmed: bool = Field(description="True if caller confirmed, False if caller declined")


@dataclass
class GatewayResult:
    status: GatewayStatus
    payload: dict[str, Any]
    event: ToolEvent
    control: str | None = None


def _parse_raw_dict(raw_args: str | dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(raw_args, dict):
        return raw_args, None
    if isinstance(raw_args, str):
        stripped = raw_args.strip()
        if stripped == "":
            return {}, None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            return None, str(exc)[:200]
        if not isinstance(parsed, dict):
            return None, "Expected a JSON object for tool arguments"
        return parsed, None
    return None, "Arguments must be a JSON string or dict"


class ToolGateway:
    """Executes tool calls through validation, policy enforcement, and confirmation gates."""

    CONFIRM_TOOL = "confirm_action"

    def __init__(
        self,
        registry: ToolRegistry,
        policy_engine: PolicyEngine | None = None,
    ) -> None:
        self.registry = registry
        self.policy_engine = policy_engine or PolicyEngine(registry)

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI tool schemas for every registry tool + the confirm_action tool."""
        confirm_schema = _ConfirmActionArgs.model_json_schema()
        confirm_schema.pop("title", None)
        confirm_tool_def: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.CONFIRM_TOOL,
                "description": (
                    "Call ONLY after the caller has clearly said yes or no to the "
                    "action you read back. confirmed=true executes the pending action, "
                    "false cancels it."
                ),
                "parameters": confirm_schema,
            },
        }
        return [spec.openai_schema() for spec in self.registry.all()] + [confirm_tool_def]

    def _finish(
        self,
        *,
        name: str,
        args: dict[str, Any],
        status: GatewayStatus,
        payload: dict[str, Any],
        ctx: ToolContext,
        start_t: float,
        reason: str | None = None,
        control: str | None = None,
    ) -> GatewayResult:
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        event = ToolEvent(
            name=name,
            args=args,
            status=status,
            result=payload,
            reason=reason,
            turn=ctx.state.turn_index,
            latency_ms=latency_ms,
        )
        ctx.state.tool_events.append(event)
        return GatewayResult(status=status, payload=payload, event=event, control=control)

    async def _check_and_execute(
        self,
        spec: ToolSpec,
        args: BaseModel,
        ctx: ToolContext,
        start_t: float,
        *,
        skip_confirmation: bool,
    ) -> GatewayResult:
        args_dict = args.model_dump()

        # Step 4: Policy check
        decision = await self.policy_engine.check(spec, args, ctx)
        if not decision.allowed:
            if decision.reason == "identity_not_verified":
                instruction = (
                    "Ask the caller for their order number and the phone number or email "
                    "on the order, then call verify_identity."
                )
            else:
                instruction = (
                    "Do not perform this action. Explain the reason politely and offer the "
                    "alternatives; offer a human if escalate is true."
                )
            payload: dict[str, Any] = {
                "error": "policy_blocked",
                "reason": decision.reason,
                "escalate": decision.escalate,
                "alternatives": decision.alternatives,
                "instruction": instruction,
            }
            return self._finish(
                name=spec.name,
                args=args_dict,
                status="blocked",
                payload=payload,
                reason=decision.reason,
                ctx=ctx,
                start_t=start_t,
            )

        # Step 5: Confirmation gate
        if spec.requires_confirmation and not skip_confirmation:
            confirm_prompt = (
                spec.confirm_template.format(**args_dict)
                if spec.confirm_template
                else f"run {spec.name} with {args}"
            )
            pending = ctx.state.pending_action
            if pending is not None and pending.name == spec.name and pending.args == args_dict:
                confirm_prompt = pending.confirm_prompt
            else:
                ctx.state.pending_action = ToolCall(
                    name=spec.name,
                    args=args_dict,
                    confirm_prompt=confirm_prompt,
                    created_turn=ctx.state.turn_index,
                )
            confirm_payload: dict[str, Any] = {
                "status": "confirmation_required",
                "confirm_prompt": confirm_prompt,
                "instruction": (
                    "Read this back to the caller in one short sentence and ask them to confirm. "
                    "Do not say it is done."
                ),
            }
            return self._finish(
                name=spec.name,
                args=args_dict,
                status="confirmation_required",
                payload=confirm_payload,
                ctx=ctx,
                start_t=start_t,
            )

        # Step 6: Execute tool
        try:
            result = await spec.fn(ctx, args)
        except Exception as exc:
            ctx.state.failure_counts[spec.name] = ctx.state.failure_counts.get(spec.name, 0) + 1
            err_payload: dict[str, Any] = {
                "error": "tool_failed",
                "detail": str(exc)[:200],
            }
            return self._finish(
                name=spec.name,
                args=args_dict,
                status="error",
                payload=err_payload,
                reason="tool_failed",
                ctx=ctx,
                start_t=start_t,
            )

        # Step 7: VERIFY state updates
        if spec.kind == ToolKind.VERIFY:
            if result.ok and result.verified_profile is not None:
                ctx.state.verified = True
                ctx.state.caller = result.verified_profile
            elif not result.ok and spec.name != "verify_identity":
                ctx.state.failure_counts["verify_identity"] = (
                    ctx.state.failure_counts.get("verify_identity", 0) + 1
                )

        # Step 8: Build result
        status: GatewayStatus = "ok" if result.ok else "error"
        res_payload: dict[str, Any] = {"ok": result.ok, **result.data}
        if result.error is not None:
            res_payload["error"] = result.error
        if not result.ok:
            ctx.state.failure_counts[spec.name] = ctx.state.failure_counts.get(spec.name, 0) + 1

        return self._finish(
            name=spec.name,
            args=args_dict,
            status=status,
            payload=res_payload,
            reason=result.error if not result.ok else None,
            control=result.control,
            ctx=ctx,
            start_t=start_t,
        )

    async def _handle_confirm(
        self,
        raw_args: str | dict[str, Any],
        ctx: ToolContext,
        start_t: float,
    ) -> GatewayResult:
        parsed_dict, parse_err = _parse_raw_dict(raw_args)
        if parsed_dict is None:
            ctx.state.failure_counts[self.CONFIRM_TOOL] = (
                ctx.state.failure_counts.get(self.CONFIRM_TOOL, 0) + 1
            )
            return self._finish(
                name=self.CONFIRM_TOOL,
                args={},
                status="error",
                payload={"error": "invalid_arguments", "detail": parse_err or "Invalid JSON"},
                reason="invalid_arguments",
                ctx=ctx,
                start_t=start_t,
            )

        try:
            confirm_args = _ConfirmActionArgs.model_validate(parsed_dict)
        except ValidationError as exc:
            ctx.state.failure_counts[self.CONFIRM_TOOL] = (
                ctx.state.failure_counts.get(self.CONFIRM_TOOL, 0) + 1
            )
            return self._finish(
                name=self.CONFIRM_TOOL,
                args=parsed_dict,
                status="error",
                payload={"error": "invalid_arguments", "detail": str(exc)[:200]},
                reason="invalid_arguments",
                ctx=ctx,
                start_t=start_t,
            )

        args_dict = confirm_args.model_dump()
        pending = ctx.state.pending_action
        if pending is None:
            return self._finish(
                name=self.CONFIRM_TOOL,
                args=args_dict,
                status="error",
                payload={"error": "nothing_pending"},
                reason="nothing_pending",
                ctx=ctx,
                start_t=start_t,
            )

        # 9a. Anti-self-confirmation turn index guard
        if ctx.state.turn_index <= pending.created_turn:
            return self._finish(
                name=self.CONFIRM_TOOL,
                args=args_dict,
                status="blocked",
                payload={
                    "error": "confirmation_must_come_from_caller",
                    "instruction": "Wait for the caller to answer yes or no.",
                },
                reason="confirmation_must_come_from_caller",
                ctx=ctx,
                start_t=start_t,
            )

        # 9b. Utterance affirmative/negative pattern guard when confirmed=True
        if confirm_args.confirmed:
            utterance = ctx.state.last_user_text()
            if not (
                _AFFIRMATIVE_RE.search(utterance) is not None
                and _NEGATIVE_RE.search(utterance) is None
            ):
                return self._finish(
                    name=self.CONFIRM_TOOL,
                    args=args_dict,
                    status="blocked",
                    payload={
                        "error": "caller_did_not_confirm",
                        "instruction": "Ask the caller to clearly say yes or no.",
                    },
                    reason="caller_did_not_confirm",
                    ctx=ctx,
                    start_t=start_t,
                )

        # 9c. Confirmed is False -> cancel pending
        if not confirm_args.confirmed:
            ctx.state.pending_action = None
            return self._finish(
                name=self.CONFIRM_TOOL,
                args=args_dict,
                status="ok",
                payload={"ok": True, "cancelled": True},
                ctx=ctx,
                start_t=start_t,
            )

        # 9d. Confirmed is True and guards pass -> clear pending, re-run 4 and 6-8
        ctx.state.pending_action = None
        spec = self.registry.get(pending.name)
        if spec is None:
            return self._finish(
                name=pending.name,
                args=pending.args,
                status="error",
                payload={"error": "unknown_tool", "tool": pending.name},
                reason="unknown_tool",
                ctx=ctx,
                start_t=start_t,
            )

        validated_args = spec.args_model.model_validate(pending.args)
        return await self._check_and_execute(
            spec,
            validated_args,
            ctx,
            start_t,
            skip_confirmation=True,
        )

    async def call(
        self,
        name: str,
        raw_args: str | dict[str, Any],
        ctx: ToolContext,
    ) -> GatewayResult:
        start_t = time.perf_counter()

        # Step 1: confirm_action routing
        if name == self.CONFIRM_TOOL:
            return await self._handle_confirm(raw_args, ctx, start_t)

        # Step 2: Lookup tool spec
        spec = self.registry.get(name)
        if spec is None:
            parsed_dict, _ = _parse_raw_dict(raw_args)
            return self._finish(
                name=name,
                args=parsed_dict or {},
                status="error",
                payload={"error": "unknown_tool", "tool": name},
                reason="unknown_tool",
                ctx=ctx,
                start_t=start_t,
            )

        # Step 3: Parse and validate arguments
        parsed_dict, parse_err = _parse_raw_dict(raw_args)
        if parsed_dict is None:
            ctx.state.failure_counts[name] = ctx.state.failure_counts.get(name, 0) + 1
            return self._finish(
                name=name,
                args={},
                status="error",
                payload={"error": "invalid_arguments", "detail": parse_err or "Invalid JSON"},
                reason="invalid_arguments",
                ctx=ctx,
                start_t=start_t,
            )

        try:
            validated_args = spec.args_model.model_validate(parsed_dict)
        except ValidationError as exc:
            ctx.state.failure_counts[name] = ctx.state.failure_counts.get(name, 0) + 1
            return self._finish(
                name=name,
                args=parsed_dict,
                status="error",
                payload={"error": "invalid_arguments", "detail": str(exc)[:200]},
                reason="invalid_arguments",
                ctx=ctx,
                start_t=start_t,
            )

        # Steps 4-8
        return await self._check_and_execute(
            spec,
            validated_args,
            ctx,
            start_t,
            skip_confirmation=False,
        )
