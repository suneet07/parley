"""Tool registry: loads built-in and per-business tools and policies."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from parley.core.brain.state import CallerProfile
from parley.core.tools.base import PolicyFn, ToolContext, ToolSpec, drain_collected

if TYPE_CHECKING:
    from parley.config import BusinessConfig

logger = structlog.get_logger(__name__)

ALWAYS_ON_BUILTINS: frozenset[str] = frozenset(
    {"search_kb", "transfer_to_human", "end_call", "create_ticket", "opt_out"}
)


@dataclass
class _CachedModule:
    tools: list[ToolSpec]
    policies: dict[str, list[PolicyFn]]
    resources_factory: Callable[[BusinessConfig], dict[str, Any]] | None
    caller_lookup: Callable[[ToolContext, str], Awaitable[CallerProfile | None]] | None


_BUILTIN_CACHE: tuple[list[ToolSpec], dict[str, list[PolicyFn]]] | None = None
_MODULE_CACHE: dict[Path, _CachedModule] = {}


def _load_builtins() -> tuple[list[ToolSpec], dict[str, list[PolicyFn]]]:
    global _BUILTIN_CACHE
    if _BUILTIN_CACHE is not None:
        return _BUILTIN_CACHE

    already_imported = "parley.core.tools.builtins" in sys.modules
    drain_collected()
    mod = importlib.import_module("parley.core.tools.builtins")
    tools, policies = drain_collected()
    if already_imported and not tools:
        tools = list(getattr(mod, "BUILTIN_TOOLS", []))
    _BUILTIN_CACHE = (tools, policies)
    return _BUILTIN_CACHE


def _load_business_module(module_path: Path) -> _CachedModule:
    resolved = module_path.resolve()
    if resolved in _MODULE_CACHE:
        return _MODULE_CACHE[resolved]

    if not resolved.exists():
        cached = _CachedModule(
            tools=[],
            policies={},
            resources_factory=None,
            caller_lookup=None,
        )
        _MODULE_CACHE[resolved] = cached
        return cached

    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    mod_name = f"parley_business_tools_{digest}"
    spec = importlib.util.spec_from_file_location(mod_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load tools module from {resolved}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    drain_collected()
    spec.loader.exec_module(module)
    mod_tools, mod_policies = drain_collected()

    resources_fn = getattr(module, "resources", None)
    lookup_caller_fn = getattr(module, "lookup_caller", None)

    cached = _CachedModule(
        tools=mod_tools,
        policies=mod_policies,
        resources_factory=resources_fn if callable(resources_fn) else None,
        caller_lookup=lookup_caller_fn if callable(lookup_caller_fn) else None,
    )
    _MODULE_CACHE[resolved] = cached
    return cached


class ToolRegistry:
    """Registry holding enabled tool specs and policy functions for a business."""

    resources_factory: Callable[[BusinessConfig], dict[str, Any]] | None
    caller_lookup: Callable[[ToolContext, str], Awaitable[CallerProfile | None]] | None

    def __init__(
        self,
        tools: list[ToolSpec],
        policies: dict[str, list[PolicyFn]] | None = None,
    ) -> None:
        self._tools: dict[str, ToolSpec] = {t.name: t for t in tools}
        self._policies: dict[str, list[PolicyFn]] = (
            {k: list(v) for k, v in policies.items()} if policies else {}
        )
        self.resources_factory = None
        self.caller_lookup = None

    @classmethod
    def for_business(cls, business: BusinessConfig) -> ToolRegistry:
        """Build a ToolRegistry for the given BusinessConfig."""
        builtin_tools, builtin_policies = _load_builtins()
        module_path = business.resolve(business.tools_module)
        cached_mod = _load_business_module(module_path)

        available: dict[str, ToolSpec] = {t.name: t for t in builtin_tools}
        for t in cached_mod.tools:
            available[t.name] = t

        unknown = [name for name in business.tools if name not in available]
        if unknown:
            raise ValueError(f"Unknown tools in business.tools: {', '.join(unknown)}")

        enabled_names = set(business.tools) | ALWAYS_ON_BUILTINS
        selected_tools = [spec for name, spec in available.items() if name in enabled_names]

        merged_policies: dict[str, list[PolicyFn]] = {}
        for name, fns in builtin_policies.items():
            if name in enabled_names:
                merged_policies.setdefault(name, []).extend(fns)
        for name, fns in cached_mod.policies.items():
            if name in enabled_names:
                merged_policies.setdefault(name, []).extend(fns)

        registry = cls(tools=selected_tools, policies=merged_policies)
        registry.resources_factory = cached_mod.resources_factory
        registry.caller_lookup = cached_mod.caller_lookup
        return registry

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def policies_for(self, name: str) -> list[PolicyFn]:
        return list(self._policies.get(name, []))
