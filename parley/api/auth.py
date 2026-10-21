"""API key authentication dependency for operator/dashboard endpoints."""

from __future__ import annotations

import hmac
from typing import Any

import structlog
from fastapi import HTTPException, Request

from parley.config import Settings, get_settings

logger = structlog.get_logger(__name__)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
_warned_empty_api_key = False


def _resolve_settings(request: Request) -> Settings:
    app: Any = getattr(request, "app", None)
    state: Any = getattr(app, "state", None) if app is not None else None
    settings: Settings | None = getattr(state, "settings", None) if state is not None else None
    return settings if settings is not None else get_settings()


def require_api_key(request: Request) -> None:
    """Validate Bearer API key, or restrict to loopback when PARLEY_API_KEY is unset."""
    global _warned_empty_api_key

    settings = _resolve_settings(request)
    configured_key = (settings.api_key or "").strip()

    if not configured_key:
        client = getattr(request, "client", None)
        host = getattr(client, "host", None) if client is not None else None
        if host in _LOOPBACK_HOSTS:
            if not _warned_empty_api_key:
                _warned_empty_api_key = True
                logger.warning(
                    "api_auth_loopback_only",
                    reason="empty_api_key",
                    host=host,
                )
            return
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: PARLEY_API_KEY is not set and client is not loopback",
        )

    headers: Any = getattr(request, "headers", None) or {}
    auth_header = str(headers.get("Authorization") or headers.get("authorization") or "")
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    if not hmac.compare_digest(token.encode("utf-8"), configured_key.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid API key")
