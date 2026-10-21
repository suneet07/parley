"""OpenTelemetry tracing setup wired into Pipecat's built-in tracing."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from parley.config import Settings

logger = structlog.get_logger(__name__)

_configured_key: tuple[str, tuple[tuple[str, str], ...]] | None = None
_tracing_enabled: bool = False


def setup(settings: Settings) -> bool:
    """Configure OpenTelemetry tracing via Pipecat if OTLP or Langfuse is configured.

    Idempotent: returns True if tracing is active, False otherwise.
    """
    global _configured_key, _tracing_enabled

    otlp_endpoint = (settings.otel_exporter_otlp_endpoint or "").strip()
    lf_pub = (settings.langfuse_public_key or "").strip()
    lf_sec = (settings.langfuse_secret_key or "").strip()

    endpoint = ""
    headers: dict[str, str] | None = None

    if otlp_endpoint:
        endpoint = otlp_endpoint
    elif lf_pub and lf_sec:
        host = (settings.langfuse_host or "https://cloud.langfuse.com").rstrip("/")
        # OTLPSpanExporter uses an explicit endpoint verbatim, so include the traces path.
        endpoint = f"{host}/api/public/otel/v1/traces"
        token = base64.b64encode(f"{lf_pub}:{lf_sec}".encode()).decode("ascii")
        headers = {"Authorization": f"Basic {token}"}
    else:
        return False

    key = (endpoint, tuple(sorted((headers or {}).items())))
    if _configured_key == key and _tracing_enabled:
        return True

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from pipecat.utils.tracing.setup import setup_tracing
    except ImportError:
        logger.info("tracing_unavailable", reason="opentelemetry_not_installed")
        return False

    try:
        exporter = (
            OTLPSpanExporter(endpoint=endpoint, headers=headers)
            if headers is not None
            else OTLPSpanExporter(endpoint=endpoint)
        )
        ok = bool(setup_tracing(service_name="parley", exporter=exporter))
    except Exception as exc:
        logger.info("tracing_setup_failed", error=str(exc))
        return False

    _configured_key = key
    _tracing_enabled = ok
    return ok


def is_enabled() -> bool:
    """Return True if tracing has been enabled in this process."""
    return _tracing_enabled


def _reset_for_tests() -> None:
    global _configured_key, _tracing_enabled
    _configured_key = None
    _tracing_enabled = False
