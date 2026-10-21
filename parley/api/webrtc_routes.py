from __future__ import annotations

import asyncio
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat_ai_prebuilt.frontend import PipecatPrebuiltUI

from parley.config import get_settings
from parley.core.pipeline.bot import run_call

logger = structlog.get_logger(__name__)

MAX_CONCURRENT_WEBRTC_CALLS = 3

router = APIRouter(tags=["webrtc"])

# Keep strong references to background WebRTC call tasks so they are not garbage-collected.
active_webrtc_tasks: set[asyncio.Task[Any]] = set()

# Pipecat 1.13 SmallWebRTCRequestHandler manages peer connections and SDP/ICE negotiation.
webrtc_request_handler = SmallWebRTCRequestHandler()


@router.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Redirect root URL to the prebuilt WebRTC client UI."""
    return RedirectResponse(url="/client/")


@router.post("/api/offer")
async def offer(offer_request: SmallWebRTCRequest, request: Request) -> dict[str, Any]:
    """Handle WebRTC SDP offer and spawn a background voice call pipeline."""
    settings = getattr(request.app.state, "settings", None) or get_settings()
    if settings.kill_switch:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable right now.",
        )

    if len(active_webrtc_tasks) >= MAX_CONCURRENT_WEBRTC_CALLS:
        raise HTTPException(
            status_code=429,
            detail="Too many concurrent WebRTC calls (limit: 3).",
        )

    async def webrtc_connection_callback(connection: SmallWebRTCConnection) -> None:
        transport = SmallWebRTCTransport(
            webrtc_connection=connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
        )
        call_id = getattr(connection, "pc_id", None) or f"webrtc-{uuid.uuid4().hex[:12]}"
        task = asyncio.create_task(
            run_call(
                transport,
                agent=request.app.state.agent,
                call_id=call_id,
                channel="webrtc",
                direction="inbound",
                store=getattr(request.app.state, "store", None),
            )
        )
        active_webrtc_tasks.add(task)
        task.add_done_callback(active_webrtc_tasks.discard)

    answer = await webrtc_request_handler.handle_web_request(
        request=offer_request,
        webrtc_connection_callback=webrtc_connection_callback,
    )
    return answer or {}


@router.patch("/api/offer")
async def ice_candidate(patch_request: SmallWebRTCPatchRequest) -> dict[str, str]:
    """Handle trickle ICE candidate patch requests for an active SmallWebRTC connection."""
    await webrtc_request_handler.handle_patch_request(patch_request)
    return {"status": "success"}


def mount_webrtc_ui(app: FastAPI) -> None:
    """Mount the Pipecat prebuilt WebRTC frontend at /client."""
    app.mount("/client", PipecatPrebuiltUI, name="client")
