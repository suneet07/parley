from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from parley.api import dashboard_routes, live, outbound_routes, twilio_routes, webrtc_routes
from parley.api.auth import require_api_key
from parley.api.store import CallStore
from parley.config import get_settings, load_business
from parley.core.outbound.queue import OutboundQueue
from parley.core.outbound.store import OutboundStore
from parley.core.outbound.worker import build_outbound_queue
from parley.core.telemetry.tracing import setup as setup_tracing

try:
    from parley.core.brain import Agent
except ImportError:  # pragma: no cover
    Agent = Any  # type: ignore[misc,assignment]

logger = structlog.get_logger(__name__)


def create_app(
    *,
    agent: Agent | None = None,
    store: CallStore | None = None,
    outbound_queue: OutboundQueue | None = None,
) -> FastAPI:
    """Create and configure the Parley FastAPI application."""
    initial_settings = get_settings()
    initial_business = load_business(initial_settings.business_config)
    owns_store = store is None
    owns_outbound_store = outbound_queue is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = getattr(app.state, "settings", None) or get_settings()
        business = getattr(app.state, "business", None) or load_business(settings.business_config)
        app.state.settings = settings
        app.state.business = business
        setup_tracing(settings)

        if getattr(app.state, "agent", None) is None:
            from parley.core.brain import Agent as BrainAgent

            app.state.agent = await BrainAgent.create(settings, business)

        if getattr(app.state, "store", None) is None:
            app.state.store = CallStore(settings.database_url)

        if getattr(app.state, "outbound_store", None) is None:
            app.state.outbound_store = OutboundStore(settings.database_url)

        if getattr(app.state, "outbound_queue", None) is None:
            app.state.outbound_queue = build_outbound_queue(
                settings=settings,
                store=app.state.outbound_store,
            )

        await app.state.store.init()
        await app.state.outbound_store.init()

        active_agent = getattr(app.state, "agent", None)
        if (
            active_agent is not None
            and hasattr(active_agent, "resources")
            and isinstance(active_agent.resources, dict)
        ):
            active_agent.resources["outbound_store"] = app.state.outbound_store

        logger.info("parley_server_started", business=business.business)
        try:
            yield
        finally:
            if owns_store and getattr(app.state, "store", None) is not None:
                await app.state.store.close()
            if owns_outbound_store and getattr(app.state, "outbound_store", None) is not None:
                await app.state.outbound_store.close()

    app = FastAPI(title="Parley Voice Agent", lifespan=lifespan)

    # Populate state immediately so TestClient works even without entering lifespan context.
    app.state.settings = initial_settings
    app.state.business = initial_business
    app.state.agent = agent
    app.state.store = store or CallStore(initial_settings.database_url)
    if outbound_queue is not None:
        app.state.outbound_queue = outbound_queue
        app.state.outbound_store = outbound_queue.store
        app.state.outbound_queue_injected = True
    else:
        db_url = store.database_url if store is not None else initial_settings.database_url
        default_outbound_store = OutboundStore(db_url)
        app.state.outbound_store = default_outbound_store
        app.state.outbound_queue = build_outbound_queue(
            settings=initial_settings,
            store=default_outbound_store,
        )
        app.state.outbound_queue_injected = False

    if agent is not None and hasattr(agent, "resources") and isinstance(agent.resources, dict):
        agent.resources["outbound_store"] = app.state.outbound_store

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        business = getattr(request.app.state, "business", initial_business)
        return {"ok": True, "business": business.business}

    @app.get("/api/calls", dependencies=[Depends(require_api_key)])
    async def list_calls(
        request: Request,
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[dict[str, Any]]:
        settings = getattr(request.app.state, "settings", initial_settings)
        call_store: CallStore = request.app.state.store
        await call_store.init()
        return await call_store.list_calls(
            tenant_id=settings.tenant_id,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/calls/{call_id}", dependencies=[Depends(require_api_key)])
    async def get_call(call_id: str, request: Request) -> dict[str, Any]:
        settings = getattr(request.app.state, "settings", initial_settings)
        call_store: CallStore = request.app.state.store
        await call_store.init()
        record = await call_store.get_call(tenant_id=settings.tenant_id, call_id=call_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Call not found")
        return record

    app.include_router(twilio_routes.router)
    app.include_router(webrtc_routes.router)
    app.include_router(dashboard_routes.router)
    app.include_router(outbound_routes.router)
    app.include_router(live.router)
    webrtc_routes.mount_webrtc_ui(app)

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the Uvicorn server hosting the Parley FastAPI app."""
    uvicorn.run(create_app(), host=host, port=port)
