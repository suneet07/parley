from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from parley.api import twilio_routes, webrtc_routes
from parley.api.store import CallStore
from parley.config import get_settings, load_business

try:
    from parley.core.brain import Agent
except ImportError:  # pragma: no cover
    Agent = Any  # type: ignore[misc,assignment]

logger = structlog.get_logger(__name__)


def create_app(
    *,
    agent: Agent | None = None,
    store: CallStore | None = None,
) -> FastAPI:
    """Create and configure the Parley FastAPI application."""
    initial_settings = get_settings()
    initial_business = load_business(initial_settings.business_config)
    owns_store = store is None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = getattr(app.state, "settings", None) or get_settings()
        business = getattr(app.state, "business", None) or load_business(settings.business_config)
        app.state.settings = settings
        app.state.business = business

        if getattr(app.state, "agent", None) is None:
            from parley.core.brain import Agent as BrainAgent

            app.state.agent = await BrainAgent.create(settings, business)

        if getattr(app.state, "store", None) is None:
            app.state.store = CallStore(settings.database_url)

        await app.state.store.init()
        logger.info("parley_server_started", business=business.business)
        try:
            yield
        finally:
            if owns_store and getattr(app.state, "store", None) is not None:
                await app.state.store.close()

    app = FastAPI(title="Parley Voice Agent", lifespan=lifespan)

    # Populate state immediately so TestClient works even without entering lifespan context.
    app.state.settings = initial_settings
    app.state.business = initial_business
    app.state.agent = agent
    app.state.store = store or CallStore(initial_settings.database_url)

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

    @app.get("/api/calls")
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

    @app.get("/api/calls/{call_id}")
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
    webrtc_routes.mount_webrtc_ui(app)

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the Uvicorn server hosting the Parley FastAPI app."""
    uvicorn.run(create_app(), host=host, port=port)
