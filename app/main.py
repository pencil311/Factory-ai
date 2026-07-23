"""FactoryPilot AI — FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.config import get_settings
from app.routers.agents import router as agents_router
from app.routers.chat import router as chat_router
from app.routers.knowledge import router as knowledge_router
from app.routers.machines import router as machines_router
from app.routers.orchestrate import router as orchestrate_router
from app.routers.pdm import router as pdm_router
from app.routers.rca import router as rca_router
from app.routers.resolve import router as resolve_router
from app.routers.sensors import router as sensors_router
from app.services.ingestion import start_ingestion, stop_ingestion
from app.services.pdm import PdmArtifactsMissingError, init_pdm_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the DB, ensure indexes, and run sensor ingestion for the app's life."""
    db.connect()
    await db.create_indexes()

    try:
        await start_ingestion()
    except Exception:
        # The API must still serve reads (and explain itself) if the sensor
        # source cannot start — a broken PLC link is not a reason to be down.
        logger.exception("Sensor ingestion failed to start; API continues without it")

    try:
        init_pdm_service()
    except PdmArtifactsMissingError as exc:
        # Loud, not fatal: the registry/sensor APIs stay up, and every /pdm/*
        # request returns a 503 carrying this exact message until models exist.
        logger.error("PdM service unavailable: %s", exc)

    yield

    await stop_ingestion()
    await db.close()


settings = get_settings()

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "Industrial maintenance platform — machine, component & sensor registry. "
        "Multi-tenant: every request is scoped by the 'X-Tenant-Id' header, "
        "falling back to DEFAULT_TENANT_ID."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(machines_router)
app.include_router(resolve_router)
app.include_router(sensors_router)
app.include_router(pdm_router)
app.include_router(knowledge_router)
app.include_router(rca_router)
app.include_router(agents_router)
app.include_router(orchestrate_router)
app.include_router(chat_router)


@app.get("/", tags=["meta"], summary="Service metadata")
async def root() -> dict:
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "status": "ok",
    }


@app.get("/health", tags=["meta"], summary="Health check")
async def health() -> dict:
    """Report API liveness and database connectivity."""
    try:
        db_ok = await db.ping()
    except Exception as exc:  # pragma: no cover - surfaced as unhealthy
        return {"status": "degraded", "database": False, "error": str(exc)}
    return {"status": "ok", "database": db_ok}
