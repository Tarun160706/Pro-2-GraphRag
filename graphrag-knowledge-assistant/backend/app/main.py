"""
GraphRAG Knowledge Intelligence Assistant — FastAPI entry point.

This module creates the application, wires up lifecycle events,
and includes API routers. Start with:

    python -m uvicorn app.main:app --reload

from the backend/ directory.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api import health, documents, search, graph, query
from app.database.database import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup and shutdown logic for the application."""
    # ── Startup ──────────────────────────────────────────────
    setup_logging()
    settings.ensure_data_dirs()
    init_db()
    logger.info("Starting %s (env=%s)", settings.app_name, settings.app_env)
    logger.info("Data directory: %s", settings.data_dir)

    yield  # application is running

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="A hybrid GraphRAG knowledge assistant combining vector and graph retrieval.",
    lifespan=lifespan,
)

# CORS — permissive during development, tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(search.router)
app.include_router(graph.router)
app.include_router(query.router)

# ── Frontend Static Files ────────────────────────────────────
frontend_dir = Path(settings.frontend_dir)
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

