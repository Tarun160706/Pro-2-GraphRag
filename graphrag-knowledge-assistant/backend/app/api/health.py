"""
Health-check endpoint.

Provides a quick way to verify the API is running and reachable.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(tags=["System"])


@router.get("/health")
async def health_check() -> dict:
    """Return service health status."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": "0.1.0",
        "environment": settings.app_env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
