"""FastAPI application factory for the Module-4 geospatial API.

Defaults resolve from the repository (configs + ``data/``); override with
``root=`` and dependencies for tests, and via environment variables for
deployment:

    AQF_DATA_ROOT   data directory (default: <repo>/data)
    AQF_REDIS_URL   optional redis:// URL for the cache backend
    AQF_CACHE_TTL   default cache TTL seconds (default 30)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI

from aqf_delhi.api import routes as R
from aqf_delhi.api.cache import build_cache
from aqf_delhi.api.store import ApiStore
from aqf_delhi.config import PROJECT_ROOT
from aqf_delhi.dashboard import dashboard_router
from aqf_delhi.dashboard.home_routes import home_router

logger = logging.getLogger(__name__)

TAGS = [
    {"name": "health", "description": "Service + store health"},
    {"name": "domain", "description": "Study domain geometry & station registry"},
    {"name": "observations", "description": "CPCB ground observations"},
    {"name": "fires", "description": "NASA FIRMS active fires"},
    {"name": "gfs", "description": "GFS asset manifests"},
    {"name": "forecasts", "description": "Module-3 bias-corrected forecasts"},
    {"name": "dashboard", "description": "Module-5 3D WebGIS dashboard"},
    {"name": "home", "description": "Unified single-page homepage (map + live panels)"},
]


def create_app(
    *,
    root: str | Path | None = None,
    store: ApiStore | None = None,
    cache=None,
    cache_ttl: int = 30,
    title: str = "Delhi NCR Air-Quality Forecast API",
) -> FastAPI:
    root = Path(root or os.environ.get("AQF_DATA_ROOT", PROJECT_ROOT / "data"))
    store = store or ApiStore(root)
    cache = cache or build_cache(os.environ.get("AQF_REDIS_URL"), ttl=cache_ttl)
    if not isinstance(cache_ttl, int):
        try:
            cache_ttl = int(os.environ.get("AQF_CACHE_TTL", cache_ttl))
        except (TypeError, ValueError):
            cache_ttl = 30

    app = FastAPI(
        title=title,
        version="0.4.0",
        description="Geospatial API over the ingestion store and bias-corrected forecasts.",
        openapi_tags=TAGS,
    )
    app.state.store = store
    app.state.cache = cache
    app.state.cache_ttl = cache_ttl

    from aqf_delhi.config import load_domain, load_stations
    from aqf_delhi.domain import build_grid

    app.state.domain = load_domain()
    app.state.grid = build_grid(app.state.domain)
    app.state.stations = load_stations()

    for router in (
        home_router,
        R.health_router,
        R.domain_router,
        R.obs_router,
        R.fires_router,
        R.gfs_router,
        R.fc_router,
        dashboard_router,
    ):
        app.include_router(router)

    logger.info(
        "API ready: root=%s cache=%s grid=%s",
        store.root, cache.name(), app.state.grid.describe(),
    )
    return app


def default_app() -> FastAPI:
    return create_app()