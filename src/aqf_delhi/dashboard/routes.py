"""Module-5: 3D WebGIS dashboard routes.

Serves the single-page dashboard (MapLibre GL front-end) and an aggregated
state payload assembled by :class:`~aqf_delhi.dashboard.state.DashboardState`
from the latest coupled / forecast / fire artifacts.
"""

from __future__ import annotations

from importlib.resources import files
from typing import Callable

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from aqf_delhi.dashboard.state import DashboardState

STATIC_ROOT = files("aqf_delhi.dashboard") / "static"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".png": "image/png",
}

dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _state_builder(request: Request) -> Callable[[], dict]:
    """Cached factory for the DashboardState bound to the request's store."""
    store = request.app.state.store
    state = DashboardState(store.root)
    return state.build


@dashboard_router.get("/state", response_model=dict)
def dashboard_state(request: Request) -> dict:
    cache = request.app.state.cache
    build = _state_builder(request)
    data = cache.get("dash.state")
    if data is None:
        grid = request.app.state.grid
        data = build(grid)
        cache.set("dash.state", data, ttl=10)
    return data


@dashboard_router.get("", response_class=HTMLResponse)
@dashboard_router.get("/", response_class=HTMLResponse)
def dashboard_index() -> HTMLResponse:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


def _static(name: str) -> Response:
    path = STATIC_ROOT / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"asset not found: {name}")
    ct = CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return Response(content=path.read_bytes(), media_type=ct)


@dashboard_router.get("/app.js")
def dashboard_app_js() -> Response:
    return _static("app.js")


@dashboard_router.get("/styles.css")
def dashboard_styles() -> Response:
    return _static("styles.css")


@dashboard_router.get("/healthz")
def dashboard_healthz() -> dict:
    return {"ok": True, "module": "5"}