"""Unified homepage: one page combining map + live CPCB + forecast + health.

Served at the root `/` so the whole system is visible from a single URL.
"""

from __future__ import annotations

from importlib.resources import files

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response

STATIC_ROOT = files("aqf_delhi.dashboard") / "static"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}

home_router = APIRouter(tags=["home"])


@home_router.get("/", response_class=HTMLResponse, summary="Unified homepage")
def home_index() -> HTMLResponse:
    html = (STATIC_ROOT / "home.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@home_router.get("/home.js", response_class=Response)
def home_js() -> Response:
    return _static("home.js")


@home_router.get("/home.css", response_class=Response)
def home_css() -> Response:
    return _static("home.css")


def _static(name: str) -> Response:
    path = STATIC_ROOT / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"asset not found: {name}")
    ct = CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return Response(content=path.read_bytes(), media_type=ct)