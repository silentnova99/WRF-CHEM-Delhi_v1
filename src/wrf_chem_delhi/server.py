"""FastAPI server for WRF-CHEM DELHI V2.

Exposes the V2 API router plus a dashboard homepage (same-session state).
Run via: python -m wrf_chem_delhi.server [--port 8001]
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from wrf_chem_delhi import MODEL_NAME, __version__
from wrf_chem_delhi.api_v2.routes import v2_router

APP_VERSION = __version__


def create_app() -> FastAPI:
    app = FastAPI(
        title="WRF-CHEM DELHI V2",
        version=APP_VERSION,
        description=MODEL_NAME,
    )
    app.include_router(v2_router)

    @app.get("/", response_class=HTMLResponse, tags=["home"])
    def home():
        return dashboard_html()

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok", "model": MODEL_NAME, "version": APP_VERSION}

    return app


def dashboard_html() -> str:
    from importlib.resources import files

    static = files("wrf_chem_delhi") / "dashboard_v2" / "static"
    try:
        return (static / "index.html").read_text(encoding="utf-8")
    except FileNotFoundError:
        return (
            "<html><body><h1>WRF-CHEM DELHI V2</h1><p>API running. "
            "Dashboard assets pending install.</p></body></html>"
        )


app = create_app()


def serve(port: int = 8001, host: str = "127.0.0.1"):
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="wrf_chem_delhi.server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8001)))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    serve(port=args.port, host=args.host)