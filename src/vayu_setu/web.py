"""VAYU-SETU live web server.

Serves:
  /              dashboard (V2 page plus VAYU-SETU status card)
  /api/*         V2 scenarios + problems (wrf_chem_delhi.api_v2)
  /api/v3/*      VAYU-SETU LIVE (real lake observations + coupled 72 h forecast)

Run:  python -m vayu_setu.web [--port 8080] [--host 127.0.0.1]
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from vayu_setu.api_live import live_router
from wrf_chem_delhi import MODEL_NAME, __version__
from wrf_chem_delhi.api_v2.routes import v2_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="VAYU-SETU + WRF-CHEM DELHI V2",
        version=__version__,
        description="Coupled 72 h Delhi NCR air-quality forecast on real data.",
    )
    app.include_router(v2_router)
    app.include_router(live_router)

    @app.get("/", response_class=HTMLResponse, tags=["home"])
    def home():
        # Government-use dashboard (full station detail, live /api/v3 data).
        html = _gov_dashboard_html()
        if html:
            return html
        return _home_html()

    @app.get("/v2", response_class=HTMLResponse, include_in_schema=False)
    def v2_home():
        # Legacy V2 problem-statement dashboard (archive).
        return _home_html()

    @app.get("/health", tags=["health"])
    def health():
        return {"status": "ok", "model": MODEL_NAME, "version": __version__,
                "live_api": "/api/v3/health"}

    return app


def _gov_dashboard_html() -> str:
    """Government-use console at / (self-contained: CSS/JS inline, no CDNs)."""
    import importlib.resources

    try:
        return (importlib.resources.files("vayu_setu") / "dashboard" / "static" / "index.html"
                ).read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return ""


def _home_html() -> str:
    try:
        from importlib.resources import files
        static = files("wrf_chem_delhi") / "dashboard_v2" / "static"
        html = (static / "index.html").read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        html = "<html><body><h1>VAYU-SETU</h1><p>API running.</p></body></html>"
    card = """
<div style="border:1px solid #2e7d32;border-radius:8px;padding:12px 16px;margin:14px 0;
            background:#f1f8e9;font-family:system-ui">
  <h3 style="margin:0 0 6px">VAYU-SETU live</h3>
  <ul style="margin:4px 0">
    <li><a href="/api/v3/health">/api/v3/health</a></li>
    <li><a href="/api/v3/current">/api/v3/current</a> (latest real observations)</li>
    <li><a href="/api/v3/forecast">/api/v3/forecast</a> (72 h coupled, PM2.5/PM10/O3 + AQI)</li>
    <li><a href="/api/v3/forecast/Delhi">/api/v3/forecast/Delhi</a></li>
    <li><a href="/api/v3/ablation">/api/v3/ablation</a></li>
    <li><a href="/docs">OpenAPI docs</a></li>
  </ul>
</div>"""
    marker = "</body>"
    if marker in html:
        return html.replace(marker, card + marker, 1)
    return html + card


app = create_app()


def serve(port: int = 8080, host: str = "127.0.0.1"):
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="vayu_setu.web")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    serve(port=args.port, host=args.host)