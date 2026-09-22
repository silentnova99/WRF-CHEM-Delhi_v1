"""VAYU-SETU live REST API (router under /api/v3).

Serves the live coupled 72 h forecast built on top of real lake observations.
Mount alongside the V2 router in `vayu_setu.web.create_app`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
from fastapi import APIRouter, HTTPException, Path, Query

from vayu_setu import live as _live
from vayu_setu.config import load_vayu_config
from vayu_setu.lake import DataLake

live_router = APIRouter(prefix="/api/v3", tags=["vayu-setu"])

MODEL_NAME = "VAYU-SETU (coupled 72 h)"


def _env():
    cfg = load_vayu_config()
    lake = DataLake(cfg)
    return cfg, lake


def _envelope(data: dict) -> dict:
    return {
        "model_name": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data,
    }


@lru_cache(maxsize=4)
def _cached_live(use_feedback: bool, _hour: str) -> dict:
    cfg, lake = _env()
    return _live.build_live(cfg, lake, use_feedback=bool(use_feedback))


def _live_payload(use_feedback: bool = True, force: bool = False) -> dict:
    if force:
        _cached_live.cache_clear()
        _live._RUN_CACHE.clear()
    hour = datetime.now(timezone.utc).strftime("%Y%m%dT%H")
    return _cached_live(use_feedback, hour)


def _stations_index(payload: dict) -> dict:
    return {s["city"].lower(): s for s in payload["stations"]}


# --------------------------------------------------------------------------- #
@live_router.get("/health")
def v3_health(force: bool = False):
    p = _live_payload(use_feedback=True, force=force)
    return _envelope({
        "status": "ok",
        "mode": "live (mode-B, real lake)",
        "run_id": p["run_id"],
        "forecast_init_utc": p["init_utc"],
        "scenario": p["scenario"],
        "use_feedback": p["use_feedback"],
        "data_as_of": p["data_as_of"],
        "anchor": p["anchor"],
        "sources": p["sources"],
        "regime_signals": p["regime_signals"],
        "n_stations": len(p["stations"]),
        "hours": len(p["series"]),
    })


@live_router.get("/current")
def v3_current():
    cfg, lake = _env()
    obs = _live.latest_observations(cfg, lake)
    return _envelope({
        "data_as_of": obs["data_as_of"].isoformat(),
        "signals": obs["signals"],
        "fires": obs["fires"],
        "stations": obs["stations"],
        "note": "Latest REAL city-hour observations in the lake (CPCB-derived extract).",
    })


@live_router.get("/forecast")
def v3_forecast(feedback: bool = Query(True), force: bool = False):
    p = _live_payload(use_feedback=feedback, force=force)
    return _envelope({
        "run_id": p["run_id"],
        "init_utc": p["init_utc"],
        "scenario": p["scenario"],
        "use_feedback": p["use_feedback"],
        "data_as_of": p["data_as_of"],
        "anchor": p["anchor"],
        "series": p["series"],
        "stations": p["stations"],
        "alerts": p["alerts"],
        "interval_note": p["interval_note"],
    })


@live_router.get("/forecast/{station_id}")
def v3_forecast_station(station_id: str, feedback: bool = Query(True)):
    p = _live_payload(use_feedback=feedback)
    st = _stations_index(p).get(station_id.lower())
    if st is None:
        raise HTTPException(status_code=404, detail=f"unknown station: {station_id}")
    return _envelope({
        "station": {k: st[k] for k in ("city", "state", "lat", "lon", "in_domain", "observed")},
        "run_id": p["run_id"],
        "init_utc": p["init_utc"],
        "scenario": p["scenario"],
        "series": st["series"],
    })


@live_router.get("/aqi")
def v3_aqi_summary():
    p = _live_payload()
    peak = max(p["series"], key=lambda s: s["aqi"])
    return _envelope({
        "run_id": p["run_id"],
        "hours": len(p["series"]),
        "peak_aqi": peak["aqi"],
        "peak_category": peak["category"],
        "peak_at": peak["time"],
        "forecast_api": "browse /api/v3/forecast for the full hourly series",
    })


@live_router.get("/aqi/{lead}")
def v3_aqi(lead: int = Path(ge=0, le=72)):
    p = _live_payload()
    lead = min(lead, len(p["series"]) - 1)
    return _envelope({**p["series"][lead], "run_id": p["run_id"], "scenario": p["scenario"]})


@live_router.get("/fires")
def v3_fires():
    p = _live_payload()
    return _envelope({"fires": p["sources"]["firms"], "run_id": p["run_id"]})


@live_router.get("/inversion")
def v3_inversion():
    p = _live_payload()
    return _envelope({
        "method": "t925-t2m lapse proxy (reduced-order) + lake inversion flag",
        "regime_signals": p["regime_signals"],
        "series": [
            {"time": s["time"], "lead_h": s["lead_h"], "trapping_idx": s["trapping_idx"]}
            for s in p["series"]
        ],
    })


@live_router.get("/alerts")
def v3_alerts():
    p = _live_payload()
    return _envelope({
        "alerts": p["alerts"],
        "run_id": p["run_id"],
        "scope": "reduced-order engine evaluation (see /current for observed status)",
    })


@live_router.get("/ablation")
def v3_ablation(force: bool = False):
    on = _live_payload(use_feedback=True, force=force)
    off = _live_payload(use_feedback=False)
    a = {s["lead_h"]: s for s in on["series"]}
    b = {s["lead_h"]: s for s in off["series"]}
    coupled = [a[h]["pm25"] for h in sorted(a)]
    uncoupled = [b[h]["pm25"] for h in sorted(b)]
    mean_c = float(np.mean(coupled))
    mean_u = float(np.mean(uncoupled))
    return _envelope({
        "run_id_on": on["run_id"],
        "run_id_off": off["run_id"],
        "scenario": on["scenario"],
        "coupled_pm25_mean_72h": round(mean_c, 2),
        "uncoupled_pm25_mean_72h": round(mean_u, 2),
        "delta_pct": round((mean_c / max(mean_u, 1e-9) - 1.0) * 100.0, 2),
        "note": "Feedback loop PM->AOD->radiation->PBL->PM: effect concentrates on high-PM hours.",
    })


_MAP_KEYS = ("pm25", "pm10", "o3", "nox", "so2", "co", "pblh_m",
             "feedback_strength", "fire_influence", "trapping_idx")


@live_router.get("/map/{key}/{lead}")
def v3_map(key: str, lead: int = Path(ge=0, le=72)):
    if key not in _MAP_KEYS:
        raise HTTPException(status_code=400, detail=f"key must be one of {_MAP_KEYS}")
    p = _live_payload()
    cfg, lake = _env()
    run, _ = _live.cached_run(cfg, lake, p["scenario"], p["use_feedback"])
    m = run.lead_map(key, lead)
    m["run_id"] = p["run_id"]
    m["scenario"] = p["scenario"]
    return _envelope(m)


@live_router.get("/model-info")
def v3_model_info():
    p = _live_payload()
    return _envelope({
        "model_name": MODEL_NAME,
        "version": p["model_version"],
        "engines": ["v2-coupled-reduced-order", "station-graph-gnn", "xgb-residual",
                    "conformal", "cpcb-aqi", "nowcast-anchor", "real-firms-exposure"],
        "capabilities": ["T+1..T+72", "coupled feedback ON/OFF", "per-station series",
                         "Indian AQI", "grid maps"],
        "disclaimer": ("Reduced-order physics-informed prototype for the SIH brief; "
                       "NOT operational WRF-Chem. Observations and FIRMS are real lake data."),
    })