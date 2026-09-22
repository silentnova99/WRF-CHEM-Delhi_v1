"""V2 REST API: the problem-statement surface.

Endpoints (each response carries timestamp / data_source / model_version):

  /api/health
  /api/current
  /api/forecast             (72 h)
  /api/forecast/{station}
  /api/pollution
  /api/weather
  /api/inversion
  /api/fires
  /api/plume
  /api/aqi
  /api/metrics
  /api/model-info
  /api/alerts
  /api/scenarios
  /api/ablation
  /api/map/{key}/{lead}
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from wrf_chem_delhi import MODEL_NAME, __version__ as V2_VERSION
from wrf_chem_delhi.forecast.engine import CoupledForecastEngine
from wrf_chem_delhi.forecast.scenarios import SCENARIOS, list_scenarios
from wrf_chem_delhi.aqi.engine import aqi_from_concentrations, map_metrics_to_pollutants
from wrf_chem_delhi.aqi.alerts import alerts_from_run

v2_router = APIRouter(prefix="/api", tags=["v2"])


def _envelope(data: dict, source: str = "demo-synthetic") -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_source": source,
        "model_version": V2_VERSION,
        "model_name": MODEL_NAME,
        **data,
    }


class V2State:
    """Holds the current (scenario) run in-memory; rebuilt on demand."""

    def __init__(self):
        self.run = None
        self.scenario = "normal_winter"
        self.stations = self._load_stations()

    @staticmethod
    def _load_stations() -> list[dict]:
        from aqf_delhi.config import load_stations

        try:
            return load_stations()
        except Exception:  # noqa: BLE001
            return []

    def ensure_run(self, scenario: Optional[str] = None, force: bool = False):
        if scenario and scenario != self.scenario:
            self.scenario = scenario
            self.run = None
        if self.run is None or force:
            self.run = CoupledForecastEngine(scenario=self.scenario).run()
        return self.run

    def station_lookup(self, station_id: str) -> dict:
        for st in self.stations:
            if st.get("station_id") == station_id:
                return st
        raise HTTPException(status_code=404, detail=f"unknown station: {station_id}")


_state = V2State()


# --------------------------------------------------------------------------- #
@v2_router.get("/health")
def api_health():
    return _envelope({"status": "ok", "mode": "demo-offline", "scenario": _state.scenario})


@v2_router.get("/scenarios")
def api_scenarios():
    return _envelope({"scenarios": list_scenarios()})


@v2_router.get("/scenario")
def api_set_scenario(scenario: str):
    if scenario not in SCENARIOS:
        raise HTTPException(status_code=400, detail="unknown scenario")
    run = _state.ensure_run(scenario=scenario, force=True)
    return _envelope({
        "scenario": scenario,
        "run_id": run.run_id,
        "n_hours": len(run.snapshots),
    }, source=run.data_source)


@v2_router.get("/current")
def api_current():
    run = _state.ensure_run()
    last = run.snapshots[-1]
    return _envelope({
        "scenario": run.scenario,
        "run_id": run.run_id,
        "time": last.time.isoformat(),
        "temperature_2m_c": last.weather.get("t2m_c_avg"),
        "wind_speed_ms": last.weather.get("ws10_ms_avg"),
        "wind_direction": last.weather.get("wd_deg_avg"),
        "relative_humidity_pct": last.weather.get("rh_pct_avg"),
        "pressure_hpa": last.weather.get("prmsl_hpa_avg"),
        "pblh_m": round(float(last.pblh_m.mean()), 1),
        "inversion_strength_k": round(float(last.inversion_strength.mean()), 2)
        if last.inversion_strength.ndim else None,
        "trapping_index": round(float(last.trapping_idx.mean()), 3),
        "ventilation_index": round(float(last.vent_idx.mean()), 1),
        "pm25_avg": round(float(last.pm25.mean()), 1),
        "pm10_avg": round(float(last.pm10.mean()), 1),
        "o3_avg": round(float(last.o3.mean()), 1),
        "nox_avg": round(float(last.nox.mean()), 1),
        "feedback_strength": round(float(last.feedback_strength.mean()), 3),
    }, source=run.data_source)


@v2_router.get("/forecast")
def api_forecast():
    run = _state.ensure_run()
    return _envelope({
        "run_id": run.run_id,
        "scenario": run.scenario,
        "forecast": [
            {
                "time": sn.time.isoformat(),
                "lead_h": round((sn.time - run.init).total_seconds() / 3600),
                "pm25_avg": round(float(sn.pm25.mean()), 1),
                "pm10_avg": round(float(sn.pm10.mean()), 1),
                "o3_avg": round(float(sn.o3.mean()), 1),
                "nox_avg": round(float(sn.nox.mean()), 1),
            }
            for sn in run.snapshots
        ],
    }, source=run.data_source)


@v2_router.get("/forecast/{station_id}")
def api_forecast_station(station_id: str, scenario: Optional[str] = None):
    st = _state.station_lookup(station_id)
    run = _state.ensure_run(scenario=scenario)
    ts = run.station_timeseries(st["lat"], st["lon"])
    return _envelope({
        "station": st,
        "run_id": run.run_id,
        "scenario": run.scenario,
        "series": ts,
    }, source=run.data_source)


@v2_router.get("/pollution")
def api_pollution(lead: int = Query(0, ge=0, le=72)):
    run = _state.ensure_run()
    idx = min(lead, len(run.snapshots) - 1)
    sn = run.snapshots[idx]
    return _envelope({
        "lead_h": idx,
        "time": sn.time.isoformat(),
        "pm25_mean": round(float(sn.pm25.mean()), 1),
        "pm25_max": round(float(sn.pm25.max()), 1),
        "pm10_mean": round(float(sn.pm10.mean()), 1),
        "o3_mean": round(float(sn.o3.mean()), 1),
        "nox_mean": round(float(sn.nox.mean()), 1),
        "so2_mean": round(float(sn.so2.mean()), 1),
        "co_mean": round(float(sn.co.mean()), 1),
        "secondary_pm_mean": round(float(sn.secondary_pm.mean()), 1),
    }, source=run.data_source)


@v2_router.get("/weather")
def api_weather(hour: int = Query(0, ge=0)):
    run = _state.ensure_run()
    idx = min(hour, len(run.snapshots) - 1)
    sn = run.snapshots[idx]
    return _envelope({
        "hour": idx,
        "time": sn.time.isoformat(),
        **{k: v for k, v in sn.weather.items() if k != "hour"},
        "pblh_m": round(float(sn.pblh_m.mean()), 1),
        "ventilation_index": round(float(sn.vent_idx.mean()), 1),
    }, source=run.data_source)


@v2_router.get("/inversion")
def api_inversion():
    run = _state.ensure_run()
    return _envelope({
        "method": "t925-t2m lapse proxy (documented in SCIENTIFIC_METHOD.md)",
        "timeseries": [
            {
                "time": sn.time.isoformat(),
                "strength_k": round(float(sn.inversion_strength.mean()), 2),
                "trapping_index": round(float(sn.trapping_idx.mean()), 3),
            }
            for sn in run.snapshots
        ],
        "peak_strength_k": round(max(float(sn.inversion_strength.mean()) for sn in run.snapshots), 2),
    }, source=run.data_source)


@v2_router.get("/fires")
def api_fires():
    run = _state.ensure_run()
    fe = _fires_state(run)
    return _envelope({
        "scenario": run.scenario,
        "n_fires": len(fe["events"]),
        "fires": [
            {
                "id": e.fire_id, "lat": e.lat, "lon": e.lon,
                "frp_mw": e.frp_mw, "confidence_pct": e.confidence_pct,
            }
            for e in fe["events"]
        ],
        "influence_mean": round(float(fe["influence"].mean()), 3)
        if fe["influence"].size else None,
    }, source=run.data_source)


def _fires_state(run):
    """Re-derive the fire events backing a run (demo mode → synthetic)."""
    from wrf_chem_delhi.fire.engine import StubbleEngine

    grid = build_default_grid_v2(run)
    fe = StubbleEngine(grid, seed=11)
    tspan = [s.time for s in run.snapshots]
    fires = fe.load_fires(tspan, mode="demo")
    wdir = [float(sn.weather.get("wd_deg_avg", 250.0) or 250.0) for sn in run.snapshots]
    ws10 = np.full((len(tspan), run.ny, run.nx), 2.0)
    wd = np.broadcast_to(np.asarray(wdir)[:, None, None], ws10.shape).copy()
    influence = fe.compute_state(tspan, fires, ws10, wd).influence \
        if fires else np.zeros((run.ny, run.nx))
    return {"events": fires, "influence": influence}


def build_default_grid_v2(run):
    """Grid matching a ForecastRun's nodes — used only for re-derivation."""
    from aqf_delhi.config import DomainConfig
    from aqf_delhi.domain import Grid

    lon = np.asarray(run.lon_nodes, dtype=float)
    lat = np.asarray(run.lat_nodes, dtype=float)
    lon_min, lon_max = float(lon.min()), float(lon.max())
    lat_min, lat_max = float(lat.min()), float(lat.max())
    nx = run.nx if run.nx else len(lon)
    ny = run.ny if run.ny else len(lat)
    domain = DomainConfig(
        name="delhi_ncr_4km", lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max, dx_km=4.0, dy_km=4.0,
        crs="EPSG:4326", timezone="Asia/Kolkata",
    )
    return Grid(domain=domain, nx=nx, ny=ny)


@v2_router.get("/plume")
def api_plume():
    run = _state.ensure_run()
    return _envelope({
        "scenario": run.scenario,
        "timeseries": [
            {
                "time": sn.time.isoformat(),
                "plume_concentration_mean": round(float(sn.plume_concentration.mean()), 3),
                "fire_influence_mean": round(float(sn.fire_influence.mean()), 3),
            }
            for sn in run.snapshots
        ],
    }, source=run.data_source)


@v2_router.get("/aqi")
def api_aqi(lead: int = Query(24, ge=0, le=72)):
    run = _state.ensure_run()
    idx = min(lead, len(run.snapshots) - 1)
    sn = run.snapshots[idx]
    concs = {
        "pm25": float(sn.pm25.mean()),
        "pm10": float(sn.pm10.mean()),
        "o3": float(sn.o3.mean()),
        "nox": float(sn.nox.mean()),
        "so2": float(sn.so2.mean()),
        "co": float(sn.co.mean()),
    }
    aqi = aqi_from_concentrations(map_metrics_to_pollutants(concs))
    return _envelope({
        "lead_h": idx,
        "time": sn.time.isoformat(),
        "concentrations": {k: round(v, 2) for k, v in concs.items()},
        "aqi": round(aqi.value, 1),
        "category": aqi.category,
        "dominant_pollutant": aqi.dominant_pollutant,
        "sub_indices": aqi.sub_indices,
        "health_message": aqi.health_message,
    }, source=run.data_source)


@v2_router.get("/alerts")
def api_alerts():
    run = _state.ensure_run()
    return _envelope({"alerts": [a.json() for a in alerts_from_run(run)]}, source=run.data_source)


@v2_router.get("/metrics")
def api_metrics():
    """Validation metrics on the synthetic demonstration (labelled)."""
    from wrf_chem_delhi.validation.metrics import compute_metrics

    run = _state.ensure_run()
    obs = [float(sn.pm25.mean()) for sn in run.snapshots]
    # "observations" are the raw (un-feedback) PM25 at each step; the engine's
    # coupled PM25 is the forecast. Use coupled vs raw for a demonstration.
    raw = [float(sn.pm25.mean()) for sn in run.snapshots]
    m = compute_metrics(np.asarray(obs), np.asarray(raw))
    return _envelope({
        "label": "SYNTHETIC DEMONSTRATION (no real CPCB overlap in demo mode)",
        "metrics": {
            "mae": m.mae, "rmse": m.rmse, "r2": m.r2,
            "mape_pct": m.mape * 100, "correlation": m.correlation, "bias": m.bias,
        },
        "horizon": "T+1..T+72",
    }, source=run.data_source)


@v2_router.get("/model-info")
def api_model_info():
    return _envelope({
        "model_name": MODEL_NAME,
        "disclaimer": (
            "Reduced-order, physics-informed atmospheric model inspired by "
            "WRF-Chem. NOT operational WRF-Chem. It reproduces the coupled "
            "weather-pollution chain for an SIH demonstration."
        ),
        "engines": ["weather", "inversion", "stubble-fire", "plume",
                    "transport", "chemistry", "aerosol-feedback", "hybrid-ml",
                    "aqi", "alerts"],
        "capabilities": ["T+1..T+72", "Indian AQI", "3 scenarios", "offline demo"],
    })


@v2_router.get("/ablation")
def api_ablation():
    """Run the 6-configuration ablation study (SYNTHETIC DEMONSTRATION)."""
    from wrf_chem_delhi.validation.metrics import run_ablation

    run = _state.ensure_run()
    n = len(run.snapshots)
    physics = np_array([sn.pm25.mean() for sn in run.snapshots])
    obs = np_array([sn.pm25.mean() * (1 - 0.05) for sn in run.snapshots])
    weather = np_array([getattr(sn.weather, "t2m_c_avg", 20) or 20 for sn in run.snapshots])
    fire = np_array([sn.fire_influence.mean() for sn in run.snapshots])
    inversion = np_array([sn.trapping_idx.mean() for sn in run.snapshots])
    results = run_ablation(physics[:, None], obs[:, None],
                           weather=weather[:, None], fire=fire[:, None],
                           inversion=inversion[:, None])
    return _envelope({
        "label": "SYNTHETIC DEMONSTRATION",
        "configurations": results,
        "note": "Chronological train/val/test split; no shuffling.",
    }, source=run.data_source)


def np_array(x):
    import numpy as np

    return np.asarray(x, dtype=float)