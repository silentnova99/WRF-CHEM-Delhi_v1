"""Unit + integration tests for the WRF-CHEM DELHI V2 reduced-order model.

Run:  $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests_v2/
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from aqf_delhi.config import load_domain
from aqf_delhi.domain import build_grid
from wrf_chem_delhi.weather.engine import synthetic_weather, make_hours
from wrf_chem_delhi.forecast.engine import CoupledForecastEngine


@pytest.fixture(scope="session")
def grid():
    return build_grid(load_domain())


@pytest.fixture(scope="session")
def hours():
    return make_hours(datetime(2026, 10, 1, 12, tzinfo=timezone.utc), list(range(0, 25)))


def test_grid_shape(grid):
    assert grid.ny == 23 and grid.nx == 21


def test_synthetic_weather_shapes(grid, hours):
    ws = synthetic_weather(grid, hours, scenario="normal_winter")
    assert ws.nt == len(hours)
    for key in ("t2m_c", "rh_pct", "prmsl_hpa", "ws10_ms", "wd_deg", "pblh_m",
                "t925_c", "t850_c"):
        arr = getattr(ws, key)
        assert arr.shape == (len(hours), grid.ny, grid.nx), key
    assert np.all(ws.pblh_m > 0)


def test_synthetic_weather_deterministic(grid, hours):
    a = synthetic_weather(grid, hours, seed=11).t2m_c
    b = synthetic_weather(grid, hours, seed=11).t2m_c
    assert np.allclose(a, b)


def test_scenario_differences(grid, hours):
    ws_s = synthetic_weather(grid, hours, scenario="strong_inversion")
    ws_n = synthetic_weather(grid, hours, scenario="normal_winter")
    assert float(ws_s.ws10_ms.mean()) < float(ws_n.ws10_ms.mean())


VOID_INIT = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize("scenario,band", [
    ("normal_winter", (120, 300)),
    ("strong_inversion", (300, 800)),
    ("stubble_plume", (350, 900)),
])
def test_forecast_peak_pm25_in_scenario_band(grid, scenario, band):
    run = CoupledForecastEngine(
        grid=grid, scenario=scenario,
        init=VOID_INIT, fhrs=list(range(0, 72 + 1, 3)) + [72],
    ).run()
    series = [float(sn.pm25.mean()) for sn in run.snapshots]
    peak = max(series)
    lo, hi = band
    assert lo <= peak <= hi, (scenario, peak)


def test_forecast_run_fields(grid):
    run = CoupledForecastEngine(grid=grid, fhrs=list(range(0, 12))).run()
    assert run.run_id
    assert run.data_source.startswith("demo-synthetic")
    assert len(run.snapshots) == 12
    sn = run.snapshots[-1]
    assert sn.pm25.shape == (grid.ny, grid.nx)
    assert sn.weather.get("t2m_c_avg") is not None
    assert run.domain_mean_series("pm25")[0] >= 0


def test_feedback_ablation_knob(grid):
    base = CoupledForecastEngine(
        grid=grid, scenario="strong_inversion", fhrs=list(range(0, 25)), use_feedback=False).run()
    fbx = CoupledForecastEngine(
        grid=grid, scenario="strong_inversion", fhrs=list(range(0, 25)), use_feedback=True).run()
    pb_base = max(float(sn.pblh_m.mean()) for sn in base.snapshots)
    pb_fb = max(float(sn.pblh_m.mean()) for sn in fbx.snapshots)
    assert pb_fb <= pb_base  # feedback suppresses PBL or neutral


def test_aqi_functional():
    from wrf_chem_delhi.aqi.engine import aqi_from_concentrations, map_metrics_to_pollutants
    concs = {"pm25": 300.0, "pm10": 400.0, "o3": 60.0, "nox": 80.0, "so2": 20.0, "co": 2.0}
    aqi = aqi_from_concentrations(map_metrics_to_pollutants(concs))
    assert aqi.category in ("Very Poor", "Severe", "Poor")
    assert aqi.value > 0


def test_alerts_from_run(grid):
    from wrf_chem_delhi.aqi.alerts import alerts_from_run
    run = CoupledForecastEngine(grid=grid, fhrs=list(range(0, 25))).run()
    alerts = alerts_from_run(run)
    assert isinstance(alerts, list)


def test_metrics_compute():
    from wrf_chem_delhi.validation.metrics import compute_metrics
    y = np.array([100.0, 110.0, 120.0])
    p = np.array([102.0, 108.0, 121.0])
    m = compute_metrics(y, p)
    assert m.mae > 0 and m.rmse > 0
    assert 0.0 <= m.r2 <= 1.0
    assert not np.isnan(m.correlation)


def test_ablation_runs(grid):
    from wrf_chem_delhi.validation.metrics import run_ablation
    run = CoupledForecastEngine(grid=grid, fhrs=list(range(0, 25))).run()
    physics = np.asarray([float(sn.pm25.mean()) for sn in run.snapshots])[:, None]
    obs = physics * 0.97
    weather = np.asarray([20.0 + (i % 6) for i in range(len(run.snapshots))])[:, None]
    fire = physics * 0.01
    inv = physics * 0.001
    res = run_ablation(physics, obs, weather=weather, fire=fire, inversion=inv)
    assert set(res) == {"A", "B", "C", "D", "E", "F"}
    assert "metrics" in res["F"]


def test_aerosol_feedback(grid):
    from wrf_chem_delhi.feedback.engine import compute_feedback
    pm = np.full((grid.ny, grid.nx), 250.0)
    pbl = np.full((grid.ny, grid.nx), 500.0)
    fb = compute_feedback(pm, pbl)
    assert float(fb.pblh_adjusted.max()) <= 500.0
    assert float(fb.feedback_strength.min()) >= 0.0


def test_chemistry_shapes(grid, hours):
    from wrf_chem_delhi.chemistry.engine import run_chemistry
    ws = synthetic_weather(grid, hours, scenario="normal_winter")
    nt, ny, nx = ws.nt, ws.ny, ws.nx
    pm = np.zeros((nt, ny, nx)) + 80.0
    voc = np.zeros((nt, ny, nx)) + 1e4
    res = run_chemistry(ws, pm, pm, np.full((nt, ny, nx), 40.0), np.zeros((nt, ny, nx)),
                        voc, np.zeros((nt, ny, nx)), np.zeros((nt, ny, nx)))
    assert res.pm25.shape == (nt, ny, nx)
    assert res.secondary_pm.max() > 0


def test_transport_step(grid):
    from wrf_chem_delhi.transport.engine import TransportModel, SPECIES
    hours = make_hours(VOID_INIT, list(range(0, 3)))
    ws = synthetic_weather(grid, hours, scenario="normal_winter")
    em = {sp: np.zeros((3, grid.ny, grid.nx)) for sp in SPECIES}
    em["pm25"][:] = 1e5
    c = {"pm25": np.full((grid.ny, grid.nx), 50.0)}
    tm = TransportModel(grid)
    out = tm.step(c, ws, 1, em, pblh_m=ws.pblh_m[1])
    assert out["pm25"].shape == (grid.ny, grid.nx)
    assert float(out["pm25"].min()) >= 0.0


def test_transport_conservative_bounded(grid):
    """Without emissions, advection+deposition should not create mass."""
    from wrf_chem_delhi.transport.engine import TransportModel, SPECIES
    hours = make_hours(VOID_INIT, list(range(0, 3)))
    ws = synthetic_weather(grid, hours, scenario="normal_winter")
    em = {sp: np.zeros((3, grid.ny, grid.nx)) for sp in SPECIES}
    tm = TransportModel(grid)
    m0 = 50.0 * grid.ny * grid.nx
    c = {"pm25": np.full((grid.ny, grid.nx), 50.0)}
    for t in (1, 2):
        out = tm.step(c, ws, t, em, pblh_m=ws.pblh_m[t])
        assert float(out["pm25"].min()) >= 0.0
        assert float(out["pm25"].sum()) <= m0 + 1e-6  # deposition may remove mass
        c = out


def test_plume_and_fire(grid, hours):
    from wrf_chem_delhi.fire.engine import StubbleEngine
    from wrf_chem_delhi.plume.engine import PlumeModel
    ws = synthetic_weather(grid, hours, scenario="stubble_plume")
    fe = StubbleEngine(grid, seed=11)
    fires = fe._synthetic_fires(hours)
    assert len(fires) > 0
    eg = fe.emissions_grid(hours, fires)
    assert eg["pm25"].shape == (len(hours), grid.ny, grid.nx)
    pl = PlumeModel(grid)
    ps = pl.compute_state(hours, fires, ws)
    assert ps.concentration.shape == (len(hours), grid.ny, grid.nx)


def test_inversion_compute(grid, hours):
    from wrf_chem_delhi.inversion.engine import compute_from_weather
    ws = synthetic_weather(grid, hours, scenario="strong_inversion")
    inv = compute_from_weather(ws)
    assert inv.strength_k.shape == (len(hours), grid.ny, grid.nx)
    assert inv.trapping_idx.shape == (len(hours), grid.ny, grid.nx)


def test_api_endpoints():
    from fastapi.testclient import TestClient
    from wrf_chem_delhi.server import create_app
    client = TestClient(create_app())
    for ep in ("/api/health", "/api/current", "/api/forecast", "/api/pollution",
               "/api/weather", "/api/inversion", "/api/fires", "/api/plume",
               "/api/aqi", "/api/alerts", "/api/model-info", "/api/scenarios", "/"):
        r = client.get(ep)
        assert r.status_code == 200, ep
        if ep.startswith("/api"):
            body = r.json()
            assert "model_version" in body, ep
            assert "timestamp" in body, ep
            assert "data_source" in body, ep


def test_api_scenario_switch(grid):
    from fastapi.testclient import TestClient
    from wrf_chem_delhi.server import create_app
    client = TestClient(create_app())
    r = client.get("/api/scenario?scenario=stubble_plume")
    assert r.status_code == 200
    assert r.json()["scenario"] == "stubble_plume"


def test_data_modes_default_demo():
    from wrf_chem_delhi.data_modes.modes import DataModeResolver
    resolver = DataModeResolver(mode="demo")
    assert resolver.resolve("weather") == "demo"
    assert resolver.resolve("fires") == "demo"
    # fallback chain never crashes
    info = resolver.get_weather()
    assert info["source"].startswith("demo-synthetic")
    fi = resolver.get_fires()
    assert fi["source"] == "demo-synthetic"