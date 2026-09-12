"""Module-2 tests: plume physics, emissions, emulator, regrid, artifacts.

All tests are offline and deterministic (synthetic meteorology / direct calls).
Network-backed GFS fetching is deliberately not exercised in CI.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from aqf_delhi.config import load_domain
from aqf_delhi.domain import build_grid
from aqf_delhi.wrf.config import load_module2
from aqf_delhi.wrf.emissions import FireDetection, fire_emissions, urban_base
from aqf_delhi.wrf.emulator import MetArrays, run_emulator
from aqf_delhi.wrf.plume import pressure_height, plume_rise
from aqf_delhi.wrf.regrid import bilinear
from aqf_delhi.wrf.coupling import (
    derive_p_sfc,
    nearest_cell,
    synthetic_met,
)


def _cfg():
    return load_module2()


def _grid():
    return build_grid(load_domain())


def _sounding(stab: float = 5.5) -> dict:
    z = np.array([0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000])
    t = 298.0 - stab * (z / 1000)
    t = np.where(z > 3000, 298.0 - stab * 3.0 + 3.0 * (z - 3000) / 1000, t)
    return {"z": z, "t": t, "ws": np.full_like(z, 5.0)}


# ---------------------------------------------------------------- plume ----

def test_plume_rise_monotonic_in_frp():
    s = _sounding()
    tops = []
    for frp in (50, 200, 800, 3000):
        p = plume_rise(frp, t_sfc_k=298.0, p_sfc_hpa=985.0,
                       z_k=s["z"], t_k=s["t"], ws_k=s["ws"],
                       max_top_m=6500.0, z_step=50.0)
        assert p.z_top_m <= 6500.0
        assert p.z_top_m > 200.0
        tops.append(p.z_top_m)
    assert tops == sorted(tops), "plume top must grow with FRP"


def test_plume_stable_caps_lower_than_unstable():
    s = _sounding()
    z = s["z"]
    stable = 298.0 + 9.0 * (np.maximum(z - 400, 0) / 2600.0)
    unstable = 298.0 - 9.0 * (z / 1000.0)
    ps = plume_rise(500, t_sfc_k=298.0, p_sfc_hpa=985.0, z_k=z,
                    t_k=stable, ws_k=s["ws"], max_top_m=6500.0)
    pu = plume_rise(500, t_sfc_k=298.0, p_sfc_hpa=985.0, z_k=z,
                    t_k=unstable, ws_k=s["ws"], max_top_m=6500.0)
    assert ps.z_top_m < pu.z_top_m
    assert ps.z_top_m < 1500.0


def test_plume_mass_histogram_normalized():
    s = _sounding()
    p = plume_rise(350, t_sfc_k=298.0, p_sfc_hpa=985.0,
                   z_k=s["z"], t_k=s["t"], ws_k=s["ws"], n_layers=12)
    assert abs(p.mb_flux_per_layer.sum() - 1.0) < 1e-9
    assert (p.mb_flux_per_layer >= 0).all()
    assert len(p.layer_edges_m) == 13
    assert p.z_centre_m <= p.z_top_m + 1.0


def test_pressure_height_monotonic():
    h925 = pressure_height(925.0, 300.0)
    h700 = pressure_height(700.0, 300.0)
    assert h925 > 700.0 and h925 < 900.0
    assert h700 > h925


# -------------------------------------------------------------- regrid -----

def test_bilinear_exact_on_linear_field():
    slat = np.linspace(27.0, 33.0, 25)
    slon = np.linspace(72.0, 79.0, 29)
    LAT, LON = np.meshgrid(slat, slon, indexing="ij")
    data = 10.0 * LAT + 3.0 * LON
    tlat = np.array([28.61, 28.70, 28.30])
    tlon = np.array([77.23, 77.20, 77.05])
    got = bilinear(slat, slon, data, tlat, tlon)
    want = 10.0 * tlat + 3.0 * tlon
    assert np.allclose(got, want, atol=1e-8)


def test_bilinear_matches_known_point():
    slat = np.linspace(28.0, 29.0, 5)
    slon = np.linspace(77.0, 78.0, 5)
    LAT, LON = np.meshgrid(slat, slon, indexing="ij")
    data = 2.0 * LAT - LON
    # exact grid point
    assert abs(bilinear(slat, slon, data, [28.5], [77.5])[0] - (2 * 28.5 - 77.5)) < 1e-8


# ----------------------------------------------------------- emissions -----

def _hours(n=24, start=None):
    start = start or datetime(2026, 9, 12, 0)
    return [start + timedelta(hours=i) for i in range(n)]


def test_fire_emissions_linear_in_frp():
    cfg, grid = _cfg(), _grid()
    hours = _hours(6)
    met = synthetic_met(cfg, grid, hours, seed=2)
    fire_low = FireDetection(28.65, 77.15, frp_mw=40.0, acq=hours[0])
    fire_high = FireDetection(28.65, 77.15, frp_mw=80.0, acq=hours[0])
    e1, _, _ = fire_emissions(cfg, grid, [fire_low], hours, met.cell_at)
    e2, _, _ = fire_emissions(cfg, grid, [fire_high], hours, met.cell_at)
    assert e1.sum() > 0
    assert abs(e1.sum() * 2.0 - e2.sum()) / e2.sum() < 1e-9
    assert e2.shape == (6, grid.ny, grid.nx)


def test_urban_base_diurnal_deterministic():
    cfg, grid = _cfg(), _grid()
    hours = _hours(48)
    a = urban_base(cfg, grid, hours)
    b = urban_base(cfg, grid, hours)
    assert np.allclose(a, b)
    assert (a >= 0).all()
    # night hours emit less than morning peak
    assert a[2].sum() < a[8].sum()


# ----------------------------------------------------------- emulator -----

def test_emulator_deterministic_and_plausible():
    cfg, grid = _cfg(), _grid()
    hours = _hours(24)
    met = synthetic_met(cfg, grid, hours, seed=3)
    r1 = run_emulator(cfg, grid, met, [])
    r2 = run_emulator(cfg, grid, met, [])
    assert np.array_equal(r1.pm25_grid, r2.pm25_grid)
    f = r1.pm25_grid
    assert np.isfinite(f).all()
    assert f.min() >= 0.0
    assert f.max() < 5000.0
    assert float(f.mean()) > 10.0


def test_emulator_pm10_scales_pm25_with_ratio():
    cfg, grid = _cfg(), _grid()
    hours = _hours(24)
    met = synthetic_met(cfg, grid, hours, seed=3)
    r = run_emulator(cfg, grid, met, [])
    # same transport kernel -> PM10 is PM2.5 scaled by the coarse ratio
    assert np.allclose(r.pm10_grid, r.pm25_grid * cfg.emissions.pm10_ratio,
                       rtol=1e-9, atol=1e-6)
    # all species deterministic
    r2 = run_emulator(cfg, grid, met, [])
    assert np.array_equal(r.pm10_grid, r2.pm10_grid)
    assert np.array_equal(r.o3_grid, r2.o3_grid)


def test_o3_diurnal_photo_chemistry():
    cfg, grid = _cfg(), _grid()
    hours = _hours(24)                       # 2026-09-12 00z .. 23z
    met = synthetic_met(cfg, grid, hours, seed=5)
    r = run_emulator(cfg, grid, met, [])
    o3 = r.o3_grid
    assert np.isfinite(o3).all()
    assert (o3 >= 0).all() and o3.max() < 500.0
    day_idx = [t for t in range(24) if 5 <= t <= 15]
    night_idx = [t for t in range(24) if t < 4 or t > 20]
    assert o3[day_idx].mean() > o3[night_idx].mean() + 20.0, \
        "photochemical proxy must peak in daytime"


def test_o3_shape_matches_pm25():
    cfg, grid = _cfg(), _grid()
    hours = _hours(9)
    met = synthetic_met(cfg, grid, hours, seed=6)
    r = run_emulator(cfg, grid, met, [])
    assert r.o3_grid.shape == r.pm25_grid.shape
    assert r.pm10_obs_demo.shape == r.pm25_grid.shape
    assert r.o3_obs_demo.shape == r.pm25_grid.shape
    assert (r.pm10_obs_demo >= 0).all() and (r.o3_obs_demo >= 0).all()


def test_emulator_fire_raises_concentration_downwind():
    cfg, grid = _cfg(), _grid()
    hours = _hours(24)
    met = synthetic_met(cfg, grid, hours, seed=4)
    base = run_emulator(cfg, grid, met, [])
    # fire upwind (west) of a mid-domain cell; synthetic westerly flow advects east
    fj, fi = nearest_cell(grid, grid.domain.lat_min + grid.domain.lat_span * 0.5,
                          grid.domain.lon_min + grid.domain.lon_span * 0.30)
    fire = [FireDetection(grid.lat_nodes[fj], grid.lon_nodes[fi],
                          frp_mw=600.0, acq=hours[1])]
    with_fire = run_emulator(cfg, grid, met, fire)
    diff = np.abs(with_fire.pm25_grid - base.pm25_grid)
    assert diff.max() > 0.1, "fire must measurably raise PM2.5 somewhere"
    row = with_fire.pm25_grid[6:, fj, :]          # after fire ignition
    east = row[:, (fi + 3):].sum()
    west = row[:, :(fi - 3)].sum()
    assert east > west


# --------------------------------------------------- synthetic met + psfc --

def test_derive_p_sfc_reasonable():
    hgt = np.array([180.0, 300.0])
    t925 = np.array([24.0, 24.0])
    p = derive_p_sfc(hgt, t925)
    assert (p > 940.0).all() and (p < 1020.0).all()
    assert p[0] > p[1], "higher terrain -> lower surface pressure"


def test_nearest_cell_clamps_outside():
    cfg, grid = _cfg(), _grid()
    j, i = nearest_cell(grid, 99.0, 200.0)   # far outside domain
    assert 0 <= j < grid.ny and 0 <= i < grid.nx


def test_synthetic_met_shapes():
    cfg, grid = _cfg(), _grid()
    hours = _hours(9)
    met = synthetic_met(cfg, grid, hours)
    assert met.t2m_c.shape == (9, grid.ny, grid.nx)
    assert met.cell_at(3, 5, 5)["t2m_c"] == float(met.t2m_c[3, 5, 5])


# ------------------------------------------------------------ artifact -----

def test_coupled_run_demo_writes_artifact(tmp_path):
    cfg = load_module2()
    info = __import__("aqf_delhi.wrf.coupling", fromlist=["coupled_run"]).coupled_run(
        init=datetime(2026, 9, 12, 0),
        fhrs=[0, 3, 6, 9, 12],
        real=False,
        cfg=cfg,
        out_root=tmp_path,
    )
    run_dir = tmp_path / info["run_id"]
    assert run_dir.is_dir()
    assert (run_dir / "_DONE").exists()
    for fname in ("meta.json", "grid.parquet", "emissions_fire.parquet", "stations.parquet"):
        assert (run_dir / fname).is_file(), fname
    g = pd.read_parquet(run_dir / "grid.parquet")
    for col in ("pm25_raw", "pm25_obs_demo", "pm10_raw", "pm10_obs_demo",
                "o3_raw", "o3_obs_demo", "inversion", "pblh_m"):
        assert col in g.columns, col
    s = pd.read_parquet(run_dir / "stations.parquet")
    for col in ("station_id", "pm25_raw", "pm25_obs_demo", "pm10_raw", "o3_raw"):
        assert col in s.columns, col
    assert s["station_id"].nunique() >= 3
    assert (g["pm10_raw"] >= g["pm25_raw"]).all()
    assert (g["o3_raw"] >= 0).all()


# ------------------------------------------------------------ namelist -----

def test_namelists_written(tmp_path):
    from aqf_delhi.config import load_domain
    from aqf_delhi.domain import build_grid
    from aqf_delhi.wrf.namelist import write_namelists

    cfg = load_module2()
    grid = build_grid(load_domain())
    paths = write_namelists(tmp_path, cfg, grid)
    txt = "".join(p.read_text(encoding="utf-8") for p in paths)
    assert "chem_opt" in txt and "plumerisefire_frq" in txt
    assert "fire_emis_ef_pm25" in txt
    assert f"e_we" in txt and f"e_sn" in txt