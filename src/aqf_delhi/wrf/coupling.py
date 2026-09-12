"""Module-2 orchestrator: GFS subset -> met fields -> coupled emulator run.

Provides the public entry points ``coupled_run`` (real or synthetic meteorology)
plus met-array builders and station sampling used by the CLI and tests.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from aqf_delhi.config import load_domain
from aqf_delhi.domain import Grid, build_grid
from aqf_delhi.wrf.config import Module2Config, load_module2
from aqf_delhi.wrf.emissions import FireDetection
from aqf_delhi.wrf.emulator import CoupledResult, MetArrays, _wd, run_emulator
from aqf_delhi.wrf.grib import GfsSubset, decode_grib, fetch_subset
from aqf_delhi.wrf.regrid import subset_to_grid

R_SPEC = 287.05
G_CONST = 9.81
Z925_STD = R_SPEC * 288.15 / G_CONST * math.log(1013.25 / 925.0)   # ~769 m


# ---------------------------------------------------------------------------
# surface pressure derivation (PRMSL not available via the filter service)
# ---------------------------------------------------------------------------

def derive_p_sfc(hgt_m: np.ndarray, t925_c: np.ndarray) -> np.ndarray:
    """Estimate surface pressure (hPa) hypsometrically from terrain + T925.

    Anchors on the 925 hPa level height in a standard atmosphere and scales
    the scale height by the cell's 925 hPa temperature.
    """
    h = R_SPEC * (t925_c + 273.15) / G_CONST
    return 925.0 * np.exp((Z925_STD - hgt_m) / h)


# ---------------------------------------------------------------------------
# GFS subset -> MetArrays
# ---------------------------------------------------------------------------

_KEYS = {
    ("2t", 2): "t2m", ("2r", 2): "rh", ("gust", 0): "gust", ("orog", 0): "hgt",
    ("t", 925): "t925", ("r", 925): "r925", ("u", 925): "u925", ("v", 925): "v925",
    ("t", 850): "t850", ("r", 850): "r850", ("u", 850): "u850", ("v", 850): "v850",
    ("t", 700): "t700", ("u", 700): "u700", ("v", 700): "v700",
    ("10u", 10): "u10", ("10v", 10): "v10",
}


def _pick(fields: dict[str, np.ndarray], short: str, level: int) -> np.ndarray:
    for k, v in fields.items():
        parts = k.split(":")
        if len(parts) == 3 and parts[0] == short and int(parts[2]) == level:
            return v
    raise KeyError(f"field {short!r} @ lev {level} not in {sorted(fields)}")


def _regrid_all(subset: GfsSubset, grid: Grid) -> dict[str, np.ndarray]:
    glat = np.array(grid.lat_nodes)
    glon = np.array(grid.lon_nodes)
    out: dict[str, np.ndarray] = {}
    for (short, level), name in _KEYS.items():
        if (short, level) in _KEYS:
            try:
                out[name] = subset_to_grid(subset.lat, subset.lon,
                                           _pick(subset.fields, short, level), glat, glon)
            except KeyError:
                continue
    return out


def met_from_gfs(
    cfg: Module2Config,
    grid: Grid,
    init: datetime,
    fhrs: list[int],
    cache_dir: Path | None = None,
) -> MetArrays:
    """Download + decode + regrid GFS subsets for every forecast hour."""
    import tempfile

    assert all(fhr >= 0 for fhr in fhrs)
    hours = [init + timedelta(hours=int(fhr)) for fhr in fhrs]
    nt = len(fhrs)
    ny, nx = grid.ny, grid.nx
    V: dict[str, np.ndarray] = {}  # name -> [nt, ny, nx]

    for k in _KEYS.values():
        V[k] = np.empty((nt, ny, nx))

    tmp_root = Path(tempfile.gettempdir()) / "opencode"
    for k, fhr in enumerate(fhrs):
        dest = None
        if cache_dir is not None:
            dest = cache_dir / init.strftime("%Y%m%d") / f"{fhr:03d}.grib2"
        b = fetch_subset(cfg.gfs, init, fhr, dest)
        if dest is None or not dest.is_file():
            dest = tmp_root / f"gfs_{init:%H}_{fhr:03d}.grib2"
            dest.write_bytes(b)
        subset = decode_grib(dest)
        glat, glon = np.array(grid.lat_nodes), np.array(grid.lon_nodes)
        all_ = _regrid_all(subset, grid)
        hgt = all_.get("hgt")
        if hgt is None:
            raise KeyError("terrain height (orog@surface) missing from GFS subset")
        V["t2m"][k] = all_["t2m"] - 273.15
        for f in ("rh", "u10", "v10", "u925", "v925", "u850", "v850", "u700", "v700"):
            V[f][k] = all_[f]
        V["t925"][k] = all_["t925"] - 273.15
        V["t850"][k] = all_["t850"] - 273.15
        V["t700"][k] = all_["t700"] - 273.15
        V["gust"][k] = all_["gust"]

    hgt_m = np.broadcast_to(hgt, (nt, ny, nx))
    ws10 = np.hypot(V["u10"], V["v10"])
    ws925 = np.hypot(V["u925"], V["v925"])
    ws850 = np.hypot(V["u850"], V["v850"])
    ws700 = np.hypot(V["u700"], V["v700"])
    prmsl = derive_p_sfc(hgt_m, V["t925"])

    return MetArrays(
        hours=hours,
        t2m_c=V["t2m"],
        rh_pct=V["rh"],
        ws10_ms=ws10,
        wd_deg=_wd(V["u10"], V["v10"]),
        gust_ms=V["gust"],
        prmsl_hpa=prmsl,
        t925_c=V["t925"],
        t850_c=V["t850"],
        t700_c=V["t700"],
        u925=V["u925"], v925=V["v925"],
        u850=V["u850"], v850=V["v850"],
        u700=V["u700"], v700=V["v700"],
    )


# ---------------------------------------------------------------------------
# synthetic meteorology (offline / deterministic tests)
# ---------------------------------------------------------------------------

def synthetic_met(
    cfg: Module2Config,
    grid: Grid,
    hours: list[datetime],
    seed: int = 7,
) -> MetArrays:
    """Plausible smooth met fields on the Delhi grid, fully deterministic."""
    rng = np.random.default_rng(seed)
    nt, ny, nx = len(hours), grid.ny, grid.nx
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    norm_lat = (jj - (ny - 1) / 2) / max((ny - 1) / 2, 1)
    norm_lon = (ii - (nx - 1) / 2) / max((nx - 1) / 2, 1)
    glide = norm_lat * 1.5 - norm_lon * 2.0

    t2m = np.empty((nt, ny, nx))
    rh = np.empty((nt, ny, nx))
    u10 = np.empty((nt, ny, nx))
    v10 = np.empty((nt, ny, nx))
    for k, h in enumerate(hours):
        diur = 0.0
        day = _dayness_scalar(h, grid)
        diur = 4.0 * (day - 0.5)
        t2m[k] = 24.0 + diur + glide + 0.8 * np.sin(ii * 0.3) + 0.8 * np.cos(jj * 0.3)
        rh[k] = 62.0 - 30.0 * (day - 0.5) + 4.0 * np.sin(ii * 0.2 + jj * 0.2)
        u10[k] = -(3.0 + 0.5 * glide)                    # westerly surface flow
        v10[k] = 0.8 * np.sin(ii * 0.5) + 0.5
    rh = np.clip(rh, 20.0, 100.0)

    u925 = u10 * 1.25
    v925 = v10 * 1.3
    u850 = u10 * 1.6 + 2.0
    v850 = v10 * 1.3
    u700 = u10 * 1.9 + 3.0
    v700 = v10 * 1.2

    hgt = 210.0 + 30.0 * norm_lat + 10.0 * norm_lon
    t925 = t2m + 4.0 + 1.0 * norm_lon
    t850 = t2m + 6.0
    t700 = t2m - 12.0
    gust = np.hypot(u10, v10) + 3.0
    ws10 = np.hypot(u10, v10)
    prmsl = derive_p_sfc(np.broadcast_to(hgt, (nt, ny, nx)), t925)

    return MetArrays(
        hours=hours,
        t2m_c=t2m, rh_pct=rh, ws10_ms=ws10,
        wd_deg=_wd(u10, v10), gust_ms=gust, prmsl_hpa=prmsl,
        t925_c=t925, t850_c=t850, t700_c=t700,
        u925=u925, v925=v925, u850=u850, v850=v850, u700=u700, v700=v700,
    )


def _dayness_scalar(h: datetime, grid: Grid) -> float:
    from aqf_delhi.wrf.emulator import _dayness
    mid_lat = grid.domain.lat_min + grid.domain.lat_span / 2
    mid_lon = grid.domain.lon_min + grid.domain.lon_span / 2
    return float(_dayness([h], mid_lon, mid_lat)[0])


# ---------------------------------------------------------------------------
# stations + artifact
# ---------------------------------------------------------------------------

def nearest_cell(grid: Grid, lat: float, lon: float) -> tuple[int, int]:
    """Clamped nearest-grid-cell (j, i) for any point inside the domain."""
    try:
        i, j = grid.index_of(lat, lon)
    except ValueError:
        d = grid.domain
        lat_c = float(np.clip(lat, d.lat_min, d.lat_max))
        lon_c = float(np.clip(lon, d.lon_min, d.lon_max))
        i, j = grid.index_of(lat_c, lon_c)
    return j, i


def station_series(
    result: CoupledResult,
    grid: Grid,
    stations: list[dict],
) -> tuple[list, dict]:
    """Per-station raw/obs series sampled from the gridded field."""
    series = []
    cells = []
    for st in stations:
        j, i = nearest_cell(grid, float(st["lat"]), float(st["lon"]))
        cells.append((j, i))
    nt = len(result.hours)
    for n, st in enumerate(stations):
        j, i = cells[n]
        for t in range(nt):
            series.append({
                "time": result.hours[t],
                "station_id": st["station_id"],
                "lat": float(st["lat"]), "lon": float(st["lon"]),
                "pm25_raw": float(result.pm25_grid[t, j, i]),
                "pm25_obs_demo": float(result.pm25_obs_demo[t, j, i]),
            })
    return series, {"cells": cells}


def coupled_run(
    init: datetime | None = None,
    fhrs: list[int] | None = None,
    real: bool = True,
    fires: list[FireDetection] | None = None,
    stations: list[dict] | None = None,
    cfg: Module2Config | None = None,
    out_root: Path | None = None,
) -> dict:
    """Run the full Module-2 chain and write a coupled-run artifact."""
    cfg = cfg or load_module2()
    grid = build_grid(load_domain())
    init = init or datetime.now()
    fhrs = fhrs or cfg.gfs.fhrs
    hours = [init + timedelta(hours=int(f)) for f in fhrs]
    init_0 = init.replace(minute=0, second=0, microsecond=0)

    from aqf_delhi.config import load_stations
    stations = stations or load_stations()

    if real:
        cache = (out_root or Path(cfg.artifact.out_root)) / "grib_cache"
        met = met_from_gfs(cfg, grid, init_0, fhrs, cache_dir=cache)
    else:
        met = synthetic_met(cfg, grid, hours)

    result = run_emulator(cfg, grid, met, fires or [])
    series, _ = station_series(result, grid, stations)

    from aqf_delhi.wrf.artifact import write_coupled_artifact
    info = write_coupled_artifact(
        out_root=Path(out_root or cfg.artifact.out_root),
        run_id=f"cm.{init_0:%Y%m%d.%H%M}",
        cfg=cfg,
        grid=grid,
        met=met,
        result=result,
        stations=stations,
        station_series=series,
        fires=fires or [],
        init=init_0,
        real=real,
    )
    return info