"""Weather engine: produces the meteorological state that drives everything.

Serves three modes (see ``wrf_chem_delhi.data_modes``):
  1. LIVE    - real GFS via the legacy ``aqf_delhi.wrf`` fetch path
  2. CACHE   - previously cached GFS artifacts on disk
  3. DEMO    - deterministic synthetic meteorology (scenario driven)

The public surface is :class:`WeatherState`, an array bundle of the essential
2-D fields over the Delhi NCR grid plus derived thermodynamic quantities
(stability, ventilation, stagnation) needed by the inversion / transport /
chemistry engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from aqf_delhi.config import load_domain
from aqf_delhi.domain import build_grid
from aqf_delhi.wrf.emulator import MetArrays

R_SPEC = 287.05  # J / (kg K)
G_ACC = 9.81  # m/s^2
CP_AIR = 1005.0  # J / (kg K)
KM_PER_DEG = 111.13


@dataclass
class WeatherState:
    """Bundle of meteorological fields over the model grid.

    Arrays are shaped ``[nt, ny, nx]`` unless noted otherwise.
    """

    hours: list[datetime]
    lat_nodes: np.ndarray
    lon_nodes: np.ndarray
    ny: int
    nx: int
    # core
    t2m_c: np.ndarray
    rh_pct: np.ndarray
    prmsl_hpa: np.ndarray
    ws10_ms: np.ndarray
    wd_deg: np.ndarray
    gust_ms: np.ndarray
    # upper air
    t925_c: np.ndarray
    t850_c: np.ndarray
    t700_c: np.ndarray
    u925: np.ndarray
    v925: np.ndarray
    u850: np.ndarray
    v850: np.ndarray
    # derived thermodynamics
    pblh_m: np.ndarray = None
    stability: np.ndarray = None  # lapse rate proxy K / km
    ventilation_idx: np.ndarray = None  # m^2/s
    stagnation_idx: np.ndarray = None  # dimensionless 0-1
    data_source: str = "demo-synthetic"
    model_version: str = "2.0.0"

    @property
    def nt(self) -> int:
        return len(self.hours)

    @property
    def time(self) -> datetime:
        return self.hours[0] if self.hours else None

    # -- per-cell accessor for the plume / fire engines ------------------- #
    def cell(self, hi: int, j: int, i: int) -> dict:
        g = {
            "t2m_c": float(self.t2m_c[hi, j, i]),
            "rh_pct": float(self.rh_pct[hi, j, i]),
            "prmsl_hpa": float(self.prmsl_hpa[hi, j, i]),
            "ws10_ms": float(self.ws10_ms[hi, j, i]),
            "wd_deg": float(self.wd_deg[hi, j, i]),
            "gust_ms": float(self.gust_ms[hi, j, i]),
            "t925_c": float(self.t925_c[hi, j, i]),
            "t850_c": float(self.t850_c[hi, j, i]),
            "t700_c": float(self.t700_c[hi, j, i]),
            "u925": float(self.u925[hi, j, i]),
            "v925": float(self.v925[hi, j, i]),
            "u850": float(self.u850[hi, j, i]),
            "v850": float(self.v850[hi, j, i]),
            "pblh_m": float(self.pblh_m[hi, j, i]),
        }
        if self.stability is not None:
            g["stability_deg_per_km"] = float(self.stability[hi, j, i])
        if self.ventilation_idx is not None:
            g["ventilation_idx"] = float(self.ventilation_idx[hi, j, i])
        return g

    def domain_mean(self, hi: int, key: str) -> float:
        arr = getattr(self, key)
        if arr is None:
            return None
        return float(np.nanmean(arr[hi]))

    def to_json(self, hi: int) -> dict:
        return {
            "hour": self.hours[hi].isoformat() if hi < len(self.hours) else None,
            "t2m_c_avg": self.domain_mean(hi, "t2m_c"),
            "rh_pct_avg": self.domain_mean(hi, "rh_pct"),
            "prmsl_hpa_avg": self.domain_mean(hi, "prmsl_hpa"),
            "ws10_ms_avg": self.domain_mean(hi, "ws10_ms"),
            "wd_deg_avg": self.domain_mean(hi, "wd_deg"),
            "gust_ms_avg": self.domain_mean(hi, "gust_ms"),
            "pblh_m_avg": self.domain_mean(hi, "pblh_m"),
            "stability_deg_per_km_avg": self.domain_mean(hi, "stability"),
            "ventilation_idx_avg": self.domain_mean(hi, "ventilation_idx"),
            "stagnation_idx_avg": self.domain_mean(hi, "stagnation_idx"),
        }


# --------------------------------------------------------------------------- #
# Wind direction from components
# --------------------------------------------------------------------------- #
def wd_from_uv(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological wind direction (degrees from which wind blows)."""
    return np.mod(270.0 - np.degrees(np.arctan2(-v, -u)), 360.0)


def dayness(hours: list[datetime], lon: float, lat: float) -> np.ndarray:
    """Fractional daylight (sin of solar elevation clipped to [0,1])."""

    def _sun(h: datetime):
        doy = h.timetuple().tm_yday
        decl = 23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 81)))
        # solar hour angle approximation at given longitude
        ha = np.radians(15.0 * (h.hour - 12.0) + (lon - 82.5) / 15.0 * 15.0)
        elev = np.sin(np.radians(lat)) * np.sin(np.radians(decl)) + np.cos(
            np.radians(lat)
        ) * np.cos(np.radians(decl)) * np.cos(ha)
        return float(np.clip(elev, 0.0, 1.0))

    return np.array([_sun(h) for h in hours])


# --------------------------------------------------------------------------- #
# DEMO synthetic weather (deterministic, scenario-aware)
# --------------------------------------------------------------------------- #
def synthetic_weather(
    grid,
    hours: list[datetime],
    *,
    seed: int = 11,
    scenario: str = "normal_winter",
) -> WeatherState:
    """Deterministic synthetic meteorology for offline DEMO mode.

    Contains a realistic diurnal cycle plus inter-hour noise so that the
    transport/chemistry/feedback chain sees a *varying* environment. The
    scenario switches modulate the PBL height / inversion / wind pattern.
    """
    rng = np.random.RandomState(seed)
    nt = len(hours)
    ny, nx = grid.ny, grid.nx

    _lat = np.asarray(grid.lat_nodes, dtype=float)
    _lon = np.asarray(grid.lon_nodes, dtype=float)
    ys = _lat[:, None]
    xs = _lon[None, :]
    norm_lat = ((_lat - 28.2) / 0.8)[:, None]
    norm_lon = ((_lon - 76.8) / 0.8)[None, :]

    day = dayness(hours, 77.2, 28.6)

    # scenario modifiers
    if scenario == "normal_winter":
        inv_strength = 2.0
        wind_scale = 1.0
        pbl_scale = 1.0
    elif scenario == "strong_inversion":
        inv_strength = 9.0
        wind_scale = 0.45
        pbl_scale = 0.55
    elif scenario == "stubble_plume":
        inv_strength = 5.0
        wind_scale = 0.75
        pbl_scale = 0.85
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    t2m = np.empty((nt, ny, nx))
    rh = np.empty((nt, ny, nx))
    prmsl = np.empty((nt, ny, nx))
    ws10 = np.empty((nt, ny, nx))
    wd = np.empty((nt, ny, nx))
    gust = np.empty((nt, ny, nx))
    t925 = np.empty((nt, ny, nx))
    t850 = np.empty((nt, ny, nx))
    t700 = np.empty((nt, ny, nx))
    u925 = np.empty((nt, ny, nx))
    v925 = np.empty((nt, ny, nx))
    u850 = np.empty((nt, ny, nx))
    v850 = np.empty((nt, ny, nx))

    for hi in range(nt):
        d = day[hi]
        diurnal = 4.0 * (d - 0.5)
        t2m[hi] = 17.0 + 6.0 * d + 1.5 * norm_lat - 1.0 * norm_lon
        # smooth slow-evolving synoptic variation
        synop = 1.2 * np.sin(2 * np.pi * (hi / 24.0) / 5.0 + seed)
        t2m[hi] += synop

        rh[hi] = np.clip(78 - 34 * (d - 0.35) - 0.2 * synop * 10, 25, 98)
        prmsl[hi] = 1018.0 + 4.0 * np.sin(2 * np.pi * hi / 24.0 / 4.0)

        wdir_base = 250.0  # WNW - NW transport from Punjab
        wd[hi] = wdir_base + 12.0 * np.sin(hi / 24.0 * 2 * np.pi) + 30.0 * norm_lon
        base_ws = (2.6 + 1.1 * np.sin(2 * np.pi * hi / 24.0 * 1.4)) * wind_scale
        ws10[hi] = base_ws + 0.8 * norm_lat + rng.normal(0, 0.3, (ny, nx))
        np.clip(ws10[hi], 0.2, None, out=ws10[hi])
        gust[hi] = ws10[hi] + 2.5

        # vertically sheared winds (transport drives advection aloft)
        u925[hi] = -3.5 * wind_scale + rng.normal(0, 0.4, (ny, nx))
        v925[hi] = -0.8 + 0.4 * np.sin(2 * np.pi * hi / 24.0) + norm_lat * 0.3
        u850[hi] = -5.5 * wind_scale + rng.normal(0, 0.5, (ny, nx))
        v850[hi] = -1.6 + 0.7 * np.sin(2 * np.pi * hi / 24.0) + norm_lat * 0.5
        t925[hi] = t2m[hi] + 4.0 + inv_strength * (d < 0.35)
        t850[hi] = t2m[hi] + 6.0
        t700[hi] = t2m[hi] - 10.0

    ws925 = np.hypot(u925, v925)
    ws850 = np.hypot(u850, v850)

    # PBL height - diurnal boundary layer modulated by scenario
    pblh = np.clip(640 + 1500 * day[:, None, None], 250, 2200) * pbl_scale
    pblh = np.broadcast_to(pblh, (nt, ny, nx)).copy()

    # atmospheric stability: approximate lapse rate between 925 and 2 m
    # (converted to K / km). Positive (cooling with height) = unstable.
    dz925 = _pressure_height(925.0, 288.15)
    stability = (t925 - t2m) / dz925 * 1000.0

    # ventilation index = PBLH * wind speed  (m^2/s)
    ventilation_idx = pblh * (ws10 + 0.35)

    # stagnation index 0..1 (Srivastava-like scaling)
    stagnation = _stagnation_index(pblh, ws10, stability, scenario)

    return WeatherState(
        hours=hours,
        lat_nodes=np.asarray(grid.lat_nodes),
        lon_nodes=np.asarray(grid.lon_nodes),
        ny=ny,
        nx=nx,
        t2m_c=t2m,
        rh_pct=rh,
        prmsl_hpa=prmsl,
        ws10_ms=ws10,
        wd_deg=wd,
        gust_ms=gust,
        t925_c=t925,
        t850_c=t850,
        t700_c=t700,
        u925=u925,
        v925=v925,
        u850=u850,
        v850=v850,
        pblh_m=pblh,
        stability=stability,
        ventilation_idx=ventilation_idx,
        stagnation_idx=stagnation,
        data_source=f"demo-synthetic ({scenario})",
    )


def _pressure_height(p_hpa: float, t_k: float) -> float:
    return R_SPEC * t_k / G_ACC * np.log(1013.25 / p_hpa)


def _stagnation_index(pblh, ws10, stability, scenario) -> np.ndarray:
    """Dimensionless stagnation proxy in [0,1].

    Elevated when PBL is shallow, winds weak and the layer is stable.
    """
    p_norm = np.clip((pblh - 200) / 2000.0, 0, 1)
    w_norm = np.clip(1 - (ws10 - 0.5) / 6.0, 0, 1)
    s_norm = np.clip(1 - (stability + 2) / 8.0, 0, 1)
    stagnation = 0.2 * (1 - p_norm) + 0.5 * w_norm + 0.3 * s_norm
    stagnation = np.clip(stagnation, 0, 1)
    if scenario == "strong_inversion":
        stagnation = np.clip(stagnation + 0.3, 0, 1)
    return stagnation


# --------------------------------------------------------------------------- #
# LIVE / CACHE: adapt legacy GFS path into WeatherState
# --------------------------------------------------------------------------- #
def weather_from_legacy_met(met: MetArrays, grid, *, data_source: str) -> WeatherState:
    """Adapt a legacy ``aqf_delhi.wrf.emulator.MetArrays`` into WeatherState.

    The legacy path already has t2m, rh, ws, wd, upper-air fields on the grid.
    We recompute PBL / stability / ventilation + stagnation here so all engines
    share the same derived thermodynamics independent of weather source.
    """
    t2m = met.t2m_c
    rh = met.rh_pct
    ws10 = met.ws10_ms
    wd = met.wd_deg
    gust = met.gust_ms
    prmsl = met.prmsl_hpa
    u925 = met.u925
    v925 = met.v925
    u850 = met.u850
    v850 = met.v850
    t925 = met.t925_c
    t850 = met.t850_c
    t700 = met.t700_c

    day = dayness(met.hours, 77.2, 28.6)
    pblh = np.clip(640 + 1500 * day[:, None, None], 250, 2200)
    pblh = np.broadcast_to(pblh, (len(met.hours), grid.ny, grid.nx)).copy()

    dz925 = _pressure_height(925.0, 288.15)
    stability = (t925 - t2m) / dz925 * 1000.0
    ventilation_idx = pblh * (ws10 + 0.35)
    scenario = "normal_winter"
    stagnation = _stagnation_index(pblh, ws10, stability, scenario)

    return WeatherState(
        hours=list(met.hours),
        lat_nodes=np.asarray(grid.lat_nodes),
        lon_nodes=np.asarray(grid.lon_nodes),
        ny=grid.ny,
        nx=grid.nx,
        t2m_c=t2m,
        rh_pct=rh,
        prmsl_hpa=prmsl,
        ws10_ms=ws10,
        wd_deg=wd,
        gust_ms=gust,
        t925_c=t925,
        t850_c=t850,
        t700_c=t700,
        u925=u925,
        v925=v925,
        u850=u850,
        v850=v850,
        pblh_m=pblh,
        stability=stability,
        ventilation_idx=ventilation_idx,
        stagnation_idx=stagnation,
        data_source=data_source,
    )


def build_default_grid():
    domain = load_domain()
    return build_grid(domain)


def make_hours(init: datetime, fhrs) -> list[datetime]:
    """Create a list of hourly datetimes starting at init (naive → aware UTC)."""
    if init.tzinfo is None:
        init = init.replace(tzinfo=timezone.utc)
    return [init + timedelta(hours=int(h)) for h in fhrs]


def weather_default_grid(hours, scenario="normal_winter", legacy_met: Optional[MetArrays] = None):
    """Entry point used by data_modes L2/L3: return WeatherState.

    If a legacy MetArrays is provided (live/cached GFS), adapt it; otherwise
    build deterministic synthetic weather.
    """
    grid = build_default_grid()
    if legacy_met is not None:
        return weather_from_legacy_met(legacy_met, grid, data_source=legacy_met.data_source)
    return synthetic_weather(grid, hours, scenario=scenario)