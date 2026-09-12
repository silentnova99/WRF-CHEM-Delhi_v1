"""Emission coupling: FRP -> PM mass, grid aggregation, vertical distribution.

Puts every active fire detection on the Delhi grid, converts its fire
radiative power to a primary PM2.5 emission flux and splits it across the
plume and surface layers using the plume-rise profile.  A diffuse urban base
source is added on top, with a deterministic diurnal cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from aqf_delhi.wrf.config import Module2Config
from aqf_delhi.wrf.plume import PlumeResult, plume_rise, pressure_height


@dataclass
class FireDetection:
    lat: float
    lon: float
    frp_mw: float
    acq: datetime


@dataclass
class FirePlume:
    frp_mw: float
    z_centre_m: float
    z_top_m: float
    surface_fraction: float     # mass fraction remaining in the PBL
    elevated_fraction: float    # mass fraction injected aloft


def plume_for(cfg: Module2Config, met_cell: dict[str, float]) -> PlumeResult:
    """Ambient profile + plume-rise for a single fire cell."""
    p = cfg.plume
    c = cfg.chemistry
    t_sfc = met_cell["t2m_c"] + 273.15
    z = np.array([
        10.0,
        pressure_height(925.0, t_sfc, c.g_const),
        pressure_height(850.0, t_sfc, c.g_const),
        pressure_height(700.0, t_sfc, c.g_const),
    ])
    t_k = np.array([t_sfc, met_cell["t925_c"] + 273.15,
                    met_cell["t850_c"] + 273.15, met_cell["t700_c"] + 273.15])
    ws_k = np.array([met_cell["ws10_ms"], met_cell["ws925_ms"],
                     met_cell["ws850_ms"], met_cell["ws700_ms"]])
    return plume_rise(
        met_cell["frp_mw"],
        t_sfc_k=float(t_sfc),
        p_sfc_hpa=float(met_cell["prmsl_hpa"]),
        z_k=z,
        t_k=t_k,
        ws_k=ws_k,
        sensible_fraction=p.sensible_fraction,
        entrainment_coef=p.entrainment_coef,
        z_step=p.z_step,
        max_top_m=p.max_top_m,
        n_layers=p.n_layers,
    )


def split_vertical(cfg: Module2Config, met_cell: dict[str, float]) -> tuple[float, float]:
    """Surface vs elevated fractions for a fire cell."""
    plm = plume_for(cfg, met_cell)
    pblh = float(met_cell.get("pblh_m", 500.0))
    edges = plm.layer_edges_m
    surf_idx = 0.5 * (edges[:-1] + edges[1:]) <= pblh
    surf = float(np.clip(plm.mb_flux_per_layer[surf_idx].sum(), 0.0, 1.0))
    return surf, 1.0 - surf


def fire_layer_profile(cfg: Module2Config, met_cell: dict[str, float]) -> FirePlume:
    plm = plume_for(cfg, met_cell)
    surf, elev = split_vertical(cfg, met_cell)
    return FirePlume(
        frp_mw=met_cell["frp_mw"],
        z_centre_m=float(plm.z_centre_m),
        z_top_m=float(plm.z_top_m),
        surface_fraction=surf,
        elevated_fraction=elev,
    )


def active_hours(fire: FireDetection, hours, lifetime_h: float) -> list[int]:
    """Indices of ``hours`` in which a fire is still emitting."""
    out = []
    for i, h in enumerate(hours):
        dt = (h - fire.acq).total_seconds() / 3600.0
        if 0.0 <= dt < lifetime_h:
            out.append(i)
    return out


def fire_emissions(
    cfg: Module2Config,
    grid,
    fires: list[FireDetection],
    hours: list[datetime],
    met_cell_at,
) -> tuple[np.ndarray, np.ndarray, list[FirePlume]]:
    """Gridded per-hour fire PM2.5 emission flux arrays.

    Returns ``(surface_gph, elevated_gph, plumes)``: ``[nt, ny, nx]`` arrays of
    grams emitted during each hour, and one :class:`FirePlume` per fire.
    """
    nt, ny, nx = len(hours), grid.ny, grid.nx
    e_sfc = np.zeros((nt, ny, nx))
    e_elev = np.zeros((nt, ny, nx))
    plumes: list[FirePlume] = []
    lifetime = cfg.emissions.fire_lifetime_h
    ef_g_per_mwh = cfg.emissions.pm25_ef * 3600.0     # g per (MW h)
    fired: set[tuple[int, int]] = set()
    for fire in fires:
        if fire.frp_mw < cfg.emissions.fire_min_frp_mw:
            continue
        try:
            j, i = grid.index_of(fire.lat, fire.lon)
        except ValueError:
            continue
        for hi in active_hours(fire, hours, lifetime):
            met_cell = {**met_cell_at(hi, j, i), "frp_mw": fire.frp_mw}
            surf, elev = split_vertical(cfg, met_cell)
            rate = fire.frp_mw * ef_g_per_mwh          # g / h
            e_sfc[hi, j, i] += rate * surf
            e_elev[hi, j, i] += rate * elev
            if (j, i) not in fired:
                plumes.append(fire_layer_profile(cfg, met_cell))
                fired.add((j, i))
    return e_sfc, e_elev, plumes


def diurnal_factor(hour_local: int) -> float:
    """Deterministic diurnal modulation of the diffuse urban source."""
    table = {
        0: 0.62, 1: 0.55, 2: 0.50, 3: 0.48, 4: 0.55,
        5: 0.80, 6: 1.10, 7: 1.25, 8: 1.32, 9: 1.20,
        10: 1.02, 11: 0.90, 12: 0.85, 13: 0.85, 14: 0.90,
        15: 0.98, 16: 1.05, 17: 1.15, 18: 1.30, 19: 1.45,
        20: 1.45, 21: 1.30, 22: 1.10, 23: 0.85,
    }
    return float(table[int(hour_local) % 24])


def urban_base(
    cfg: Module2Config,
    grid,
    hours: list[datetime],
    lon_tz: float = 77.2,
) -> np.ndarray:
    """Diffuse urban primary PM2.5 emission per cell per hour ([nt, ny, nx] g/h)."""
    base_km2 = cfg.emissions.urban_base_kg_km2_day * 1000.0 / 24.0   # g/km2/h
    cell_km2 = grid.lat_res_km() * grid.lon_res_km()
    nt = len(hours)
    out = np.zeros((nt, grid.ny, grid.nx))
    for hi, h in enumerate(hours):
        local_hour = (h.utcoffset().total_seconds() // 3600 + h.hour) % 24 if h.tzinfo else h.hour
        local_hour = round(local_hour + (lon_tz - 82.5) / 15.0) % 24
        out[hi] = base_km2 * cell_km2 * diurnal_factor(local_hour)
    return out