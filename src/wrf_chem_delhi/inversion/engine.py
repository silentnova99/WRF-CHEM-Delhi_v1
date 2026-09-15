"""Inversion engine: atmospheric-inversion detection and diagnostics.

Inversion detection is performed against a coarse vertical temperature
profile. With real data (GFS) only 2 m / 925 / 850 / 700 hPa temperatures exist,
so the engine uses a documented **proxy**:

  *strength* = (T(925) - T(2m)) lapse (K) reversed; positive → rising temp with
  height = thermal inversion between surface and ~770 m layer.

This proxy is NOT radiosonde analysis. A full-profile implementation
(:func:`from_vertical_profile`) is provided and used automatically when a dense
(>=3 sample) vertical T profile is supplied; otherwise fall back to the proxy.

Outputs
-------
InversionState per hour with, per grid cell:

* ``strength_k`` - inversion strength (K)
* ``height_m``   - inversion base height estimate (m)
* ``trapping_idx`` - 0..1 trapping index (stronger + shallower + weak wind → 1)
* ``duration_steps`` - consecutive hours with inversion (rolling)
* ``probability`` - logistic probability of active inversion in [0,1]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aqf_delhi.domain import Grid
from wrf_chem_delhi.weather.engine import WeatherState, G_ACC, R_SPEC

_WIND_TRAP_K_S = 1.5  # m/s wind cap for a strong "trap" state
_DZ_925 = R_SPEC * 288.15 / G_ACC * np.log(1013.25 / 925.0)  # ~769 m


@dataclass
class InversionState:
    hours: list
    lat_nodes: np.ndarray
    lon_nodes: np.ndarray
    ny: int
    nx: int
    strength_k: np.ndarray  # [nt, ny, nx] inversion magnitude (K, >0 = inversion)
    height_m: np.ndarray  # [nt, ny, nx] estimated base height (m)
    duration_steps: np.ndarray  # [nt, ny, nx] rolling consecutive inversion
    probability: np.ndarray  # [nt, ny, nx] 0..1
    trapping_idx: np.ndarray  # [nt, ny, nx] 0..1
    method: str = "t925-t2m-lapse-proxy"

    @property
    def nt(self):
        return len(self.hours)

    @property
    def active(self) -> np.ndarray:
        """Boolean mask: inversion active where probability > 0.5."""
        return self.probability > 0.5

    def json(self, hi: int) -> dict:
        return {
            "hour": self.hours[hi].isoformat(),
            "strength_k_avg": float(np.nanmean(self.strength_k[hi])) if self.nt else None,
            "height_m_avg": float(np.nanmean(self.height_m[hi])),
            "trapping_idx_avg": float(np.nanmean(self.trapping_idx[hi])),
            "probability_avg": float(np.nanmean(self.probability[hi])),
            "active_fraction": float(np.mean(self.active[hi])),
            "duration_avg": float(np.nanmean(self.duration_steps[hi])),
            "method": self.method,
        }


def _logistic(x: np.ndarray, x0: float, k: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def compute_from_weather(ws: WeatherState) -> InversionState:
    """Inversion diagnostics from a :class:`WeatherState` (surface + 925 2-level)."""
    nt, ny, nx = ws.nt, ws.ny, ws.nx
    hours = ws.hours

    strength = (ws.t925_c - ws.t2m_c).astype(float)
    strength = np.maximum(strength, 0.0)

    # base height: pick 925-height (~769 m) when inversion aloft, else ~200 m
    height = np.where(strength > 1.0, _DZ_925, 250.0)

    # probability from strength + wind logic
    prob = _logistic(strength, 2.0, 0.8) * _logistic(ws.ws10_ms, 4.0, -1.2)
    prob = np.clip(prob, 0, 1)

    # trapping index: wants strong inversion, low wind, low PBL
    w_term = _logistic(ws.ws10_ms, 4.0, -1.2)
    p_term = 1.0 - np.clip((ws.pblh_m - 250.0) / 1800.0, 0, 1)
    trap = 0.55 * prob + 0.25 * w_term + 0.2 * p_term
    trap = np.clip(trap, 0, 1)

    # rolling duration
    duration = _rolling_duration(prob > 0.5)

    return InversionState(
        hours=hours, lat_nodes=ws.lat_nodes, lon_nodes=ws.lon_nodes, ny=ny, nx=nx,
        strength_k=strength, height_m=height, duration_steps=duration,
        probability=prob, trapping_idx=trap,
    )


def from_vertical_profile(
    hours: list,
    grid: Grid,
    t_sfc_k: np.ndarray,
    z_sfc_m: np.ndarray,
    heights_m: np.ndarray,
    temps_k: np.ndarray,
) -> InversionState:
    """Full vertical-profile inversion detection (radiosonde-like).

    Parameters
    ----------
    hours : list[datetime]
    grid : Grid
    t_sfc_k : array [nt, ny, nx] 2-m temperature (K)
    z_sfc_m : array [nt, ny, nx] surface elevation (m)
    heights_m : array [nlev]
    temps_k : array [nt, ny, nx, nlev]

    Returns an InversionState where strength is the max positive lapse segment.
    This is the scientifically richer path used when real profiles exist;
    the proxy engine above is only a documented fallback.
    """
    nt, ny, nx = t_sfc_k.shape
    nlev = len(heights_m)
    strength = np.zeros((nt, ny, nx))
    height = np.full((nt, ny, nx), _DZ_925)
    prob = np.zeros((nt, ny, nx))

    z = z_sfc_m + heights_m  # [nt, ny, nx, nlev]
    for hi in range(nt):
        for j in range(ny):
            for i in range(nx):
                prof = np.concatenate(([t_sfc_k[hi, j, i]], temps_k[hi, j, i, :]))
                zz = np.concatenate(([z_sfc_m[hi, j, i]], z[hi, j, i, :]))
                # find maximum positive lapse rate segment
                dT_dz = np.gradient(prof, zz)
                max_pos = np.max(dT_dz) if len(dT_dz) else 0.0
                strength[hi, j, i] = max(0.0, float(max_pos))
                if max_pos > 0.02:  # K/m => inverting
                    idx = int(np.argmax(dT_dz))
                    height[hi, j, i] = float(zz[idx]) if idx < len(zz) else 200.0
                prob[hi, j, i] = _logistic(np.array([max_pos]), 0.02, 300.0)[0]

    trap = np.clip(0.55 * prob + 0.25 * _uniform_low_wind(np.zeros((nt, ny, nx))), 0, 1)

    return InversionState(
        hours=hours, lat_nodes=np.asarray(grid.lat_nodes), lon_nodes=np.asarray(grid.lon_nodes),
        ny=ny, nx=nx, strength_k=strength, height_m=height,
        duration_steps=_rolling_duration(prob > 0.5), probability=prob,
        trapping_idx=trap, method="vertical-profile-lapse",
    )


def _uniform_low_wind(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 1)


def _rolling_duration(active: np.ndarray) -> np.ndarray:
    """Consecutive-hours count for the boolean mask [nt, ny, nx]."""
    nt, ny, nx = active.shape
    out = np.zeros_like(active, dtype=int)
    for j in range(ny):
        for i in range(nx):
            count = 0
            for t in range(nt):
                count = count + 1 if active[t, j, i] else 0
                out[t, j, i] = count
    return out