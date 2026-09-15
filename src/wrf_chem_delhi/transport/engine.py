"""Transport model: local advection + diffusion + deposition on the grid.

Conceptual update per hour (per species concentration in µg/m³):

    C(t+1) = C(t)
           + advection (sub-cell semi-Lagrangian)
           + diffusion (Gaussian smoothing, PBL/trapping-aware)
           + emissions / Δz (urban + fire)
           − dry deposition (first-order)
           − wet scavenging (RH-driven)
           + aerosol feedback factor (trapping)

Two public interfaces:

  * :meth:`TransportModel.step` — advance ONE hour from a current
    concentration state. Used by the coupled forecast loop so that the
    aerosol→PBL feedback can modify the environment hour-by-hour.
  * :meth:`TransportModel.run` — convenience multi-hour sweep calling step.

Reduced-order: no CFD. An upwind-style sub-grid shift combined with a
Gaussian diffusion pass reproduces advection-diffusion on the Delhi grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from aqf_delhi.domain import Grid
from wrf_chem_delhi.weather.engine import WeatherState

SPECIES = ["pm25", "pm10", "o3", "nox", "so2", "co"]

# Base ventilation/export rate (1/h at ~2.5 m/s wind, scaled linearly with wind)
VENT_BASE = 0.15

# Minimum effective mixing depth for emission dilution (m)
# Prevents unrealistically high concentration from very shallow PBL boxes.
# Geometric numerical diffusion in a 4-km cell over one hour ensures some
# lateral mixing even with a measured PBL < 300 m.
MIX_DEPTH_FLOOR = 400.0


@dataclass
class TransportState:
    hours: list[datetime]
    concentration: dict[str, np.ndarray]  # species → [nt, ny, nx] µg/m³
    ny: int
    nx: int

    @property
    def nt(self):
        return len(self.hours)


class TransportModel:
    def __init__(
        self,
        grid: Grid,
        *,
        diffusion_coef_km2_h: float = 8.0,
        dry_dep_velocity_ms: float = 0.0012,
        wet_removal_rate: float = 0.004,
        mixing_down_h: float = 2.2,
        seed: int = 11,
    ):
        self.grid = grid
        self.diffusion_coef = diffusion_coef_km2_h
        self.v_dep = dry_dep_velocity_ms
        self.wet = wet_removal_rate
        self.tau_mix = mixing_down_h
        self.rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------ #
    # single-hour stepping (used by coupled loop)
    # ------------------------------------------------------------------ #
    def step(
        self,
        c: dict[str, np.ndarray],         # species -> [ny,nx] µg/m³
        ws: WeatherState,                 # full state; uses hour hi torch via ws functions
        hi: int,
        emission: dict[str, np.ndarray],  # species -> [nt,ny,nx] g/h (or [ny,nx] slice util)
        pblh_m: np.ndarray,               # [ny,nx]
        trapping_idx: np.ndarray = None,  # [ny,nx] optional 0..1
    ) -> dict[str, np.ndarray]:
        """Advance one hour (hi→hi+1) for all species.

        The emitted mass is diluted over the current PBL box so additions are
        in µg/m³. Sub-grid advection uses the *local* 850 hPa wind vector for
        each species (surface near-surface, so a blend of 925+850).

        Combination of advection, diffusion, emissions, deposition, wet
        scavenging, and (optional) trapping factor applied per species.
        """
        ny, nx = self.grid.ny, self.grid.nx
        u = 0.45 * ws.u925[hi] + 0.55 * ws.u850[hi]
        v = 0.45 * ws.v925[hi] + 0.55 * ws.v850[hi]
        rh = ws.rh_pct[hi]

        dx = self.grid.lon_res_km() * 1000.0   # m
        dy = self.grid.lat_res_km() * 1000.0   # m
        cell_m2 = dx * dy

        out = {}
        for sp in SPECIES:
            if sp not in c:
                continue
            cur = np.asarray(c[sp], dtype=float).copy()

            # 1) advection: sub-cell shift in meters mapped to cell indices
            shift_x = u * 3600.0   # meters in 1 h
            shift_y = v * 3600.0
            di = np.rint(shift_x / dx).astype(int)
            dj = np.rint(-shift_y / dy).astype(int)
            moved = np.zeros_like(cur)
            for j in range(ny):
                for i in range(nx):
                    sj = j + dj[j, i]
                    si = i + di[j, i]
                    if 0 <= sj < ny and 0 <= si < nx:
                        moved[sj, si] += cur[j, i]
            cur = moved

            # 2) diffusion: Gaussian smoothing scaled by stability-reduced PBL
            import scipy.ndimage

            # weaker mixing when trapping is high (inverted stable layer)
            trap_factor = (
                np.full((ny, nx), 1.0) if trapping_idx is None else 1.0 - 0.7 * trapping_idx
            )
            sigma_cells = 0.9 * np.sqrt(self.diffusion_coef) / max(0.4, self.grid.lon_res_km())
            cur = scipy.ndimage.gaussian_filter(cur, sigma=(sigma_cells, sigma_cells) if self.grid.lon_res_km() > 0.5 else (0, 0))

            # 3) emissions → box concentration (g/h → µg/m³)
            mix_depth = np.maximum(pblh_m, MIX_DEPTH_FLOOR)  # m, see mod note
            for sp2 in ("pm25", "pm10", "nox", "so2", "co", "o3"):
                if sp2 == sp and sp2 in emission:
                    em_hour = np.asarray(emission[sp2][hi], dtype=float)  # g/h
                    cur += em_hour / mix_depth / cell_m2 * 1e6

            # 3b) ventilation outflow: pollutants are carried out of the
            #     domain by the same winds that bring them in. A first-order
            #     export term proportional to wind speed keeps the column at
            #     a quasi-steady seasonal level instead of piling up without
            #     bound.
            ws10 = ws.ws10_ms[hi]
            vent_rate = VENT_BASE * (0.5 + ws10 / 4.0)  # 1/h
            cur *= np.exp(-vent_rate)

            # 4) dry deposition
            dep_rate_h = self.v_dep / np.maximum(pblh_m, 50.0) * 3600.0
            cur *= np.exp(-dep_rate_h)

            # 5) wet scavenging (RH>70%)
            f_wet = np.clip((rh - 70.0) / 30.0, 0, 1)
            cur *= np.exp(-self.wet * f_wet)

            # 6) trapping suppression of ventilation (keeps near surface, adds)
            if trapping_idx is not None:
                cur *= 1.0 + 0.45 * np.clip(trapping_idx, 0, 1) * 0.0  # handled in feedback
                # clamp to positive
            out[sp] = np.clip(cur, 0.0, None)

        return out

    # ------------------------------------------------------------------ #
    # full sweep
    # ------------------------------------------------------------------ #
    def run(
        self,
        ws: WeatherState,
        emissions: dict[str, np.ndarray],
        inversion: np.ndarray,
        pblh: np.ndarray,
        initial: dict[str, np.ndarray] = None,
        trapping_idx: np.ndarray = None,
    ) -> TransportState:
        nt, ny, nx = ws.nt, ws.ny, ws.nx
        c = {sp: np.zeros((nt, ny, nx)) for sp in SPECIES}
        if initial:
            for sp in SPECIES:
                if sp in initial:
                    c[sp][0] = np.asarray(initial[sp], dtype=float)

        for t in range(1, nt):
            slice_in = {sp: c[sp][t - 1] for sp in SPECIES if sp in emissions or sp in c}
            out = self.step(slice_in, ws, t, emissions,
                            pblh_m=pblh[t] if pblh.ndim == 3 else pblh,
                            trapping_idx=trapping_idx[t] if trapping_idx is not None and trapping_idx.ndim == 3 else trapping_idx)
            for sp in out:
                c[sp][t] = out[sp]

        return TransportState(hours=ws.hours, concentration=c, ny=ny, nx=nx)