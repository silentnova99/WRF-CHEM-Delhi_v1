"""Plume model: buoyant rise + Gaussian dispersion from fire sources.

Uses the existing legacy 1-D integral Freitas-style plume rise
(``aqf_delhi.wrf.plume.plume_rise``) for the vertical injection profile and
adds a Gaussian horizontal-dispersion (:ref:`plume width`, centreline falloff)
so each fire's smoke reaches downwind stations with a physically-plausible
concentration influence.

Reduced-order: no latent-heat plume, no moist dynamics. Clearly labelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from aqf_delhi.domain import Grid
from aqf_delhi.wrf.plume import plume_rise, pressure_height
from wrf_chem_delhi.fire.engine import FireEvent
from wrf_chem_delhi.weather.engine import WeatherState

_KM_PER_DEG = 111.13


@dataclass
class PlumeRiseResult:
    frp_mw: float
    z_center_m: float
    z_top_m: float
    injection_profile: np.ndarray  # n_layers (mass fraction / layer)
    n_layers: int = 12

    def to_json(self) -> dict:
        return {
            "frp_mw": self.frp_mw,
            "z_center_m": round(self.z_center_m, 1),
            "z_top_m": round(self.z_top_m, 1),
            "n_layers": self.n_layers,
        }


@dataclass
class PlumeState:
    hours: list[datetime]
    fires: list[FireEvent]
    rises: list[PlumeRiseResult]
    # horizontal concentration influence over grid 0..1 per hour
    concentration: np.ndarray  # [nt, ny, nx]
    transport_dir_deg: np.ndarray  # [nt, ny, nx] mean transport direction
    width_km: np.ndarray  # [nt, ny, nx] Gaussian sigma width (km)
    ny: int
    nx: int

    def json(self, hi: int) -> dict:
        active = [i for i, e in enumerate(self.fires)
                  if 0 <= (self.hours[hi] - e.acq).total_seconds() / 3600.0 < 24]
        return {
            "hour": self.hours[hi].isoformat(),
            "n_plumes": len(active),
            "mean_ztop": float(np.nanmean([self.rises[i].z_top_m for i in active])) if active else None,
            "peak_concentration": float(np.nanmax(self.concentration[hi])),
            "mean_transport_dir": float(np.nanmean(self.transport_dir_deg[hi])),
        }


class PlumeModel:
    def __init__(self, grid: Grid, *, n_layers: int = 12, z_step: float = 50.0, max_top_m: float = 6500.0):
        self.grid = grid
        self.n_layers = n_layers
        self.z_step = z_step
        self.max_top_m = max_top_m

    def rise(self, fire: FireEvent, met_cell: dict) -> PlumeRiseResult:
        t_sfc_c = met_cell["t2m_c"] + 273.15
        p_sfc = met_cell.get("prmsl_hpa", 1010.0)
        z_k = [
            0.0,
            pressure_height(925.0, t_sfc_c),
            pressure_height(850.0, t_sfc_c),
            pressure_height(700.0, t_sfc_c),
        ]
        t_k = [t_sfc_c, met_cell["t925_c"] + 273.15, met_cell["t850_c"] + 273.15, met_cell["t700_c"] + 273.15]
        ws_k = [met_cell["ws10_ms"],
                np.hypot(met_cell["u925"], met_cell["v925"]),
                np.hypot(met_cell["u850"], met_cell["v850"]),
                met_cell["ws10_ms"] + 3.0]
        res = plume_rise(
            fire.frp_mw,
            t_sfc_k=t_sfc_c,
            p_sfc_hpa=p_sfc,
            z_k=np.asarray(z_k, dtype=float),
            t_k=np.asarray(t_k, dtype=float),
            ws_k=np.asarray(ws_k, dtype=float),
            sensible_fraction=0.45,
            entrainment_coef=0.6,
            z_step=self.z_step,
            max_top_m=self.max_top_m,
            n_layers=self.n_layers,
        )
        return PlumeRiseResult(
            frp_mw=fire.frp_mw,
            z_center_m=float(res.z_centre_m),
            z_top_m=float(res.z_top_m),
            injection_profile=np.asarray(res.mb_flux_per_layer),
            n_layers=self.n_layers,
        )

    def compute_state(
        self,
        hours: list[datetime],
        fires: list[FireEvent],
        ws: WeatherState,
        fire_concentration: np.ndarray = None,
    ) -> PlumeState:
        """Gaussian dispersion: centerline downwind plume cells.

        ``concentration[h,j,i]`` = weight for arriving smoke at cell (j,i) hour h,
        driven by wind speed/direction at that hour resolved toward each fire.
        """
        nt, ny, nx = ws.nt, ws.ny, ws.nx
        rises = []
        conc = np.zeros((nt, ny, nx))
        tdir = np.zeros((nt, ny, nx))
        width = np.zeros((nt, ny, nx))

        lats = np.asarray(self.grid.lat_nodes, dtype=float)[:, None]
        lons = np.asarray(self.grid.lon_nodes, dtype=float)[None, :]
        lat_1d = np.asarray(self.grid.lat_nodes, dtype=float)
        lon_1d = np.asarray(self.grid.lon_nodes, dtype=float)

        for ev in fires:
            # met at fire's nearest cell (use hour 0 for thermal estimate)
            fi = min(max(int(np.argmin(np.abs(lat_1d - ev.lat))), 0), ny - 1)
            ci = min(max(int(np.argmin(np.abs(lon_1d - ev.lon))), 0), nx - 1)
            met0 = ws.cell(0, fi, ci)
            res = self.rise(ev, met0)
            rises.append(res)

            for hi in range(nt):
                met = ws.cell(hi, fi, ci)
                ws10 = max(0.2, met["ws10_ms"])
                wdir = met["wd_deg"]
                # unit vector of flow (transport toward downwind)
                ux = np.sin(np.radians(wdir))  # NOTE: meteorological dir
                uy = np.cos(np.radians(wdir))

                dlat = lats - ev.lat
                dlon = (lons - ev.lon) * np.cos(np.radians(ev.lat))
                x_km = dlon * _KM_PER_DEG * ux + dlat * _KM_PER_DEG * uy
                y_km = dlat * _KM_PER_DEG * ux - dlon * _KM_PER_DEG * uy

                # downwind coordinate must be positive to be under plume
                sigma_y = max(6.0, res.z_top_m / 100.0 + 4.0)  # ~ plume width
                dist_km = x_km
                gaussian = np.exp(-0.5 * (y_km / sigma_y) ** 2)
                along = np.exp(-dist_km / max(30.0, ws10 * 12.0)) if np.any(dist_km >= 0) else np.zeros_like(x_km)
                # only downwind side (x_km >= 0)
                along = np.where(x_km >= -2.0, along, 0.0)
                conc_cell = gaussian * along
                conc[hi] += conc_cell * (ev.frp_mw / 50.0)
                tdir[hi] = wdir
                width[hi] = sigma_y

        # normalize to 0..1
        cmax = conc.max() if conc.max() > 0 else 1.0
        conc = conc / cmax
        return PlumeState(
            hours=hours, fires=fires, rises=rises, concentration=conc,
            transport_dir_deg=tdir, width_km=width, ny=ny, nx=nx,
        )