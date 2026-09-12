"""Offline coupled WRF-Chem emulator (Module-2 core).

Converts meteorology + fire emissions + a diffuse urban base into a gridded
raw PM2.5 field that Module-3 reads in place of the synthetic model output.
The emulator tracks two reservoirs (surface well-mixed and elevated lofted
smoke), applies first-order aging/wet loss, advects fire smoke with 2-D
Gaussian puffs driven by the grid-mean upper wind, and converts the grid to
station series for the downstream bias-correction stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.ndimage import gaussian_filter

from aqf_delhi.wrf.config import Module2Config
from aqf_delhi.wrf.emissions import FireDetection, fire_emissions, urban_base


def _wd(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Meteorological wind direction (deg) from u/v grids."""
    return (270.0 - np.degrees(np.arctan2(-v, -u))) % 360.0


def _dayness(hours: list[datetime], lon: float, lat: float) -> np.ndarray:
    """Fractional daytime for each hour [nt], from solar geometry."""
    out = np.zeros(len(hours))
    for i, h in enumerate(hours):
        utc = h.hour + h.minute / 60.0
        lst = (utc + lon * 24.0 / 360.0) % 24.0
        decl = -23.44 * np.cos(np.radians(360.0 / 365.0 * (h.timetuple().tm_yday + 10)))
        ha = np.radians((lst - 12.0) * 15.0)
        sin_elev = (np.sin(np.radians(lat)) * np.sin(np.radians(decl))
                    + np.cos(np.radians(lat)) * np.cos(np.radians(decl)) * np.cos(ha))
        out[i] = float(np.clip(sin_elev, 0.0, 1.0))
    return out


def _shift(arr: np.ndarray, dj: int, di: int) -> np.ndarray:
    """Translate an array by integer cells; zero-fill vacated edges."""
    out = np.zeros_like(arr)
    ny, nx = arr.shape
    j0, j1 = max(dj, 0), min(max(ny + dj, 0), ny)
    i0, i1 = max(di, 0), min(max(nx + di, 0), nx)
    if j1 <= j0 or i1 <= i0:
        return out
    out[j0:j1, i0:i1] = arr[j0 - dj:j1 - dj, i0 - di:i1 - di]
    return out


@dataclass
class MetArrays:
    """Regridded GFS fields on the Delhi grid (all [nt, ny, nx])."""
    hours: list[datetime]
    t2m_c: np.ndarray = field(repr=False)
    rh_pct: np.ndarray = field(repr=False)
    ws10_ms: np.ndarray = field(repr=False)
    wd_deg: np.ndarray = field(repr=False)
    gust_ms: np.ndarray = field(repr=False)
    prmsl_hpa: np.ndarray = field(repr=False)
    t925_c: np.ndarray = field(repr=False)
    t850_c: np.ndarray = field(repr=False)
    t700_c: np.ndarray = field(repr=False)
    u925: np.ndarray = field(repr=False)
    v925: np.ndarray = field(repr=False)
    u850: np.ndarray = field(repr=False)
    v850: np.ndarray = field(repr=False)
    u700: np.ndarray = field(repr=False)
    v700: np.ndarray = field(repr=False)

    def cell_at(self, hi: int, j: int, i: int) -> dict[str, float]:
        def _s(a): return float(a[hi, j, i])
        return dict(
            t2m_c=_s(self.t2m_c), rh_pct=_s(self.rh_pct),
            ws10_ms=_s(self.ws10_ms), wd_deg=_s(self.wd_deg),
            gust_ms=_s(self.gust_ms), prmsl_hpa=_s(self.prmsl_hpa),
            t925_c=_s(self.t925_c), t850_c=_s(self.t850_c), t700_c=_s(self.t700_c),
            ws925_ms=float(np.hypot(self.u925[hi, j, i], self.v925[hi, j, i])),
            ws850_ms=float(np.hypot(self.u850[hi, j, i], self.v850[hi, j, i])),
            ws700_ms=float(np.hypot(self.u700[hi, j, i], self.v700[hi, j, i])),
            pblh_m=500.0,
        )


@dataclass
class CoupledResult:
    hours: list[datetime]
    pm25_grid: np.ndarray           # [nt, ny, nx] µg/m³ raw model
    pm25_obs_demo: np.ndarray       # [nt, ny, nx] synthetic obs (demo)
    pm10_grid: np.ndarray           # [nt, ny, nx] µg/m³ raw model
    pm10_obs_demo: np.ndarray
    o3_grid: np.ndarray             # [nt, ny, nx] µg/m³ photochemical proxy
    o3_obs_demo: np.ndarray
    pblh: np.ndarray
    inversion: np.ndarray
    vent_idx: np.ndarray
    ws10_ms: np.ndarray


def run_emulator(
    cfg: Module2Config,
    grid,
    met: MetArrays,
    fires: list[FireDetection],
) -> CoupledResult:
    """Run the offline coupled emulator and return the gridded species fields.

    Primary particulates (PM2.5, PM10) share one transport kernel: emitted
    mass fills surface/elevated reservoirs with puff advection, decay, wet
    loss and inversion handling.  O3 is a secondary photochemical proxy driven
    by solar geometry, temperature and the same stability/inversion state.
    """
    seed = cfg.emulator.seed
    rng = np.random.default_rng(seed)
    nt = len(met.hours)
    ny, nx = grid.ny, grid.nx
    cell_m2 = (grid.lat_res_km() * 1000.0) * (grid.lon_res_km() * 1000.0)
    ratio = cfg.emissions.pm10_ratio

    mid_lat = grid.domain.lat_min + grid.domain.lat_span / 2
    mid_lon = grid.domain.lon_min + grid.domain.lon_span / 2
    day = _dayness(met.hours, mid_lon, mid_lat)[:, None, None]
    day = np.broadcast_to(day, (nt, ny, nx)).copy()

    inv = ((met.t925_c - met.t2m_c) > 3.0).astype(float)
    inv[met.ws10_ms > 4.0] = 0.0
    inv *= (day < 0.15).astype(float)
    pblh = np.clip(240.0 + 1560.0 * day - 720.0 * inv, 180.0, 2200.0)
    vent = pblh * (met.ws10_ms + 0.35)

    e_urb = urban_base(cfg, grid, met.hours)
    e_sfc_fire, e_elev_fire, _ = fire_emissions(cfg, grid, fires, met.hours, met.cell_at)

    # per-species emission grids: PM10 is PM2.5 scaled by the coarse ratio
    emit = {
        "pm25": (e_sfc_fire, e_elev_fire, e_urb),
        "pm10": (e_sfc_fire * ratio, e_elev_fire * ratio, e_urb * ratio),
    }
    c_sfc = {s: np.zeros((nt, ny, nx)) for s in emit}
    c_elev = {s: np.zeros((nt, ny, nx)) for s in emit}

    tau_mix = 2.2                     # hours for elevated smoke to mix down
    decay = cfg.emulator.decay_rate_h
    wet = cfg.emulator.wet_removal_per_rh01
    inv_bias = cfg.emulator.inversion_bias_mult
    sigma_km = cfg.emulator.diffusion_sigma_km
    cap = min(cfg.emulator.advect_cap_h, nt - 1)
    dlat = grid.lat_res_km()
    dlon = grid.lon_res_km()

    for t in range(nt):
        for s, (e_f_sfc, e_f_elev, e_u) in emit.items():
            c = c_sfc[s]
            ce = c_elev[s]
            c[t] += e_u[t] / (pblh[t] * cell_m2 + 1e-9) * 1e6

            for k in range(max(0, t - cap), t):
                df = t - k
                decay_fire = (1.0 - decay) ** df
                u_mean = float(np.mean(met.u850[k]))
                v_mean = float(np.mean(met.v850[k]))
                dx_km = u_mean * 3.6 * df
                dy_km = v_mean * 3.6 * df
                dj = int(round(dy_km / dlat))
                di = int(round(dx_km / dlon))
                sigma_px = sigma_km * np.sqrt(df) / max(dlon, 0.5)
                if e_f_sfc[k].sum() > 0:
                    p_sfc = gaussian_filter(e_f_sfc[k], sigma_px)
                    c[t] += _shift(p_sfc, dj, di) * decay_fire / (pblh[k] * cell_m2 + 1e-9) * 1e6
                if e_f_elev[k].sum() > 0:
                    p_elev = gaussian_filter(e_f_elev[k], sigma_px * 1.6)
                    ce[t] += _shift(p_elev, dj, di) * decay_fire / (pblh[k] * cell_m2 + 1e-9) * 1e6

            mix_down = np.clip(1.0 / tau_mix, 0.0, 0.5)
            c[t] += ce[t] * mix_down
            ce[t] *= (1.0 - mix_down)

            c[t] = gaussian_filter(c[t], max(sigma_km / max(dlon, 0.5) * 0.5, 0.3))

            c[t] *= np.exp(-decay)

            wet_frac = np.clip((met.rh_pct[t] - 70.0) / 30.0, 0.0, 1.0)
            c[t] *= (1.0 - wet * wet_frac)

            c[t] = np.where(inv[t].astype(bool), c[t] * inv_bias, c[t])

    # --- secondary O3 photochemical proxy (µg/m³, no emitted mass) ---------
    ch = cfg.chemistry
    tempf = np.clip((met.t2m_c - 20.0) / ch.o3_temp_k_c, 0.0, 1.0)
    o3 = (ch.o3_bg_ug_m3 * (0.35 + 0.65 * day)
          + ch.o3_photo_ug_m3 * day * tempf
          - ch.o3_titration_ug_m3 * (1.0 - day))
    o3 = np.where(inv > 0.0, o3 * ch.o3_inv_mult, o3)
    o3 = np.maximum(o3, 5.0)

    c_pm25 = c_sfc["pm25"]
    c_pm10 = c_sfc["pm10"]
    c_obs = c_pm25 + rng.normal(0.0, cfg.emulator.obs_noise_ug_m3, c_pm25.shape)
    c_obs = np.maximum(c_obs, 0.0)
    c_obs += cfg.emulator.urban_offset_ug_m3 * ((met.ws10_ms < 1.5) & (inv > 0.0)).astype(float)

    p10_obs = c_pm10 + rng.normal(0.0, cfg.emulator.obs_noise_ug_m3 * 1.1, c_pm10.shape)
    p10_obs = np.maximum(p10_obs, 0.0)
    p10_obs += cfg.emulator.urban_offset_ug_m3 * 1.3 * ((met.ws10_ms < 1.5) & (inv > 0.0)).astype(float)

    o3_obs = np.maximum(o3 + rng.normal(0.0, 3.0, o3.shape), 0.0)

    return CoupledResult(
        hours=met.hours,
        pm25_grid=c_pm25,
        pm25_obs_demo=c_obs,
        pm10_grid=c_pm10,
        pm10_obs_demo=p10_obs,
        o3_grid=o3,
        o3_obs_demo=o3_obs,
        pblh=pblh,
        inversion=inv,
        vent_idx=vent,
        ws10_ms=met.ws10_ms,
    )