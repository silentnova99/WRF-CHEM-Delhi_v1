"""1-D integral fire plume rise (Freitas et al. 2007 style).

Given a fire's convective heat release and the ambient sounding, integrates
the Boussinesq plume equations (mass, momentum, energy) in height:

    dM/dz  = 2 r ( alpha*|w| + alpha_wind*|u_a(z)| )     M = r^2 w
    dKm/dz = r^2 g (T'/T0)                                Km = r^2 w^2
    dEn/dz = -(dT_a/dz) (r^2 w) / T0                      En = r^2 w (T'/T0)

The fire heat release sets the initial energy flux
``En0 = Q_conv/(rho0 cp T0)``. Entrainment grows with updraft speed *and*
ambient wind (bent-over dilution). The plume terminates where the updraft is
spent (``w <= 0.2 m/s``) or a hard ceiling is reached.  The returned vertical
mass histogram (centred on the peak-updraft height) is used by the emission
coupling to distribute smoke across model layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_SPEC = 287.05  # J/(kg K) dry air


@dataclass
class PlumeResult:
    frp_mw: float
    z_centre_m: float                 # height of peak updraft (injection centre)
    z_top_m: float                    # plume top (all smoke retained below)
    z_neutral_m: float                # first height where buoyancy excess <= 0
    mb_flux_per_layer: np.ndarray     # [n_layers] fraction of mass per layer
    layer_edges_m: np.ndarray         # [n_layers + 1]


def pressure_height(p_hpa: float, t_sfc_k: float, g: float = 9.81) -> float:
    """Geopotential height (m) of a pressure level via barometric integration.

    Uses a constant scale height from ``T_sfc`` for the ~800–1000 hPa levels
    used to build plume ambient profiles.
    """
    scale = R_SPEC * t_sfc_k / g
    return float(scale * np.log(1013.25 / p_hpa))


def plume_rise(
    frp_mw: float,
    *,
    t_sfc_k: float,
    p_sfc_hpa: float,
    z_k: np.ndarray,
    t_k: np.ndarray,
    ws_k: np.ndarray,
    sensible_fraction: float = 0.45,
    entrainment_coef: float = 0.6,
    wind_entrainment_coef: float = 0.35,
    z_step: float = 50.0,
    max_top_m: float = 6500.0,
    n_layers: int = 12,
    area_per_frp_km2_mw: float = 0.01,
    min_area_km2: float = 0.005,
    r0: float | None = None,
) -> PlumeResult:
    """Integrate plume rise and return the vertical injection histogram.

    ``r0`` defaults to the equivalent radius of the burned area which itself
    scales with FRP (``area ~ area_per_frp_km2_mw * FRP``), so bigger fires are
    broader *and* more buoyant, which is what lifts smoke highest.
    """
    frp_mw = float(frp_mw)
    assert frp_mw >= 0.0 and len(z_k) >= 2
    z_k = np.asarray(z_k, dtype=float)
    t_k = np.asarray(t_k, dtype=float)
    ws_k = np.asarray(ws_k, dtype=float)

    cp = 1005.0
    g = 9.81
    t0 = float(t_sfc_k)
    rho0 = (p_sfc_hpa * 100.0) / (R_SPEC * t0)
    q_conv = sensible_fraction * frp_mw * 1.0e6            # W (MW -> W)
    en0 = q_conv / (rho0 * cp * t0)
    f0 = g * en0                                           # buoyancy flux (m4/s3)
    if r0 is None:
        area = max(area_per_frp_km2_mw * frp_mw, min_area_km2) * 1.0e6
        r0 = float(np.sqrt(area / np.pi))
    r0 = float(r0)
    if frp_mw <= 0.0:
        m, km, en = 1.0, 1.0, 0.0
        z = 1.0
    else:
        w0 = max(float((f0 / r0) ** (1.0 / 3.0)), 0.3)
        m, km, en = (r0 * r0) * w0, (r0 * r0) * w0 * w0, en0
        z = 1.0
    w_max = 0.0
    z_centre, z_neutral, z_top = 1.0, max_top_m, max_top_m

    for _ in range(int((max_top_m - z) / z_step) + 1):
        w = km / max(m, 1e-12)
        r2 = max((m * m) / max(km, 1e-12), 1e-6)
        theta = en / max(m, 1e-12)
        u_a = float(np.interp(z, z_k, ws_k))
        lapse = float(np.interp(z, z_k, np.gradient(t_k, z_k)))
        ent = entrainment_coef * abs(w) + wind_entrainment_coef * abs(u_a)

        m += z_step * 2.0 * float(np.sqrt(r2)) * ent
        km += z_step * r2 * g * theta
        en += z_step * -(lapse / t0) * r2 * w

        z += z_step
        if z >= max_top_m or m <= 0.0:
            break
        if w > w_max:
            w_max = w
            z_centre = z
        if z_neutral == max_top_m and theta <= 0.0:
            z_neutral = z
        if w < 0.2:
            break
    z_top = float(min(z, max_top_m))
    if z_neutral == max_top_m:
        z_neutral = z_top
    if frp_mw <= 0.0:
        z_centre, z_neutral, z_top = 1.0, 1.0, 1.0

    edges = np.linspace(0.0, z_top, n_layers + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    span = max((z_top - z_centre) * 0.6, 200.0)
    frac = np.exp(-0.5 * ((mid - z_centre) / span) ** 2)
    frac = frac / max(frac.sum(), 1e-12)
    if z_top <= 1.0:
        frac[:] = 1.0 / n_layers
    return PlumeResult(
        frp_mw=frp_mw,
        z_centre_m=z_centre,
        z_top_m=z_top,
        z_neutral_m=z_neutral,
        mb_flux_per_layer=frac,
        layer_edges_m=edges,
    )