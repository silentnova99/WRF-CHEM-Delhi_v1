"""Feature engineering tower (Module-3 section 3.3).

Converts raw per-station inputs — physical-model fields, observed
pollutants, fire forcing, and calendar/geometric context — into the flat
feature vector consumed by the T-GCN and the XGBoost head.

Cyclical calendar terms (hour-of-day, day-of-year) use sin/cos encoding so
midnight/new-year wrap-around is preserved.

The ``FeatureSpec`` enumerates the ordered feature names; :func:`FeatureSpec.from_raw`
builds the [T, N, F] tensor for a given raw input stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Raw input keys fed by the dataset loader (model output + obs + fire).
RAW_KEYS = [
    # physical model output
    "pm25_raw",       # WRF-Chem PM2.5 (µg/m3)
    "pm10_raw",       # WRF-Chem PM10
    "o3_raw",         # WRF-Chem ozone
    "pblh_m",         # PBL height (m)
    "gamma",          # lapse rate proxy (K/100m, +stable/inversion)
    "ri_bulk",        # bulk Richardson number
    "rh_pct",
    "ws_ms",          # 10-m wind speed
    "wd_deg",         # 10-m wind direction
    "aod",            # aerosol optical depth
    "t2m_c",          # 2-m temperature
    # fire forcing
    "frp_local_mw",   # FIRMS FRP aggregated per station footprint
    "fire_exposure",  # downwind-integrated fire exposure (g/m2·s proxy)
    # station observations (lagged, from CPCB)
    "obs_pm25",       # observed PM2.5 (µg/m3)
    "obs_pm10",
    "obs_o3",
    "obs_manual",
]

CYCLIC = {"hour", "doy"}


@dataclass
class FeatureSpec:
    """Ordered feature plan. ``include:list[str]`` controls the engineering."""

    include: list[str] = field(
        default_factory=lambda: [
            "raw",
            "hour_cyclic",
            "doy_cyclic",
            "lag1",
            "lag24",
            "rolling24",
            "inv_sig",
            "logscale",
            "trend",
        ]
    )

    @property
    def names(self) -> list[str]:
        names: list[str] = []
        if "raw" in self.include:
            names += RAW_KEYS
        if "hour_cyclic" in self.include:
            names += ["hour_sin", "hour_cos"]
        if "doy_cyclic" in self.include:
            names += ["doy_sin", "doy_cos"]
        if "lag1" in self.include:
            names += ["lag1_pm25", "lag1_pm10", "lag1_o3"]
        if "lag24" in self.include:
            names += ["lag24_pm25"]
        if "rolling24" in self.include:
            names += ["roll24_pm25", "roll24_pm10"]
        if "inv_sig" in self.include:
            names += ["inv_strength", "vent_coef"]
        if "logscale" in self.include:
            names += ["log_obs_pm25"]
        if "trend" in self.include:
            names += ["mdl_trend", "mdl_obs_bias"]
        return names

    @property
    def n_features(self) -> int:
        return len(self.names)

    def from_raw(
        self, raw: dict[str, np.ndarray], *, hours: np.ndarray | None, doys: np.ndarray | None
    ) -> np.ndarray:
        """Build [T, N, F] features from aligned per-station arrays.

        Each entry in ``raw`` is [T, N]. ``hours``/``doys`` are int arrays [T].
        ``obs_pm25`` etc. must be *lagged/imputed* already by the loader.
        """
        cols: dict[str, np.ndarray] = {}
        t, n = list(raw.values())[0].shape

        if "raw" in self.include:
            for key in RAW_KEYS:
                arr = raw.get(key)
                cols[key] = _safe_dtype(arr, t, n)

        if "hour_cyclic" in self.include:
            h = np.asarray(hours, dtype=float)
            cols["hour_sin"] = np.broadcast_to(np.sin(2 * math.pi * h / 24)[:, None], (t, n))
            cols["hour_cos"] = np.broadcast_to(np.cos(2 * math.pi * h / 24)[:, None], (t, n))

        if "doy_cyclic" in self.include:
            d = np.asarray(doys, dtype=float)
            cols["doy_sin"] = np.broadcast_to(np.sin(2 * math.pi * d / 365.25)[:, None], (t, n))
            cols["doy_cos"] = np.broadcast_to(np.cos(2 * math.pi * d / 365.25)[:, None], (t, n))

        obs = raw.get("obs_pm25")
        if "lag1" in self.include and obs is not None:
            cols["lag1_pm25"] = _lag(obs, 1, t, n)
            if raw.get("obs_pm10") is not None:
                cols["lag1_pm10"] = _lag(raw["obs_pm10"], 1, t, n)
            if raw.get("obs_o3") is not None:
                cols["lag1_o3"] = _lag(raw["obs_o3"], 1, t, n)

        if "lag24" in self.include and obs is not None:
            cols["lag24_pm25"] = _lag(obs, 24, t, n)

        if "rolling24" in self.include:
            if obs is not None:
                cols["roll24_pm25"] = _rolling(obs, 24, t, n)
            if raw.get("obs_pm10") is not None:
                cols["roll24_pm10"] = _rolling(raw["obs_pm10"], 24, t, n)

        if "inv_sig" in self.include:
            gamma = raw.get("gamma")
            ri = raw.get("ri_bulk")
            ws = raw.get("ws_ms")
            pblh = raw.get("pblh_m")
            if gamma is not None and ri is not None:
                cols["inv_strength"] = _inversion_strength(gamma, ri)
            if ws is not None and pblh is not None:
                cols["vent_coef"] = _ventilation(pblh, ws)

        if "logscale" in self.include and obs is not None:
            cols["log_obs_pm25"] = np.log1p(np.clip(obs, 0.0, None))

        if "trend" in self.include:
            pm = raw.get("pm25_raw")
            if pm is not None:
                cols["mdl_trend"] = np.clip(pm - _lag(pm, 1, t, n), -300.0, 300.0)
            if pm is not None and obs is not None:
                cols["mdl_obs_bias"] = pm - obs

        return np.stack([cols[name] for name in self.names], axis=-1)


# --------------------------------------------------------------------------- #
def _safe_dtype(arr, t: int, n: int) -> np.ndarray:
    if arr is None:
        return np.zeros((t, n), dtype=float)
    return np.asarray(arr, dtype=float).reshape(t, n)


def _lag(arr: np.ndarray, k: int, t: int, n: int) -> np.ndarray:
    out = np.zeros((t, n), dtype=float)
    out[k:] = np.asarray(arr)[:-k]
    return out


def _rolling(arr: np.ndarray, w: int, t: int, n: int) -> np.ndarray:
    vals = np.asarray(arr, dtype=float)
    out = np.zeros((t, n), dtype=float)
    cum = np.cumsum(np.nan_to_num(vals), axis=0)
    for i in range(t):
        lo = max(0, i - w + 1)
        out[i] = (cum[i] - cum[lo - 1]) / (i - lo + 1) if lo > 0 else cum[i] / (i + 1)
    return out


def _inversion_strength(gamma: np.ndarray, ri: np.ndarray) -> np.ndarray:
    """0..1 trapping strength: logistic of stability + Richardson."""
    g = np.asarray(gamma, dtype=float)
    r = np.asarray(ri, dtype=float)
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))

    return sig((g - 0.5) / 0.3) * sig((r - 0.6) / 0.4)


def _ventilation(pblh_m: np.ndarray, ws_ms: np.ndarray) -> np.ndarray:
    return np.asarray(pblh_m, dtype=float) * np.asarray(ws_ms, dtype=float) / 1000.0