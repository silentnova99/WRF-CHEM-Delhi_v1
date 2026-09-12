"""Deterministic synthetic experiment for Module-3 (offline verification).

Simulates a physically-motivated "truth" PM field (seasonal + diurnal +
meteorology + stubble-fire downwind smoke), a *biased* WRF-Chem style model
field, and noisy CPCB station observations. The bias is structured —
winter inversion episodes cause model under-prediction, and densely urban
stations carry a micro-UHI offset — exactly the effects the ST-GNN+XGBoost
engine is designed to remove. Seeded RNG ⇒ reproducible demo runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from aqf_delhi.ml.graph import haversine_km


@dataclass
class SyntheticData:
    times: np.ndarray            # datetime64[h] [T]
    station_ids: list[str]
    lat: np.ndarray              # [N]
    lon: np.ndarray              # [N]
    raw: dict[str, np.ndarray]   # per-key [T, N] arrays aligned to FeatureSpec
    target: np.ndarray           # obs PM2.5 [T, N]
    fire_meta: dict              # source info for inspection
    split_train_end: int
    split_val_end: int
    split_test_end: int

    def split(self, which: str):
        """Return contiguous (raw, target) window for train|val|test."""
        lo = {"train": 0, "val": self.split_train_end, "test": self.split_val_end}[which]
        hi = {"train": self.split_train_end, "val": self.split_val_end, "test": self.split_test_end}[which]
        out = {k: v[lo:hi] for k, v in self.raw.items()}
        return out, self.target[lo:hi], self.times[lo:hi]


# --------------------------------------------------------------------------- #
class SyntheticDelhiExperiment:
    def __init__(
        self,
        stations: list[dict],
        *,
        n_days: int = 90,
        start: datetime | None = None,
        seed: int = 7,
    ) -> None:
        self.stations = stations
        self.n_days = n_days
        self.start = start or datetime(2026, 9, 1)
        self.rng = np.random.default_rng(seed)

    def generate(self) -> SyntheticData:
        start = self.start
        times = [start + timedelta(hours=h) for h in range(self.n_days * 24)]
        t = len(times)
        doys = np.array([ts.timetuple().tm_yday for ts in times], dtype=float)
        hours = np.array([ts.timetuple().tm_hour for ts in times], dtype=float)
        n = len(self.stations)
        lat = np.array([float(s["lat"]) for s in self.stations])
        lon = np.array([float(s["lon"]) for s in self.stations])

        # ---- meteorology ----
        dayness = 0.5 * (1.0 + np.cos(2 * math.pi * (hours - 13) / 24))       # [T]
        # winter inversion weighting: Oct (274)..Jan forward ramp
        winter = np.clip(_winter_weight(doys), 0.0, 1.0)                       # [T]
        inv_occ = (1.0 - dayness) * (0.15 + 0.85 * winter)                     # [T]
        ws_base = 2.2 + 1.6 * np.cos(2 * math.pi * (hours - 14) / 24)
        ws = np.clip(ws_base * (1 - 0.35 * inv_occ), 0.4, 8.0)                 # [T]
        wd = 300.0 + 20.0 * self.rng.normal(size=t)                            # WNW winds [T]
        pblh = 150 + 900 * dayness * (1 + 0.15 * winter) - 300 * inv_occ       # [T]
        pblh = np.clip(pblh, 80, 1500)
        gamma = 0.05 - 0.9 * dayness + 1.35 * inv_occ                          # K/100m [T]
        ri_bulk = 0.15 + 3.4 * inv_occ + 0.5 * self.rng.normal(size=t)         # [T]
        t2m = 32 - 6 * _winter_weight(doys) - 4 * dayness + self.rng.normal(size=t)
        rh = 45 + 25 * (1 - dayness) + 10 * winter
        vent = np.clip(pblh * ws / 1000.0, 0.05, 4.0)                          # [T]
        aod = 0.25 + 0.6 * inv_occ + 0.15 * winter + 0.15 * self.rng.normal(size=t)

        # ---- fire forcing (Punjab/Haryana stubble, upwind NW) ----
        fire_srcs = [(28.95, 76.90), (29.20, 76.60), (28.60, 76.55)]
        burn_season = np.array([1.0 if 274 <= d <= 334 else 0.25 for d in doys])
        burn_hour = np.clip(1.0 - np.abs(hours - 15) / 4.0, 0.05, 1.0)        # peak ~15:00
        burn = burn_season * burn_hour                                          # [T]
        frp_local = np.zeros((t, n))
        fire_exposure = np.zeros((t, n))
        for (slat, slon) in fire_srcs:
            d = _hav(slat, slon, lat, lon)
            radial = np.exp(-((d / 25.0) ** 2))                                # [N]
            frp_local += radial[None, :] * (120.0 * burn)[:, None]
            # downwind integration along prevailing 120-deg transport
            ux, uy = math.cos(math.radians(120)), math.sin(math.radians(120))
            dx, dy = (slat - lat), (slon - lon)
            along = dx * uy + dy * ux                                          # [N]
            cross_sq = (dx * ux - dy * uy) ** 2
            plume = np.exp(-cross_sq / 800.0) * np.exp(-np.maximum(along, 0) / 90.0)
            fire_exposure += plume[None, :] * (1.0 * burn)[:, None]

        # ---- truth PM2.5 ----
        urban = np.array([float(s.get("urban", 1.0)) for s in self.stations])  # [N]
        seasonal_level = 40 + 120 * _winter_weight(doys)                        # [T]
        diurnal_pm = 12 + 10 * (1 - dayness)                                    # [T]
        meteor_scale = (1 + 1.6 * inv_occ) / (0.6 + 0.5 * vent / vent.max())
        base_truth = seasonal_level + diurnal_pm                                # [T]
        truth = np.broadcast_to(
            (base_truth[:, None] * np.clip(meteor_scale, 0.3, 2.5)[:, None]),
            (t, n),
        ).copy()
        truth += 18.0 * urban[None, :]                                          # local emissions
        truth += 260.0 * fire_exposure                                          # stubble smoke
        truth += self.rng.normal(0.0, 4.0, size=(t, n))
        truth = np.clip(truth, 5.0, 900.0)

        # ---- biased WRF-Chem model field ----
        b_mult = 0.74 + 0.22 * (vent / vent.max())[:, None] - 0.12 * inv_occ[:, None]
        b_add = (-6.0 + 12.0 * inv_occ)[:, None]
        uhi_bias = 5.0 * urban[None, :]
        model = truth * b_mult + b_add + uhi_bias + self.rng.normal(0, 6, size=(t, n))
        model = np.clip(model, 3.0, 1000.0)
        # PM10 & O3 (correlated spreads)
        model_pm10 = 1.55 * model * np.clip(0.9 + 0.2 * inv_occ[:, None], 0.8, 1.3)
        model_o3 = 120 * (0.4 + 0.6 * dayness) - 30 * inv_occ
        o3_bias = 0.9 * model_o3 - 4
        o3_bias = np.clip(o3_bias[:, None] * np.ones((1, n)) + self.rng.normal(0, 4, (t, n)), 4, 160)

        # ---- CPCB observations (noisy truth) ----
        obs = truth + self.rng.normal(0.0, 10.0, size=(t, n))
        spike = (self.rng.random((t, n)) < 0.015)
        obs = np.where(spike, obs + self.rng.normal(60, 25, size=(t, n)), obs)
        obs = np.clip(obs, 0.0, 1200.0)

        raw = {
            "pm25_raw": model,
            "pm10_raw": model_pm10,
            "o3_raw": o3_bias,
            "pblh_m": _column(pblh, n),
            "gamma": _column(gamma, n),
            "ri_bulk": _column(ri_bulk, n),
            "rh_pct": _column(rh, n),
            "ws_ms": _column(ws, n),
            "wd_deg": _column(wd, n),
            "aod": _column(aod, n),
            "t2m_c": _column(t2m, n),
            "frp_local_mw": frp_local,
            "fire_exposure": fire_exposure,
            "obs_pm25": obs,                    # [T,N] observation field (imputed)
            "obs_manual": np.zeros((t, n)),
        }
        # observed PM10 (proxy) + ozone proxies feed lag features only
        raw["obs_pm10"] = np.clip(1.5 * truth + self.rng.normal(0, 15, (t, n)), 0, 1500)
        raw["obs_o3"] = np.clip(model_o3[:, None] * np.ones((1, n)) + self.rng.normal(0, 5, (t, n)), 0, 160)

        # impute the first row of obs (feature builder lags; keep raw consistent)
        raw["obs_pm25"][0] = raw["obs_pm25"][1]

        tres = t - 144                  # 72 h validation + 72 h test after train
        peak_doy = int(doys[np.argmax(burn)])
        return SyntheticData(
            times=np.array(times, dtype="datetime64[h]"),
            station_ids=[s["station_id"] for s in self.stations],
            lat=lat, lon=lon, raw=raw, target=obs,
            fire_meta={"sources": fire_srcs, "peak_burn_doy": peak_doy},
            split_train_end=tres,
            split_val_end=t - 72,
            split_test_end=t,
        )


# --------------------------------------------------------------------------- #
def _winter_weight(doy: np.ndarray) -> np.ndarray:
    """Seasonal pollution weight: high Oct–Feb, low mid-year. [0,1]."""
    d = np.asarray(doy, dtype=float)
    w = np.clip((d - 260) / 40.0, 0.0, 1.0)   # late-Sep → Oct ramp
    w = np.where(d < 60, 0.7 + 0.3 * np.clip(d / 60.0, 0, 1), w)   # Jan tail
    w = np.where(
        (d > 120) & (d < 260),
        0.35 + 0.15 * np.sin((d - 120) / 140.0 * math.pi),          # summer dip
        w,
    )
    return np.clip(w, 0.0, 1.0)


def _column(arr: np.ndarray, n: int) -> np.ndarray:
    return np.asarray(arr, dtype=float)[:, None] * np.ones((1, n))


def _hav(slat: float, slon: float, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return np.array(
        [
            haversine_km(slat, slon, la, lo)
            for la, lo in zip(lat.ravel(), lon.ravel())
        ]
    )