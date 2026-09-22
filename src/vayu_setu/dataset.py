"""RealData datasets + reduced-order physical surrogate "raw model".

Bridges the aligned feature lake into the existing `aqf_delhi.ml` interface.

The "raw model" fields are produced by a *documented reduced-order physical
surrogate* at city level (persistence-decay + meteorology-driven emissions +
fire exposure + aerosol-radiation-PBL feedback trapping). The ML stages bias-
correct this field against real CPCB observations — the same role WRF-Chem
plays operationally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import VayuConfig
from .lake import DataLake

KMH_TO_MS = 1.0 / 3.6

SURROGATE_PARAMS = {
    "pm25_decay_base": 0.35,       # hourly fraction lost (ventilation + chemistry)
    "urban_emis_sfc": 9.0,         # ug/m3 per hour urban contribution, calm+cold
    "fire_to_pm": 0.045,           # upwind FRP MW -> ug/m3 over Delhi region
    "cold_boost_per_C": 0.03,      # surface-emission boost per degC below 18 C
    "nox_to_o3_dep": 0.035,        # PM2.5 (NOx proxy) -> O3 suppression
    "o3_bg": 38.0,                 # background O3 ug/m3
    "o3_solar_k": 40.0,            # peak daytime photochemical production
}


@dataclass
class SplitData:
    x: np.ndarray                  # [T, N, F]
    y: np.ndarray                  # [T, N] target observations (raw scale)
    hours: np.ndarray              # [T]
    doys: np.ndarray               # [T]
    times: np.ndarray              # [T] datetime64

    @property
    def t(self) -> int:
        return self.x.shape[0]

    @property
    def n(self) -> int:
        return self.x.shape[1]


@dataclass
class DataSplits:
    train: SplitData
    validation: SplitData
    test: SplitData
    station_ids: List[str]
    lat: np.ndarray
    lon: np.ndarray
    species: str
    feature_names: List[str]
    surrogate_params: Dict[str, float]
    coupling_features: bool


def shift_lag(arr: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(np.asarray(arr, dtype=float), np.nan)
    out[k:] = np.asarray(arr)[:-k]
    return out


def nnz(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return np.where(np.isfinite(a), a, 0.0)


def _pivot(f: pd.DataFrame, column: str, cities: List[str]) -> np.ndarray:
    sub = f[["_t", "city", column]].copy()
    sub[column] = pd.to_numeric(sub[column], errors="coerce").astype(float)
    sub = sub.dropna(subset=[column])
    if sub.empty:
        return None
    try:
        piv = sub.pivot_table(index="_t", columns="city", values=column, aggfunc="first")
    except Exception:
        return None
    piv = piv.reindex(columns=cities)
    return piv.to_numpy(dtype=float)


def build_splits(cfg: VayuConfig, lake: DataLake, split_frames: Dict[str, pd.DataFrame],
                 cities: List[str], species: str = "pm25",
                 coupling_features: bool = True) -> DataSplits:
    """Convert enriched split frames into standardized-free [T,N,F] tensors."""
    from aqf_delhi.ml.features import FeatureSpec

    obs_columns = {"pm25": "pm25_ugm3", "pm10": "pm10_ugm3", "o3": "o3_ugm3"}
    obs_col = obs_columns[species]
    spec = FeatureSpec()
    sp = dict(SURROGATE_PARAMS)

    def tensorize(f: pd.DataFrame) -> SplitData:
        f = f.copy()
        f["_t"] = pd.to_datetime(f["observed_at_utc"], errors="coerce").dt.tz_localize(None).dt.floor("h")
        f = f.sort_values(["city", "_t"]).reset_index(drop=True)
        grid_all = pd.DataFrame({"_t": np.unique(f["_t"].to_numpy())})
        t = grid_all["_t"].to_numpy()
        T = len(t)
        N = len(cities)
        cols = [c for c in f.columns if c not in {"city", "_t", "observed_at_utc", "state", "lat", "lon"}]
        cols = [c for c in cols if pd.to_numeric(f[c], errors="coerce").notna().any()]

        mats: Dict[str, np.ndarray] = {}
        for c in cols:
            m = _pivot(f, c, cities)
            if m is None:
                continue
            # forward-fill within node
            fill = np.full((T, N), np.nan)
            ordered = f[f["city"] == cities[0]][["_t"]].drop_duplicates() if False else None
            for ci, city in enumerate(cities):
                sub = f[f["city"] == city].set_index("_t").reindex(grid_all["_t"])
                col = pd.to_numeric(sub[c], errors="coerce").ffill().bfill().to_numpy(dtype=float)
                fill[:, ci] = col
            mats[c] = fill

        def m(name: str) -> np.ndarray:
            return mats.get(name, np.zeros((T, N), dtype=float))

        ws = m("ws_kmh") * KMH_TO_MS
        wd = np.radians(m("wd_deg"))
        u10 = ws * np.sin(wd)
        v10 = ws * np.cos(wd)
        rh = m("rh_pct")
        t2m = m("temp_2m_c")
        aod = m("aod_fused") if "aod_fused" in mats else m("aod")
        solar = m("solar_radiation_wm2")
        inv_flag = m("temp_inversion_flag")
        stagn = m("wind_stagnation")
        pbl_base = m("pblh_base_m")
        pbl_coup = m("pblh_coupled_m")
        fb_strength = m("feedback_strength")
        frp_up = m("upwind_frp")
        frp_close = m("close_frp_50km")
        min_dist = m("min_distance_km")
        min_dist = np.where(min_dist > 0, min_dist, 300.0)
        obs = m(obs_col)

        # --- reduced-order surrogate raw model (physics) ----------------------
        # No-leakage rule: the surrogate only sees OBSERVATIONS FROM THE PREVIOUS
        # HOUR (obs_pm25/obs_pm10 below), never the current-hour target y(t).
        obs_prev = nnz(shift_lag(m("pm25_ugm3"), 1))
        obs10_prev = nnz(shift_lag(m("pm10_ugm3"), 1))
        obs_o3_cur = nnz(shift_lag(m("o3_ugm3"), 1))
        vert = np.clip((pbl_coup if coupling_features else pbl_base) * ws / 1e3, 0.05, 30.0)
        loss = np.clip(sp["pm25_decay_base"] / (1.0 + 0.25 * vert) + 0.05 * (rh > 70.0), 0.02, 0.85)
        cold = np.clip((18.0 - t2m) / 8.0, 0.0, 1.5)
        stagn_v = 0.6 + 0.8 * np.clip(stagn, 0, 1)
        urban = sp["urban_emis_sfc"] * (1.0 + sp["cold_boost_per_C"] * 8.0 * cold) * stagn_v
        dist_decay = np.exp(-(min_dist / 300.0))
        fire = sp["fire_to_pm"] * (frp_up * dist_decay + 0.5 * frp_close)
        if coupling_features:
            trap = 1.0 + 0.35 * np.clip(fb_strength, 0.0, 1.0) * np.clip((obs_prev - 80.0) / 120.0, 0.0, 1.0)
        else:
            trap = np.ones_like(obs_prev)
        raw_pm = np.maximum(obs_prev * (1.0 - loss) + urban + fire, 0.0) * trap

        raw_pm10 = np.maximum(obs10_prev * (1.0 - loss) + 1.55 * urban + 1.6 * fire, 0.0) * trap

        day_sc = np.clip(solar / 400.0, 0.0, 1.0)
        o3_raw = np.clip(sp["o3_bg"] + sp["o3_solar_k"] * day_sc * (1.0 + 0.02 * np.clip(t2m - 15, -3, 10))
                         - sp["nox_to_o3_dep"] * obs_prev * (1.0 - day_sc), 0.0, 200.0)

        gamma = 0.10 * inv_flag + 0.05 * stagn - 0.01 * np.clip(ws, 0, 10)
        ri_bulk = 0.25 * inv_flag / (1.0 + np.clip(ws, 0, 8)) + 1e-3
        obs_pm25 = obs_prev
        obs_pm10 = obs10_prev
        obs_o3 = obs_o3_cur

        raw: Dict[str, np.ndarray] = {
            "pm25_raw": raw_pm,
            "pm10_raw": raw_pm10,
            "o3_raw": o3_raw,
            "pblh_m": pbl_coup if coupling_features else pbl_base,
            "gamma": gamma,
            "ri_bulk": ri_bulk,
            "rh_pct": rh,
            "ws_ms": ws,
            "wd_deg": m("wd_deg"),
            "aod": aod,
            "t2m_c": t2m,
            "frp_local_mw": 0.5 * frp_close + 0.3 * frp_up,
            "fire_exposure": sp["fire_to_pm"] * dist_decay * (frp_up + 0.5 * frp_close),
            "obs_pm25": obs_pm25,
            "obs_pm10": obs_pm10,
            "obs_o3": obs_o3,
            "obs_manual": np.zeros((T, N), dtype=float),
        }
        hours = pd.to_datetime(t).hour.to_numpy(dtype=float)
        doys = pd.to_datetime(t).dayofyear.to_numpy(dtype=float)
        x = spec.from_raw(raw, hours=hours, doys=doys)
        y = obs
        return SplitData(x=x, y=y, hours=hours, doys=doys, times=t)

    parts = {name: tensorize(f) for name, f in split_frames.items()}
    return DataSplits(
        train=parts["train"], validation=parts["validation"], test=parts["test"],
        station_ids=cities, lat=np.array([]), lon=np.array([]),
        species=species, feature_names=spec.names, surrogate_params=dict(sp),
        coupling_features=coupling_features,
    )


def load_split_frames(cfg: VayuConfig, lake: DataLake) -> Dict[str, pd.DataFrame]:
    return {
        "train": pd.read_parquet(lake.features("train", "features.parquet")),
        "validation": pd.read_parquet(lake.features("validation", "features.parquet")),
        "test": pd.read_parquet(lake.features("test", "features.parquet")),
    }


def stations_frame(lake: DataLake) -> pd.DataFrame:
    return pd.read_parquet(lake.processed("station", "station_registry.parquet"))