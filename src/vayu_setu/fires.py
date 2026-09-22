"""FIRMS active-fire processing — NW-India filtering + hourly fire features.

Produces a real fire feature record per hour (Delhi NCR + upwind Punjab/Haryana):

    fire_count, total_FRP, mean_FRP, max_FRP, upwind_FRP, upwind_fire_count,
    close_FRP_50km, distance_to_Delhi, FRP_6h, FRP_12h, FRP_24h, FRP_48h

Raw detections are retained untouched in the lake; only copies are filtered.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import VayuConfig
from .lake import DataLake
from .qc import FireCleaner

DELHI = (28.6139, 77.2090)          # Delhi (New Delhi) reference point
NW_BBOX = {"lat_min": 27.5, "lat_max": 33.0, "lon_min": 73.5, "lon_max": 78.8}


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    r = 6371.0
    la1, la2 = np.radians(lat1), np.radians(lat2)
    dla = np.radians(lat2 - lat1)
    dlo = np.radians(lon2 - lon1)
    a = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def bearing_deg(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    la1, la2 = np.radians(lat1), np.radians(lat2)
    dlo = np.radians(lon2 - lon1)
    y = np.sin(dlo) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlo)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def load_firms_raw(lake: DataLake) -> pd.DataFrame:
    frames = []
    raw = lake.raw_dir("firms")
    for f in sorted(raw.glob("*.csv")):
        df = pd.read_csv(f, usecols=["latitude", "longitude", "acq_date", "acq_time",
                                     "frp", "confidence", "satellite", "instrument", "daynight"],
                         low_memory=False)
        frames.append(df.assign(origin_file=f.name))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def filter_nw_index(df: pd.DataFrame) -> pd.Series:
    b = NW_BBOX
    return (
        (df["latitude"].between(b["lat_min"], b["lat_max"]))
        & (df["longitude"].between(b["lon_min"], b["lon_max"]))
    )


def add_dist_bearing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["distance_km"] = haversine_km(out["latitude"].to_numpy(), out["longitude"].to_numpy(),
                                      DELHI[0], DELHI[1])
    out["bearing_deg"] = bearing_deg(out["latitude"].to_numpy(), out["longitude"].to_numpy(),
                                     DELHI[0], DELHI[1])
    return out


def _wind_from_bearing(direction_rad: float, bearing: np.ndarray) -> np.ndarray:
    """Met wind 'from' direction -> True if fire bearing is upwind of Delhi."""
    if direction_rad is None:
        return None
    # wind FROM wd; air flows toward wd+180. A fire is upwind if the air
    # arriving at Delhi crossed the fire: fire bearing within ±90 of wd.
    wd = np.degrees(direction_rad)
    diff = np.abs((bearing - wd + 180) % 360 - 180)
    return diff <= 90


def hourly_fire_features(cfg: VayuConfig, lake: DataLake, timeline: pd.DatetimeIndex,
                         wind_from_deg: Optional[pd.Series] = None) -> pd.DataFrame:
    """Aggregate cleaned NW-India fires to the canonical hourly timeline.

    `wind_from_deg` indexed by hour gives dynamic upwind weighting; if absent a
    fixed north-west sector (bearing 270-360) is used (documented).
    """
    raw = load_firms_raw(lake)
    if raw.empty:
        return pd.DataFrame()
    cleaner = FireCleaner(cfg)
    cleaned = cleaner.clean(raw)["df"]
    nw = cleaned[filter_nw_index(cleaned)].copy()
    nw = add_dist_bearing(nw)
    hourly = nw.set_index(pd.to_datetime(nw["observed_at_utc"], utc=True).dt.tz_localize(None)).sort_index()

    idx = pd.Index(timeline, name="observed_at_utc")
    fires = pd.DataFrame(index=idx).assign(
        fire_count=0, total_frp=0.0, mean_frp=0.0, max_frp=0.0,
        upwind_fire_count=0, upwind_frp=0.0, close_frp_50km=0.0, min_distance_km=np.nan,
    )
    if not hourly.empty:
        wd_series = wind_from_deg if wind_from_deg is not None and len(wind_from_deg) else None
        for ts, grp in hourly.groupby(hourly.index):
            if ts not in idx:
                continue
            upwind = None
            if wd_series is not None and ts in wd_series.index:
                upwind = _wind_from_bearing(np.radians(float(wd_series.get(ts))), grp["bearing_deg"].to_numpy())
            # fallback sector when no wind data (fixed north-west sector)
            if upwind is None:
                upwind = (grp["bearing_deg"].to_numpy() >= 270) & (grp["bearing_deg"].to_numpy() <= 360)
            close = grp["distance_km"].to_numpy() <= 50.0
            fires.loc[ts] = {
                "fire_count": int(len(grp)),
                "total_frp": float(grp["frp"].sum()),
                "mean_frp": float(grp["frp"].mean()),
                "max_frp": float(grp["frp"].max()),
                "upwind_fire_count": int(upwind.sum()),
                "upwind_frp": float(grp.loc[upwind, "frp"].sum()) if upwind.any() else 0.0,
                "close_frp_50km": float(grp.loc[close, "frp"].sum()) if close.any() else 0.0,
                "min_distance_km": float(grp["distance_km"].min()),
            }
    # rolling accumulation windows
    for win in (6, 12, 24, 48):
        fires[f"frp_{win}h"] = fires["total_frp"].rolling(win, min_periods=1).sum()
    fires["hourly_iso"] = fires.index.strftime("%Y-%m-%d %H:00:00")
    return fires.reset_index()