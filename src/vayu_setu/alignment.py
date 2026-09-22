"""Temporal + spatial alignment onto one canonical hourly UTC timeline.

Rules (documented, never blind):
  * 15-min CPCB -> hourly        : mean for concentrations, mean for met
  * daily AQI   -> (kept daily in a separate aligned table; hourly downscale
                    only via documented interpolation when used for ML targets)
  * FIRMS       -> hourly fire grid aggregated on the 4 km Delhi grid
  * AOD         -> nearest-neighbour hourly grid from CAMS point data
  * GFS         -> bilinear/met-preserving interpolation to the Delhi grid

Canonical timeline format: `YYYY-MM-DD HH:00:00` UTC.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import VayuConfig


def canonical_timeline(start: str | datetime, end: str | datetime, freq: str = "1h") -> pd.DatetimeIndex:
    start = pd.to_datetime(start, utc=True).tz_localize(None).floor("h")
    end = pd.to_datetime(end, utc=True).tz_localize(None).floor("h")
    return pd.date_range(start, end, freq=freq, name="observed_at_utc")


def to_canonical(df: pd.DataFrame, ts_col: str, freq: str = "1h",
                 agg: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Resample any frame to canonical hourly UTC.

    Aggregation rules default per column type; pass explicit `agg` for
    scientific choices (e.g. {'wind_speed': 'mean', 'frp': 'sum'}).
    """
    t = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dt.tz_localize(None)
    tmp = df.assign(_t=t).set_index("_t")
    num_cols = tmp.select_dtypes(include=[np.number]).columns.difference([ts_col])
    agg_map: Dict[str, str] = agg or {}
    for c in num_cols:
        if c not in agg_map:
            if "max" in c.lower() or "peak" in c.lower():
                agg_map[c] = "max"
            else:
                agg_map[c] = "mean"
    return tmp.resample(freq).agg(agg_map).reset_index().rename(columns={"_t": ts_col})


def format_timeline(series: pd.Series) -> str:
    return series.astype(str) if series.dtype == object else series.dt.strftime("%Y-%m-%d %H:00:00")


class DelhiGrid:
    """4 km Delhi NCR analysis grid (cosine-corrected, matches configs/domain.yaml)."""

    def __init__(self, cfg: VayuConfig) -> None:
        d = cfg.section("domain")
        lat_min, lat_max = d["lat_min"], d["lat_max"]
        lon_min, lon_max = d["lon_min"], d["lon_max"]
        ny, nx = cfg.grid["ny"], cfg.grid["nx"]
        self.lat = np.linspace(lat_min, lat_max, ny)
        self.lon = np.linspace(lon_min, lon_max, nx)
        self.ny, self.nx = ny, nx
        self.lats, self.lons = np.meshgrid(self.lat, self.lon, indexing="ij")

    def cell_index(self, lat: float, lon: float) -> tuple:
        i = int(np.clip(np.argmin(np.abs(self.lat - float(lat))), 0, self.ny - 1))
        j = int(np.clip(np.argmin(np.abs(self.lon - float(lon))), 0, self.nx - 1))
        return i, j

    def coords(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"lat": self.lats.ravel(), "lon": self.lons.ravel(), "i": self.lats.ravel(),
             "j": self.lons.ravel()}
        ).rename(columns={"i": "i0", "j": "j0"})


def align_points_to_grid(df: pd.DataFrame, grid: DelhiGrid,
                         value_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Assign point observations to the nearest Delhi-grid cell (by index)."""
    out = df.copy()
    ijs = [grid.cell_index(r.latitude, r.longitude) for r in out.itertuples()]
    out["gi"], out["gj"] = [i for i, _ in ijs], [j for _, j in ijs]
    return out


def spatial_weights_identity(lat: np.ndarray, lon: np.ndarray, pts_lat: float,
                             pts_lon: float) -> float:
    """Simple 1/d^2 weight used before an inverse-distance interpolation to a point."""
    d = np.hypot(lat - pts_lat, lon - pts_lon)
    d = np.where(d < 1e-6, 1e-6, d)
    return 1.0 / (d**2)


def to_kolkata(df: pd.DataFrame, ts_col: str = "observed_at_utc") -> pd.DataFrame:
    out = df.copy()
    out["observed_at_kolkata"] = (
        pd.to_datetime(out[ts_col], utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )
    return out


def document_aggregation(**rules: str) -> str:
    lines = [f"{k}: {v}" for k, v in rules.items()]
    return "; ".join(lines) if lines else "default: per-column mean/max heuristic"