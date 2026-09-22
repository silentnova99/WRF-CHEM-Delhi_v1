"""Coupled feature store builder.

Reads the aligned, cleaned frames and writes chronological splits:

    data/features/train/ | validation/ | test/

Each split carries the enriched long frame (city, time, targets, physics,
met, AOD, fire features + lags) as parquet, plus a station-registry table
reused by the graph builder and the API.

Split boundaries are strictly chronological (no future leakage) and are
recorded per split in `_split_meta.json`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import VayuConfig
from .lake import DataLake
from .physics import physics_features_from_frame

ENRICHED_SOURCE = "data/interim/aligned/enriched_cities.parquet"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


CANONICAL_COLUMNS: Dict[str, str] = {
    "City": "city",
    "State": "state",
    "Latitude": "lat",
    "Longitude": "lon",
    "Datetime": "observed_at_utc",
    "Temp_2m_C": "temp_2m_c",
    "Humidity_Percent": "rh_pct",
    "Dew_Point_C": "dew_point_c",
    "Wind_Speed_10m_kmh": "ws_kmh",
    "Wind_Speed_80m_kmh": "ws80_kmh",
    "Wind_Speed_120m_kmh": "ws120_kmh",
    "Wind_Dir_10m": "wd_deg",
    "Wind_Gusts_kmh": "gusts_kmh",
    "Wind_Stagnation": "wind_stagnation",
    "Precipitation_mm": "precip_mm",
    "Pressure_MSL_hPa": "pressure_msl_hpa",
    "Surface_Pressure_hPa": "surface_pressure_hpa",
    "Solar_Radiation_Wm2": "solar_radiation_wm2",
    "Direct_Radiation_Wm2": "direct_radiation_wm2",
    "Diffuse_Radiation_Wm2": "diffuse_radiation_wm2",
    "UV_Index": "uv_index",
    "Cloud_Cover_Percent": "cloud_cover_pct",
    "Is_Daytime": "is_daytime",
    "Sunshine_Seconds": "sunshine_seconds",
    "PM2_5_ugm3": "pm25_ugm3",
    "PM10_ugm3": "pm10_ugm3",
    "PM_Ratio": "pm_ratio",
    "CO_ugm3": "co_ugm3",
    "NO2_ugm3": "no2_ugm3",
    "SO2_ugm3": "so2_ugm3",
    "O3_ugm3": "o3_ugm3",
    "Dust_ugm3": "dust_ugm3",
    "AOD": "aod",
    "Temp_Inversion": "temp_inversion_flag",
    "Festival_Period": "festival_period",
    "Crop_Burning_Season": "crop_burning_season",
    "Wind_Stagnation": "wind_stagnation",
    "US_AQI": "us_aqi",
    "AQI_Category": "aqi_category_num",
    "PM25_Category_India": "pm25_category_india",
}

DROP_ALL_NAN = {"Temp_80m_C", "Temp_120m_C", "Temp_180m_C", "Inversion_Strength_C", "NH3_ugm3"}
DROP_ARTIFACTS = {"Year", "Month", "Day", "Hour", "Day_of_Week", "Day_Name", "Week_of_Year",
                  "Is_Weekend", "Quarter", "Season", "Time_of_Day", "Humidity_Category",
                  "Wind_Category", "Rain_mm", "Is_Raining", "Heavy_Rain", "Cloud_Low_Percent",
                  "Cloud_Mid_Percent", "Cloud_High_Percent", "US_AQI_PM25", "US_AQI_PM10",
                  "US_AQI_NO2", "US_AQI_O3", "US_AQI_CO", "EU_AQI", "EU_AQI_PM25", "EU_AQI_PM10",
                  "Sunshine_Seconds"}


def load_enriched_frame(cfg: VayuConfig, path: Optional[Path] = None) -> pd.DataFrame:
    """Read the canonical enriched city-hour frame (cache if absent => build)."""
    p = path or cfg.resolve(ENRICHED_SOURCE)
    if p.exists():
        return pd.read_parquet(p)
    raise FileNotFoundError(
        f"Enriched frame not found at {p}. Run the alignment/QC stage first."
    )


def build_enriched_frame(cfg: VayuConfig, lake: DataLake,
                         raw_path: Optional[Path] = None) -> pd.DataFrame:
    """Convert the real INDIA_AQI extract into the canonical enriched frame."""
    src = raw_path or (lake.raw_dir("cpcb") / "INDIA_AQI_COMPLETE_20251126.csv")
    if not src.exists():
        raise FileNotFoundError(f"CPCB extract not present in lake: {src}")
    df = pd.read_csv(src, low_memory=False)

    keep_cols: List[str] = []
    for c in df.columns:
        if c in DROP_ALL_NAN or c in DROP_ARTIFACTS:
            continue
        keep_cols.append(c)
    df = df[keep_cols]
    df = df.rename(columns=CANONICAL_COLUMNS)
    df["city"] = df["city"].str.strip()
    # Drop rare near-empty columns that slipped through
    df = df.drop(columns=[c for c in ["ws80_kmh", "ws120_kmh", "aqi_category_num"]
                          if c in df.columns and df[c].isna().all()])

    # canonical timestamp
    t = pd.to_datetime(df["observed_at_utc"], errors="coerce", utc=True).dt.tz_localize(None)
    df = df[(t.notna())].copy()
    df["observed_at_utc"] = t[t.notna()]
    df["_t"] = df["observed_at_utc"].dt.floor("h")
    for c in df.select_dtypes(include=[object]).columns:
        df[c] = df[c].astype(str).str.strip()
    for c in ["pm25_ugm3", "pm10_ugm3", "o3_ugm3", "co_ugm3", "no2_ugm3", "so2_ugm3",
              "temp_2m_c", "rh_pct", "ws_kmh", "wd_deg", "pressure_msl_hpa",
              "solar_radiation_wm2", "cloud_cover_pct", "precip_mm", "aod"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["city", "_t"]).reset_index(drop=True)
    return df


def add_fire_features(df: pd.DataFrame, fire_frame: pd.DataFrame,
                      node_static: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Merge hourly NW-India fire features onto the city-hour frame.

    Fire exposure per city is distance-decayed from the NW fire region: cities
    upwind (north-west) of Delhi receive higher weight using city lon/lat.
    """
    if fire_frame is None or fire_frame.empty:
        for c in ["fire_count", "total_frp", "mean_frp", "max_frp", "upwind_frp",
                  "close_frp_50km", "frp_6h", "frp_12h", "frp_24h", "frp_48h", "min_distance_km"]:
            df[c] = 0.0
        return df
    ff = fire_frame[["observed_at_utc", "fire_count", "total_frp", "mean_frp", "max_frp",
                     "upwind_frp", "close_frp_50km", "frp_6h", "frp_12h", "frp_24h",
                     "frp_48h", "min_distance_km"]].copy()
    ff["observed_at_utc"] = pd.to_datetime(ff["observed_at_utc"], errors="coerce").dt.floor("h")
    df = df.merge(ff, how="left", on="observed_at_utc", suffixes=("", "_fire"))
    for c in ["fire_count", "total_frp", "mean_frp", "max_frp", "upwind_frp",
              "close_frp_50km", "frp_6h", "frp_12h", "frp_24h", "frp_48h", "min_distance_km"]:
        df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0)
    return df


def add_lags(df: pd.DataFrame, max_lag: int = 48, targets=("pm25_ugm3", "pm10_ugm3", "o3_ugm3")) -> pd.DataFrame:
    """No-leakage lags: lagged target values within a city (filled forward only)."""
    out = df.copy()
    for city, g in out.groupby("city", sort=False):
        idx = g.index
        for target in targets:
            vals = pd.to_numeric(g[target], errors="coerce").ffill()
            for lag in [1, 3, 6, 12, 24, 48]:
                if lag <= max_lag:
                    out.loc[idx, f"{target}_lag{lag}"] = vals.shift(lag).values
        out.loc[idx, "pm25_roll24"] = pd.to_numeric(g["pm25_ugm3"], errors="coerce").rolling(
            24, min_periods=1).mean().values
    return out


def write_feature_splits(cfg: VayuConfig, lake: DataLake) -> Dict[str, Any]:
    """Build and cache enriched frame, then write chronological splits."""
    enriched = build_enriched_frame(cfg, lake)
    enriched = add_lags(enriched, max_lag=cfg.section("ml").get("lags_max_hours", 48))

    # apply physics-derived features
    enriched = physics_features_from_frame(cfg, enriched)
    enriched["_t"] = pd.to_datetime(enriched["observed_at_utc"], errors="coerce")

    aligned_dir = lake.ensure(lake.interim("aligned"))
    lake.write_parquet(enriched, aligned_dir / "enriched_cities.parquet")
    stations = (
        enriched.groupby("city", as_index=False)
        .agg(lat=("lat", "first"), lon=("lon", "first"),
             state=("state", "first"), n_hours=("_t", "count"))
    )
    lake.write_parquet(stations, lake.processed("station", "station_registry.parquet"))

    t_min = enriched["_t"].min()
    t_max = enriched["_t"].max()
    ml_cfg = cfg.section("ml")
    # chronological test = last 12 months, validation = the 12 months before, train = rest
    test_start = t_max - pd.DateOffset(months=12) - pd.Timedelta(hours=1)
    val_start = t_max - pd.DateOffset(months=24) - pd.Timedelta(hours=1)

    def write_split(name: str, mask: pd.Series) -> None:
        split = enriched[mask].drop(columns=["_t"]).reset_index(drop=True)
        out_dir = lake.features(name)
        lake.write_parquet(split, out_dir / "features.parquet")
        meta = {
            "split": name,
            "n_rows": int(len(split)),
            "cities": sorted(split["city"].unique().tolist()),
            "t_start": str(split["observed_at_utc"].min()),
            "t_end": str(split["observed_at_utc"].max()),
            "generated_at": _utcnow(),
        }
        lake.write_json(meta, out_dir / "_split_meta.json")

    write_split("train", enriched["_t"] < val_start)
    write_split("validation", (enriched["_t"] >= val_start) & (enriched["_t"] < test_start))
    write_split("test", enriched["_t"] >= test_start)

    return {
        "generated_at": _utcnow(),
        "t_min": str(t_min),
        "t_max": str(t_max),
        "test_start": str(test_start),
        "val_start": str(val_start),
        "stations": int(len(stations)),
        "rows": int(len(enriched)),
    }