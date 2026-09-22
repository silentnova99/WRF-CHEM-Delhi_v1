"""Quality control: cleaning + reporting with loud failure on critical gaps.

Cleaning rules (never modify raw files; cleaned copies live under
data/interim/cleaned/{source}/):

  * duplicate timestamps -> keep last, flag/count
  * impossible concentrations (PM > hard ceiling, < explicit lower bound)
  * negative concentrations where physically invalid
  * sensor spikes via median-absolute-deviation (MAD) filter
  * ambiguous/invalid timestamps -> coerce to canonical hourly UTC
  * station outages -> per (station, month) uptime table
  * unit consistency -> explicit units metadata carried alongside
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import VayuConfig


@dataclass
class QcReport:
    source: str
    total_rows: int = 0
    valid_rows: int = 0
    missing_rows: int = 0
    duplicate_rows: int = 0
    invalid_rows: int = 0
    missing_percentage: float = 0.0
    temporal_coverage: Dict[str, Any] = field(default_factory=dict)
    spatial_coverage: Dict[str, Any] = field(default_factory=dict)
    stations: int = 0
    flagged_columns: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    generated_at: str = ""
    status: str = "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status,
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "missing_rows": self.missing_rows,
            "duplicate_rows": self.duplicate_rows,
            "invalid_rows": self.invalid_rows,
            "missing_percentage": self.missing_percentage,
            "temporal_coverage": self.temporal_coverage,
            "spatial_coverage": self.spatial_coverage,
            "stations": self.stations,
            "flagged_columns": self.flagged_columns,
            "anomalies": self.anomalies,
            "generated_at": self.generated_at,
        }


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class StationCleaner:
    """Cleans a long-form station observation frame.

    Expected columns: ``observed_at_utc`` (or ``timestamp``), ``location_name``
    (or ``station_id``), ``location_lat``/``location_lon`` (optional),
    plus pollutant columns ``pm25``, ``pm10``, ``o3``, ``no2``, ``co``, ``so2``.
    """

    POLLUTANTS = ("pm25", "pm10", "o3", "no2", "so2", "co")

    def __init__(self, cfg: VayuConfig) -> None:
        self.cfg = cfg
        self.max_conc = cfg.section("sources").get("cpcb", {}).get("max_value_ug_m3", 2000.0)
        self.lower_bound = cfg.section("sources").get("cpcb", {}).get("explicit_lower_bound", 0.0)
        self.mad_sigma = cfg.section("sources").get("cpcb", {}).get("mad_cutoff_sigma", 5.0)

    def clean(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Returns dict {df, dropped, report, flags}."""
        df = df.copy()
        report = QcReport(source="cpcb", generated_at=_utcnow())
        report.total_rows = len(df)
        flags = pd.DataFrame(index=df.index)
        flags["is_invalid"] = False

        # --- duplicate timestamps (per station when station id available) ---
        dup_cols = [c for c in ("observed_at_utc", "timestamp", "datetime") if c in df.columns]
        if dup_cols:
            key = [dup_cols[0]]
            if "location_name" in df.columns:
                key.append("location_name")
            report.duplicate_rows = int(df.duplicated(subset=key).sum())
            df = df[~df.duplicated(subset=key, keep="last")].copy()

        # --- timestamp coercion to canonical hourly UTC ---
        ts_col = dup_cols[0] if dup_cols else None
        if ts_col:
            parsed = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            bad_ts = parsed.isna()
            flags["is_invalid"] |= bad_ts
            report.invalid_rows += int(bad_ts.sum())
            df = df.loc[~bad_ts].copy()
            df[ts_col] = parsed.dt.tz_localize(None).dt.floor("h")

        # --- physical range checks on pollutants ---
        for p in self.POLLUTANTS:
            if p not in df.columns:
                continue
            numeric = pd.to_numeric(df[p], errors="coerce")
            out_of_range = (numeric < self.lower_bound) | (numeric > self.max_conc)
            flags["is_invalid"] |= out_of_range
            report.invalid_rows += int(out_of_range.sum())
            report.flagged_columns.append(p) if out_of_range.any() else None

        # --- sensor spikes via MAD (per station, per pollutant) ---
        for p in self.POLLUTANTS:
            if p not in df.columns:
                continue
            numeric = pd.to_numeric(df[p], errors="coerce")
            grp = df.groupby("location_name", dropna=False)[numeric.name] if "location_name" in df.columns else None
            if grp is not None:
                med = grp.transform("median")
                mad = (numeric - med).abs().groupby(df["location_name"]).transform("median")
            else:
                med = numeric.median()
                mad = (numeric - med).abs().median()
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (numeric - med) / (1.4826 * mad)
            spike = (z.abs() > self.mad_sigma) & numeric.notna()
            flags["is_invalid"] |= spike
            report.flagged_columns.append(f"{p}_spike") if spike.any() else None

        # --- missingness summary (on cleaned set before dropping) ---
        tmp = df.copy()
        for p in self.POLLUTANTS:
            if p in tmp.columns:
                m = tmp[p].isna().sum()
                report.missing_rows += int(m)
        had_missing = int(df.missing_rows.sum()) if hasattr(df, "missing_rows") else 0

        valid_mask = ~flags["is_invalid"]
        dropped = df.loc[~valid_mask].copy()
        df = df.loc[valid_mask].copy()

        report.valid_rows = len(df)
        report.invalid_rows = int(report.invalid_rows)
        report.missing_percentage = round(100.0 * report.missing_rows / max(report.total_rows, 1), 3)
        report.stations = int(df["location_name"].nunique()) if "location_name" in df.columns else 1

        if ts_col:
            report.temporal_coverage = {
                "start": str(df[ts_col].min()),
                "end": str(df[ts_col].max()),
                "n_hours": int(df[ts_col].nunique()),
            }
        if "location_lat" in df.columns and "location_lon" in df.columns:
            report.spatial_coverage = {
                "lat_range": [round(float(df["location_lat"].min()), 4), round(float(df["location_lat"].max()), 4)],
                "lon_range": [round(float(df["location_lon"].min()), 4), round(float(df["location_lon"].max()), 4)],
                "n_stations": report.stations,
            }
        else:
            report.spatial_coverage = {"n_stations": report.stations}

        if report.missing_percentage > 70.0:
            report.anomalies.append(f"missing_percentage {report.missing_percentage}% exceeds 70%")
        if report.valid_rows == 0:
            report.status = "FAIL"
            report.anomalies.append("NO_VALID_ROWS: downstream modelling must not proceed")
        elif report.missing_percentage > 30.0:
            report.status = "DEGRADED"
        return {"df": df, "dropped": dropped, "report": report}

    @staticmethod
    def station_uptime(df: pd.DataFrame, ts_col: str = "observed_at_utc") -> pd.DataFrame:
        """Per-station monthly uptime fraction."""
        if ts_col not in df.columns or "location_name" not in df.columns:
            return pd.DataFrame()
        t = pd.to_datetime(df[ts_col], errors="coerce")
        month = t.dt.to_period("M")
        return (
            df.groupby(["location_name", month])
            .size()
            .rename("n")
            .to_frame()
            .reset_index()
        )


class FireCleaner:
    """Cleans raw FIRMS fire detections."""

    def __init__(self, cfg: VayuConfig) -> None:
        self.cfg = cfg
        firms_cfg = cfg.section("sources").get("firms", {})
        self.frp_min, self.frp_max = firms_cfg.get("frp_range_mw", [0.0, 5000.0])
        self.conf_min = firms_cfg.get("confidence_min", 20.0)

    def clean(self, df: pd.DataFrame) -> Dict[str, Any]:
        import re

        df = df.copy()
        report = QcReport(source="firms", generated_at=_utcnow())
        report.total_rows = len(df)
        flags = pd.DataFrame(index=df.index)
        flags["is_invalid"] = False

        if "acq_date" in df.columns and "acq_time" in df.columns:
            hour = df["acq_time"] // 100
            minute = df["acq_time"] % 100
            try:
                dt = pd.to_datetime(df["acq_date"], errors="coerce") + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(minute, unit="m")
            except Exception:
                dt = pd.to_datetime(df["acq_date"], errors="coerce")
            df["observed_at_utc"] = dt.dt.tz_localize("UTC").dt.tz_localize(None).dt.floor("h")
            bad = dt.isna()
            report.invalid_rows += int(bad.sum())
            flags["is_invalid"] |= bad
        else:
            report.anomalies.append("missing acq_date/acq_time")

        for c in ("latitude", "longitude"):
            if c in df.columns:
                num = pd.to_numeric(df[c], errors="coerce")
                bad = num.isna() | (num.abs() > 90 if c == "latitude" else (num.abs() > 180))
                flags["is_invalid"] |= bad

        if "frp" in df.columns:
            frp = pd.to_numeric(df["frp"], errors="coerce")
            bad = (frp < self.frp_min) | (frp > self.frp_max) | frp.isna()
            flags["is_invalid"] |= bad
            report.invalid_rows += int(bad.sum())

        if "confidence" in df.columns:
            def parse_conf(v):
                if isinstance(v, str):
                    m = re.search(r"\d+", v)
                    return int(m.group()) if m else None
                try:
                    return int(v)
                except Exception:
                    return 100
            conf = df["confidence"].map(parse_conf)
            bad = conf < self.conf_min
            flags["is_invalid"] |= bad
            report.invalid_rows += int(bad.sum())
            df["confidence"] = conf.fillna(self.conf_min).astype(int)

        valid = ~flags["is_invalid"]
        dropped = df.loc[~valid].copy()
        df = df.loc[valid].copy()
        report.valid_rows = len(df)
        report.missing_percentage = round(100.0 * report.invalid_rows / max(report.total_rows, 1), 3)
        report.temporal_coverage = {
            "start": str(df["observed_at_utc"].min()) if "observed_at_utc" in df.columns else None,
            "end": str(df["observed_at_utc"].max()) if "observed_at_utc" in df.columns else None,
            "n_hours": int(df["observed_at_utc"].nunique()) if "observed_at_utc" in df.columns else 0,
        }
        report.spatial_coverage = {
            "lat_range": [round(float(df["latitude"].min()), 4), round(float(df["latitude"].max()), 4)],
            "lon_range": [round(float(df["longitude"].min()), 4), round(float(df["longitude"].max()), 4)],
        } if {"latitude", "longitude"}.issubset(df.columns) else {}
        if report.valid_rows == 0:
            report.status = "FAIL"
            report.anomalies.append("NO_VALID_FIRES")
        return {"df": df, "dropped": dropped, "report": report}


def assert_critical_data(report: QcReport, min_rows: int = 1):
    """Fail loudly rather than silently training on empty/stale data."""
    if report.valid_rows < min_rows:
        raise RuntimeError(
            f"CRITICAL: source '{report.source}' has only {report.valid_rows} valid rows "
            f"(need >= {min_rows}). Refusing to proceed. {report.anomalies}"
        )


def tables_to_markdown(tables: Dict[str, pd.DataFrame]) -> str:
    out: List[str] = []
    for name, df in tables.items():
        out.append(f"\n### {name}\n")
        out.append(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False))
    return "\n".join(out)