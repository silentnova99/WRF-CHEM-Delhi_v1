"""Parquet-backed repository for the Module-4 API.

Reads the Module-1 ingestion store (``data/ingest/{source}/{date}/{run_id}``)
and Module-3 forecast artifacts (``data/forecasts/{run_id}``). Consumers of a
partition are gated by the ``_DONE`` marker, mirroring ``ParquetStore``
semantics. All methods return plain JSON-serializable structures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DONE_MARKER = "_DONE"
SOURCES = ("cpcb", "firms", "gfs")


def _iso(series: pd.Series) -> list[str]:
    return [t.isoformat() if t is not None else None for t in series]


class ApiStore:
    def __init__(
        self,
        root: str | Path,
        *,
        ingest_rel: str = "ingest",
        forecasts_rel: str = "forecasts",
        done_marker: str = DONE_MARKER,
    ) -> None:
        self.root = Path(root)
        self.ingest_dir = self.root / ingest_rel
        self.forecasts_dir = self.root / forecasts_rel
        self.done_marker = done_marker
        self._reads = 0

    # ------------------------------------------------------------------ #
    # partition discovery
    # ------------------------------------------------------------------ #
    def _partitions(self, source: str) -> list[tuple[str, str]]:
        base = self.ingest_dir / source
        if not base.is_dir():
            return []
        out: list[tuple[str, str]] = []
        for done in base.rglob(self.done_marker):
            run_id_dir = done.parent
            date_dir = run_id_dir.parent
            if date_dir.name and run_id_dir.name:
                out.append((date_dir.name, run_id_dir.name))
        out.sort()
        return out

    def latest_partition(self, source: str) -> tuple[str, str] | None:
        parts = self._partitions(source)
        return parts[-1] if parts else None

    def partitions_desc(self, source: str) -> list[str]:
        return [f"{d}/{r}" for d, r in reversed(self._partitions(source))]

    @property
    def reads(self) -> int:
        return self._reads

    # ------------------------------------------------------------------ #
    # raw partition reads
    # ------------------------------------------------------------------ #
    def _read_partition(self, source: str, date: str, run_id: str) -> pd.DataFrame:
        self._reads += 1
        part = self.ingest_dir / source / date / run_id
        fp = part / "records.parquet"
        if not (part / self.done_marker).is_file() or not fp.is_file():
            raise FileNotFoundError(f"incomplete partition: {source}/{date}/{run_id}")
        return pd.read_parquet(fp)

    # ------------------------------------------------------------------ #
    # observations (CPCB)
    # ------------------------------------------------------------------ #
    def observations_latest(
        self, *, metric: str | None = None, station_id: str | None = None
    ) -> list[dict]:
        latest = self.latest_partition("cpcb")
        if latest is None:
            return []
        date, run_id = latest
        df = self._read_partition("cpcb", date, run_id)
        for col in ("observed_at_utc", "ingest_ts"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
        if metric:
            df = df[df["metric"].eq(metric)]
        if station_id:
            df = df[df["station_id"].eq(station_id)]
        if df.empty:
            return []
        keep = ["station_id", "metric", "observed_at_utc"]
        idx = df.groupby(["station_id", "metric"], dropna=False)[
            "observed_at_utc"
        ].idxmax()
        rows = df.loc[idx]
        return [self._obs_to_dict(r) for r in rows.to_dict("records")]

    def observations_timeseries(
        self,
        *,
        station_id: str,
        metric: str,
        from_utc: str | None = None,
        to_utc: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        latest = self.latest_partition("cpcb")
        if latest is None:
            return []
        date, run_id = latest
        df = self._read_partition("cpcb", date, run_id)
        df["observed_at_utc"] = pd.to_datetime(df["observed_at_utc"], utc=True)
        df = df[df["station_id"].eq(station_id) & df["metric"].eq(metric)]
        if from_utc:
            df = df[df["observed_at_utc"] >= pd.to_datetime(from_utc, utc=True)]
        if to_utc:
            df = df[df["observed_at_utc"] <= pd.to_datetime(to_utc, utc=True)]
        df = df.sort_values("observed_at_utc").tail(int(limit))
        return [self._obs_to_dict(r) for r in df.to_dict("records")]

    @staticmethod
    def _obs_to_dict(row: dict) -> dict:
        return {
            "source": row.get("source"),
            "observed_at_utc": (
                row.get("observed_at_utc").isoformat()
                if isinstance(row.get("observed_at_utc"), (pd.Timestamp, np.datetime64))
                else row.get("observed_at_utc")
            ),
            "ingest_ts": (
                row.get("ingest_ts").isoformat()
                if isinstance(row.get("ingest_ts"), (pd.Timestamp, np.datetime64))
                else row.get("ingest_ts")
            ),
            "quality_flag": row.get("quality_flag", "OK"),
            "station_id": row.get("station_id"),
            "station_name": row.get("station_name"),
            "lat": float(row.get("lat") or 0.0),
            "lon": float(row.get("lon") or 0.0),
            "metric": row.get("metric"),
            "value": float(row.get("value") or 0.0),
            "unit": row.get("unit"),
            "sensor_type": row.get("sensor_type"),
        }

    # ------------------------------------------------------------------ #
    # fires (FIRMS)
    # ------------------------------------------------------------------ #
    def fires_recent(
        self,
        *,
        limit: int = 500,
        min_lon: float | None = None,
        min_lat: float | None = None,
        max_lon: float | None = None,
        max_lat: float | None = None,
    ) -> list[dict]:
        latest = self.latest_partition("firms")
        if latest is None:
            return []
        date, run_id = latest
        df = self._read_partition("firms", date, run_id)
        df["acq_datetime"] = pd.to_datetime(df["acq_datetime"], utc=True)
        if min_lon is not None:
            df = df[df["lon"] >= min_lon]
        if max_lon is not None:
            df = df[df["lon"] <= max_lon]
        if min_lat is not None:
            df = df[df["lat"] >= min_lat]
        if max_lat is not None:
            df = df[df["lat"] <= max_lat]
        df = df.sort_values("acq_datetime", ascending=False).head(int(limit))
        rows = []
        for r in df.to_dict("records"):
            rows.append(
                {
                    "fire_id": r.get("fire_id"),
                    "satellite": r.get("satellite"),
                    "lat": float(r.get("lat") or 0.0),
                    "lon": float(r.get("lon") or 0.0),
                    "acq_datetime": r["acq_datetime"].isoformat(),
                    "frp_mw": float(r.get("frp_mw") or 0.0),
                    "brightness_kelvin": r.get("brightness_kelvin"),
                    "confidence_percent": r.get("confidence_percent"),
                    "day_night": r.get("day_night", "D"),
                    "grid_i": r.get("grid_i"),
                    "grid_j": r.get("grid_j"),
                    "assigned_grid_id": r.get("assigned_grid_id"),
                }
            )
        return rows

    # ------------------------------------------------------------------ #
    # GFS asset manifests
    # ------------------------------------------------------------------ #
    def gfs_runs(self) -> list[str]:
        runs: set[str] = set()
        for date, run_id in self._partitions("gfs"):
            df = self._read_partition("gfs", date, run_id)
            if "run_id" in df.columns:
                runs.update(str(x) for x in df["run_id"].unique())
        return sorted(runs)

    def gfs_assets(
        self, *, run_id: str | None = None, variable: str | None = None, limit: int = 1000
    ) -> list[dict]:
        rows: list[dict] = []
        for date, rid in self._partitions("gfs"):
            df = self._read_partition("gfs", date, rid)
            if run_id and "run_id" in df.columns:
                df = df[df["run_id"].eq(run_id)]
            if variable and "variable" in df.columns:
                df = df[df["variable"].eq(variable)]
            for r in df.to_dict("records"):
                rows.append(
                    {
                        "run_id": r.get("run_id"),
                        "init_utc": (
                            r["init_utc"].isoformat()
                            if isinstance(r.get("init_utc"), (pd.Timestamp, np.datetime64))
                            else r.get("init_utc")
                        ),
                        "lead_h": int(r.get("lead_h") or 0),
                        "valid_utc": (
                            r["valid_utc"].isoformat()
                            if isinstance(r.get("valid_utc"), (pd.Timestamp, np.datetime64))
                            else r.get("valid_utc")
                        ),
                        "variable": r.get("variable"),
                        "level_str": r.get("level_str"),
                        "level_kind": r.get("level_kind"),
                        "grid": r.get("grid"),
                        "asset_uri": r.get("asset_uri"),
                        "missing_fraction": float(r.get("missing_fraction") or 0.0),
                    }
                )
            if len(rows) >= int(limit):
                break
        return rows[: int(limit)]

    # ------------------------------------------------------------------ #
    # forecast artifacts (Module-3)
    # ------------------------------------------------------------------ #
    def forecast_runs(self) -> list[str]:
        base = self.forecasts_dir
        if not base.is_dir():
            return []
        runs = [
            done.parent.name
            for done in base.rglob(self.done_marker)
            if done.parent.name
        ]
        return sorted(runs, reverse=True)

    def forecast_latest(self) -> str | None:
        runs = self.forecast_runs()
        return runs[0] if runs else None

    def _forecast_df(self, run_id: str) -> pd.DataFrame:
        part = self.forecasts_dir / run_id
        fp = part / "forecast.parquet"
        if not (part / self.done_marker).is_file() or not fp.is_file():
            raise FileNotFoundError(f"unknown or incomplete forecast run: {run_id}")
        df = pd.read_parquet(fp)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        return df

    def forecast_report(self, run_id: str) -> dict | None:
        part = self.forecasts_dir / run_id
        fp = part / "report.json"
        if not fp.is_file():
            return None
        with fp.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def forecast_series(
        self, run_id: str, station_id: str | None = None, limit: int = 10000
    ) -> list[dict]:
        df = self._forecast_df(run_id)
        if station_id:
            df = df[df["station_id"].eq(station_id)]
        df = df.sort_values(["time", "station_id"]).head(int(limit))
        return self._forecast_rows(df)

    def forecast_field(self, run_id: str, lead: int) -> list[dict]:
        df = self._forecast_df(run_id)
        times = sorted(df["time"].unique())
        if not (0 <= lead < len(times)):
            raise IndexError(f"lead {lead} out of range [0, {len(times) - 1}]")
        t = times[lead]
        return self._forecast_rows(df[df["time"] == t])

    @staticmethod
    def _forecast_rows(df: pd.DataFrame) -> list[dict]:
        rows = []
        for r in df.to_dict("records"):
            rows.append(
                {
                    "time": r["time"].isoformat(),
                    "lat": float(r.get("lat") or 0.0),
                    "lon": float(r.get("lon") or 0.0),
                    "station_id": r.get("station_id"),
                    "obs_pm25": r.get("obs_pm25"),
                    "pm25_raw": r.get("pm25_raw"),
                    "engine_pm25": float(r.get("engine_pm25") or 0.0),
                    "engine_lo": float(r.get("engine_lo") or 0.0),
                    "engine_hi": float(r.get("engine_hi") or 0.0),
                }
            )
        return rows