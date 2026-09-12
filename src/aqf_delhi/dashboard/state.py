"""Module-5 dashboard state aggregation.

Reads the latest coupled (Module-2), forecast (Module-3) and fire artifacts
straight from the parquet store and packs them into one compact JSON payload
the WebGIS dashboard can render without further requests:

- coupled PM2.5 grids (per forecast hour, 2-decimal flat arrays + dims/bbox)
- latest-met grid (for one overlay)
- active fire points
- per-station latest raw/obs values
- latest bias-corrected forecast summary
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aqf_delhi.domain import Grid


def _round_list(arr: np.ndarray, nd: int = 2) -> list:
    return np.round(np.asarray(arr, dtype=float), nd).ravel().tolist()


class DashboardState:
    """Reads an aggregated state dict from a coupled-run artifact directory."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.coupled_root = self.data_root / "coupled"
        self.forecast_root = self.data_root / "forecasts"

    # -- discovery ---------------------------------------------------------

    def latest_coupled_run(self) -> tuple[str, Path] | None:
        if not self.coupled_root.is_dir():
            return None
        runs = sorted(
            (p for p in self.coupled_root.iterdir()
             if p.is_dir() and (p / "_DONE").is_file()),
            key=lambda p: p.name, reverse=True,
        )
        if not runs:
            return None
        return runs[0].name, runs[0]

    def latest_forecast_run(self) -> tuple[str, Path] | None:
        if not self.forecast_root.is_dir():
            return None
        runs = sorted(
            (p for p in self.forecast_root.iterdir()
             if p.is_dir() and (p / "_DONE").is_file()),
            key=lambda p: p.name, reverse=True,
        )
        if not runs:
            return None
        return runs[0].name, runs[0]

    # -- individual blocks -------------------------------------------------

    def coupled_block(self, grid: Grid) -> dict:
        found = self.latest_coupled_run()
        if found is None:
            return {}
        run_id, run_dir = found
        gm = pd.read_parquet(run_dir / "grid.parquet")
        gm = gm.sort_values(["time", "j", "i"]).reset_index(drop=True)
        ny, nx = int(gm["j"].max()) + 1, int(gm["i"].max()) + 1

        hour_arrays, hour_labels = [], []
        for k, g in gm.groupby("time", sort=True):
            hours_sorted = g.sort_values(["j", "i"]).reset_index(drop=True)
            hour_labels.append(pd.Timestamp(k).strftime("%Y-%m-%dT%H:%M"))
            blk = {
                "pm25_raw": _round_list(hours_sorted["pm25_raw"].to_numpy()),
                "pm25_obs_demo": _round_list(hours_sorted["pm25_obs_demo"].to_numpy()),
                "inversion": _round_list(hours_sorted["inversion"].to_numpy(), 0),
            }
            if "pm10_raw" in hours_sorted.columns:
                blk["pm10_raw"] = _round_list(hours_sorted["pm10_raw"].to_numpy())
            if "o3_raw" in hours_sorted.columns:
                blk["o3_raw"] = _round_list(hours_sorted["o3_raw"].to_numpy())
            hour_arrays.append(blk)

        last_t = gm["time"].max()
        last_grid = gm[gm["time"] == last_t].sort_values(["j", "i"]).reset_index(drop=True)
        lat_nodes = gm.sort_values("j")["lat"].unique()
        lon_nodes = gm.sort_values("i")["lon"].unique()
        return {
            "run_id": run_id,
            "ny": ny, "nx": nx,
            "lat_nodes": _round_list(lat_nodes, 3),
            "lon_nodes": _round_list(lon_nodes, 3),
            "times": hour_labels,
            "hours": hour_arrays,
            "met_last": {
                "t2m_c": _round_list(last_grid["t2m_c"], 1),
                "ws10_ms": _round_list(last_grid["ws10_ms"], 1),
                "wd_deg": _round_list(last_grid["wd_deg"], 0),
                "pblh_m": _round_list(last_grid["pblh_m"], 0),
            },
        }

    def fires_block(self) -> list[dict]:
        found = self.latest_coupled_run()
        if found is None:
            return []
        _, run_dir = found
        fp = run_dir / "emissions_fire.parquet"
        if not fp.is_file():
            return []
        df = pd.read_parquet(fp)
        if df.empty:
            return []
        return df.to_dict(orient="records")

    def stations_block(self, grid: Grid) -> list[dict]:
        found = self.latest_coupled_run()
        if found is None:
            return []
        _, run_dir = found
        sp = run_dir / "stations.parquet"
        if not sp.is_file():
            return []
        df = pd.read_parquet(sp)
        if df.empty:
            return []
        latest = df.sort_values("time").groupby("station_id").last().reset_index()
        return [
            {
                "station_id": r.station_id,
                "lat": float(r.lat), "lon": float(r.lon),
                "pm25_raw": float(r.pm25_raw),
                "pm25_obs_demo": float(r.pm25_obs_demo),
            }
            for r in latest.itertuples()
        ]

    def forecast_block(self) -> dict:
        found = self.latest_forecast_run()
        if found is None:
            return {}
        run_id, run_dir = found
        report_path = run_dir / "report.json"
        body = {}
        if report_path.is_file():
            try:
                body = json.loads(report_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                body = {}
        report = body.get("report", body)
        return {
            "run_id": run_id,
            "stations": report.get("stations"),
            "rmse_improvement_pct": report.get("rmse_improvement_pct"),
            "raw_rmse": (report.get("raw") or {}).get("rmse"),
            "engine_rmse": (report.get("engine") or {}).get("rmse"),
        }

    def build(self, grid: Grid) -> dict:
        d = grid.domain
        return {
            "server_time": pd.Timestamp.now().isoformat(),
            "domain": {
                "name": d.name,
                "lat_min": d.lat_min, "lat_max": d.lat_max,
                "lon_min": d.lon_min, "lon_max": d.lon_max,
                "ny": grid.ny, "nx": grid.nx,
            },
            "coupled": self.coupled_block(grid),
            "fires": self.fires_block(),
            "stations": self.stations_block(grid),
            "forecasts": self.forecast_block(),
        }