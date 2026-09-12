"""Coupled-run artifact writer (parquet + meta + ``_DONE`` marker)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aqf_delhi.wrf.config import Module2Config
from aqf_delhi.wrf.emissions import FireDetection
from aqf_delhi.wrf.emulator import CoupledResult, MetArrays


def write_coupled_artifact(
    *,
    out_root: Path,
    run_id: str,
    cfg: Module2Config,
    grid,
    met: MetArrays,
    result: CoupledResult,
    stations: list[dict],
    station_series: list[dict],
    fires: list[FireDetection],
    init,
    real: bool,
) -> dict:
    """Write one coupled run under ``out_root / run_id``."""
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    nt, ny, nx = len(met.hours), grid.ny, grid.nx

    meta = {
        "run_id": run_id,
        "init": init.isoformat() if hasattr(init, "isoformat") else str(init),
        "real_grib": real,
        "grid": {"ny": ny, "nx": nx,
                 "lat_min": grid.domain.lat_min, "lat_max": grid.domain.lat_max,
                 "lon_min": grid.domain.lon_min, "lon_max": grid.domain.lon_max,
                 "dx_km": grid.lon_res_km(), "dy_km": grid.lat_res_km()},
        "fhrs": [int(h.hour) * 1 for h in met.hours],
        "fire_count": len(fires),
        "config": cfg.model_dump(),
    }
    import json
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str),
                                       encoding="utf-8")

    grid_rows = []
    for t, h in enumerate(met.hours):
        for j in range(ny):
            for i in range(nx):
                grid_rows.append({
                    "time": h, "i": i, "j": j,
                    "lat": grid.lat_nodes[j], "lon": grid.lon_nodes[i],
                    "pm25_raw": float(result.pm25_grid[t, j, i]),
                    "pm25_obs_demo": float(result.pm25_obs_demo[t, j, i]),
                    "t2m_c": float(met.t2m_c[t, j, i]),
                    "rh_pct": float(met.rh_pct[t, j, i]),
                    "ws10_ms": float(met.ws10_ms[t, j, i]),
                    "wd_deg": float(met.wd_deg[t, j, i]),
                    "gust_ms": float(met.gust_ms[t, j, i]),
                    "prmsl_hpa": float(met.prmsl_hpa[t, j, i]),
                    "pblh_m": float(result.pblh[t, j, i]),
                    "inversion": float(result.inversion[t, j, i]),
                    "vent_idx": float(result.vent_idx[t, j, i]),
                })
    pd.DataFrame(grid_rows).to_parquet(run_dir / "grid.parquet", index=False)

    fire_rows = [
        {
            "lat": f.lat, "lon": f.lon, "frp_mw": f.frp_mw,
            "acq": f.acq.isoformat() if hasattr(f.acq, "isoformat") else str(f.acq),
        }
        for f in fires
    ]
    pd.DataFrame(fire_rows).to_parquet(run_dir / "emissions_fire.parquet", index=False)

    if station_series:
        pd.DataFrame(station_series).to_parquet(run_dir / "stations.parquet", index=False)

    (run_dir / "_DONE").write_text("ok", encoding="utf-8")
    return {
        "run_id": run_id,
        "path": str(run_dir),
        "fires": len(fires),
        "hours": len(met.hours),
    }