"""Forecast artifact writer (Module-3 → Module-4 hand-off).

Publishes a completed forecast run — point predictions, conformal
intervals, the raw-model baseline and the observed field — as one long
parquet (time × station) partition under ``{base}/forecasts/{run_id}/``,
gated by the canonical ``_DONE`` marker:

    {base}/forecasts/{run_id}/forecast.parquet
    {base}/forecasts/{run_id}/report.json
    {base}/forecasts/{run_id}/_DONE
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DONE_MARKER = "_DONE"
FORECAST_FILE = "forecast.parquet"
REPORT_FILE = "report.json"

COLUMNS = [
    "time",
    "station_id",
    "lat",
    "lon",
    "obs_pm25",
    "pm25_raw",
    "engine_pm25",
    "engine_lo",
    "engine_hi",
]


def write_forecast(
    base_dir: str | Path,
    run_id: str,
    *,
    times,
    station_ids: list[str],
    lat,
    lon,
    obs_pm25,
    pm25_raw,
    engine_pm25,
    engine_lo,
    engine_hi,
    report: dict | None = None,
) -> Path:
    """Write one forecast run atomically (data, report, then ``_DONE``)."""
    times = pd.to_datetime(np.asarray(times))
    n_leads, n_stations = np.asarray(engine_pm25).shape
    assert times.shape == (n_leads,), "times must have one entry per lead"
    assert len(station_ids) == n_stations
    assert np.asarray(obs_pm25).shape == (n_leads, n_stations)
    assert np.asarray(pm25_raw).shape == (n_leads, n_stations)
    assert np.asarray(engine_lo).shape == (n_leads, n_stations)
    assert np.asarray(engine_hi).shape == (n_leads, n_stations)

    rows_per_lead = []
    for k in range(n_leads):
        for j in range(n_stations):
            rows_per_lead.append(
                [
                    times[k],
                    station_ids[j],
                    float(lat[j]),
                    float(lon[j]),
                    float(obs_pm25[k, j]),
                    float(pm25_raw[k, j]),
                    float(engine_pm25[k, j]),
                    float(engine_lo[k, j]),
                    float(engine_hi[k, j]),
                ]
            )
    df = pd.DataFrame(rows_per_lead, columns=COLUMNS)

    part = Path(base_dir) / run_id
    part.mkdir(parents=True, exist_ok=True)
    tmp = part / f".tmp-{run_id}.parquet"
    df.to_parquet(tmp, index=False, compression="zstd")
    (part / FORECAST_FILE).write_bytes(tmp.read_bytes())
    tmp.unlink(missing_ok=True)

    meta = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "leads": int(n_leads),
        "stations": int(n_stations),
    }
    if report:
        meta.update({"report": report})
    (part / REPORT_FILE).write_text(json.dumps(meta, indent=2, default=str))

    (part / DONE_MARKER).write_text(
        f"written_at={datetime.now(timezone.utc).isoformat()}\n"
        f"rows={len(df)}\n"
    )
    logger.info("published forecast %s (%d rows) -> %s", run_id, len(df), part)
    return part