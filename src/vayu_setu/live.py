"""VAYU-SETU live forecast service (Mode-B, servicing endpoints).

The 72 h coupled forecast is produced by the V2 reduced-order engine
(`wrf_chem_delhi.forecast.engine.CoupledForecastEngine`), initialized at the
current UTC hour. The engine's mean level is anchored to the most recent REAL
observations in the lake via a per-species DC offset (nowcast anchoring); the
offset is only applied when fresh, and its age is always reported.

Real FIRMS detections (lake `data/raw/firms`), the fetched GFS cycle metadata
(`data/raw/gfs`) and the latest CPCB-derived city-hour observations are surfaced
as live metadata / exposure signals. Validation-interval halfwidths come from
`reports/train_report_{species}.json`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd

from vayu_setu.config import VayuConfig

SPECIES = ("pm25", "pm10", "o3")
TARGET_COL = {"pm25": "pm25_ugm3", "pm10": "pm10_ugm3", "o3": "o3_ugm3"}
GROUP_KEYS = ("pm25", "pm10", "o3", "nox", "so2", "co")

NCR_LAT = (28.4, 29.1)
NCR_LON = (76.7, 77.5)
ANCHOR_MAX_AGE_DAYS = 3.0


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _num(series) -> float | None:
    v = pd.to_numeric(series, errors="coerce")
    if v is None or pd.isna(v):
        return None
    return round(float(v), 2)


def load_features_frame(cfg: VayuConfig, lake) -> pd.DataFrame:
    f = pd.read_parquet(lake.features("test", "features.parquet"))
    f["_t"] = pd.to_datetime(f["observed_at_utc"], errors="coerce")
    return f


@lru_cache(maxsize=1)
def load_firms_raw(lake) -> pd.DataFrame:
    from vayu_setu.fires import load_firms_raw as _load

    return _load(lake)


@lru_cache(maxsize=1)
def train_halfwidth(cfg: VayuConfig, species: str) -> Optional[float]:
    rp = cfg.resolve(f"reports/train_report_{species}.json")
    if not rp.exists():
        return None
    d = json.loads(rp.read_text(encoding="utf-8"))
    return d.get("hybrid", {}).get("interval_halfwidth")


def latest_observations(cfg: VayuConfig, lake, window_h: int = 72) -> dict[str, Any]:
    """Newest real city-hour observations + regime signals from the lake."""
    f = load_features_frame(cfg, lake)
    t_max = f["_t"].max()
    recent = f[f["_t"] >= t_max - pd.Timedelta(hours=window_h)]
    rows = []
    for city, g in recent.groupby("city"):
        last = g.sort_values("_t").iloc[-1]
        rows.append({
            "city": str(city),
            "state": str(last.get("state", "")),
            "lat": round(float(last["lat"]), 4),
            "lon": round(float(last["lon"]), 4),
            "time": last["_t"].isoformat(),
            **{s: _num(last[TARGET_COL[s]]) for s in SPECIES},
            "temperature_c": _num(last.get("temp_2m_c")),
            "rh_pct": _num(last.get("rh_pct")),
            "wind_stagnation": int(pd.to_numeric(last.get("wind_stagnation"), errors="coerce") or 0),
            "inversion": int(pd.to_numeric(last.get("temp_inversion_flag"), errors="coerce") or 0),
            "aod": _num(last.get("aod")),
        })
    signals = {
        "inversion_hours": int(pd.to_numeric(recent.get("temp_inversion_flag", 0), errors="coerce").fillna(0).gt(0).sum()),
        "biomass_season_hours": int(pd.to_numeric(recent.get("crop_burning_season", 0), errors="coerce").fillna(0).gt(0).sum()),
        "stagnation_hours": int(pd.to_numeric(recent.get("wind_stagnation", 0), errors="coerce").fillna(0).gt(0).sum()),
        "mean_aod": round(float(pd.to_numeric(recent.get("aod", np.nan), errors="coerce").mean()), 3),
    }
    fires = live_fires_summary(lake)
    scenario = pick_scenario(signals, fires)
    return {"data_as_of": t_max, "stations": rows, "signals": signals,
            "fires": fires, "scenario": scenario}


def pick_scenario(signals: dict[str, Any], fires: dict[str, Any]) -> str:
    if fires.get("frp_sum_mw", 0) > 2000.0:
        return "stubble_plume"
    if signals.get("inversion_hours", 0) > 0 and signals.get("inversion_hours", 0) >= signals.get("biomass_season_hours", 0):
        return "strong_inversion"
    return "normal_winter"


def live_fires_summary(lake) -> dict[str, Any]:
    """Real FIRMS detections in the lake, filtered to the NCR area."""
    try:
        df = load_firms_raw(lake)
    except Exception:  # noqa: BLE001
        return {"available": False, "n_detections": 0, "frp_sum_mw": 0.0}
    lat_col = "lat" if "lat" in df.columns else "latitude"
    lon_col = "lon" if "lon" in df.columns else "longitude"
    frp_col = "frp_mw" if "frp_mw" in df.columns else "frp"
    time_col = "observed_at_utc" if "observed_at_utc" in df.columns else \
        ("acq_time_dt" if "acq_time_dt" in df.columns else "acq_date")
    ncr = df[(df[lat_col] >= NCR_LAT[0]) & (df[lat_col] <= NCR_LAT[1]) &
             (df[lon_col] >= NCR_LON[0]) & (df[lon_col] <= NCR_LON[1])].copy()
    frp = pd.to_numeric(ncr.get(frp_col, 0.0), errors="coerce").fillna(0.0)
    t_max_raw = pd.to_datetime(ncr[time_col], errors="coerce").max()
    return {
        "available": True,
        "n_detections": int(len(ncr)),
        "frp_sum_mw": round(float(frp.sum()), 1),
        "last_detection_at": t_max_raw.isoformat() if pd.notna(t_max_raw) else None,
        "area": "NCR box lat 28.4-29.1 lon 76.7-77.5",
    }


_RUN_CACHE: dict[tuple, Any] = {}


def cached_run(cfg: VayuConfig, lake, scenario: str, feedback: bool) -> tuple[Any, datetime]:
    """Engine run cached per (hour, scenario, feedback) — cheap (≈1 s) but idempotent."""
    from wrf_chem_delhi.forecast.engine import CoupledForecastEngine

    init = now_utc().replace(minute=0, second=0, microsecond=0)
    key = (init.strftime("%Y%m%dT%H"), scenario, feedback)
    if key not in _RUN_CACHE:
        engine = CoupledForecastEngine(scenario=scenario, init=init, seed=11, use_feedback=feedback)
        _RUN_CACHE[key] = (engine.run(), init)
    return _RUN_CACHE[key]


def _engine(cfg: VayuConfig, lake, scenario: str, feedback: bool) -> tuple[Any, datetime]:
    return cached_run(cfg, lake, scenario, feedback)


def compute_anchor(cfg: VayuConfig, lake, run) -> dict[str, Any]:
    """Per-species DC offset: newest observed NCR mean vs engine T+1 mean."""
    f = load_features_frame(cfg, lake)
    t_max = f["_t"].max()
    age_days = (now_utc().replace(tzinfo=None) - pd.Timestamp(t_max).to_pydatetime()).total_seconds() / 86400.0
    ncr = f[(f._t >= t_max - pd.Timedelta(hours=1)) &
            (f["lat"] >= NCR_LAT[0]) & (f["lat"] <= NCR_LAT[1]) &
            (f["lon"] >= NCR_LON[0]) & (f["lon"] <= NCR_LON[1])]
    offsets: dict[str, float] = {}
    for s in SPECIES:
        obs = pd.to_numeric(ncr[TARGET_COL[s]], errors="coerce").dropna()
        if obs.empty:
            offsets[s] = 0.0
            continue
        t1 = float(np.nanmean(np.asarray(run.snapshots[1].__getattribute__(s), dtype=float)))
        offsets[s] = round(float(obs.mean()) - t1, 2)
    fresh = age_days <= ANCHOR_MAX_AGE_DAYS
    mode = "applied" if fresh else "stale-not-applied"
    if not fresh:
        offsets = {s: 0.0 for s in SPECIES}
    return {
        "mode": mode,
        "age_days": round(age_days, 1),
        "data_as_of": t_max.isoformat(),
        "anchor_max_age_days": ANCHOR_MAX_AGE_DAYS,
        "offsets_ugm3": offsets,
        "note": "DC correction so the forecast is continuous with the last REAL observations; "
                "stale anchors are never applied (re-ingest CPCB via `python scripts/run_vayu.py ingest`).",
    }


def build_live(cfg: VayuConfig, lake, use_feedback: bool = True) -> dict[str, Any]:
    obs = latest_observations(cfg, lake)
    run, init = _engine(cfg, lake, obs["scenario"], use_feedback)
    anchor = compute_anchor(cfg, lake, run)
    offsets = anchor["offsets_ugm3"]
    hw = {s: train_halfwidth(cfg, s) for s in SPECIES}

    ts = {k: [float(v) for v in run.domain_mean_series(k)] for k in GROUP_KEYS}
    pblh = run.domain_mean_series("pblh_m")
    fdbk = run.domain_mean_series("feedback_strength")
    trap = run.domain_mean_series("trapping_idx")
    ffire = run.domain_mean_series("fire_influence")
    sec = run.domain_mean_series("secondary_pm")
    invs = run.domain_mean_series("inversion_strength")
    vent = run.domain_mean_series("vent_idx")
    plume = run.domain_mean_series("plume_concentration")

    from wrf_chem_delhi.aqi.engine import aqi_from_concentrations, map_metrics_to_pollutants

    series = []
    n = len(run.snapshots)
    for k in range(n):
        anchor_add = {s: offsets.get(s, 0.0) for s in SPECIES}
        concs = {s: round(max(0.0, ts[s][k] + anchor_add[s]), 2) for s in SPECIES}
        aqi = aqi_from_concentrations(map_metrics_to_pollutants({
            **concs,
            "nox": ts["nox"][k], "so2": ts["so2"][k], "co": ts["co"][k],
        }))
        lead = round((sn_time(run, k) - init).total_seconds() / 3600.0)
        slice_ = {
            "time": sn_time(run, k).isoformat(),
            "lead_h": lead,
            **concs,
            **{s: round(max(0.0, concs[s] - hw[s]), 2) if hw[s] else None for s in SPECIES},
            **{f"{s}_hi": round(concs[s] + hw[s], 2) if hw[s] else None for s in SPECIES},
            "aqi": round(aqi.value, 1),
            "category": aqi.category,
            "nox": round(ts["nox"][k], 2),
            "so2": round(ts["so2"][k], 2),
            "co": round(ts["co"][k], 2),
            "pblh_m": round(pblh[k], 1),
            "feedback_strength": round(fdbk[k], 4),
            "trapping_idx": round(trap[k], 3),
            "fire_influence": round(ffire[k], 4),
            "secondary_pm": round(sec[k], 2),
            "inversion_strength": round(invs[k], 2),
            "vent_idx": round(vent[k], 3),
            "plume_concentration": round(plume[k], 2),
        }
        series.append(slice_)

    stations = []
    for st in obs["stations"]:
        s = run.station_timeseries(st["lat"], st["lon"])
        st_series = []
        for k in (24, 48, 72):
            if k >= n:
                continue
            off = {sp: offsets.get(sp, 0.0) for sp in SPECIES}
            lead = round((sn_time(run, k) - init).total_seconds() / 3600.0)
            st_series.append({
                "lead_h": lead,
                "time": sn_time(run, k).isoformat(),
                **{sp: round(max(0.0, s[sp][k] + off[sp]), 2) for sp in SPECIES},
            })
        in_domain = (NCR_LAT[0] - 0.2 <= st["lat"] <= NCR_LAT[1] + 0.2 and
                     NCR_LON[0] - 0.2 <= st["lon"] <= NCR_LON[1] + 0.2)
        stations.append({
            "city": st["city"],
            "state": st["state"],
            "lat": st["lat"],
            "lon": st["lon"],
            "in_domain": bool(in_domain),
            "observed": {k2: st.get(k2) for k2 in SPECIES},
            "series": st_series,
        })

    alerts = []
    try:
        from wrf_chem_delhi.aqi.alerts import alerts_from_run
        alerts = [a.json() for a in alerts_from_run(run)]
    except Exception:  # noqa: BLE001
        alerts = []

    return {
        "generated_at_utc": now_utc().isoformat(),
        "run_id": run.run_id,
        "init_utc": init.isoformat(),
        "forecast_init_hour_utc": int(init.hour),
        "scenario": obs["scenario"],
        "use_feedback": bool(use_feedback),
        "model_version": run.model_version,
        "data_as_of": obs["data_as_of"].isoformat(),
        "sources": {
            "pollution_observations": "real CPCB-derived lake (Mode B)",
            "weather": run.data_source,
            "gfs_cycle": gfs_cycle_meta(),
            "firms": obs["fires"],
        },
        "anchor": anchor,
        "regime_signals": obs["signals"],
        "series": series,
        "stations": stations,
        "alerts": alerts,
        "interval_note": "90% conformal intervals from validation residuals (train_report_*.json); "
                         "applied additively on anchored concentrations.",
    }


def sn_time(run, k: int) -> datetime:
    t = run.snapshots[k].time
    if t.tzinfo is None:
        return t.replace(tzinfo=timezone.utc)
    return t


def gfs_cycle_meta() -> dict[str, Any]:
    try:
        from pathlib import Path
        root = Path("data/raw/gfs")
        files = sorted(p for p in root.rglob("*.grib2")) if root.exists() else []
        cycles: dict[str, int] = {}
        for p in files:
            cyc = p.relative_to(root).parts[0]
            cycles[cyc] = cycles.get(cyc, 0) + 1
        return {
            "available": bool(files),
            "files": len(files),
            "cycles": {k: v for k, v in sorted(cycles.items())},
            "caveat": "objective GFS subset fetched by the pipeline; the V2 weather state is "
                      "reduced-order (documented demosynthetic) while live-GFS wiring is pending.",
        }
    except Exception:  # noqa: BLE001
        return {"available": False, "files": 0, "cycles": {}}