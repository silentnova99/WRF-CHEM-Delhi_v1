"""Alert engine: rule-driven health / pollution alerts.

Alert types (per problem statement):
  HIGH_PM        | PM2.5 exceeds a trigger (e.g. > 120 µg/m³)
  SEVERE_AQI     | Indian AQI >= 301 (Very Poor) or >= 401 (Severe)
  INVERSION      | trapping index > 0.6
  STAGNANT_WIND  | wind speed < 1.5 m/s for > 3 consecutive hours
  STUBBLE_PLUME  | plume concentration + fire influence above threshold
  RAPID_PM_RISE  | ΔPM2.5 > 40 µg/m³ in 6 h

Alerts carry: id, type, severity (info|warning|critical), time, message,
data (relevant metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from wrf_chem_delhi.aqi.engine import aqi_from_concentrations, map_metrics_to_pollutants

SEVERITY = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class Alert:
    alert_id: str
    alert_type: str
    severity: str
    timestamp: datetime
    message: str
    data: dict = field(default_factory=dict)

    def json(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "type": self.alert_type,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "data": self.data,
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def evaluate_alerts(
    *,
    pm25_series: list[float],
    aqi_series: list[float],
    hours: list[datetime],
    trapping_series: list[float],
    ws10_series: list[float],
    plume_conc_series: list[float],
    fire_influence_series: list[float],
    pm25_hi: float = 120.0,
    aqi_hi: float = 301.0,
    trap_hi: float = 0.6,
    wind_lo: float = 1.5,
    plung_hi: float = 0.35,
    fire_hi: float = 0.4,
    rise_hi: float = 40.0,
) -> list[Alert]:
    """Evaluate rule-based alerts over a forecast/series horizon.

    Every parameter has a sensible default; the caller passes the time series
    (typically per-domain or per-station). Returns sorted list by severity.
    """
    alerts: list[Alert] = []
    n = len(pm25_series)
    if n == 0:
        return alerts

    # ------- HIGH_PM -------
    peak_pm = max(pm25_series)
    if peak_pm > pm25_hi:
        idx = pm25_series.index(peak_pm)
        alerts.append(Alert(
            alert_id=f"HIGH_PM-{idx}",
            alert_type="HIGH_PM",
            severity="warning" if peak_pm < 201 else "critical",
            timestamp=hours[idx] if idx < len(hours) else _now(),
            message=f"PM2.5 reached {peak_pm:.0f} µg/m³ (> {pm25_hi}); health advisory.",
            data={"pm25": peak_pm, "threshold": pm25_hi},
        ))

    # ------- SEVERE_AQI -------
    peak_aqi = max(aqi_series)
    if peak_aqi >= aqi_hi:
        idx = aqi_series.index(peak_aqi)
        alerts.append(Alert(
            alert_id=f"SEVERE_AQI-{idx}",
            alert_type="SEVERE_AQI",
            severity="critical" if peak_aqi >= 401 else "warning",
            timestamp=hours[idx] if idx < len(hours) else _now(),
            message=f"AQI reached {peak_aqi:.0f} (threshold {aqi_hi}); restrict outdoor activity.",
            data={"aqi": peak_aqi, "threshold": aqi_hi},
        ))

    # ------- INVERSION -------
    if trapping_series:
        peak_trap = max(trapping_series)
        if peak_trap > trap_hi:
            idx = trapping_series.index(peak_trap)
            alerts.append(Alert(
                alert_id=f"INVERSION-{idx}",
                alert_type="INVERSION",
                severity="warning",
                timestamp=hours[idx] if idx < len(hours) else _now(),
                message=f"Trapping index {peak_trap:.2f} (> {trap_hi}); plume/pollution trapped near surface.",
                data={"trapping_index": peak_trap, "threshold": trap_hi},
            ))

    # ------- STAGNANT_WIND -------
    if ws10_series:
        consec = 0
        for i, w in enumerate(ws10_series):
            consec = consec + 1 if w < wind_lo else 0
            if consec >= 3:
                alerts.append(Alert(
                    alert_id="STAGNANT_WIND",
                    alert_type="STAGNANT_WIND",
                    severity="warning",
                    timestamp=hours[i] if i < len(hours) else _now(),
                    message=f"Wind < {wind_lo:.1f} m/s for 3+ h (stagnation).",
                    data={"ws10_ms": w, "consecutive_hours": consec},
                ))
                break

    # ------- STUBBLE_PLUME -------
    if plume_conc_series and fire_influence_series:
        peak_comb = [
            p * (1 + 1.2 * f)
            for p, f in zip(plume_conc_series, fire_influence_series)
        ]
        peak = max(peak_comb)
        if peak > plung_hi + fire_hi:
            idx = peak_comb.index(peak)
            alerts.append(Alert(
                alert_id=f"STUBBLE_PLUME-{idx}",
                alert_type="STUBBLE_PLUME",
                severity="warning",
                timestamp=hours[idx] if idx < len(hours) else _now(),
                message="Elevated plume + fire influence detected; stubble smoke arriving.",
                data={"plume_conc": plume_conc_series[idx],
                      "fire_influence": fire_influence_series[idx]},
            ))

    # ------- RAPID_PM_RISE -------
    if len(pm25_series) >= 7:
        i6 = 6
        pm_now = pm25_series[i6]
        pm_past = pm25_series[0]
        delta = pm_now - pm_past
        if delta > rise_hi:
            alerts.append(Alert(
                alert_id="RAPID_PM_RISE",
                alert_type="RAPID_PM_RISE",
                severity="warning",
                timestamp=hours[i6] if i6 < len(hours) else _now(),
                message=f"PM2.5 rose {delta:.0f} µg/m³ over 6 h (> {rise_hi}).",
                data={"delta_6h": delta, "pm25_now": pm_now, "pm25_6h_ago": pm_past},
            ))

    alerts.sort(key=lambda a: SEVERITY[a.severity], reverse=True)
    return alerts


def alerts_from_run(run, hi: int = None) -> list[Alert]:
    """Convenience: evaluate alerts against a whole ForecastRun.

    Uses domain-mean series. `hi` may limit to a lead; by default uses all.
    """
    n = len(run.snapshots)
    if hi is not None:
        n = min(hi + 1, n)
    pm25 = run.domain_mean_series("pm25")[:n]
    o3 = run.domain_mean_series("o3")[:n]
    days = range_compact(run, n)
    aqi = []
    for i in range(n):
        score = aqi_from_concentrations(
            map_metrics_to_pollutants(_snapshot_concs(run, i))
        )
        aqi.append(score.value)
    t = run.domain_mean_series("trapping_idx")[:n]
    w = [float(sn.weather.get("ws10_ms_avg") or sn.weather.get("ws10_ms", 2.0))
         for sn in run.snapshots[:n]] or [2.0] * n
    p = run.domain_mean_series("plume_concentration")[:n]
    f = run.domain_mean_series("fire_influence")[:n]
    wet = [2.0] * n  # not used directly
    h = [sn.time for sn in run.snapshots[:n]]
    return evaluate_alerts(
        pm25_series=pm25, aqi_series=aqi, hours=h, trapping_series=t,
        ws10_series=w, plume_conc_series=p, fire_influence_series=f,
    )


def _snapshot_concs(run, i: int) -> dict:
    sn = run.snapshots[i]
    return {k: float(np.nanmean(np.asarray(getattr(sn, k), dtype=float))) for k in
            ("pm25", "pm10", "o3", "nox", "so2", "co")}


def range_compact(run, n: int) -> range:
    return range(n)