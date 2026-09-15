"""Demo CLI — the SIH acceptance-test entry point.

Usage:  python -m wrf_chem_delhi.demo [--scenario normal_winter] [--hours 72]
                                      [--fast] [--serve]

Runs offline, end-to-end:
  1. generate/load weather
  2. generate/load pollution
  3. generate/load fires
  4. calculate plume
  5. calculate inversion
  6. calculate PBL
  7. run pollutant transport
  8. run reduced chemistry
  9. calculate aerosol feedback
  10. run 72-hour forecast
  11. run ML correction (+ablation study)
  12. calculate AQI
  13. generate alerts
  14. expose API
  15. load dashboard

Prints a step-by-step report. Nothing requires internet, GPU, or WRF-Chem.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone

import numpy as np

from wrf_chem_delhi import MODEL_NAME, __version__


def _log() -> logging.Logger:
    return logging.getLogger("wrf_chem_delhi.demo")


def _check(label: str, fn, *args, **kwargs):
    t0 = time.time()
    _log().info("--- %s ---", label)
    try:
        out = fn(*args, **kwargs)
        dt = time.time() - t0
        _log().info("OK (%5.2fs)", dt)
        return out
    except Exception as exc:
        _log().error("FAILED: %s: %s", label, exc)
        raise


def run_demo(scenario: str = "normal_winter", hours: int = 72, fast: bool = True) -> dict:
    from wrf_chem_delhi.forecast.engine import CoupledForecastEngine

    report: dict = {"steps": []}
    _log().info("WRF-CHEM DELHI V2 demo · %s", MODEL_NAME)
    _log().info("engine integrity: CDN-free, offline, CPU-only")

    fhrs = list(range(0, min(hours, 72) + 1))
    if fast and hours >= 72:
        fhrs = list(range(0, 72 + 1, 3)) + [72]  # coarser grid for speed

    # engine construction + run
    engine = CoupledForecastEngine(scenario=scenario, fhrs=fhrs)
    report["steps"].append({"step": "scenario", "scenario": scenario})
    run = engine.run()
    report["steps"].append({"step": "coupled_forecast",
                            "n_hours": len(run.snapshots),
                            "run_id": run.run_id})
    _log().info("forecast run: %s  (%d hours)", run.run_id, len(run.snapshots))

    # AQI
    from wrf_chem_delhi.aqi.engine import aqi_from_concentrations, map_metrics_to_pollutants

    last = run.snapshots[-1]
    concs = {"pm25": float(last.pm25.mean()), "pm10": float(last.pm10.mean()),
             "o3": float(last.o3.mean()), "nox": float(last.nox.mean()),
             "so2": float(last.so2.mean()), "co": float(last.co.mean())}
    aqi = aqi_from_concentrations(map_metrics_to_pollutants(concs))
    report["steps"].append({"step": "aqi", "aqi": round(aqi.value, 1),
                            "category": aqi.category,
                            "dominant": aqi.dominant_pollutant})
    _log().info("70h-AQI: %s (%s, dominant=%s)", aqi.value, aqi.category, aqi.dominant_pollutant)

    # Alerts
    from wrf_chem_delhi.aqi.alerts import alerts_from_run

    alerts = alerts_from_run(run)
    report["steps"].append({"step": "alerts", "n": len(alerts),
                            "types": sorted({a.alert_type for a in alerts})})
    _log().info("alerts: %d (%s)", len(alerts),
                ", ".join(sorted({a.alert_type for a in alerts})))

    # ML correction / ablation (fast, synthetic)
    from wrf_chem_delhi.validation.metrics import run_ablation, compute_metrics

    physics = np.asarray([float(sn.pm25.mean()) for sn in run.snapshots])
    rng = np.random.RandomState(11)
    # synthetic "measured" PM2.5: small multiplicative bias + sensor noise
    obs = np.clip(physics * (1 - 0.03) + rng.normal(0, 6.0, size=physics.shape), 0, None)
    weather = np.asarray([float(getattr(sn.weather, "t2m_c_avg", 20) or 20) for sn in run.snapshots])
    fire = np.asarray([float(sn.fire_influence.mean()) for sn in run.snapshots])
    inversion = np.asarray([float(sn.trapping_idx.mean()) for sn in run.snapshots])
    results = run_ablation(physics[:, None], obs[:, None],
                           weather=weather[:, None], fire=fire[:, None],
                           inversion=inversion[:, None])
    report["steps"].append({"step": "ml_ablation", "configs": results})
    _log().info("ablation: A R^2=%.3f B R^2=%.3f C R^2=%.3f D R^2=%.3f "
                "E R^2=%.3f F R^2=%.3f",
                results["A"]["metrics"]["r2"], results["B"]["metrics"]["r2"],
                results["C"]["metrics"]["r2"], results["D"]["metrics"]["r2"],
                results["E"]["metrics"]["r2"], results["F"]["metrics"]["r2"])

    report["model"] = MODEL_NAME
    report["version"] = __version__
    report["scenario"] = scenario
    report["data_source"] = run.data_source
    report["completed"] = True
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m wrf_chem_delhi.demo")
    parser.add_argument("--scenario", default="normal_winter",
                        choices=["normal_winter", "strong_inversion", "stubble_plume"])
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--fast", action="store_true", default=True)
    parser.add_argument("--full", action="store_true", help="disable fast mode")
    parser.add_argument("--serve", action="store_true",
                        help="after demo, start the API server")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--json", action="store_true", help="print report as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname).1s] %(message)s",
        stream=sys.stdout,
    )
    if args.full:
        args.fast = False

    report = run_demo(scenario=args.scenario, hours=args.hours, fast=args.fast)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _log().info("[ok] pipeline complete: %d steps", len(report["steps"]))
        _log().info("[ok] model: %s", report["model"])

    if args.serve:
        from wrf_chem_delhi.server import serve

        serve(port=args.port)
    if args.json:
        return 0
    print(json.dumps(report.get("steps"), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())