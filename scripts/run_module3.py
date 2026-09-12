"""Module-3 CLI — train & evaluate the ST-GNN + XGBoost bias-correction engine.

Usage:
    python scripts/run_module3.py demo --quick        # fast synthetic check
    python scripts/run_module3.py demo --full         # full synthetic pipeline
    python scripts/run_module3.py aqi --pm25 182.4 --pm10 301 --o3 41.2

Prints the raw-model vs engine metric report (RMSE/MBE/MAE/SPE, coverage).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")


def _demo(args) -> int:
    from aqf_delhi.ml.train import run_pipeline

    report = run_pipeline(seeded=True, torch_seed=args.seed, quick=args.quick)
    print(json.dumps(report, indent=2))
    if report["engine"]["rmse"] >= report["raw"]["rmse"]:
        print("WARNING: engine did not beat raw model — check hyper-parameters",
              file=sys.stderr)
        return 2
    return 0


def _aqi(args: Any) -> int:
    from aqf_delhi.ml.aqi import aqi_from_concentrations

    conc = {}
    for name in ("pm25", "pm10", "o3"):
        value = getattr(args, name)
        if value is not None:
            conc[{"pm25": "PM2.5", "pm10": "PM10", "o3": "O3"}[name]] = value
    score = aqi_from_concentrations(conc)
    print(f"AQI={score.value} bucket='{score.bucket}' dominant={score.dominant_pollutant}")
    print("sub-indices:", score.sub_indices)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    demo = sub.add_parser("demo", help="train+evaluate synthetic pipeline")
    demo.add_argument("--quick", action="store_true", help="fast config (default)")
    demo.add_argument("--full", action="store_true", help="full-size synthetic run")
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(func=_demo)

    aqi = sub.add_parser("aqi", help="compute IND-AQI from concentrations")
    aqi.add_argument("--pm25", type=float)
    aqi.add_argument("--pm10", type=float)
    aqi.add_argument("--o3", type=float)
    aqi.set_defaults(func=_aqi)

    args = p.parse_args(argv or sys.argv[1:])
    if getattr(args, "cmd", None) == "demo" and args.quick and args.full:
        p.error("pass either --quick or --full, not both")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())