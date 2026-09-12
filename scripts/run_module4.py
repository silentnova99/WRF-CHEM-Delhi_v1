"""Module-4 CLI: publish forecasts and serve the geospatial API.

    python scripts/run_module4.py forecast --quick
    python scripts/run_module4.py serve --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _approx(p: float | None, default: float) -> float:
    return float(p) if p is not None else default


def cmd_forecast(args) -> None:
    from aqf_delhi.ml.train import run_pipeline

    outdir = Path(args.outdir)
    report = run_pipeline(seeded=True, torch_seed=int(args.seed), quick=args.quick, outdir=outdir)
    print(json.dumps({k: report[k] for k in ("run_id", "stations", "rmse_improvement_pct")}, indent=2))
    print(f"published -> {outdir / report['run_id']}")


def cmd_serve(args) -> None:
    import uvicorn

    from aqf_delhi.api.app import create_app

    app = create_app(root=args.root)
    uvicorn.run(app, host=args.host, port=int(args.port), log_level=args.log_level)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_module4", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_fc = sub.add_parser("forecast", help="run + publish a forecast artifact")
    p_fc.add_argument("--quick", action="store_true", help="quick (45-day) experiment")
    p_fc.add_argument("--full", action="store_true", help="full (90-day) experiment")
    p_fc.add_argument("--seed", type=int, default=7)
    p_fc.add_argument("--outdir", type=str, default="data/forecasts")
    p_fc.set_defaults(func=cmd_forecast)

    p_sv = sub.add_parser("serve", help="start the FastAPI server")
    p_sv.add_argument("--root", type=str, default=None, help="data directory (default <repo>/data)")
    p_sv.add_argument("--host", type=str, default="127.0.0.1")
    p_sv.add_argument("--port", type=int, default=8000)
    p_sv.add_argument("--log-level", type=str, default="info")
    p_sv.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    if args.cmd == "forecast":
        args.quick = not args.full
    logging.basicConfig(level=logging.INFO)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())