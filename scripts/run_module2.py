"""Module-2 CLI: real GRIB coupled runs, namelists, plume demos.

Examples
--------
    python scripts/run_module2.py grib --init 2026-09-12 00   --fhr 0 3 6 12
    python scripts/run_module2.py run  --init 2026-09-12 00   --real
    python scripts/run_module2.py run  --init 2026-09-12 00   --demo
    python scripts/run_module2.py namelist --out data/coupled/namelist
    python scripts/run_module2.py plume --frp 320
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(prog="run_module2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("grib", help="fetch + decode a GFS subset")
    pg.add_argument("--init", nargs="+", required=True, help="YYYY-MM-DD [HH]")
    pg.add_argument("--fhr", nargs="+", type=int, default=[0], required=True)
    pg.add_argument("--cache", type=str, default="data/coupled/grib_cache")
    pg.set_defaults(run=_cmd_grib)

    pr = sub.add_parser("run", help="full coupled emulator run")
    pr.add_argument("--init", nargs="+", required=True, help="YYYY-MM-DD [HH]")
    pr.add_argument("--real", action="store_true", help="use real GFS subsets")
    pr.add_argument("--demo", action="store_true", help="use synthetic meteorology")
    pr.add_argument("--fhrs", type=int, nargs="+", default=None)
    pr.add_argument("--out", type=str, default="data/coupled")
    pr.set_defaults(run=_cmd_run)

    pn = sub.add_parser("namelist", help="write WRF-Chem namelists")
    pn.add_argument("--out", type=str, default="data/coupled/namelist")
    pn.set_defaults(run=_cmd_namelist)

    pp = sub.add_parser("plume", help="plume-rise sensitivity demo")
    pp.add_argument("--frp", type=float, default=320.0)
    pp.set_defaults(run=_cmd_plume)

    args = ap.parse_args()
    return args.run(args) or 0


def _parse_init(tokens: list[str]) -> datetime:
    if len(tokens) == 1:
        return datetime.strptime(tokens[0], "%Y-%m-%d")
    return datetime.strptime(f"{tokens[0]} {tokens[1]}", "%Y-%m-%d %H")


def _cmd_grib(args) -> None:
    from aqf_delhi.wrf.config import load_module2
    from aqf_delhi.wrf.grib import filter_url, fetch_subset, decode_grib

    cfg = load_module2()
    ini = _parse_init(args.init)
    for fhr in args.fhr:
        cache = Path(args.cache) / ini.strftime("%Y%m%d") / f"{fhr:03d}.grib2"
        fetch_subset(cfg.gfs, ini, fhr, cache)
        ds = decode_grib(cache)
        print(f"fhr {fhr:03d}  {len(ds.fields)} fields  lat "
              f"{ds.lat[0]:.2f}..{ds.lat[-1]:.2f}  lon {ds.lon[0]:.2f}..{ds.lon[-1]:.2f}")
        for k, v in ds.fields.items():
            print(f"    {k:<28} {v.shape}  min {v.min():.1f}  max {v.max():.1f}")


def _cmd_run(args) -> None:
    from aqf_delhi.wrf.coupling import coupled_run

    ini = _parse_init(args.init)
    real = args.real and not args.demo
    info = coupled_run(
        init=ini,
        fhrs=args.fhrs,
        real=real,
        out_root=Path(args.out),
    )
    print(f"coupled run {info['run_id']} -> {info['path']}")
    print(f"    hours={info['hours']}  fires={info['fires']}")


def _cmd_namelist(args) -> None:
    from aqf_delhi.wrf.config import load_module2
    from aqf_delhi.domain import build_grid
    from aqf_delhi.config import load_domain

    cfg = load_module2()
    grid = build_grid(load_domain())
    from aqf_delhi.wrf.namelist import write_namelists

    paths = write_namelists(Path(args.out), cfg, grid)
    for p in paths:
        print(f"wrote {p}")


def _cmd_plume(args) -> None:
    import numpy as np

    from aqf_delhi.wrf.plume import plume_rise

    z = np.array([0, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000])
    t = 298.0 - 5.5 * (z / 1000)
    t = np.where(z > 3000, 298.0 - 16.5 + 3.0 * (z - 3000) / 1000, t)
    ws = np.full_like(z, 5.0)
    for frp in (50, 150, args.frp, 1200):
        pr = plume_rise(frp, t_sfc_k=298.0, p_sfc_hpa=985.0, z_k=z, t_k=t, ws_k=ws)
        print(f"FRP {frp:6.0f} MW -> z_top {pr.z_top_m:6.0f} m  "
              f"z_centre {pr.z_centre_m:6.0f} m  mass_sfc_frac "
              f"{pr.mb_flux_per_layer[int(0.5 * len(pr.mb_flux_per_layer)):].sum():.2f}")


if __name__ == "__main__":
    raise SystemExit(main())