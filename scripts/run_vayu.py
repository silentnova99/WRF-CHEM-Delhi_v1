"""VAYU-SETU pipeline runner.

Usage:
    python scripts/run_vayu.py audit            # discovery audit report
    python scripts/run_vayu.py ingest           # Mode-B acquisition into data/raw
    python scripts/run_vayu.py manifest         # registry + dataset manifest + size report
    python scripts/run_vayu.py qc               # clean + QC reports (firmly loud on failure)
    python scripts/run_vayu.py align            # enriched frame + fire features + canonical timeline
    python scripts/run_vayu.py features         # chronological feature splits
    python scripts/run_vayu.py train [--species pm25] [--quick]   # baselines + ST-GNN + XGB + conformal
    python scripts/run_vayu.py ablation [--quick]                 # coupled vs uncoupled ablation
    python scripts/run_vayu.py forecast         # v2 coupled forecast engine demo (72h, coupling on/off)
    python scripts/run_vayu.py report           # write leakage + validation + size reports
    python scripts/run_vayu.py serve [--port 8080]  # VAYU-SETU live web (dashboard + /api/v3)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vayu")


def main() -> None:
    ap = argparse.ArgumentParser(description="VAYU-SETU data/validation pipeline")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit")
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("ingest")
    p.add_argument("--skip-gfs", action="store_true")
    p.add_argument("--skip-aod", action="store_true")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("manifest")
    p.add_argument("--human", action="store_true")
    p.set_defaults(fn=cmd_manifest)

    p = sub.add_parser("qc")
    p.set_defaults(fn=cmd_qc)

    p = sub.add_parser("align")
    p.add_argument("--rebuild", action="store_true")
    p.set_defaults(fn=cmd_align)

    p = sub.add_parser("features")
    p.set_defaults(fn=cmd_features)

    p = sub.add_parser("train")
    p.add_argument("--species", default="pm25")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("ablation")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(fn=cmd_ablation)

    p = sub.add_parser("forecast")
    p.set_defaults(fn=cmd_forecast)

    p = sub.add_parser("report")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args()
    args.fn(args)


def _env():
    from vayu_setu.config import load_vayu_config
    from vayu_setu.lake import DataLake

    cfg = load_vayu_config()
    lake = DataLake(cfg)
    return cfg, lake


def cmd_audit(args) -> None:
    cfg, _ = _env()
    from vayu_setu.discovery import write_audit

    write_audit(cfg, str(cfg.resolve("reports/data_discovery_audit.yaml")),
                str(cfg.resolve("reports/data_discovery_audit.md")))
    logger.info("audit written to reports/data_discovery_audit.{md,yaml}")


def cmd_ingest(args) -> None:
    cfg, lake = _env()
    from vayu_setu.ingest import run_data_ingest
    from vayu_setu.registry import SourceRegistry

    registry = SourceRegistry(cfg)
    report = run_data_ingest(cfg, registry, lake,
                             do_gfs=not args.skip_gfs, do_aod=not args.skip_aod)
    registry.to_yaml(cfg.resolve(cfg.raw.get("registry_path", "reports/data_registry.yaml")))
    registry.to_json(cfg.resolve("reports/data_registry.json"))
    print(report)


def cmd_manifest(args) -> None:
    cfg, lake = _env()
    from vayu_setu.lake import compute_storage_sizes
    from vayu_setu.manifest import write_human_report, write_manifest

    mpath = cfg.resolve(cfg.raw.get("manifest_path", "reports/dataset_manifest.yaml"))
    write_manifest(cfg, lake, mpath)
    if args.human:
        write_human_report(cfg, lake, cfg.resolve("reports/dataset_manifest.md"))
    size = compute_storage_sizes(cfg)
    import yaml
    cfg.resolve("reports/dataset_size_report.yaml").parent.mkdir(parents=True, exist_ok=True)
    (cfg.resolve("reports/dataset_size_report.yaml")).write_text(yaml.safe_dump(size, sort_keys=False), encoding="utf-8")
    logger.info("manifest: %s", mpath)
    print(size)


def cmd_qc(args) -> None:
    cfg, lake = _env()
    from vayu_setu.lake import normalize_timestamps
    from vayu_setu.qc import FireCleaner, StationCleaner, assert_critical_data
    from vayu_setu.registry import SourceRegistry

    registry = SourceRegistry.from_yaml(cfg.resolve(cfg.raw.get("registry_path", "reports/data_registry.yaml")))
    out_root = lake.ensure(lake.interim("cleaned"))
    reports = {}

    # ---- CPCB enriched extract (real file) --------------------------------- #
    cpcb_raw = lake.raw_dir("cpcb") / "INDIA_AQI_COMPLETE_20251126.csv"
    if cpcb_raw.exists():
        import pandas as pd
        df = pd.read_csv(cpcb_raw, low_memory=False)
        # normalise into the long cleaner format
        df = df.rename(columns={
            "Datetime": "observed_at_utc", "City": "location_name",
            "Latitude": "location_lat", "Longitude": "location_lon",
            "PM2_5_ugm3": "pm25", "PM10_ugm3": "pm10", "O3_ugm3": "o3",
            "NO2_ugm3": "no2", "CO_ugm3": "co", "SO2_ugm3": "so2",
        })
        keep = [c for c in ["observed_at_utc", "location_name", "location_lat", "location_lon",
                            "pm25", "pm10", "o3", "no2", "co", "so2"] if c in df.columns]
        cleaned = StationCleaner(cfg).clean(df[keep])
        lake.write_parquet(cleaned["df"], out_root / "cpcb" / "cpcb_enriched_cleaned.parquet")
        reports["cpcb"] = cleaned["report"].to_dict()
        assert_critical_data(cleaned["report"])

    # ---- FIRMS NW-India raw detections ------------------------------------- #
    from vayu_setu.fires import load_firms_raw
    firms_raw = load_firms_raw(lake)
    if not firms_raw.empty:
        cleaned = FireCleaner(cfg).clean(firms_raw)
        assert_critical_data(cleaned["report"], min_rows=100)
        lake.write_parquet(cleaned["df"], out_root / "firms" / "firms_all_sat_cleaned.parquet")
        reports["firms"] = cleaned["report"].to_dict()
    else:
        reports["firms"] = {"status": "DATA_UNAVAILABLE", "anomalies": ["no raw FIRMS files in lake"]}

    # ---- QC report files ---------------------------------------------------- #
    import json
    qc_path = cfg.resolve("reports/quality_report.json")
    qc_path.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                   "sources": reports}, indent=2, default=str), encoding="utf-8")
    print(json.dumps(reports, indent=2, default=str))


def cmd_align(args) -> None:
    cfg, lake = _env()
    from vayu_setu.fires import hourly_fire_features, load_firms_raw
    from vayu_setu.qc import FireCleaner
    from vayu_setu.features import build_enriched_frame, add_fire_features
    from vayu_setu.alignment import canonical_timeline

    enriched = build_enriched_frame(cfg, lake)
    raw = load_firms_raw(lake)
    fire_clean = FireCleaner(cfg).clean(raw)["df"] if not raw.empty else __import__("pandas").DataFrame()
    t0 = enriched["_t"].min()
    t1 = enriched["_t"].max()
    timeline = canonical_timeline(t0, t1)
    fire_feats = hourly_fire_features(cfg, lake, timeline)
    out = add_fire_features(enriched, fire_feats)
    aligned_dir = lake.ensure(lake.interim("aligned"))
    lake.write_parquet(out, aligned_dir / "enriched_fire.parquet")
    lake.write_parquet(fire_clean, lake.interim("aligned", "fires_nw_aligned.parquet"))
    logger.info("aligned %d rows; fire frame %d rows", len(out), 0 if fire_feats.empty else len(fire_feats))


def cmd_features(args) -> None:
    cfg, lake = _env()
    from vayu_setu.features import write_feature_splits

    meta = write_feature_splits(cfg, lake)
    print(meta)


def cmd_train(args) -> None:
    cfg, lake = _env()
    import json

    from vayu_setu.dataset import load_split_frames, stations_frame, build_splits
    from vayu_setu.features import write_feature_splits
    from vayu_setu.train import run_baselines, train_hybrid

    if not lake.features("train", "_split_meta.json").exists():
        write_feature_splits(cfg, lake)
    frames = load_split_frames(cfg, lake)
    stations = stations_frame(lake)
    cities = stations["city"].tolist()
    ds = build_splits(cfg, lake, frames, cities, species=args.species,
                      coupling_features=True)
    from vayu_setu.dataset import DataSplits
    ds.lat = stations.set_index("city").reindex(cities)["lat"].to_numpy()
    ds.lon = stations.set_index("city").reindex(cities)["lon"].to_numpy()

    result = train_hybrid(cfg, lake, ds, stations,
                          hidden=16 if args.quick else 32,
                          epochs=50 if args.quick else 60,
                          stride=6 if args.quick else 4)
    # baselines
    from vayu_setu.train import BaselineXGBoost
    xgb = BaselineXGBoost(seed=42)
    xgb.fit(ds)
    bl = run_baselines(ds, xgb_model=xgb)
    combined = {"species": args.species, "hybrid": {k: v for k, v in result.items()
                                                    if k not in ("yhat", "lo", "hi", "yobs", "times")},
                "baselines": bl}
    out = cfg.resolve(f"reports/train_report_{args.species}.json")
    out.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
    import numpy as np
    np.savez(cfg.resolve(f"reports/forecasts_{args.species}.npz"),
             yhat=result["yhat"], lo=result["lo"], hi=result["hi"],
             yobs=result["yobs"], times=result["times"].astype("datetime64[m]").astype(str))
    print(json.dumps(combined, indent=2, default=str))


def cmd_ablation(args) -> None:
    cfg, lake = _env()
    import json

    from vayu_setu.dataset import load_split_frames, stations_frame, build_splits
    from vayu_setu.features import write_feature_splits
    from vayu_setu.train import train_hybrid

    if not lake.features("train", "_split_meta.json").exists():
        write_feature_splits(cfg, lake)
    frames = load_split_frames(cfg, lake)
    stations = stations_frame(lake)
    cities = stations["city"].tolist()

    results = {}
    for coupling in (True, False):
        ds = build_splits(cfg, lake, frames, cities, species="pm25", coupling_features=coupling)
        ds.lat = stations.set_index("city").reindex(cities)["lat"].to_numpy()
        ds.lon = stations.set_index("city").reindex(cities)["lon"].to_numpy()
        r = train_hybrid(cfg, lake, ds, stations,
                         hidden=16 if args.quick else 32,
                         epochs=50 if args.quick else 60, stride=6 if args.quick else 4)
        results["coupled" if coupling else "uncoupled"] = r["engine"]
        if coupling:
            results["_coupled_meta"] = {"coverage": r["interval_coverage_100"],
                                        "halfwidth": r["interval_halfwidth"]}
    c_rmse = results["coupled"]["rmse"]
    u_rmse = results["uncoupled"]["rmse"]
    results["_delta"] = {
        "coupled_rmse": c_rmse,
        "uncoupled_rmse": u_rmse,
        "delta_pct": round((1 - c_rmse / u_rmse) * 100, 3),
    }
    out = cfg.resolve("reports/ablation_coupling.json")
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(json.dumps(results, indent=2, default=str))


def cmd_forecast(args) -> None:
    """Demonstrate the coupled 72-h forecast engine with feedback ON vs OFF."""
    import numpy as np

    cfg, lake = _env()
    from wrf_chem_delhi.forecast.engine import CoupledForecastEngine

    init = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    result = {}
    for use_feedback in (True, False):
        engine = CoupledForecastEngine(scenario="stubble_plume", init=init,
                                       seed=11, use_feedback=use_feedback)
        run = engine.run()
        pm = np.array(run.domain_mean_series("pm25"))
        pbl = np.array(run.domain_mean_series("pblh_m"))
        fb = np.array(run.domain_mean_series("feedback_strength"))
        result["coupled" if use_feedback else "uncoupled"] = {
            "run_id": run.run_id,
            "pm25_mean_72h": round(float(pm.mean()), 2),
            "pm25_peak": round(float(pm.max()), 2),
            "pblh_mean_72h": round(float(pbl.mean()), 1),
            "feedback_strength_mean": round(float(fb.mean()), 4),
        }
    result["_note"] = "FULL COUPLED closes the PM->AOD->radiation->PBL->PM loop (use_feedback=True); COUPLING DISABLED holds met PBL fixed (ablation)."
    import json
    out = cfg.resolve("reports/forecast_engine_ablation.json")
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


def cmd_serve(args) -> None:
    """Launch the VAYU-SETU live web API (dashboard + /api + /api/v3)."""
    from vayu_setu.web import serve

    logger.info("VAYU-SETU live web server on http://%s:%s", args.host, args.port)
    serve(port=args.port, host=args.host)


def cmd_report(args) -> None:
    cfg, lake = _env()
    import json

    import numpy as np
    import pandas as pd
    import yaml

    from vayu_setu.validate import (leakage_report, metrics_by_group,
                                    per_station_metrics, write_validation_report)

    leak = leakage_report(cfg, lake)
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": leak.get("timeline", []),
        "leakage_checks": leak.get("checks", []),
        "models": {},
        "stations": {},
        "episodes": {},
        "ablation": {},
        "engine_ablation": {},
        "data_size": {},
    }
    stations = __import__("vayu_setu.dataset", fromlist=["stations_frame"]).stations_frame(lake)
    cities = stations["city"].tolist()

    for spec in ("pm25", "pm10", "o3"):
        rp = cfg.resolve(f"reports/train_report_{spec}.json")
        fp = cfg.resolve(f"reports/forecasts_{spec}.npz")
        if not rp.exists():
            continue
        d = json.loads(rp.read_text(encoding="utf-8"))
        for k, v in d["baselines"].items():
            v["species"] = spec
            payload["models"].setdefault(k, {})[spec] = {k2: v2 for k2, v2 in v.items()
                                                         if k2 in ("rmse", "mae", "bias", "correlation", "r2")
                                                         or k2 == "rmse"}
        payload["models"].setdefault("vayu-setu", {})[spec] = {
            k2: round(v2, 3) for k2, v2 in d["hybrid"]["engine"].items() if k2 != "label"
        }
        payload["models"]["vayu-setu"][spec]["coverage_100"] = round(d["hybrid"]["interval_coverage_100"], 2)
        payload["models"]["vayu-setu"][spec]["conformal_halfwidth"] = round(d["hybrid"]["interval_halfwidth"], 3)

        if fp.exists():
            nz = np.load(fp)
            yobs, yhat = nz["yobs"], nz["yhat"]
            payload["stations"][spec] = per_station_metrics(yobs, yhat, cities, nz["lo"], nz["hi"])
            # episode masks at forecast lead timestamps
            leads_t = pd.to_datetime(np.array(nz["times"].tolist(), dtype="datetime64[ns]") if False else nz["times"])
            leads_t = np.array(list(nz["times"]), dtype="datetime64[ns]")
            fld = pd.to_datetime(leads_t)
            test_f = pd.read_parquet(lake.features("test", "features.parquet"))
            test_f["_t"] = pd.to_datetime(test_f["observed_at_utc"], errors="coerce")
            eps = []
            for gname, col, sense in (("inversion", "temp_inversion_flag", ">"), ("biomass_burning", "crop_burning_season", ">"),
                                      ("high_aod", "aod", ">"), ("stagnant_wind", "wind_stagnation", ">")):
                if col not in test_f.columns:
                    continue
                vals = pd.to_numeric(test_f[col], errors="coerce")
                thr = 0.6 if col == "aod" else 0.0
                active = (vals > thr).to_numpy()
                sub = test_f[active]
                idx = sub["_t"].isin(fld).to_numpy()
                mask = np.zeros((len(leads_t), len(cities)), dtype=bool)
                for i, ts in enumerate(fld):
                    rows = sub[sub["_t"] == ts]
                    if rows.empty:
                        continue
                    for c in rows["city"].tolist():
                        if c in cities:
                            mask[i, cities.index(c)] = True
                eps.append(metrics_by_group(yobs, yhat, mask, gname))
            payload["episodes"][spec] = eps

    ab = cfg.resolve("reports/ablation_coupling.json")
    if ab.exists():
        d = json.loads(ab.read_text(encoding="utf-8"))
        payload["ablation"] = {"coupled_rmse": d["_delta"]["coupled_rmse"],
                               "uncoupled_rmse": d["_delta"]["uncoupled_rmse"],
                               "rmse_delta_pct": d["_delta"]["delta_pct"]}
    eb = cfg.resolve("reports/forecast_engine_ablation.json")
    if eb.exists():
        d = json.loads(eb.read_text(encoding="utf-8"))
        payload["engine_ablation"] = {"coupled_pm25_mean_72h": d["coupled"]["pm25_mean_72h"],
                                      "uncoupled_pm25_mean_72h": d["uncoupled"]["pm25_mean_72h"],
                                      "feedback_mean": d["coupled"]["feedback_strength_mean"]}
    sz = cfg.resolve("reports/dataset_size_report.yaml")
    from vayu_setu.lake import compute_storage_sizes
    payload["data_size"] = compute_storage_sizes(cfg)

    yaml_path = cfg.resolve("reports/validation_report.yaml")
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False), encoding="utf-8")
    md_path = write_validation_report(cfg, yaml_path, payload)
    print(f"validation reports written: {yaml_path} | {md_path}")
    print(yaml.safe_dump({k: v for k, v in payload.items() if k != "stations"}, sort_keys=False))


if __name__ == "__main__":
    main()