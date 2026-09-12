"""Training/evaluation pipeline (Module-3 section 3.5).

``run_pipeline`` wires: registry stations → graph → synthetic experiment →
feature engineering → GNN fit → residual head → conformal calibration →
72-h rolling forecast on the held-out test season → metric report.

The report compares the *raw biased model field* against the *bias-corrected
engine output*, giving the acceptance gates (RMSE ↓, MBE → 0).
"""

from __future__ import annotations

import logging

import numpy as np

from aqf_delhi.config import load_stations
from aqf_delhi.ml import metrics as M
from aqf_delhi.ml.datagen import SyntheticDelhiExperiment
from aqf_delhi.ml.ensemble import BiasCorrectionEngine
from aqf_delhi.ml.features import FeatureSpec
from aqf_delhi.ml.graph import StationGraphBuilder

logger = logging.getLogger(__name__)


def _feature_tensor(experiment, spec: FeatureSpec, lo: int, hi: int) -> np.ndarray:
    import pandas as pd

    t = experiment.times[lo:hi].astype("datetime64[s]").astype("int64")
    idx = pd.to_datetime(t, unit="s")
    hours = idx.hour.astype(float).to_numpy()
    doys = idx.dayofyear.astype(float).to_numpy()
    raw = {k: v[lo:hi] for k, v in experiment.raw.items()}
    x = spec.from_raw(raw, hours=hours, doys=doys)
    return x, hours, doys


def _standardize(
    x_tr, x_va, x_te
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = x_tr.mean(axis=(0, 1), keepdims=True)
    sd = x_tr.std(axis=(0, 1), keepdims=True) + 1e-6
    return (x_tr - mu) / sd, (x_va - mu) / sd, (x_te - mu) / sd, mu.ravel(), sd.ravel()


def _station_static(stations) -> np.ndarray:
    lat = np.array([float(s["lat"]) for s in stations])
    lon = np.array([float(s["lon"]) for s in stations])
    urban = np.array([float(s.get("urban", 1.0)) for s in stations])
    return np.stack(
        [urban, (lat - 28.4) / 0.8, (lon - 77.2) / 1.2], axis=1
    )


def run_pipeline(
    *,
    seeded: bool = True,
    torch_seed: int = 7,
    quick: bool = True,
    outdir: str | None = None,
) -> dict:
    """Run the full pipeline; when *outdir* is set, publish the forecast
    artifact (see :mod:`aqf_delhi.ml.publish`) and add ``run_id`` to the
    returned report."""
    import torch

    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)

    stations = load_stations()
    if quick:
        # spread subset across NCR for the offline demo
        keep = [0, 2, 3, 8, 11, 12, 17, 19, 22, 26, 28, 32]
        stations = [stations[i] for i in keep]
    static = _station_static(stations)

    graph = StationGraphBuilder(k=6, wind_bearing_deg=300, wind_season_months=(10, 11)).build(
        stations, month=10
    )
    logger.info("graph: %d nodes, %.0f edges", len(graph), graph.adjacency().sum() / 2)

    experiment = SyntheticDelhiExperiment(
        stations, n_days=45 if quick else 90, seed=torch_seed
    ).generate()

    spec = FeatureSpec()
    x_tr, h_tr, d_tr = _feature_tensor(experiment, spec, 0, experiment.split_train_end)
    x_va, h_va, d_va = _feature_tensor(experiment, spec, experiment.split_train_end, experiment.split_val_end)
    x_te, h_te, d_te = _feature_tensor(experiment, spec, experiment.split_val_end, experiment.split_test_end)
    x_tr, x_va, x_te, mu_f, sd_f = _standardize(x_tr, x_va, x_te)

    y_tr = experiment.target[0 : experiment.split_train_end]
    y_va = experiment.target[experiment.split_train_end : experiment.split_val_end]
    y_te = experiment.target[experiment.split_val_end : experiment.split_test_end]

    # sanity: y alignment uses same end indices as feature tensors
    assert y_tr.shape[0] == x_tr.shape[0] and y_va.shape[0] == x_va.shape[0]

    # Standardize targets with the obs_pm25 feature-channel stats so the
    # autoregressive loop stays in one consistent space.
    from aqf_delhi.ml.features import RAW_KEYS

    obs_idx = RAW_KEYS.index("obs_pm25")
    y_mu, y_sd = mu_f[obs_idx], sd_f[obs_idx]
    y_tr_s = (y_tr - y_mu) / y_sd
    y_va_s = (y_va - y_mu) / y_sd

    engine = BiasCorrectionEngine(
        graph, spec, context=24, hidden=16 if quick else 24,
        seed=torch_seed,
    )
    engine.set_feature_scaling(mu_f, sd_f)
    engine.set_target_scaling(y_mu, y_sd)
    epochs = 50 if quick else 80
    if quick:
        engine.fit(x_tr, y_tr_s, x_va, y_va_s, hours_tr=h_tr, doys_tr=d_tr,
                   hours_va=h_va, doys_va=d_va, epochs=epochs, lr=3e-3,
                   spe_lambda=0.05, stride=6, station_static=static, verbose=False)
    else:
        engine.fit(x_tr, y_tr_s, x_va, y_va_s, hours_tr=h_tr, doys_tr=d_tr,
                   hours_va=h_va, doys_va=d_va, epochs=epochs, lr=1e-3,
                   spe_lambda=0.05, stride=4, station_static=static)

    # 72-h rolling forecast over the test window start
    n_leads = 72
    init = x_va[-24:]  # context: last 24h of validation
    y_init = y_va[-1]
    leads_h = h_te[:n_leads]
    leads_d = d_te[:n_leads]
    yhat, lo, hi = engine.forecast_rolling_rescaled(
        init, y_init, x_te[:n_leads], leads_h, leads_d, station_static=static
    )
    y_obs = y_te[:n_leads]

    # raw (uncorrected) model baseline aligned to same leads
    raw_te = experiment.raw["pm25_raw"][experiment.split_val_end : experiment.split_val_end + n_leads]

    rep_raw = M.report(y_obs, raw_te, "raw-model")
    rep_eng = M.report(y_obs, yhat, "engine")
    report = {
        "quick": quick,
        "stations": len(stations),
        "graph_nodes": len(graph),
        "raw": rep_raw,
        "engine": rep_eng,
        "interval_halfwidth": engine.conformal_halfwidth,
        "interval_coverage_90": M.interval_coverage(y_obs.ravel(), lo.ravel(), hi.ravel()),
        "rmse_improvement_pct": (1 - rep_eng["rmse"] / rep_raw["rmse"]) * 100,
        "mbe_abs_after": abs(rep_eng["mbe"]),
    }
    logger.info(
        "report: raw RMSE=%.1f MBE=%+.1f | engine RMSE=%.1f MBE=%+.1f (Δ RMSE %+.1f%%)",
        rep_raw["rmse"], rep_raw["mbe"], rep_eng["rmse"], rep_eng["mbe"],
        report["rmse_improvement_pct"],
    )
    if outdir is not None:
        from datetime import datetime, timezone

        from aqf_delhi.ml.publish import write_forecast

        run_id = f"fc.{datetime.now(timezone.utc):%Y%m%d.%H%M%S}"
        write_forecast(
            outdir,
            run_id,
            times=experiment.times[
                experiment.split_val_end : experiment.split_val_end + n_leads
            ],
            station_ids=experiment.station_ids,
            lat=experiment.lat,
            lon=experiment.lon,
            obs_pm25=y_obs,
            pm25_raw=raw_te,
            engine_pm25=yhat,
            engine_lo=lo,
            engine_hi=hi,
            report=report,
        )
        report["run_id"] = run_id
    return report