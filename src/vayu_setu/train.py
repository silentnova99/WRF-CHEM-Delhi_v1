"""Training: baselines + ST-GNN + XGBoost residual + conformal on REAL data.

Reuses `aqf_delhi.ml` (TemporalGCN, XGBResidualHead, BiasCorrectionEngine,
FeatureSpec, metrics) with the real data tensors produced by
`vayu_setu.dataset`. Baselines (persistence, seasonal/hourly climatology,
direct XGBoost) are implemented here so the hybrid can be compared against
them on identical test leads.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aqf_delhi.ml import metrics as M
from aqf_delhi.ml.ensemble import BiasCorrectionEngine
from aqf_delhi.ml.features import FeatureSpec, RAW_KEYS
from aqf_delhi.ml.graph import StationGraphBuilder

from .config import VayuConfig
from .dataset import DataSplits
from .lake import DataLake

logger = logging.getLogger(__name__)

CONTEXT = 24
LEADS = 72


def standardize_on_train(x_tr: np.ndarray, x_va: np.ndarray, x_te: np.ndarray):
    """Standardize using ONLY training statistics (no leakage)."""
    mu = x_tr.mean(axis=(0, 1), keepdims=True)
    sd = x_tr.std(axis=(0, 1), keepdims=True) + 1e-6
    return (x_tr - mu) / sd, (x_va - mu) / sd, (x_te - mu) / sd, mu.ravel(), sd.ravel()


def station_static(ds: DataSplits, station_frame: pd.DataFrame) -> np.ndarray:
    cities = ds.station_ids
    sf = station_frame.set_index("city").reindex(cities)
    lat = sf["lat"].to_numpy(dtype=float)
    lon = sf["lon"].to_numpy(dtype=float)
    urban = np.ones(len(cities))
    return np.stack([urban, (lat - 28.4) / 0.8, (lon - 77.2) / 1.2], axis=1)


def build_graph_from_splits(ds: DataSplits, station_frame: pd.DataFrame, month: int = 10):
    stations = []
    for name in ds.station_ids:
        row = station_frame.set_index("city").loc[name]
        stations.append({"station_id": name, "name": name, "lat": float(row["lat"]), "lon": float(row["lon"])})
    builder = StationGraphBuilder(k=min(6, len(stations) - 1), transport_radius_km=800.0,
                                  wind_bearing_deg=300.0, wind_season_months=(9, 12))
    return builder.build(stations, month=month)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def baseline_persistence(ds: DataSplits, split: str = "test", lead: int = 1) -> np.ndarray:
    """Persistence for a 72-h forecast: hold the last known observation (at the
    forecast origin) constant across all leads. (Carrying lag-1 obs through the
    whole horizon would be an unrealistic 'nowcast' baseline.)"""
    split_d = getattr(ds, split)
    y = split_d.y
    out = np.empty((LEADS, y.shape[1]), dtype=float)
    out[:] = y[0][None, :]
    return out


def _n(ds: DataSplits) -> int:
    return int(ds.train.n)


def _nodes_latlon(ds: DataSplits):
    n = _n(ds)
    if len(getattr(ds, "lat", [])) == n and len(getattr(ds, "lon", [])) == n:
        return np.asarray(ds.lat, dtype=float), np.asarray(ds.lon, dtype=float)
    return np.zeros(n), np.zeros(n)


def baseline_climatology(ds: DataSplits, spec: str = "pm25") -> np.ndarray:
    """Seasonal/hourly climatology computed from TRAIN only (no leakage)."""
    tr = ds.train
    hours = tr.hours.astype(int)
    doys = tr.doys.astype(int)
    df = pd.DataFrame({"city": np.tile(np.arange(tr.n), tr.t), "hour": np.repeat(hours, tr.n),
                       "month": np.repeat(((doys - 1) // 30).clip(0, 11), tr.n),
                       "y": tr.y.ravel()})
    climat = df.groupby(["city", "hour", "month"])["y"].mean().to_dict()
    te = ds.test
    h = te.hours.astype(int)
    d = te.doys.astype(int)
    mm = ((d - 1) // 30).clip(0, 11)
    out = np.full_like(te.y, np.nan)
    for t in range(te.t):
        for c in range(te.n):
            out[t, c] = climat.get((c, int(h[t]), int(mm[t])), np.nan)
    train_node_mean = tr.y.mean(axis=0)  # train-only fallback
    out = np.where(np.isfinite(out), out, train_node_mean[None, :])
    return out[:LEADS]


class BaselineXGBoost:
    """Direct XGBoost regressor on the standardized feature tensor.

    Features: flattened [context, N, F] window per node + node-static
    (lat / lon / urban); trained globally across nodes.
    """

    def __init__(self, seed: int = 7, n_estimators: int = 300, max_depth: int = 6) -> None:
        from xgboost import XGBRegressor
        self.model = XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1,
        )
        self.seed = seed

    def _windows(self, ds: DataSplits, split: str):
        sd = getattr(ds, split)
        t, n, f = sd.x.shape
        xw = np.stack([sd.x[i - CONTEXT:i].reshape(n, -1) for i in range(CONTEXT, t)])
        yw = sd.y[CONTEXT:]
        nodes = np.tile(np.arange(n), len(xw))
        return xw.reshape(len(xw) * n, -1), yw.ravel(), nodes

    def fit(self, ds: DataSplits) -> None:
        xs, ys, nodes = self._windows(ds, "train")
        n = _n(ds)
        lat, lon = _nodes_latlon(ds)
        self.static = np.stack([np.ones(n), (lat - 28.4) / 0.8, (lon - 77.2) / 1.2], axis=1)
        xs = np.hstack([xs, self.static[nodes]])
        self.model.fit(xs, ys)

    def forecast(self, ds: DataSplits, n_leads: int = LEADS) -> np.ndarray:
        """Rolling forecast on test: +1h autoregressive with lagged-obs closure."""
        te = ds.test
        tr = ds.train
        t, n, f = te.x.shape
        # initialize context = last CONTEXT rows of validation
        init = np.concatenate([tr.x[-CONTEXT:], te.x], axis=0)
        ctx = init[:CONTEXT].copy()
        leads_out = np.zeros((n_leads, n))
        obs_idx = RAW_KEYS.index("obs_pm25")
        base = len(RAW_KEYS) + 4  # after raw + cyclical
        lag1 = base + 4  # lag1_pm25 position in FeatureSpec.names
        names = self._feature_index()
        lag1_idx = names.index("lag1_pm25") if "lag1_pm25" in names else None

        for k in range(n_leads):
            feats = ctx.transpose(1, 0, 2).reshape(n, -1)  # per-node window [N, C*F]
            lon = np.array([ds.lon[i] for i in range(n)]) if len(ds.lon) else np.zeros(n)
            lat = np.array([ds.lat[i] for i in range(n)]) if len(ds.lat) else np.zeros(n)
            static = np.stack([np.ones(n), (lat - 28.4) / 0.8, (lon - 77.2) / 1.2], axis=1)
            feats_full = np.hstack([feats, static])
            yhat = self.model.predict(feats_full)
            leads_out[k] = yhat
            # advance context with next known physical frame from test, closing obs channel
            if k < te.t - 1:
                nxt = te.x[k + 1].copy()
                if obs_idx < f:
                    nxt[:, obs_idx] = yhat
                # place yhat into raw pm25 channel too (autoregressive physical closure)
                nxt[:, 0] = yhat
                if lag1_idx is not None:
                    nxt[:, lag1_idx] = yhat
                ctx = np.concatenate([ctx[1:], nxt[None, :, :]], axis=0)
        return leads_out

    def _feature_index(self) -> List[str]:
        return FeatureSpec().names


# --------------------------------------------------------------------------- #
# Hybrid training driver
# --------------------------------------------------------------------------- #
def train_hybrid(
    cfg: VayuConfig,
    lake: DataLake,
    ds: DataSplits,
    station_frame: pd.DataFrame,
    *,
    context: int = CONTEXT,
    hidden: int = 32,
    epochs: int = 60,
    lr: float = 1e-3,
    stride: int = 4,
    seed: int = 42,
    n_leads: int = LEADS,
    verbose: bool = True,
) -> Dict[str, Any]:
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    spec = FeatureSpec()
    static = station_static(ds, station_frame)
    graph = build_graph_from_splits(ds, station_frame, month=10)

    x_tr, x_va, x_te, mu_f, sd_f = standardize_on_train(ds.train.x, ds.validation.x, ds.test.x)
    y_tr, y_va, y_te = ds.train.y, ds.validation.y, ds.test.y

    obs_idx = RAW_KEYS.index("obs_pm25")
    y_mu, y_sd = mu_f[obs_idx], sd_f[obs_idx]
    y_tr_s = (y_tr - y_mu) / y_sd
    y_va_s = (y_va - y_mu) / y_sd

    engine = BiasCorrectionEngine(graph, spec, context=context, hidden=hidden, seed=seed)
    engine.set_feature_scaling(mu_f, sd_f)
    engine.set_target_scaling(y_mu, y_sd)
    engine.fit(x_tr, y_tr_s, x_va, y_va_s,
               hours_tr=ds.train.hours, doys_tr=ds.train.doys,
               hours_va=ds.validation.hours, doys_va=ds.validation.doys,
               epochs=epochs, lr=lr, spe_lambda=0.05, stride=stride,
               station_static=static, verbose=verbose)

    init = x_va[-context:]
    yhat, lo, hi = engine.forecast_rolling_rescaled(
        init, y_va[-1], x_te[:n_leads],
        ds.test.hours[:n_leads], ds.test.doys[:n_leads],
        station_static=static,
    )
    yobs = y_te[:n_leads]
    rep_eng = M.report(yobs, yhat, "vayu-setu")
    coverage = M.interval_coverage(yobs.ravel(), lo.ravel(), hi.ravel())
    return {
        "engine": rep_eng,
        "interval_halfwidth": engine.conformal_halfwidth,
        "interval_coverage_100": coverage * 100,
        "n_nodes": _n(ds),
        "species": ds.species,
        "coupling": ds.coupling_features,
        "yhat": yhat,
        "lo": lo,
        "hi": hi,
        "yobs": yobs,
        "times": ds.test.times[:n_leads],
        "run_id": f"run.{datetime.now(timezone.utc):%Y%m%d.%H%M%S}",
    }


# --------------------------------------------------------------------------- #
# Baseline comparison runner
# --------------------------------------------------------------------------- #
def run_baselines(ds: DataSplits, xgb_model: Optional[BaselineXGBoost] = None) -> Dict[str, Any]:
    pers = baseline_persistence(ds, "test", lead=1)
    clim = baseline_climatology(ds, ds.species)
    reports = {"persistence": M.report(ds.test.y[:LEADS], pers, "persistence"),
               "climatology": M.report(ds.test.y[:LEADS], clim, "climatology")}
    if xgb_model is not None:
        xf = xgb_model.forecast(ds, n_leads=LEADS)
        reports["xgb_direct"] = M.report(ds.test.y[:LEADS], xf, "xgb_direct")
    return reports