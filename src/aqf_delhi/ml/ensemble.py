"""Bias-correction ensemble engine (Module-3 sections 3.1, 3.4, 3.5).

Pipeline:
    fit(GNN on windows) → residual pool → fit(XGB residual head)
        → split-conformal calibration → rolling 72-h forecast

Forecasting closes the observation loop autoregressively: once the lead
starts we feed our own previous prediction back into the lagged-obs
features, matching how the operational system operates after T+0.

Calendar (hour-of-day / day-of-year) is attached to every window via the
dataset time axis, so the residual head sees true cyclical terms.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from aqf_delhi.ml.features import FeatureSpec
from aqf_delhi.ml.gnn import TemporalGCN
from aqf_delhi.ml.graph import StationGraph
from aqf_delhi.ml.xgb_head import XGBResidualHead

logger = logging.getLogger(__name__)


def make_windows(
    x: np.ndarray, y: np.ndarray, context: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Slide (context, next-step) windows → (windows, targets, window_ends)."""
    t = x.shape[0]
    ends = list(range(context, t, stride))
    windows = np.stack([x[i - context : i] for i in ends])
    targets = y[ends]
    return windows, targets, ends


def _cal_values(hours: np.ndarray, doys: np.ndarray, idx: int) -> dict[str, float]:
    h = int(hours[idx] if idx < len(hours) else hours[-1])
    d = int(doys[idx] if idx < len(doys) else doys[-1])
    return {
        "hour_sin": math.sin(2 * math.pi * h / 24),
        "hour_cos": math.cos(2 * math.pi * h / 24),
        "doy_sin": math.sin(2 * math.pi * d / 365.25),
        "doy_cos": math.cos(2 * math.pi * d / 365.25),
    }


class BiasCorrectionEngine:
    def __init__(
        self,
        graph: StationGraph,
        feature_spec: FeatureSpec,
        context: int = 24,
        hidden: int = 24,
        seed: int = 7,
        create_gnn=None,
    ) -> None:
        self.graph = graph
        self.spec = feature_spec
        self.context = context
        self.seed = seed
        adj = graph.adjacency(normalized=True)
        self.gnn = (create_gnn or TemporalGCN)(
            n_features=feature_spec.n_features, hidden=hidden, out_dim=1, adj=adj
        )
        self.xgb = XGBResidualHead(seed=seed)
        self.conformal_halfwidth = 0.0
        self.eta = 1.0
        self._scale_mu: np.ndarray | None = None   # per-channel [F]
        self._scale_sd: np.ndarray | None = None
        self._y_mu = 0.0
        self._y_sd = 1.0

    def set_feature_scaling(self, mu: np.ndarray, sd: np.ndarray) -> None:
        """Register per-channel standardization stats for the auto-regressive loop."""
        self._scale_mu = np.asarray(mu, dtype=float).ravel()
        self._scale_sd = np.asarray(sd, dtype=float).ravel()

    def set_target_scaling(self, mu: float, sd: float) -> None:
        """Target-space stats (must match the obs_pm25 feature channel stats)."""
        self._y_mu = float(mu)
        self._y_sd = float(sd) if float(sd) > 1e-9 else 1.0

    def _unscale(self, value: np.ndarray) -> np.ndarray:
        return value * self._y_sd + self._y_mu

    def _standardize(self, value: np.ndarray, channel: int) -> np.ndarray:
        if self._scale_sd is None:
            return value
        return (value - self._scale_mu[channel]) / self._scale_sd[channel]

    # ------------------------------------------------------------------ #
    def fit(
        self,
        x_tr: np.ndarray,
        y_tr: np.ndarray,
        x_va: np.ndarray,
        y_va: np.ndarray,
        *,
        hours_tr: np.ndarray,
        doys_tr: np.ndarray,
        hours_va: np.ndarray,
        doys_va: np.ndarray,
        epochs: int = 60,
        lr: float = 1e-3,
        spe_lambda: float = 0.05,
        stride: int = 4,
        sample_weight: np.ndarray | None = None,
        station_static: np.ndarray | None = None,
        verbose: bool = True,
    ) -> dict:
        import torch

        from aqf_delhi.ml.gnn import hybrid_loss

        windows, targets, ends = make_windows(x_tr, y_tr, self.context, stride=stride)
        adj = torch.from_numpy(self.graph.adjacency(normalized=True).astype(np.float32))
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.gnn.to(dev)

        w = (
            torch.ones(windows.shape[2], device=dev)  # per-node weights
            if sample_weight is None
            else torch.as_tensor(sample_weight, dtype=torch.float32, device=dev)
        )
        w_b = w.unsqueeze(0)
        xt = torch.from_numpy(windows.astype(np.float32)).to(dev)
        yt = torch.from_numpy(targets.astype(np.float32)).to(dev)

        opt = torch.optim.Adam(self.gnn.parameters(), lr=lr)
        history: list[float] = []
        self.gnn.train()
        for ep in range(1, epochs + 1):
            opt.zero_grad()
            b = xt.shape[0]
            preds = torch.stack([self.gnn(xt[i], adj) for i in range(b)]).squeeze(-1)
            loss = hybrid_loss(preds, yt, w_b, spe_lambda=spe_lambda)
            loss.backward()
            opt.step()
            history.append(float(loss.item()))
            if verbose and ep % 10 == 0:
                logger.info("gnn epoch %d loss=%.4f", ep, history[-1])

        self._fit_residual_head(
            x_va, y_va, hours_va, doys_va, station_static=station_static, low_mem=True
        )
        logger.info("ensemble fitted: context=%d hidden=%d", self.context, self.gnn.hidden)
        return {"loss_history": history, "epochs": epochs}

    # ------------------------------------------------------------------ #
    def _fit_residual_head(
        self, x_va, y_va, hours_va, doys_va, *, station_static, low_mem: bool = True
    ) -> None:
        wva, yva, ends = make_windows(
            x_va, y_va, self.context, stride=max(1, len(y_va) // 160)
        )
        residuals: list[np.ndarray] = []
        feats: list[np.ndarray] = []
        for i in range(len(wva)):
            g = self._gnn_forward(wva[i])
            residuals.append(yva[i] - g)
            feats.append(self._head_features(wva[i], g, _cal_values(hours_va, doys_va, ends[i]),
                                             station_static))
        res = np.concatenate(residuals).ravel()
        fstack = np.concatenate(feats, axis=0)
        rng = np.random.default_rng(self.seed)
        mask = rng.random(len(fstack)) < (0.5 if low_mem else 1.0)
        self.xgb.fit(fstack[mask], res[mask])
        med = np.median(res)
        self.conformal_halfwidth = float(np.quantile(np.abs(res - med), 0.90))
        logger.info(
            "residual head fitted on %d rows; conformal half-width=%.2f",
            int(mask.sum()), self.conformal_halfwidth,
        )

    # ------------------------------------------------------------------ #
    def forecast_rolling(
        self,
        x_init: np.ndarray,
        y_init: np.ndarray,
        x_future: np.ndarray,
        hours_seq: np.ndarray,
        doys_seq: np.ndarray,
        *,
        station_static=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rolling forecast over ``len(hours_seq)`` leads.

        ``x_future[lead]`` carries the *live* physical-model frame for that
        lead (standardized); its observed-lag channels are overlaid with the
        model's own previous prediction (autoregressive observation closure),
        mirroring operational behaviour where the analysis lags the forecast.
        Returns (point [leads,N], lo [leads,N], hi [leads,N]).
        """
        from aqf_delhi.ml.features import RAW_KEYS

        obs_idx = RAW_KEYS.index("obs_pm25")
        base = len(RAW_KEYS)
        lag1_pos = base + 4           # after the 4 cyclical channels

        x = x_init
        leads: list[np.ndarray] = []
        lows: list[np.ndarray] = []
        highs: list[np.ndarray] = []
        for k in range(len(hours_seq)):
            g = self._gnn_forward(x)
            cal = _cal_values(hours_seq, doys_seq, k)
            fx = self._head_features(x, g, cal, station_static)
            yhat = g + self.eta * self.xgb.predict(fx)
            leads.append(yhat)
            hw = self.conformal_halfwidth
            lows.append(yhat - hw)
            highs.append(yhat + hw)

            # advance one lead: keep live physical fields, close obs channel
            frame = np.array(x_future[k], copy=True)   # [N, F]
            frame[:, obs_idx] = yhat
            if lag1_pos < len(frame[0]):
                frame[:, lag1_pos] = yhat
            x = np.concatenate([x[1:], frame[None, :, :]], axis=0)
        return np.stack(leads), np.stack(lows), np.stack(highs)

    def forecast_rolling_rescaled(self, *args, **kwargs):
        """Same as :meth:`forecast_rolling` but returns raw-scale arrays."""
        leads, lows, highs = self.forecast_rolling(*args, **kwargs)
        return self._unscale(leads), self._unscale(lows), self._unscale(highs)

    # ------------------------------------------------------------------ #
    def _gnn_forward(self, window_x: np.ndarray) -> np.ndarray:
        import torch

        dev = next(self.gnn.parameters()).device
        self.gnn.eval()
        xt = torch.from_numpy(window_x.astype(np.float32)).to(dev)
        adj = torch.from_numpy(self.graph.adjacency(normalized=True).astype(np.float32)).to(dev)
        with torch.no_grad():
            out = self.gnn(xt, adj)
        return out.squeeze(-1).cpu().numpy()

    def _head_features(
        self, window_x, gnn_pred, cal: dict[str, float], station_static
    ) -> np.ndarray:
        n = window_x.shape[1]
        cols = [window_x[-1], gnn_pred[:, None]]
        for k in ("hour_sin", "hour_cos", "doy_sin", "doy_cos"):
            cols.append(np.full((n, 1), cal[k]))
        cols.append(station_static if station_static is not None else np.zeros((n, 1)))
        return np.concatenate(cols, axis=1)