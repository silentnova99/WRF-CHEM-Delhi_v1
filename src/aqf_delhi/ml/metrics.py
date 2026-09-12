"""Performance metrics for the bias-correction engine.

All functions accept flattened [M] or spatial [T, N] arrays and return
scalars. ``spe`` implements the spatial pattern error (Module-3 gate).
"""

from __future__ import annotations

import numpy as np


def mbe(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) - np.asarray(yhat)))


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def wrmse_from_weights(asarray: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return float(np.sqrt((w * (np.asarray(asarray) - np.asarray(yhat)) ** 2).sum()))


def spe(y: np.ndarray, yhat: np.ndarray) -> float:
    """Spatial pattern error: 1 − mean Pearson corr per time-step.

    ``y``/``yhat`` are [T, N]; rows with zero variance are skipped.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    rhos: list[float] = []
    for a, b in zip(y, yhat):
        am, bm = a - a.mean(), b - b.mean()
        denom = (np.sqrt((am**2).sum() * (bm**2).sum()))
        if denom < 1e-9:
            continue
        rhos.append(float((am * bm).sum() / denom))
    if not rhos:
        return 1.0
    return float(1.0 - np.mean(rhos))


def interval_coverage(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.1
) -> float:
    """Empirical coverage of (1−alpha) prediction intervals."""
    inside = (np.asarray(y) >= lo) & (np.asarray(y) <= hi)
    return float(inside.mean())


def seasonal_weights(
    doys: np.ndarray, hours: np.ndarray, lead: np.ndarray | None = None
) -> np.ndarray:
    """Sample weights: +fire season (Oct/Nov), +mid-range leads if given."""
    w = np.ones_like(np.asarray(doys, dtype=float)) * 0.7
    season = np.asarray(((doys >= 274) & (doys <= 334)) * 1.0 + 0.3)
    w = season
    if lead is not None:
        w = w * (1.0 + 0.5 * ((lead >= 24) & (lead <= 48)))
    return w


def report(y_true, y_pred, label: str) -> dict:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    spatial = None if yt.ndim != 2 or yp.ndim != 2 else spe(yt, yp)
    return {
        "label": label,
        "mbe": mbe(yt, yp),
        "mae": mae(yt, yp),
        "rmse": rmse(yt, yp),
        "spe": spatial,
    }