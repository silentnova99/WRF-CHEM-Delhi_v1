"""XGBoost residual head (Module-3 section 3.4).

Trains on the *residual* r = y_obs − ŷ_GNN using context + calendar +
static features (never y_obs directly, to prevent leakage). The boosted
trees learn the structured model-vs-truth bias that a smooth GNN cannot
capture (discrete winter-inversion offsets, station micro-UHI deltas).
"""

from __future__ import annotations

import numpy as np

try:  # xgboost is an optional dependency of the ML package
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - surfaced on first use

    class XGBRegressor:  # type: ignore[no-redef]
        def __init__(self, *a, **k):  # pragma: no cover
            raise RuntimeError(
                "xgboost is not installed; run `pip install xgboost` "
                "(add the `ml` extra)."
            )


class XGBResidualHead:
    def __init__(
        self,
        n_estimators: int = 300,
        eta: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.7,
        seed: int = 7,
    ) -> None:
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            eta=eta,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------ #
    def fit(
        self,
        features: np.ndarray,
        residual: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """``features`` [M, F] → learn residual r = y − ŷ_GNN."""
        residual = np.asarray(residual, dtype=float).ravel()
        self.model.fit(features, residual, sample_weight=sample_weight, verbose=False)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)

    # ------------------------------------------------------------------ #
    def residual_features(
        self,
        window_x: np.ndarray,      # [T, N, F] context features at lead time
        gnn_pred: np.ndarray,      # [N] GNN point estimate
        calendar: dict[str, np.ndarray],  # broadcast scalars per lead
        station_static: np.ndarray,       # [N, S]
    ) -> np.ndarray:
        """Build the [N, Fx] table for one lead across nodes."""
        n = window_x.shape[1]
        last = window_x[-1]                    # [N, F]
        cols: list[np.ndarray] = []
        cols.append(last)
        cols.append(gnn_pred[:, None])
        for key in ("hour_sin", "hour_cos", "doy_sin", "doy_cos"):
            cols.append(np.full((n, 1), calendar[key]))
        cols.append(station_static)
        return np.concatenate(cols, axis=1)

    def set_feature_names(self, names: list[str]) -> None:
        self.feature_names = names