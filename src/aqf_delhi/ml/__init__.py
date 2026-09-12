"""Module 3 — Spatiotemporal ML bias-correction & downscaling engine.

Implements the ST-GNN + XGBoost ensemble blueprint: graph construction,
feature engineering, T-GCN spatial-temporal network, residual gradient
head, conformal calibration and hybrid loss (WRMSE + spatial-pattern
error). Ships with a deterministic synthetic-data harness so the full
training pipeline is verifiable before live WRF-Chem output is streamed
through the Phase-1 store.
"""

from aqf_delhi.ml import aqi, ensemble, features, gnn, graph, metrics, xgb_head

__all__ = ["aqi", "ensemble", "features", "gnn", "graph", "metrics", "xgb_head"]


def run_demo(torch_seed: int = 7, quick: bool = True) -> dict:
    """Train + evaluate the full engine on synthetic data.

    Returns a metrics report dict (see ``ml.train.run_pipeline``).
    """
    from aqf_delhi.ml.train import run_pipeline

    return run_pipeline(seeded=True, torch_seed=torch_seed, quick=quick)