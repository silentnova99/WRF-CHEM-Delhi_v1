"""Hybrid ML layer (V2).

Re-exports the reduced-order ML correction + ablation machinery. The heavy
GNN+XGBoost engine from ``aqf_delhi.ml`` remains available for the real-data
(historical/live) mode; this module provides the CPU-only, physics-informed
T+1 model and the chronological ablation study required by the SIH spec.
"""

from wrf_chem_delhi.validation.metrics import (
    T1Model,
    GnnCorrection,
    compute_metrics,
    run_ablation,
    train_and_forecast_72h,
    ForecastMetrics,
)

__all__ = [
    "T1Model",
    "GnnCorrection",
    "compute_metrics",
    "run_ablation",
    "train_and_forecast_72h",
    "ForecastMetrics",
]