"""VAYU-SETU — data acquisition, validation and coupled-feature pipeline
for the Delhi NCR 72-hour air-quality forecasting system.

This package orchestrates the data-first execution order mandated by the
project:

  discovery -> download -> provenance validation -> manifest -> cleaning ->
  temporal/spatial alignment -> data lake -> physics emulator -> features ->
  baselines -> ST-GNN -> XGBoost residual -> conformal -> forecast ->
  validation/ablation -> API/dashboard.

It reuses the physics/ML engines already shipped in `aqf_delhi` and
`wrf_chem_delhi`; nothing here duplicates them.
"""

__version__ = "0.1.0"