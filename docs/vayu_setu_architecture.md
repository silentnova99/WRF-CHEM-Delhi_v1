# VAYU-SETU architecture

Physics-informed coupled air-pollution/weather forecast prototype for Delhi NCR —
built on top of the existing WRF-Chem Delhi V1/V2 codebase (`aqf_delhi`,
`wrf_chem_delhi`). This document is the runtime blueprint for
`src/vayu_setu/`, `scripts/run_vayu.py` and `configs/vayu_setu.yaml`.

## Pipeline (strict order, no future leakage)

```
audit   -> discovery + Kaggle/reachability audit report (reports/data_discovery_audit.*)
ingest  -> Mode-B acquisition into the lake (data/raw/{cpcb,firms,gfs,aod})
manifest-> registry + dataset manifest + dataset-size report
qc      -> cleaning + loud failure on critical gaps (data/interim/cleaned, reports/quality_report.json)
align   -> canonical hourly-UTC timeline + NW-India fire features (data/interim/aligned)
features-> chronological feature splits (data/features/{train,validation,test}) + station registry
train   -> species models (data + baselines)
ablation-> coupled vs uncoupled comparison (reports/ablation_coupling.json)
forecast-> V2 reduced-order coupled-engine demo, feedback ON/OFF (reports/forecast_engine_ablation.json)
report  -> validation + leakage + episode metrics (reports/validation_report.{yaml,md})
```

## Data lake layout

| tier | path | description |
|------|------|-------------|
| raw | `data/raw/{cpcb,firms,gfs,aod,emissions}` | untouched originals (real sources, checksums) |
| interim/cleaned | cleaned CPCB + FIRMS copies + QC report | |
| interim/aligned | enriched city-hour frame + fire features | |
| processed/station | station registry (29 cities, lat/lon/state/n_hours) | |
| features/{train,validation,test} | parquet splits + `_split_meta.json` | |

Mode B is active: Kaggle is unreachable from this environment (no credentials),
so authoritative local + live sources are used (documented per record in
`src/vayu_setu/discovery.py::DISCOVERY_RECORDS`).

## Chronological split (computed, not configured)

| split | start | end | rows |
|-------|-------|-----|------|
| train | 2022-08-05 | 2023-11-26 | 333,326 |
| validation | 2023-11-26 | 2024-11-26 | 254,736 |
| test | 2024-11-26 | 2025-11-26 | 254,098 |

Normalization, imputation and climatology are fitted on `train` exclusively;
`reports/leakage_report.yaml` verifies the chronology gap at every boundary.

## Model stack

1. **Reduced-order physical surrogate** (documented `SURROGATE_PARAMS` in
   `src/vayu_setu/dataset.py`) provides the `RAW_KEYS` physical fields exactly
   as WRF-Chem would operationally — *lagged* observations only (no current-hour
   target in any feature channel).
2. **ST-GNN** (`aqf_delhi.ml.gnn.TemporalGCN`, 24h context) on the station graph
   (k-NN + transport/seasonal wind edges) predicts standardized PM2.5/10/O3.
3. **XGBoost residual head** (`aqf_delhi.ml.xgb_head`) corrects GNN residuals.
4. **Conformal** q90 half-width from validation residuals → prediction intervals.
5. Rolling 72-hour forecast closes observed channels with the model's own
   previous prediction (operational autoregressive closure).

Coupling flag toggles the physics-feedback channels (`pblh_coupled`,
`feedback_strength`, coupling-aware `aod_fused` trapping) in the surrogate and
the ML input — the ablation measures their marginal value.

## Key modules

- `src/vayu_setu/config.py` — runtime config, domain grid (28.2–29.0 N,
  76.8–77.6 E, 4 km).
- `src/vayu_setu/ingest.py` — GFS NOMADS subregion fetch (0.25°, 72 h) +
  Open-Meteo CAMS AOD, bounded, idempotent.
- `src/vayu_setu/fires.py` — NW-India (lat 27.5–33.0, lon 73.5–78.8) hourly
  FRP/upwind/close-50km features from real NASA FIRMS NRT exports.
- `src/vayu_setu/physics.py` — AOD = k·PM·f(RH), Beer–Lambert dimming, surface
  cooling, coupled PBL retention/trapping; `coupling=False` freezes the loop
  (ablation).
- `src/vayu_setu/dataset.py` — leakage-free tensor builder bridging to the
  existing `aqf_delhi.ml` interface.
- `src/vayu_setu/train.py` — baselines (persistence at forecast origin,
  climatology, direct XGBoost) + hybrid runner.
- `src/vayu_setu/validate.py` — leakage checks, per-station + episode metrics,
  validation-report renderer.