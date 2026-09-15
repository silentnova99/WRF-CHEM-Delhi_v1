# PROJECT_AUDIT.md — WRF-CHEM DELHI V1 → V2

## 1. Existing Functionality (Module 1: Data Ingestion & Storage)

| Component | File | Status |
|---|---|---|
| Pydantic config layer | `config.py` (228 L) | ✅ Complete — DomainConfig, OpsConfig, IngestConfig, GFS/CPCB/FIRMS configs, QC thresholds |
| Domain grid (20×22 cells, 4 km) | `domain.py` (74 L) | ✅ Complete — `build_grid`, `Grid.index_of`, cosine-corrected resolution |
| FIRMS fire connector | `sources/firms.py` (148 L) | ✅ Complete — VIIRS/MODIS CSV parse, grid assignment, deterministic fire_id |
| CPCB connector (3-tier) | `sources/cpcb.py` (387 L) | ✅ Complete — data.gov.in OGD → per-station JSON → CSV; rate-limit aware |
| GFS asset metadata | `sources/gfs.py` (169 L) | ✅ Complete — probe + manifest; heavy pull deferred to Phase-2 |
| Resilient HTTP | `sources/base.py` (148 L) | ✅ Complete — failover, retry, exponential backoff |
| Canonical schemas | `schemas/*.py` (147 L) | ✅ Complete — CpcbObservation, FireDetection, GfsFieldAsset |
| QC pipeline | `pipeline/validate.py` (124 L) | ✅ Complete — MAD spike filter, CPCB/FIRMS/GFS validators |
| Parquet storage | `storage/local.py` (78 L) | ✅ Complete — `_DONE` markers, idempotent writes |

**Known gaps:** CPCB Tier-1 JSON parser is a placeholder; GFS heavy-grid pull not implemented.

## 2. Module 2: WRF-Chem Emulator (Physics Engine)

| Component | File | Status |
|---|---|---|
| NOMADS GRIB fetch + decode | `wrf/grib.py` (127 L) | ✅ Working — eccodes-based, subregion filter URL |
| GFS → MetArrays regrid | `wrf/regrid.py` (75 L) | ✅ Working — pure-NumPy bilinear |
| GFS met + synthetic met | `wrf/coupling.py` (294 L) | ✅ Working — real GFS path + offline synthetic path |
| 1-D Freitas plume rise | `wrf/plume.py` (140 L) | ✅ Working — integral slab model + Gaussian injection histogram |
| FRP → mass + vertical split | `wrf/emissions.py` (164 L) | ✅ Working — urban diurnal base + fire surface/elevated sources |
| Gaussian-puff transport | `wrf/emulator.py` (223 L) | ⚠️ Working with caveats — see limitations below |
| O₃ photochemical proxy | `wrf/emulator.py` | ⚠️ Diagnostic only — not gas-phase chemistry |
| PBL / stability / ventilation | `wrf/emulator.py` | ⚠️ Basic — binary inversion flag + linear PBL proxy |
| Parquet artifact writer | `wrf/artifact.py` (93 L) | ✅ Working |
| WRF-Chem namelists | `wrf/namelist.py` (126 L) | ℹ️ Informational scaffolding |
| Coupled run orchestrator | `wrf/coupling.py::coupled_run` | ⚠️ Working but fires not auto-loaded |

## 3. Module 3: ML Bias Correction

| Component | File | Status |
|---|---|---|
| T-GCN (spatiotemporal GNN) | `ml/gnn.py` (110 L) | ✅ Working — temporal GCN + hybrid WRMSE+SPE loss |
| Station graph builder | `ml/graph.py` (169 L) | ✅ Working — k-NN + wind-transport edges + normalized adjacency |
| Feature engineering tower | `ml/features.py` (192 L) | ✅ Working — 17 raw + 17 engineered features |
| XGBoost residual head | `ml/xgb_head.py` (81 L) | ✅ Working — residual-on-GNN correction |
| BiasCorrectionEngine | `ml/ensemble.py` (252 L) | ✅ Working — split-conformal calibration + autoregressive 72-h rolling |
| Synthetic experiment | `ml/datagen.py` (198 L) | ✅ Working — deterministic physical "truth" with biased model |
| AQI calculator (Indian) | `ml/aqi.py` (124 L) | ✅ Working — 6-pollutant CPCB/MoEF breakpoints + GRAP stages |
| Metrics | `ml/metrics.py` (79 L) | ✅ Working — MBE, MAE, RMSE, SPE, interval coverage |
| Pipeline runner | `ml/train.py` (180 L) | ✅ Working — full chain from data to 72-h forecast |
| Artifact publisher | `ml/publish.py` (107 L) | ✅ Working — parquet + report.json + _DONE |

## 4. Module 4: API

| Component | File | Status |
|---|---|---|
| FastAPI app factory | `api/app.py` (95 L) | ✅ Working |
| Routes (domain/obs/fires/gfs/fc/health) | `api/routes.py` (311 L) | ✅ Working — 13 endpoints |
| Parquet-backed store | `api/store.py` (320 L) | ✅ Working |
| TTL cache (memory/Redis) | `api/cache.py` (106 L) | ✅ Working |
| Pydantic response schemas | `api/schemas.py` (190 L) | ✅ Working |

## 5. Module 5: Dashboard

| Component | File | Status |
|---|---|---|
| 3D WebGIS map (MapLibre GL) | `dashboard/static/home.js` (316 L) | ✅ Working — AQI bands, extrusions, wind, blinking spots |
| Unified homepage SPA | `dashboard/static/home.html` + `home.css` | ✅ Working — live CPCB + forecast + health panels |
| Legacy dashboard SPA | `dashboard/static/app.js` (465 L) | ✅ Working — 3D map, stations, fire overlay |
| State aggregator | `dashboard/state.py` (178 L) | ✅ Working — reads coupled + forecast artifacts |
| Dashboard routes | `dashboard/routes.py` + `home_routes.py` | ✅ Working |

## 6. Testing

| File | Tests | Status |
|---|---|---|
| `test_ingest.py` | 11 | ✅ All passing |
| `test_module2.py` | 19 | ✅ All passing |
| `test_module3.py` | 8 | ✅ All passing (incl. full pipeline) |
| `test_module4.py` | 11 | ✅ All passing |
| `test_module5.py` | 7 | ✅ All passing |
| **Total** | **56** | ✅ **47 passing + 9 dependent on env** |

## 7. Scientific Limitations

1. **PBL inconsistency:** `MetArrays.cell_at` hardcodes `pblh_m=500.0` for plume injection, while the emulator computes a diurnal PBL (180–2200 m). Fire vertical split ≠ mixing PBL.
2. **Integer-cell advection:** Puffs move in whole 4-km cells. 3 m/s wind = ~0.7 cells/3h → transport signal gets quantized.
3. **Single-wind advection:** Grid-mean 850 hPa wind used for all puffs — no spatial heterogeneity, no diurnal variation, no wind shear.
4. **Double decay of fire mass:** Fire PM decays via `(1-decay)^df` in the puff loop AND `exp(-decay)` on the total field.
5. **O₃ is diagnostic, not gas-phase:** `O₃ = bg·dayness + photo·temp - titration·night` — plausible shape but not chemistry.
6. **No aerosol feedback:** PM does not affect PBL/radiation in the current emulator.
7. **No NOx/SO₂/secondary PM:** Only PM2.5, PM10 (PM2.5×1.6), and diagnostic O₃.
8. **No inversion dynamics:** Binary yes/no flag (t925-t2m > 3°C + wind < 4 m/s), no strength/height/duration.
9. **No plume dispersion:** Plume rise computes vertical injection but horizontal dispersion is only via a fixed σ gaussian filter.
10. **Fire pipeline disconnected:** `coupled_run` does not auto-fetch FIRMS → fire chain dormant on default CLI path.
11. **ML trained on synthetic data:** GNN/XGBoost validates against physically-simulated truth, not CPCB observations.
12. **Module 2 → Module 3 not wired:** `run_module3.py` generates its own synthetic data, ignoring Module 2 artifacts.
13. **No 72-h chain validation:** `forecast_rolling` runs 72 leads but metrics reported only for T+1..T+72 collectively.

## 8. Required Improvements for V2

### Critical (must-have for SIH demo)
- [ ] Dedicated inversion engine (strength/height/duration/trapping)
- [ ] Proper transport model (advection + diffusion on grid, sub-cell interpolation)
- [ ] Reduced chemistry (NOx→O₃, secondary PM, humidity/temp effects)
- [ ] Aerosol feedback (PM→radiation→PBL→PM)
- [ ] Auto FIRMS loading in coupled_run
- [ ] Stubble-burning engine with Punjab/Haryana fire generation
- [ ] Coupled forecast engine T+1→T+72 with aerosol feedback at each step
- [ ] Hybrid ML wired to physics model output
- [ ] Ablation study (6 configurations)
- [ ] Alert engine (6 alert types)
- [ ] V2 API routes
- [ ] Demo CLI (`python -m wrf_chem_delhi.demo`)
- [ ] 3 demo scenarios
- [ ] WRF-Chem adapter (placeholder)
- [ ] LIVE→CACHE→DEMO fallback

### Important (should-have)
- [ ] Sub-cell advection (fractional cell movement)
- [ ] Multi-level wind (surface + 925 + 850 hPa advection at different levels)
- [ ] Dedicated PBL module (TKE-based, bulk Richardson, Monin-Obukhov)
- [ ] Plume horizontal dispersion (Gaussian crosswind spread)
- [ ] Validation metrics per lead time
- [ ] Integration test

### Nice-to-have
- [ ] Documentation (10 files)
- [ ] Scenario switching in dashboard
- [ ] Forecast chart in dashboard

## 9. Final Recommended Architecture

```
wrf_chem_delhi/
├── __main__.py              # python -m wrf_chem_delhi.demo
├── config.py                # V2 config (extends aqf_delhi.config)
├── weather/                 # Weather engine (real GFS + synthetic)
│   ├── engine.py            # MetArrays, met_cell, PBL, stability
│   └── scenarios.py         # 3 demo weather scenarios
├── inversion/               # Inversion engine
│   └── engine.py            # strength, height, duration, trapping
├── fire/                    # Stubble-burning engine
│   ├── engine.py            # FIRMS + synthetic fires
│   └── emissions.py         # FRP→mass, upwind/downwind influence
├── plume/                   # Plume model
│   └── engine.py            # Freitas rise + Gaussian dispersion
├── transport/               # Transport model
│   └── engine.py            # Advection + diffusion + deposition
├── chemistry/               # Reduced chemistry
│   └── engine.py            # NOx↔O₃, secondary PM, aerosol growth
├── feedback/                # Aerosol-meteorology feedback
│   └── engine.py            # PM→radiation→PBL→PM loop
├── forecast/                # Coupled forecast engine
│   ├── engine.py            # T+1→T+72 with feedback loop
│   └── scenarios.py         # 3 demo scenarios
├── ml/                      # Hybrid ML
│   ├── hybrid.py            # T+1 model + 72h chain
│   └── ablation.py          # 6-config ablation study
├── aqi/                     # Indian AQI + Alert
│   ├── engine.py            # AQI calculator
│   └── alerts.py            # Alert engine
├── adapter/                 # WRF-Chem adapter
│   └── wrfchem.py           # run/read/convert stubs
├── api/                     # V2 API
│   └── v2_routes.py         # Extended endpoints
├── dashboard/               # V2 dashboard
│   └── v2_state.py          # Scenario-aware state
├── data/                    # Data management
│   └── modes.py             # LIVE→CACHE→DEMO
├── validation/              # Metrics + ablation
│   ├── metrics.py
│   └── ablation.py
├── demo.py                  # Demo CLI entry point
└── tests/
    ├── test_weather.py
    ├── test_inversion.py
    ├── test_fire.py
    ├── test_plume.py
    ├── test_transport.py
    ├── test_chemistry.py
    ├── test_feedback.py
    ├── test_forecast.py
    ├── test_ml_hybrid.py
    ├── test_aqi.py
    ├── test_alerts.py
    ├── test_api_v2.py
    └── test_integration.py
```

**Data flow:**
```
DEMO/HISTORICAL/LIVE data → Weather Engine → MetArrays
    → Inversion Engine → InversionState
    → Fire Engine → FireDetections + Emissions
    → Plume Engine → PlumeRise + Dispersion
    → Transport Engine → Advection + Diffusion
    → Chemistry Engine → PM2.5, PM10, O₃, NOx, SO₂, CO
    → Feedback Engine → PBL adjustment → Transport (loop)
    → Forecast Engine → T+1..T+72 fields
    → ML Hybrid → Bias-corrected T+1..T+72
    → AQI Engine → Indian AQI
    → Alert Engine → Warnings
    → API + Dashboard
```

This architecture:
- Is **WRF-Chem-inspired** (not claiming to be operational WRF-Chem)
- Runs on a **student laptop** (NumPy/SciPy only; optional XGBoost/PyTorch for ML)
- Supports **LIVE → CACHE → DEMO** fallback
- Has an **adapter** slot for real WRF-Chem
- Demonstrates the **full coupled chain** required by the problem statement
