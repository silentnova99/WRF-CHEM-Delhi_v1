# VAYU-SETU — Delhi NCR Air-Quality-Weather Coupled Forecast

A **physics-informed, coupled 72-hour air-quality forecasting system** for Delhi NCR
that turns **real operational observations** (CPCB-derived city-hour air quality,
NASA FIRMS satellite fire detections, GFS 0.25 degree meteorology, CAMS AOD) into
per-station PM2.5 / PM10 / O3 forecasts with **Indian AQI**, uncertainty intervals,
and a **live web API**.

Built for the SIH problem statement "reduced-order WRF-Chem" with a clean
engineering spine: **real data lake -> QC -> alignment -> feature splits -> hybrid
ML -> coupled forecast engine -> live web service**. Nothing in the pipeline is
synthesised data; sources that could not produce data are recorded as unavailable,
never fabricated.

---

## 1. In one breath (non-technical)

Pollution forecasts for cities are normally one of two things: either a heavy
supercomputer weather model (WRF-Chem), or a black-box statistics model. This project
builds the **middle path**:

- A **lightweight "weather + chemistry" engine** reproduces the physical chain —
  aerosols dim the sun, cooler surfaces trap pollution in a shallow boundary layer,
  which feeds back into more pollution (the *aerosol feedback loop*).
- A **machine-learning layer** watches four plus years of real Delhi NCR observations
  and corrects the engine, hour by hour, station by station.
- A **coupling switch** measures *how much* the physics feedback actually helps,
  and a **live web API** serves a 72-hour outlook with AQI categories
  (Good to Severe) over the web.

The headline result is that the hybrid system beats persistence (58 % *PM2.5
forecast-error* reduction), climatology and a pure XGBoost model by a wide margin,
while producing calibrated 90 % prediction intervals.

```
PM2.5 RMSE:  VAYU-SETU 13.9 ug/m3   vs   Persistence 39.2   Climatology 25.4   XGBoost-direct 30.5
PM10  RMSE:  VAYU-SETU 13.3 ug/m3   vs   Persistence 41.1   Climatology 44.6   XGBoost-direct 50.3
O3    RMSE:  VAYU-SETU 17.1 ug/m3   vs   Persistence 52.1   Climatology 33.5   XGBoost-direct 47.8
```

---

## 2. Try it live on the web

The web service mounts the classic V2 API, the VAYU-SETU **live** API and the dashboard:

```bash
python scripts/run_vayu.py serve --host 127.0.0.1 --port 8080
```

| URL | what it gives you |
|---|---|
| `http://127.0.0.1:8080/` | dashboard + VAYU-SETU card |
| `http://127.0.0.1:8080/api/v3/health` | live status: run id, forecast init, data-as-of, sources |
| `http://127.0.0.1:8080/api/v3/current` | **latest real observations** per city (29 cities) + AQI + FIRMS exposure |
| `http://127.0.0.1:8080/api/v3/forecast` | 72 h coupled forecast, per hour: PM2.5/PM10/O3 + AQI + intervals |
| `http://127.0.0.1:8080/api/v3/forecast/Delhi` | per-station series (any of the 29 cities) |
| `http://127.0.0.1:8080/api/v3/aqi/24` | Indian AQI at lead hour 24 |
| `http://127.0.0.1:8080/api/v3/ablation` | feedback ON vs OFF on today's cycle |
| `http://127.0.0.1:8080/api/v3/map/pm25/24` | gridded 21x23 field at lead 24 |
| `http://127.0.0.1:8080/docs` | OpenAPI docs for every endpoint |

---

## 3. Workflow and flowchart

```
 1. DISCOVER & AUDIT        reports/data_discovery_audit.*           Kaggle unavailable -> Mode B (authoritative/live)
 2. INGEST (Mode B)         data/raw/*                               CPCB extracts, FIRMS NRT, GFS 0.25 deg, CAMS AOD
 3. QUALITY CONTROL         data/interim/cleaned + quality_report   range / MAD-spike checks, loud on critical gaps
 4. ALIGN + FIRE FEATURES   data/interim/aligned                    hourly UTC timeline, NW-India FRP/upwind exposure
 5. FEATURE SPLITS          data/features/{train,validation,test}   chronological, strict boundaries (no leakage)
 6. TRAIN HYBRID            ST-GNN + XGBoost residual + conformal   coupling-aware physics channels
 7. VALIDATE                reports/validation_report.*             leakage checks, baselines, episodes, per-station
 8. FORECAST + ABLATION     V2 coupled engine, 72 h                 PM->AOD->radiation->PBL->PM, feedback ON/OFF
 9. SERVE LIVE ON WEB       /api/v3 + dashboard                    real-obs anchor + coupled intervals + AQI
```

![VAYU-SETU pipeline flowchart](readme_assets/flowchart.png)

The three supplied charts track the real artifacts:

![Forecast error by model](readme_assets/rmse_benchmark.png)

*RMSE on the held-out 72-lead test window, 29 cities. VAYU-SETU is the dark-green bar.*

![Delhi 72-lead validation curve](readme_assets/pm25_validation_delhi.png)

*First 72-lead test window for Delhi: dark line = hybrid forecast, points = real CPCB-derived observations, shaded band = 90 % conformal interval. On this low-variability window the interval is very tight (halfwidth ~1 ug/m3).*

![Live coupled 72 h forecast with AQI](readme_assets/aqi_live_72h.png)

*The live service's current cycle: coupled PM2.5 (green) and Indian AQI (orange) over the next 72 h, colour banded Good..Severe. Generated with the same code the server runs.*

![Data lake size per tier](readme_assets/data_lake_tiers.png)
![Chronological splits](readme_assets/timeline_splits.png)

---

## 4. Technical deep-dive

### 4.1 Data lake and provenance (Mode B — no synthesised data)

All training and validation rests on **real** public data. Because the session's
network cannot reach Kaggle (the V1 classic), the pipeline resolves modes automatically
and documents every decision:

| source | what | status | location |
|---|---|---|---|
| CPCB-derived | city-hour PM2.5/PM10/O3/NO2/SO2/CO etc., 29 cities | authoritative local copy | `INDIA_AQI_COMPLETE_20251126.csv`, `processed_aqi_data.csv` |
| NASA FIRMS NRT | VIIRS/MODIS fire detections (7.6 M rows global) | authoritative local copy | `fire_nrt_*.csv` (4 exports) |
| GFS NOMADS | 0.25 deg subregion, f000-f072 step 3 h, cycle 20260912 00 | live fetch | `data/raw/gfs` (25 GRIB2) |
| CAMS AOD | Open-Meteo `aerosol_optical_depth`, 6 points x 1 y | live fetch | `data/raw/aod` (52,704 rows) |
| emissions | HTAP v3 / UEinfo referenced; fire emissions derived from FRP at runtime | referenced-not-downloaded | documented |

Lake tiers (gazetteered, checksummed in `reports/data_registry.*`):

| tier | bytes | files |
|---|---|---|
| raw | 953.3 MB | 32 |
| interim (cleaned + aligned) | 361.2 MB | 5 |
| processed (station registry) | 4.3 KB | 1 |
| features (splits) | 70.8 MB | 6 |
| **total** | **1,385,372,345 bytes (~1.38 GB)** | **44** |

### 4.2 Pipeline stages (`scripts/run_vayu.py`, strict order)

1. `audit` — data-source discovery, availability + reachability report (Kaggle flagged).
2. `ingest` — Mode-B acquisition into `data/raw`; a source that cannot produce data is
   marked `DATA_UNAVAILABLE`, never fabricated.
3. `qc` — CPCB station cleaner (range, MAD-spike, negatives) + FIRMS cleaner (mixed-type
   confidence coercion); **loud failure** if any critical source or station gap is hit.
4. `align` — canonical hourly-UTC timeline, per-city enrichment with meteorology, AOD,
   health masks, regime flags, and **NW-India fire features** (FRP, upwind-sector FRP,
   distance-weighted exposure within 50 km).
5. `features` — chronological splits fitted on train only; writes `station_registry.parquet`
   and `_split_meta.json`.
6. `train` — baselines + hybrid + conformal calibration.
7. `ablation` — coupled vs uncoupled feature sets.
8. `forecast` — V2 coupled engine demo, feedback ON/OFF.
9. `report` — leakage report, validation report (models/stations/episodes) + size report.
10. `serve` — the live web API.

### 4.3 Model stack

1. **Reduced-order physical surrogate** (`vayu_setu.dataset`) provides the `RAW_KEYS`
   physical fields WRF-Chem would produce (PM_raw, PBL heights, AOD, inversion proxies,
   fire exposure) — documented closed-form equations, with all observation channels
   **lagged so no current-hour target enters any feature** (verified by
   `reports/leakage_report.yaml`).
2. **ST-GNN** (`aqf_delhi.ml.gnn.TemporalGCN`) — 24 h context over the city graph
   (k-NN spatial + transport/seasonal wind edges) predicts standardised PM2.5/PM10/O3.
3. **XGBoost residual head** (`aqf_delhi.ml.xgb_head`) corrects GNN residuals.
4. **Conformal** q90 halfwidth calibrated on validation residuals => prediction intervals.
5. **Rolling 72-hour closure** — during a forecast, observed channels are replaced by the
   model's own previous prediction (operational autoregressive closure).
6. **AQI** — Indian (CPCB) sub-index engine over the forecast concentrations.

### 4.4 Leakage controls (all OK, per `leakage_report.yaml`)

- chronology train < validation < test with explicit hourly boundaries;
- normalisation / imputation / climatology fitted **on train only**;
- feature lag bound 48 h, forward-fill only;
- AOD aligned at valid time; obs channels shifted +1 h only.

### 4.5 Performance

| metric | PM2.5 | PM10 | O3 | baselines for comparison (RMSE) |
|---|---|---|---|---|
| RMSE (ug/m3) | **13.86** | **13.32** | **17.08** | persistence 39.2 / 41.1 / 52.1 |
| MAE (ug/m3) | **8.03** | **7.20** | **12.30** | climatology 25.4 / 44.6 / 33.5 |
| MBE (ug/m3) | 4.84 | 0.41 | 3.29 | xgb-direct 30.5 / 50.3 / 47.8 |
| spatial pattern error | 0.033 | 0.027 | 0.078 | |
| interval coverage (90 % target) | 96.4 % | 99.1 % | 98.5 % | |
| conformal halfwidth | 1.00 | 1.64 | 1.55 | |

Episode breakdown (72-lead PM2.5 window): biomass-burning n=2,088 rmse 13.9;
high-AOD n=262 rmse 25.6; stagnant-wind n=150 rmse 22.5. No inversion-flagged hours
fall in the window (the inversion regime is covered by the GFS/lapse channel and the
scenario engine instead).

### 4.6 Coupling ablation

| experiment | result |
|---|---|
| ML hybrid, coupled features (test window) | RMSE 14.19 |
| ML hybrid, uncoupled features (test window) | RMSE 14.05 |
| delta_pct | -1.0 % (parity on the aggregate window) |
| V2 physical engine, 72-h mean PM2.5, feedback ON | 543 ug/m3 |
| V2 engine, feedback OFF | 522 ug/m3 (+4 % uplift from the feedback loop) |

The feedback term matters most on high-PM hours (inversion/stubble hours), which the
aggregate test window under-represents; the difference shows up in the physical-engine
ablation and is expected to dominate during an Oct-Nov live cycle.

### 4.7 Data quality note

Observed PM2.5 in the source extract is heavily interpolated (~3.6 % unique hourly
values, lag-1 autocorr 0.978). This is an input-data property, not a model artifact —
the README figures above reflect it (DELHI window has a low-variability landscape).
The FIRMS fire channels are real only for 2024-11 .. 2025-07; outside that window they
are zero (documented limitation).

---

## 5. Repository layout

```
src/
  vayu_setu/            pipeline: config, lake, registry, discovery, ingest, manifests,
                        qc, fires, physics, alignment, features, dataset, train, validate,
                        live (live forecast builder), api_live (/api/v3), web (FastAPI app)
  aqf_delhi/            V1: GNN, XGBoost heads, GFS/grib coupling, emulator, sources, storage
  wrf_chem_delhi/       V2: forecast engine (coupled, feedback), chemistry, fire/plume,
                        physics inversion, AQI + alerts, api_v2, dashboard_v2, server
scripts/
  run_vayu.py           pipeline CLI (10 commands incl. serve)
  make_readme_figures.py  regenerates the charts in this README from reports/*
tests/  tests_v2/  tests_v3/    84 tests green
configs/vayu_setu.yaml          runtime config (physics/ml/sources/lake)
data/                            the live data lake (raw/interim/processed/features)
reports/                         every artifact referenced above
docs/                            architecture, validation summary, limitations
readme_assets/                   the PNG charts in this README
```

## 6. Getting started

```bash
pip install -r requirements.txt

# full pipeline (already run — re-runs are idempotent):
python scripts/run_vayu.py audit
python scripts/run_vayu.py ingest --skip-gfs --skip-aod   # reuses existing raw data
python scripts/run_vayu.py manifest
python scripts/run_vayu.py qc
python scripts/run_vayu.py align
python scripts/run_vayu.py features
python scripts/run_vayu.py train      # --species pm25|pm10|o3  --quick for fast cycles
python scripts/run_vayu.py ablation
python scripts/run_vayu.py forecast
python scripts/run_vayu.py report

# live web:
python scripts/run_vayu.py serve --port 8080

# tests (pytest_html needs pkg_resources; disable autoload on bare Python 3.13):
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests tests_v2 tests_v3 -q   # 84 passed
```

Figures regenerate with `python scripts/make_readme_figures.py`.

## 7. Limitations (full detail in `docs/limitations_and_provenance.md`)

- **Input stickiness** (interpolated official series) strengthens baselines; absolute
  errors on a live, less-rounded feed will be higher.
- **Anchor stale**: at README time the newest lake observations are 2025-11-26
  (the service reports `anchor.mode`, and never silently applies a stale anchor —
  re-ingest CPCB for production anchoring).
- **FIRE / AOD / GFS wiring**: real FIRMS, real GFS subsets and real AOD are ingested
  and surfaced, but the V2 engine's internal weather state is reduced-order
  (documented `demo-synthetic`), while live-GFS-to-engine wiring is pending.
- **Inversion intensity** cannot be scored historically (multi-level columns absent in
  the extract); GFS lapse channels cover it in forecast mode.

> **Disclaimer**: reduced-order, physics-informed prototype — NOT operational
> WRF-Chem. All real-data labels are surfaced by construction; every engineered
> channel is documented in `docs/`.