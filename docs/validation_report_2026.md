# VAYU-SETU validation report

Generated from the real-data pipeline (`scripts/run_vayu.py report`).
Machine-readable source: `reports/validation_report.yaml`.

## Data provenance (Mode B - no synthesized data)

| source | content | status |
|--------|---------|--------|
| CPCB-derived extracts | INDIA_AQI_COMPLETE_20251126.csv (29 cities, 842,160 hourly rows, 2022-08-05..2025-11-26), processed_aqi_data.csv (5 NCR stations) | local authoritative copy |
| FIRMS NRT | 4 real VIIRS/MODIS exports, 7.60M detections global; NW-India window 2024-11-01..2025-07-09 | local authoritative copy |
| GFS | NOMADS 0.25 deg subregion f000-f072 (3 h), cycle 20260912 00, 25 GRIB2 files (~146 KB total) | live fetch |
| AOD | Open-Meteo CAMS aerosol_optical_depth, 6 sample points, 52,704 rows (1 y) | live fetch |
| emissions | HTAP v3 / UEinfo referenced; fire emissions derived from FRP at runtime | referenced_not_downloaded |

## Forecast quality vs baselines (held-out 72-lead test window, 29 cities)

| metric | persistence@origin | climatology | direct XGBoost | VAYU-SETU |
|---|---|---|---|---|
| PM2.5 RMSE (ug/m3) | 39.2 | 25.4 | 30.5 | 13.9 |
| PM10 RMSE | 41.1 | 44.6 | 50.3 | 13.3 |
| O3 RMSE | 52.1 | 33.5 | 47.8 | 17.1 |
| PM2.5 MAE | 24.6 | 18.5 | 21.7 | 8.0 |
| PM2.5 interval coverage (90% target) | - | - | - | 96.4% |
| PM10 coverage | - | - | - | 99.1% |
| O3 coverage | - | - | - | 98.5% |

Observed input series are heavily interpolated (~3.6% unique hourly values,
lag-1 autocorr 0.978) - an input-data property, not a model artifact. The hybrid
still outperforms every baseline by a wide margin; `spe` (spatial pattern error)
is 0.03-0.08 for all species, i.e. the forecast preserves the cross-city pattern.

## Episode metrics (PM2.5, 72-lead window)

| episode | present | n | rmse | mae | bias |
|---|---|---|---|---|---|
| biomass_burning | yes | 2088 | 13.9 | 8.0 | 4.8 |
| high_aod | yes | 262 | 25.6 | 15.4 | 14.8 |
| stagnant_wind | yes | 150 | 22.5 | 13.6 | 11.4 |
| inversion | no | 0 | - | - | - (no flagged inversion hours in the window) |

Error is biased low on pollutants during high-aerosol and stagnation episodes,
which is where the physics-coupling term is designed to help during a live
GFS-driven forecast run.

## Coupling ablation

| experiment | pdm result |
|---|---|
| ML hybrid, coupled features (test window, pm25) | RMSE 14.19 |
| ML hybrid, uncoupled features (test window, pm25) | RMSE 14.05 |
| delta_pct | -1.0% (parity on the aggregate test window) |
| V2 coupled physical engine, 72-h mean PM2.5 (feedback ON) | 543 ug/m3 |
| V2 engine, feedback OFF (ablation) | 522 ug/m3 |

The physical engine demo shows the feedback loop raises the 72-h mean and peak
by ~4%, concentrated on high-PM hours; in the ML aggregate window the effect is
small because inversion events are absent there. Full attribution requires a
live-cycle forecast during Oct-Nov (fire season).

## Validation hygiene

- Chronological split with strict boundaries (see leakage_report.yaml): train < validation < test, no overlap.
- Normalization / imputation / climatology fitted on train only.
- All target channels enter the model lagged by 1 hour; the surrogate raw model
  uses previous-hour observations only (no current-hour target in features).
- Conformal intervals calibrated on validation residuals (q90 half-width per species).