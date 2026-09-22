# VAYU-SETU validation report

generated: 2026-09-19T18:34:24Z

## Chronological split (no leakage)

| split | start | end |
|---|---|---|
| train | 2022-08-05 00:00:00 | 2023-11-26 21:00:00 |
| validation | 2023-11-26 22:00:00 | 2024-11-26 21:00:00 |
| test | 2024-11-26 22:00:00 | 2025-11-26 23:00:00 |

## Test-window metrics (72-h forecast)

| model | rmse | mae | bias | correlation | r2 |
|---|---|---|---|---|---|
| persistence | - | - | - | - | - |
| climatology | - | - | - | - | - |
| xgb_direct | - | - | - | - | - |
| vayu-setu | - | - | - | - | - |

## Episode metrics (test window, per species)

| species | episode | present | n | rmse | mae | bias |
|---|---|---|---|---|---|---|
| pm25 | inversion | False | - | - | - | - |
| pm25 | biomass_burning | True | 2088 | 13.86 | 8.028 | 4.844 |
| pm25 | high_aod | True | 262 | 25.566 | 14.551 | 12.608 |
| pm25 | stagnant_wind | True | 150 | 22.454 | 12.033 | 7.972 |
| pm10 | inversion | False | - | - | - | - |
| pm10 | biomass_burning | True | 2088 | 13.319 | 7.197 | 0.407 |
| pm10 | high_aod | True | 262 | 23.357 | 11.404 | 4.011 |
| pm10 | stagnant_wind | True | 150 | 23.443 | 10.499 | 0.652 |
| o3 | inversion | False | - | - | - | - |
| o3 | biomass_burning | True | 2088 | 17.078 | 12.304 | 3.287 |
| o3 | high_aod | True | 262 | 19.502 | 14.633 | 10.82 |
| o3 | stagnant_wind | True | 150 | 13.861 | 9.92 | 3.298 |

## Coupling ablation

- coupled rmse: 14.189645707593504
- uncoupled rmse: 14.04583762779889
- delta (coupled advantages -): -1.024%
