# Provenance, limitations and known gaps

## What is real vs derived

- **Real observations**: CPCB-derived city-hour observations (INDIA_AQI_COMPLETE
  + processed_aqi_data extracts), NASA FIRMS active-fire detections, live GFS
  subregions from NOMADS, CAMS AOD via Open-Meteo.
- **Derived (documented reduced-order)**: `pm25_raw/pm10_raw/o3_raw` "model"
  fields (persistence-decay + met-driven emissions + fire exposure + coupling
  trapping), `pblh_base_m/pblh_coupled_m`, `aod_fused`, `feedback_strength`,
  inversion/lapse proxies. These stand in for WRF-Chem output during training;
  the ML stages bias-correct them identically to how they would correct
  WRF-Chem fields operationally.
- **Nothing is fabricated**: any source that could not produce data is recorded
  as `DATA_UNAVAILABLE` / `referenced_not_downloaded`, never synthesized.

## Known gaps / limitations

1. **Input stickiness**: the CPCB-derived series are heavily interpolated
   (~3.6% unique values; lag-1 autocorr 0.978). Persistence-based baselines are
   therefore strong and absolute RMSE numbers look optimistic versus what a
   live, less-rounded feed would give.
2. **Fire coverage**: local FIRMS exports cover NW-India only for
   2024-11-01..2025-07-09. Outside that window fire features are zero (early
   years have no fire detections). A real NRT subscription would fill this.
3. **Inversion flag**: `Temp_Inversion` is a presence/absence flag; the
   multi-level temperature columns and `Inversion_Strength_C` are 100% missing
   in the source extract, so inversion *intensity* cannot be scored from
   history. The forecast loop instead uses GFS multi-level data (fetched) for
   stability and lapse.
4. **Multi-level met history**: Open-Meteo archive does not serve
   temperature at 80/120/180 m; only surface fields were available for the
   historical feature frame.
5. **Kaggle mode**: Kaggle API is unavailable (no credentials) so the session
   runs Mode B (authoritative/live) only; documented in DISCOVERY_RECORDS.
6. **data.gov.in OGD**: unreachable from this network; live CPCB pull could not
   be exercised (local 2026-09-12 partition used instead).
7. **Ablation granularity**: on the aggregate test window the coupled vs
   uncoupled feature sets differ by ~1% (parity). The physics term is expected
   to matter most during inversion/stubble episodes (Oct-Nov), which the same
   window does not exercise; the physical-engine demo shows the sign and size
   of the effect (~4% mean uplift) in a stubble-plume scenario.
8. **Conformal coverage** is conservative (96-99% vs 90% target) - the q90
   residual quantile is computed on a small validation window.

## Runtime notes

- Pipeline runner: `python scripts/run_vayu.py <stage>`.
- Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests tests_v2 tests_v3 -q` (84 passed).
- Config: `configs/vayu_setu.yaml`; lake root is `data/` under the repo (override with `AQF_DATA_ROOT`).