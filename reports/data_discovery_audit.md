# VAYU-SETU data-source discovery audit

Generated: 2026-09-19T18:31:44Z

## CPCB

### Options

| candidate | provider | coverage | verdict |
|---|---|---|---|
| Delhi Air Quality Time Series Data 2025 to 2026-05 | Kaggle/jaspreetsingh9652 | 2025-02 -> 2026-05, 4 CAAQMS stations | reject-fallback |
| India Real-Time AQI (Hourly) | Kaggle/imrancoder786 | 2016-2026 all India hourly | reject-unvetted |
| Air Quality Data in India 2015-2020 (CPCB NAMP daily) | Kaggle/rohanrao | 2015-2020 daily, multi-city | reject-coarse |

### Authoritative sources
- airquality.cpcb.gov.in CCR repository (per-station history)
- data.gov.in OGD resource 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69 (live OGD API - already ingested in repo, 2026-09-12 partition present)
- Local extracts on disk: processed_aqi_data.csv (Delhi NCR 5 stations, 2019-2024 hourly), INDIA_AQI_COMPLETE_20251126.csv (29 cities 2022-2025 hourly + met + AOD)

**Decision:** USE_AUTHORITATIVE_LOCAL — local real CPCB-derived extracts + live OGD partition; Kaggle rejected as convenience-only.

**Status:** `DATA_AVAILABLE` (primary: `local`)

## GFS

### Options

| candidate | provider | coverage | verdict |
|---|---|---|---|
| India Historical Climate Data 2014-2023 (Open-Meteo ERA5) | Kaggle/thearkknight |  | reject-not-gfs |
| Meterological factors of india (2023, ERA5 with PBLH) | Kaggle/moonknightmarvel |  | reject-single-year |

### Authoritative sources
- NOMADS filter_gfs_0p25.pl (live, verified reachable, ~4 KB/hour subregion)
- NCAR RDA ncep-gfs-0-25-degree-historical-archive (2015-2026)
- AWS noaa-gfs-bdp-pds (rolling 4-week s3 mirror)

**Decision:** USE_NOMADS_LIVE — verified subregion GRIB2 fetch for operational cycles; history via Open-Meteo ERA5 archive for the training era (documented as ERA5/GFS-analogue met) and RDA for full GFS backfill.

**Status:** `DATA_AVAILABLE` (primary: `nomads`)

## FIRMS

### Options

| candidate | provider | coverage | verdict |
|---|---|---|---|
| FIRE FROM SPACE: INDIA 8 YEARS | Kaggle/sherkhan15 | 2012-2020 VIIRS 375m + MODIS India | verified-real |
| NASA VIIRS Fire History (Global NRT) | Kaggle/yatharth22bce11044 |  | verified-real |

### Authoritative sources
- NASA FIRMS archive (firms.modaps.eosdis.nasa.gov) — local exports present: fire_nrt_SV-C2_565336.csv (VIIRS S-NPP), fire_nrt_J1V-C2_565335.csv (NOAA-20), fire_nrt_M-C61_565334.csv (MODIS), fire_nrt_SV-C2_634473.csv (later VIIRS window)

**Decision:** USE_LOCAL_FIRMS — real NASA FIRMS NRT exports on disk; Kaggle mirrors equivalent; FIRMS bbox grid download for top-ups.

**Status:** `DATA_AVAILABLE` (primary: `local`)

## AOD

### Options

| candidate | provider | coverage | verdict |
|---|---|---|---|
| Kolkata Open-Meteo AQ (cams_global AOD/pollutants) | Kaggle/nitirajkulkarni |  | reject-city |

### Authoritative sources
- Open-Meteo Air Quality API CAMS aerosol_optical_depth (keyless, 2022-08+; verified reachable)
- MODIS MAIAC MCD19A2.061 (1 km daily, via LP DAAC / Earth Engine)
- Local: AOD column inside INDIA_AQI_COMPLETE_20251126.csv (2022-2025)

**Decision:** USE_OPEN_METEO_CAMS + LOCAL_AOD_COLUMN — calibrated emissivity/attenuation factor uses actual AOD; no fabrication.

**Status:** `DATA_AVAILABLE` (primary: `open_meteo_cams`)

## EMISSIONS

### Options

| candidate | provider | coverage | verdict |
|---|---|---|---|

### Authoritative sources
- HTAP v3.x mosaic (JRC-EDGAR, REAS-fill, 0.1 deg) — ~200 GB, out of budget for prototype
- UEinfo gridded inventories Delhi/NCR 1 km
- configs/emission_factors.yaml (already in repo, Wooster-2005 FRP->fuel, species EFs)

**Decision:** REFERENCED_NOT_DOWNLOADED — fire emissions derived from FRP at runtime via existing engine; urban baseline from emission_factors.yaml. HTAP/UEinfo documented for production hand-off.

**Status:** `DATA_UNAVAILABLE` (primary: `referenced`)
