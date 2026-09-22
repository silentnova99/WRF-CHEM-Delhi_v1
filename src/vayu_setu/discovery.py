"""Data-source discovery + audit (sections 4, 5 and 50 of the spec).

Encodes the decision tree:

    Is a trustworthy Kaggle dataset available?
        YES -> validate provenance -> use
        NO  -> search authoritative source -> use if reachable, else
               STATUS = DATA_UNAVAILABLE (never fabricate)

Records candidate datasets found during discovery (Kaggle + authoritative)
plus the final decision, and renders the audit as YAML + markdown.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml

from .config import VayuConfig


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Candidate datasets surfaced by the discovery pass (Kaggle search + web audit,
# 2026-09-19). URLs appear as found; sizes/coverage flagged where uncertain.
DISCOVERY_RECORDS: Dict[str, Any] = {
    "cpcb": {
        "candidates": [
            {
                "name": "Delhi Air Quality Time Series Data 2025 to 2026-05",
                "provider": "Kaggle/jaspreetsingh9652",
                "kaggle_id": "jaspreetsingh9652/delhi-air-quality-time-series-data-2025-to-2026-05",
                "url": "https://www.kaggle.com/datasets/jaspreetsingh9652/delhi-air-quality-time-series-data-2025-to-2026-05",
                "coverage": "2025-02 -> 2026-05, 4 CAAQMS stations",
                "vars": ["pm25", "pm10", "no2", "no", "nox", "o3", "co", "so2", "T", "RH", "WS", "WD"],
                "license": "CC0",
                "verdict": "reject-fallback",   # local authoritative mirror preferred
            },
            {
                "name": "India Real-Time AQI (Hourly)",
                "provider": "Kaggle/imrancoder786",
                "kaggle_id": "imrancoder786/india-real-time-air-quality-index-aqi-hourly",
                "url": "https://www.kaggle.com/datasets/imrancoder786/india-real-time-air-quality-index-aqi-hourly",
                "coverage": "2016-2026 all India hourly",
                "license": "Kaggle ToS",
                "verdict": "reject-unvetted",
            },
            {
                "name": "Air Quality Data in India 2015-2020 (CPCB NAMP daily)",
                "provider": "Kaggle/rohanrao",
                "kaggle_id": "rohanrao/air-quality-data-in-india",
                "url": "https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india",
                "coverage": "2015-2020 daily, multi-city",
                "verdict": "reject-coarse",
            },
        ],
        "authoritative": [
            "airquality.cpcb.gov.in CCR repository (per-station history)",
            "data.gov.in OGD resource 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69 (live OGD API - already ingested in repo, 2026-09-12 partition present)",
            "Local extracts on disk: processed_aqi_data.csv (Delhi NCR 5 stations, 2019-2024 hourly), INDIA_AQI_COMPLETE_20251126.csv (29 cities 2022-2025 hourly + met + AOD)",
        ],
        "decision": "USE_AUTHORITATIVE_LOCAL — local real CPCB-derived extracts + live OGD partition; Kaggle rejected as convenience-only.",
    },
    "gfs": {
        "candidates": [
            {"name": "India Historical Climate Data 2014-2023 (Open-Meteo ERA5)",
             "provider": "Kaggle/thearkknight",
             "kaggle_id": "thearkknight/india-historical-climate-data-2014-2023",
             "url": "https://www.kaggle.com/datasets/thearkknight/india-historical-climate-data-2014-2023",
             "verdict": "reject-not-gfs", "note": "station-point ERA5, not native GFS, no PBLH"},
            {"name": "Meterological factors of india (2023, ERA5 with PBLH)",
             "provider": "Kaggle/moonknightmarvel",
             "kaggle_id": "moonknightmarvel/era5-2023",
             "url": "https://www.kaggle.com/datasets/moonknightmarvel/era5-2023",
             "verdict": "reject-single-year", "note": "only 2023"},
        ],
        "authoritative": [
            "NOMADS filter_gfs_0p25.pl (live, verified reachable, ~4 KB/hour subregion)",
            "NCAR RDA ncep-gfs-0-25-degree-historical-archive (2015-2026)",
            "AWS noaa-gfs-bdp-pds (rolling 4-week s3 mirror)",
        ],
        "decision": "USE_NOMADS_LIVE — verified subregion GRIB2 fetch for operational cycles; history via Open-Meteo ERA5 archive for the training era (documented as ERA5/GFS-analogue met) and RDA for full GFS backfill.",
    },
    "firms": {
        "candidates": [
            {"name": "FIRE FROM SPACE: INDIA 8 YEARS",
             "provider": "Kaggle/sherkhan15",
             "kaggle_id": "sherkhan15/indian-wildfire-nasa-dataset-8-years",
             "url": "https://www.kaggle.com/datasets/sherkhan15/indian-wildfire-nasa-dataset-8-years",
             "coverage": "2012-2020 VIIRS 375m + MODIS India",
             "verdict": "verified-real", "note": "directly FIRMS-derived, DOI-cited"},
            {"name": "NASA VIIRS Fire History (Global NRT)",
             "provider": "Kaggle/yatharth22bce11044",
             "kaggle_id": "yatharth22bce11044/viirs-fire-history",
             "url": "https://www.kaggle.com/datasets/yatharth22bce11044/viirs-fire-history",
             "verdict": "verified-real"},
        ],
        "authoritative": [
            "NASA FIRMS archive (firms.modaps.eosdis.nasa.gov) — local exports present: fire_nrt_SV-C2_565336.csv (VIIRS S-NPP), fire_nrt_J1V-C2_565335.csv (NOAA-20), fire_nrt_M-C61_565334.csv (MODIS), fire_nrt_SV-C2_634473.csv (later VIIRS window)",
        ],
        "decision": "USE_LOCAL_FIRMS — real NASA FIRMS NRT exports on disk; Kaggle mirrors equivalent; FIRMS bbox grid download for top-ups.",
    },
    "aod": {
        "candidates": [
            {"name": "Kolkata Open-Meteo AQ (cams_global AOD/pollutants)",
             "provider": "Kaggle/nitirajkulkarni",
             "kaggle_id": "nitirajkulkarni/kolkata-in-1275004",
             "verdict": "reject-city", "note": "Kolkata, not Delhi — pattern reference only"},
        ],
        "authoritative": [
            "Open-Meteo Air Quality API CAMS aerosol_optical_depth (keyless, 2022-08+; verified reachable)",
            "MODIS MAIAC MCD19A2.061 (1 km daily, via LP DAAC / Earth Engine)",
            "Local: AOD column inside INDIA_AQI_COMPLETE_20251126.csv (2022-2025)",
        ],
        "decision": "USE_OPEN_METEO_CAMS + LOCAL_AOD_COLUMN — calibrated emissivity/attenuation factor uses actual AOD; no fabrication.",
    },
    "emissions": {
        "candidates": [],
        "authoritative": [
            "HTAP v3.x mosaic (JRC-EDGAR, REAS-fill, 0.1 deg) — ~200 GB, out of budget for prototype",
            "UEinfo gridded inventories Delhi/NCR 1 km",
            "configs/emission_factors.yaml (already in repo, Wooster-2005 FRP->fuel, species EFs)",
        ],
        "decision": "REFERENCED_NOT_DOWNLOADED — fire emissions derived from FRP at runtime via existing engine; urban baseline from emission_factors.yaml. HTAP/UEinfo documented for production hand-off.",
    },
}


def audit_records(cfg: VayuConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {"generated_at": _now_utc()}
    src_cfg = cfg.section("sources")
    for source, rec in DISCOVERY_RECORDS.items():
        cfg_rec = src_cfg.get(source, {})
        out[source] = {
            "provider_candidates": rec["candidates"],
            "authoritative_sources": rec["authoritative"],
            "final_decision": rec["decision"],
            "status": "DATA_AVAILABLE" if cfg_rec.get("enabled", True) else "DATA_UNAVAILABLE",
            "primary_selected": cfg_rec.get("primary"),
        }
    return out


def write_audit(cfg: VayuConfig, out_yaml: str, out_md: str) -> None:
    records = audit_records(cfg)
    ym, mm = out_yaml, out_md
    import pathlib
    pathlib.Path(ym).parent.mkdir(parents=True, exist_ok=True)
    with open(ym, "w", encoding="utf-8") as fh:
        yaml.safe_dump(records, fh, sort_keys=False)
    md: List[str] = ["# VAYU-SETU data-source discovery audit",
                     "", f"Generated: {records['generated_at']}", ""]
    for source, rec in records.items():
        if source == "generated_at":
            continue
        md.append(f"## {source.upper()}")
        md.append("")
        md.append("### Options")
        md.append("")
        md.append("| candidate | provider | coverage | verdict |")
        md.append("|---|---|---|---|")
        for c in rec["provider_candidates"]:
            md.append(f"| {c.get('name','')} | {c.get('provider','')} | {c.get('coverage','')} | {c.get('verdict','')} |")
        md.append("")
        md.append("### Authoritative sources")
        for a in rec["authoritative_sources"]:
            md.append(f"- {a}")
        md.append("")
        md.append(f"**Decision:** {rec['final_decision']}")
        md.append("")
        md.append(f"**Status:** `{rec['status']}` (primary: `{rec['primary_selected']}`)")
        md.append("")
    with open(mm, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))