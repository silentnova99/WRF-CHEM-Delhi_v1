"""CPCB ground-station observation canonical schema.

1 row per (station, metric, observation time). Units are normalized to
µg/m³ for particulates and ppb for gaseous species at ingest time.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from aqf_delhi.schemas.common import CanonicalRecord


class CpcbObservation(CanonicalRecord):
    station_id: str
    station_name: str
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    metric: str  # PM2.5 | PM10 | O3 | NO2 | SO2 | CO | AQI
    value: float
    unit: str  # ug/m3 | ppb | mg/m3 | index
    sensor_type: str = "continuous"  # continuous | manual
    source_api_ver: str = "unknown"
    raw_payload: dict = Field(default_factory=dict)