"""NASA FIRMS active-fire / Fire Radiative Power (FRP) canonical schema."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from aqf_delhi.schemas.common import CanonicalRecord

Satellite = Literal["VIIRS_SNPP", "VIIRS_NOAA20", "MODIS_AQUA", "MODIS_TERRA"]


class FireDetection(CanonicalRecord):
    fire_id: str
    satellite: Satellite
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    acq_datetime: datetime
    frp_mw: float = Field(ge=0.0)          # Fire Radiative Power, megawatts
    brightness_kelvin: float = Field(gt=200.0, lt=700.0)
    confidence_percent: float = Field(ge=0.0, le=100.0)
    scan_km: float = Field(ge=0.0)
    track_km: float = Field(ge=0.0)
    day_night: Literal["D", "N"] = "D"
    grid_i: int | None = None               # assigned 4 km cell column
    grid_j: int | None = None               # assigned 4 km cell row
    assigned_grid_id: str | None = None