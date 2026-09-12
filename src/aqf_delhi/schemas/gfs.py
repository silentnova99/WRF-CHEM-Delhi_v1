"""GFS model field asset canonical schema (gridded remains on object store).

Phase-1 stores *asset metadata* (raster URIs + level-1 stats) in the
tabular store; the heavy grids stay as COG/NetCDF objects. This keeps the
canonical table small while preserving full reprocessing provenance.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, field_validator

from aqf_delhi.schemas.common import CanonicalRecord

LevelKind = Literal["surface", "pressure", "height_agl", "model_level"]


class GfsFieldAsset(CanonicalRecord):
    run_id: str                       # e.g. "gfs.20260912.00z"
    init_utc: datetime
    lead_h: int = Field(ge=0)
    valid_utc: datetime
    variable: str                     # e.g. "TMP"
    level_str: str                    # e.g. "2 m above ground" | "850 mb"
    level_kind: LevelKind
    grid: str = "0p25"                # 0.25 degree product
    crs: str = "EPSG:4326"
    asset_uri: str                    # object-store path to raster/NetCDF
    sha256: str | None = None
    rows: int | None = None
    cols: int | None = None
    min_val: float | None = None
    max_val: float | None = None
    mean_val: float | None = None
    missing_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("variable")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @property
    def h_key(self) -> str:
        """Stable dedup key for a GFS (run, lead, variable, level)."""
        return f"{self.run_id}|{self.lead_h:03d}|{self.variable}|{self.level_str}"