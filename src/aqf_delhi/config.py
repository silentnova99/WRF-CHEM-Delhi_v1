"""Configuration loading & validation.

Reads ``configs/domain.yaml`` and ``configs/ingest.yaml`` into typed
pydantic models. Raises clear errors on the import path so every pipeline
component shares one validated view of the world.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


# --------------------------------------------------------------------------- #
# Domain
# --------------------------------------------------------------------------- #
class DomainConfig(BaseModel):
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    dx_km: float
    dy_km: float
    crs: str = "EPSG:4326"
    timezone: str = "Asia/Kolkata"

    @property
    def lat_span(self) -> float:
        return self.lat_max - self.lat_min

    @property
    def lon_span(self) -> float:
        return self.lon_max - self.lon_min

    def validate_bounds(self) -> None:
        if not (self.lat_min < self.lat_max and self.lon_min < self.lon_max):
            raise ValueError("domain bounds are inverted (min >= max)")


class OpsConfig(BaseModel):
    log_level: str = "INFO"
    retries: int = Field(default=3, ge=0)
    retry_backoff_seconds: float = Field(default=2.0, ge=0.0)
    request_timeout_seconds: float = Field(default=60.0, ge=1.0)


class RootDomainConfig(BaseModel):
    domain: DomainConfig
    ops: OpsConfig = OpsConfig()


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
class StorageConfig(BaseModel):
    base_dir: str = "data/ingest"
    partition_layout: str = "{source}/{date}/{run_id}"
    parquet_compression: str = "zstd"


class GFSEndpointConfig(BaseModel):
    enabled: bool = True
    model: str = "gfs"
    product: str = "pgrb2.0p25"
    fhr_start: int = 0
    fhr_end: int = 72
    fhr_step: int = 3
    cycles: list[int] = [0, 6, 12, 18]
    variables: list[str] = []
    endpoints: list[str]
    file_pattern: str = "gfs.t{cyc:02d}z.pgrb2.0p25.f{fhr:03d}"

    @property
    def forecast_hours(self) -> list[int]:
        return list(range(self.fhr_start, self.fhr_end + 1, self.fhr_step))


class DataGovinConfig(BaseModel):
    api_key_env: str = "DATA_GOVIN_API_KEY"
    resource_id: str = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
    base_url: str = "https://api.data.gov.in/resource"
    page_size: int = Field(default=10000, ge=1, le=10000)
    max_pages: int = Field(default=200, ge=1)
    page_delay_seconds: float = Field(default=0.6, ge=0.0)
    filters: dict[str, str | list[str]] = Field(default_factory=dict)


class CPcbConfig(BaseModel):
    enabled: bool = True
    poll_interval_minutes: int = 15
    endpoints: list[str]
    fallback_ttl_hours: int = 6
    min_station_coverage: float = Field(default=0.6, ge=0.0, le=1.0)
    data_govin: DataGovinConfig = DataGovinConfig()


class FirmsCollection(BaseModel):
    viirs_nrt: str = "VIIRS_SNPP_NRT"
    viirs_noaa20_nrt: str = "VIIRS_NOAA20_NRT"
    modis_nrt: str = "MODIS_NRT"


class FirmsConfig(BaseModel):
    enabled: bool = True
    api_base: str = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    map_key_env: str = "FIRMS_MAP_KEY"
    collections: FirmsCollection = FirmsCollection()
    default_collection: str = "VIIRS_SNPP_NRT"
    bbox: str = "76.8,28.2,77.6,29.0"
    day_window: int = 1
    poll_interval_minutes: int = 30


class GFSQualityConfig(BaseModel):
    max_missing_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    lat_band: list[float] = [20.0, 32.0]
    log_band: list[float] = [60.0, 100.0]

    @field_validator("lat_band", "log_band")
    @classmethod
    def _pair(cls, v: list[float]) -> list[float]:
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError(f"band must be [min, max], got {v}")
        return v


class CpcbQualityConfig(BaseModel):
    max_value_ug_m3: float = Field(default=2000.0, gt=0.0)
    explicit_lower_bound: float = 0.0
    mad_cutoff_sigma: float = Field(default=5.0, gt=0.0)


class FirmsQualityConfig(BaseModel):
    frp_range_mw: list[float] = [0.0, 5000.0]
    confidence_min: float = Field(default=20.0, ge=0.0, le=100.0)

    @field_validator("frp_range_mw")
    @classmethod
    def _frp_pair(cls, v: list[float]) -> list[float]:
        if len(v) != 2 or v[0] > v[1]:
            raise ValueError(f"frp_range_mw must be [min, max], got {v}")
        return v


class QualityConfig(BaseModel):
    gfs: GFSQualityConfig = GFSQualityConfig()
    cpcb: CpcbQualityConfig = CpcbQualityConfig()
    firms: FirmsQualityConfig = FirmsQualityConfig()


class GrapConfig(BaseModel):
    stage_thresholds: dict[int, int] = {1: 201, 2: 301, 3: 401, 4: 451}
    min_stations_pct: float = Field(default=60.0, ge=0.0, le=100.0)
    persistence_hours: int = 3

    @model_validator(mode="after")
    def _sorted_stages(self) -> "GrapConfig":
        if sorted(self.stage_thresholds) != list(self.stage_thresholds):
            raise ValueError("GRAP stage_thresholds must be keyed in stage order")
        return self


class RootIngestConfig(BaseModel):
    storage: StorageConfig = StorageConfig()
    sources: dict[str, Any] = {}
    quality: QualityConfig = QualityConfig()
    grap: GrapConfig = GrapConfig()

    @property
    def gfs(self) -> GFSEndpointConfig:
        return GFSEndpointConfig(**self.sources.get("gfs", {}))

    @property
    def cpcb(self) -> CPcbConfig:
        return CPcbConfig(**self.sources.get("cpcb", {}))

    @property
    def firms(self) -> FirmsConfig:
        return FirmsConfig(**self.sources.get("firms", {}))


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping: {path}")
    return data


def load_domain(config_dir: Path = DEFAULT_CONFIG_DIR) -> DomainConfig:
    root = RootDomainConfig(**_load_yaml(config_dir / "domain.yaml"))
    root.domain.validate_bounds()
    return root.domain


def load_ops(config_dir: Path = DEFAULT_CONFIG_DIR) -> OpsConfig:
    root = RootDomainConfig(**_load_yaml(config_dir / "domain.yaml"))
    return root.ops


def load_ingest(config_dir: Path = DEFAULT_CONFIG_DIR) -> RootIngestConfig:
    root = RootIngestConfig(**_load_yaml(config_dir / "ingest.yaml"))
    # Force validation of nested source configs immediately (fail-fast on boot).
    _ = (root.gfs, root.cpcb, root.firms)
    return root


def load_stations(config_dir: Path = DEFAULT_CONFIG_DIR) -> list[dict[str, Any]]:
    """Load the CPCB station registry CSV as list-of-dicts."""
    import csv

    path = config_dir / "stations.csv"
    if not path.is_file():
        raise FileNotFoundError(f"station registry not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))