"""HTTP response models for the Module-4 geospatial API.

Read-only envelopes: every model is intentionally permissive (``extra``
ignored) so schema evolutions in the parquet store never break the API.
Times are returned as ISO-8601 UTC strings.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _ReadModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


# --------------------------------------------------------------------------- #
# Registry & domain
# --------------------------------------------------------------------------- #
class Station(_ReadModel):
    station_id: str
    name: str
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)
    district: str | None = None
    state: str | None = None
    is_manual: bool = False
    source: str | None = None


class DomainOut(_ReadModel):
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    dx_km: float
    dy_km: float
    crs: str
    timezone: str
    nx: int
    ny: int
    res_lon_km: float
    res_lat_km: float


class GridCell(_ReadModel):
    i: int
    j: int
    lat: float
    lon: float


# --------------------------------------------------------------------------- #
# Observations (CPCB)
# --------------------------------------------------------------------------- #
class Observation(_ReadModel):
    source: str | None = None
    observed_at_utc: str
    ingest_ts: str | None = None
    quality_flag: str = "OK"
    station_id: str
    station_name: str
    lat: float
    lon: float
    metric: str
    value: float
    unit: str
    sensor_type: str | None = None


class ObservationPoint(_ReadModel):
    t: str
    value: float


class ObservationsLatest(_ReadModel):
    generated_at_utc: str
    count: int
    partitions: list[str]
    rows: list[Observation]


class ObservationsSeries(_ReadModel):
    station_id: str
    metric: str
    unit: str | None = None
    count: int
    points: list[ObservationPoint]


# --------------------------------------------------------------------------- #
# Fires (FIRMS)
# --------------------------------------------------------------------------- #
class FireDetection(_ReadModel):
    fire_id: str
    satellite: str
    lat: float
    lon: float
    acq_datetime: str
    frp_mw: float
    brightness_kelvin: float | None = None
    confidence_percent: float | None = None
    day_night: str | None = None
    grid_i: int | None = None
    grid_j: int | None = None
    assigned_grid_id: str | None = None


# --------------------------------------------------------------------------- #
# GFS manifests
# --------------------------------------------------------------------------- #
class GfsAsset(_ReadModel):
    run_id: str
    init_utc: str
    lead_h: int
    valid_utc: str
    variable: str
    level_str: str
    level_kind: str
    grid: str
    asset_uri: str
    missing_fraction: float = 0.0


# --------------------------------------------------------------------------- #
# Forecasts (Module-3 artifacts)
# --------------------------------------------------------------------------- #
class ForecastMetric(_ReadModel):
    label: str
    mbe: float
    mae: float
    rmse: float
    spe: float | None = None


class ForecastRunSummary(_ReadModel):
    run_id: str
    created_utc: str | None = None
    stations: int | None = None
    raw: ForecastMetric | None = None
    engine: ForecastMetric | None = None
    interval_halfwidth: float | None = None
    interval_coverage_90: float | None = None
    rmse_improvement_pct: float | None = None
    mbe_abs_after: float | None = None


class ForecastPoint(_ReadModel):
    time: str
    lat: float
    lon: float
    station_id: str
    obs_pm25: float | None = None
    pm25_raw: float | None = None
    engine_pm25: float
    engine_lo: float
    engine_hi: float


class ForecastSeries(_ReadModel):
    run_id: str
    station_id: str
    start_utc: str
    count: int
    points: list[ForecastPoint]


class ForecastField(_ReadModel):
    run_id: str
    lead: int
    time: str
    count: int
    cells: list[ForecastPoint]


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
class SourceHealth(_ReadModel):
    completed_partitions: int
    latest: str | None = None


class HealthOut(_ReadModel):
    status: str
    cache: str
    cache_entries: int
    sources: dict[str, SourceHealth]
    forecasts: ForecastRunSummary | None = None