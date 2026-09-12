"""HTTP routes for the Module-4 geospatial API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from fastapi import APIRouter, HTTPException, Query, Request

from aqf_delhi.api import schemas as S
from aqf_delhi.api.store import ApiStore

CACHE_DEFAULT_TTL = 30


def _ctx(request: Request) -> tuple[ApiStore, object]:
    return request.app.state.store, request.app.state.cache


def _cached(cache, key: str, fn: Callable, ttl: int = CACHE_DEFAULT_TTL):
    hit = cache.get(key)
    if hit is not None:
        return hit
    value = fn()
    cache.set(key, value, ttl)
    return value


# --------------------------------------------------------------------------- #
# domain + registry
# --------------------------------------------------------------------------- #
domain_router = APIRouter(tags=["domain", "stations"])


@domain_router.get("/domain", response_model=S.DomainOut, summary="Study domain + grid")
def get_domain(request: Request) -> S.DomainOut:
    store, cache = _ctx(request)
    domain, grid = request.app.state.domain, request.app.state.grid
    return S.DomainOut(
        name=domain.name,
        lat_min=domain.lat_min,
        lat_max=domain.lat_max,
        lon_min=domain.lon_min,
        lon_max=domain.lon_max,
        dx_km=domain.dx_km,
        dy_km=domain.dy_km,
        crs=domain.crs,
        timezone=domain.timezone,
        nx=grid.nx,
        ny=grid.ny,
        res_lon_km=grid.lon_res_km(),
        res_lat_km=grid.lat_res_km(),
    )


@domain_router.get("/domain/grid", response_model=list[S.GridCell])
def get_grid(request: Request) -> list[S.GridCell]:
    _, cache = _ctx(request)
    return _cached(
        cache, "grid", lambda: [
            S.GridCell(i=i, j=j, lat=lat, lon=lon)
            for j, lat in enumerate(request.app.state.grid.lat_nodes)
            for i, lon in enumerate(request.app.state.grid.lon_nodes)
        ],
    )


@domain_router.get("/stations", response_model=list[S.Station])
def get_stations(request: Request) -> list[S.Station]:
    _, cache = _ctx(request)
    return _cached(
        cache,
        "stations",
        lambda: [
            S.Station(
                station_id=r["station_id"],
                name=r["name"],
                lat=float(r["lat"]),
                lon=float(r["lon"]),
                district=r.get("district"),
                state=r.get("state"),
                is_manual=str(r.get("is_manual", "")).lower() == "true",
                source=r.get("source"),
            )
            for r in request.app.state.stations
        ],
        3600,
    )


# --------------------------------------------------------------------------- #
# observations (CPCB)
# --------------------------------------------------------------------------- #
obs_router = APIRouter(prefix="/observations", tags=["observations"])


@obs_router.get(
    "/latest", response_model=S.ObservationsLatest, summary="Latest reading per station+metric"
)
def observations_latest(
    request: Request,
    metric: str | None = Query(None),
    station_id: str | None = Query(None),
) -> S.ObservationsLatest:
    store, cache = _ctx(request)
    key = f"obs.latest.{metric or '*'}.{station_id or '*'}"
    rows = _cached(cache, key, lambda: store.observations_latest(metric=metric, station_id=station_id))
    return S.ObservationsLatest(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        count=len(rows),
        partitions=store.partitions_desc("cpcb")[:10],
        rows=[S.Observation(**r) for r in rows],
    )


@obs_router.get("/timeseries", response_model=S.ObservationsSeries)
def observations_timeseries(
    request: Request,
    station_id: str = Query(..., min_length=1),
    metric: str = Query("PM2.5"),
    from_utc: str | None = Query(None, alias="from"),
    to_utc: str | None = Query(None, alias="to"),
    limit: int = Query(1000, ge=1, le=10000),
) -> S.ObservationsSeries:
    store, cache = _ctx(request)
    key = f"obs.ts.{station_id}.{metric}.{from_utc}.{to_utc}.{limit}"
    rows = _cached(
        cache,
        key,
        lambda: store.observations_timeseries(
            station_id=station_id, metric=metric, from_utc=from_utc, to_utc=to_utc, limit=limit
        ),
        ttl=15,
    )
    if not rows:
        raise HTTPException(status_code=404, detail=f"no observations for {station_id}/{metric}")
    unit = rows[0].get("unit")
    return S.ObservationsSeries(
        station_id=station_id,
        metric=metric,
        unit=unit,
        count=len(rows),
        points=[S.ObservationPoint(t=r["observed_at_utc"], value=r["value"]) for r in rows],
    )


# --------------------------------------------------------------------------- #
# fires (FIRMS)
# --------------------------------------------------------------------------- #
fires_router = APIRouter(prefix="/fires", tags=["fires"])


@fires_router.get("/recent", response_model=list[S.FireDetection])
def fires_recent(
    request: Request,
    limit: int = Query(500, ge=1, le=5000),
    min_lon: float | None = Query(None),
    min_lat: float | None = Query(None),
    max_lon: float | None = Query(None),
    max_lat: float | None = Query(None),
) -> list[S.FireDetection]:
    store, cache = _ctx(request)
    key = f"fires.{limit}.{min_lon}.{min_lat}.{max_lon}.{max_lat}"
    rows = _cached(
        cache,
        key,
        lambda: store.fires_recent(
            limit=limit, min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat
        ),
        ttl=60,
    )
    return [S.FireDetection(**r) for r in rows]


# --------------------------------------------------------------------------- #
# GFS manifests
# --------------------------------------------------------------------------- #
gfs_router = APIRouter(prefix="/gfs", tags=["gfs"])


@gfs_router.get("/runs", response_model=list[str])
def gfs_runs(request: Request) -> list[str]:
    store, cache = _ctx(request)
    return _cached(cache, "gfs.runs", store.gfs_runs, ttl=120)


@gfs_router.get("/assets", response_model=list[S.GfsAsset])
def gfs_assets(
    request: Request,
    run_id: str | None = Query(None),
    variable: str | None = Query(None),
    limit: int = Query(1000, ge=1, le=10000),
) -> list[S.GfsAsset]:
    store, cache = _ctx(request)
    key = f"gfs.assets.{run_id or '*'}.{variable or '*'}.{limit}"
    rows = _cached(
        cache, key, lambda: store.gfs_assets(run_id=run_id, variable=variable, limit=limit)
    )
    return [S.GfsAsset(**r) for r in rows]


# --------------------------------------------------------------------------- #
# forecasts (Module-3 artifacts)
# --------------------------------------------------------------------------- #
fc_router = APIRouter(prefix="/forecasts", tags=["forecasts"])


@fc_router.get("/runs", response_model=list[str])
def forecasts_runs(request: Request) -> list[str]:
    store, cache = _ctx(request)
    return _cached(cache, "fc.runs", store.forecast_runs, ttl=15)


@fc_router.get("/latest", response_model=S.ForecastRunSummary)
def forecasts_latest(request: Request) -> S.ForecastRunSummary:
    store, cache = _ctx(request)
    run_id = store.forecast_latest()
    if run_id is None:
        raise HTTPException(status_code=404, detail="no forecast runs published yet")
    report = store.forecast_report(run_id) or {}
    body = report.get("report", {})
    summary = S.ForecastRunSummary(
        run_id=run_id,
        created_utc=report.get("created_utc"),
        stations=body.get("stations"),
        raw=S.ForecastMetric(**body["raw"]) if body.get("raw") else None,
        engine=S.ForecastMetric(**body["engine"]) if body.get("engine") else None,
        interval_halfwidth=body.get("interval_halfwidth"),
        interval_coverage_90=body.get("interval_coverage_90"),
        rmse_improvement_pct=body.get("rmse_improvement_pct"),
        mbe_abs_after=body.get("mbe_abs_after"),
    )
    cache.set("fc.latest", summary.model_dump(mode="json"), 15)
    return summary


@fc_router.get("/{run_id}/series", response_model=S.ForecastSeries)
def forecast_series(
    request: Request,
    run_id: str,
    station_id: str | None = Query(None),
    limit: int = Query(10000, ge=1, le=200000),
) -> S.ForecastSeries:
    store, cache = _ctx(request)
    try:
        rows = store.forecast_series(run_id, station_id=station_id, limit=limit)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"forecast run not found: {run_id}")
    if not rows:
        raise HTTPException(status_code=404, detail=f"no series for {run_id}/{station_id or '*'}")
    return S.ForecastSeries(
        run_id=run_id,
        station_id=rows[0]["station_id"],
        start_utc=rows[0]["time"],
        count=len(rows),
        points=[S.ForecastPoint(**r) for r in rows],
    )


@fc_router.get("/{run_id}/map", response_model=S.ForecastField)
def forecast_field(
    request: Request, run_id: str, lead: int = Query(0, ge=0)
) -> S.ForecastField:
    store, cache = _ctx(request)
    try:
        rows = store.forecast_field(run_id, lead)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"forecast run not found: {run_id}")
    except IndexError:
        raise HTTPException(status_code=404, detail=f"lead {lead} out of range")
    if not rows:
        raise HTTPException(status_code=404, detail=f"no cells at lead {lead}")
    return S.ForecastField(
        run_id=run_id, lead=lead, time=rows[0]["time"], count=len(rows),
        cells=[S.ForecastPoint(**r) for r in rows],
    )


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #
health_router = APIRouter(tags=["health"])


@health_router.get("/health", response_model=S.HealthOut)
def health(request: Request) -> S.HealthOut:
    store, cache = _ctx(request)
    out = {}
    for source in ("cpcb", "firms", "gfs"):
        parts = store._partitions(source)
        out[source] = S.SourceHealth(
            completed_partitions=len(parts), latest=f"{parts[-1][0]}/{parts[-1][1]}" if parts else None
        )
    latest_run = store.forecast_latest()
    summary = None
    if latest_run:
        report = store.forecast_report(latest_run) or {}
        body = report.get("report", {})
        summary = S.ForecastRunSummary(
            run_id=latest_run,
            stations=body.get("stations"),
            engine=S.ForecastMetric(**body["engine"]) if body.get("engine") else None,
            rmse_improvement_pct=body.get("rmse_improvement_pct"),
        )
    return S.HealthOut(
        status="ok",
        cache=cache.name(),
        cache_entries=cache.entries(),
        sources=out,
        forecasts=summary,
    )