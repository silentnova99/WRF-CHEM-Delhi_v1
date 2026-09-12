"""Module-4 API tests (offline, no live services, no network)."""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def _stations(subset: list[int] | None = None):
    from aqf_delhi.config import load_stations

    st = load_stations()
    return [st[i] for i in (subset if subset else [0, 1, 2])]


def _build_ingest(data_root: Path) -> None:
    from aqf_delhi.schemas.common import CanonicalRecord
    from aqf_delhi.schemas.cpcb import CpcbObservation
    from aqf_delhi.schemas.firms import FireDetection
    from aqf_delhi.schemas.gfs import GfsFieldAsset
    from aqf_delhi.storage.local import ParquetStore

    store = ParquetStore(data_root / "ingest")
    stations = _stations()
    t0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)

    obs: list[CanonicalRecord] = []
    for k in range(3):
        ts = t0 + timedelta(hours=k)
        for j, st in enumerate(stations):
            for metric, unit in (("PM2.5", "ug/m3"), ("PM10", "ug/m3")):
                obs.append(
                    CpcbObservation(
                        source="cpcb",
                        observed_at_utc=ts,
                        ingest_ts=ts,
                        station_id=st["station_id"],
                        station_name=st["name"],
                        lat=float(st["lat"]),
                        lon=float(st["lon"]),
                        metric=metric,
                        value=100.0 + 10 * j + 5 * k,
                        unit=unit,
                        raw_payload={"test": True},
                    )
                )
    store.write(obs, source="cpcb", date="2026-09-01", run_id="cpcb.2026-09-01.120000")

    fires = [
        FireDetection(
            source="firms",
            observed_at_utc=t0,
            ingest_ts=t0,
            fire_id="f0001",
            satellite="VIIRS_SNPP",
            lat=28.8, lon=77.0,
            acq_datetime=t0,
            frp_mw=120.0,
            brightness_kelvin=330.0,
            confidence_percent=95.0,
            scan_km=0.75,
            track_km=0.75,
        ),
        FireDetection(
            source="firms",
            observed_at_utc=t0,
            ingest_ts=t0,
            fire_id="f0002",
            satellite="MODIS_AQUA",
            lat=28.5, lon=77.4,
            acq_datetime=t0,
            frp_mw=45.0,
            brightness_kelvin=310.0,
            confidence_percent=70.0,
            scan_km=0.75,
            track_km=0.75,
        ),
        FireDetection(
            source="firms",
            observed_at_utc=t0,
            ingest_ts=t0,
            fire_id="f0003",
            satellite="VIIRS_SNPP",
            lat=31.9, lon=76.3,
            acq_datetime=t0,
            frp_mw=10.0,
            brightness_kelvin=305.0,
            confidence_percent=40.0,
            scan_km=0.75,
            track_km=0.75,
        ),
    ]
    store.write(fires, source="firms", date="2026-09-01", run_id="firms.2026-09-01.120000")

    assets = []
    for fhr in (0, 6):
        for var in ("TMP", "RH"):
            assets.append(
                GfsFieldAsset(
                    source="gfs",
                    observed_at_utc=t0,
                    ingest_ts=t0,
                    run_id="gfs.2026-09-01.00z",
                    init_utc=t0,
                    lead_h=fhr,
                    valid_utc=t0 + timedelta(hours=fhr),
                    variable=var,
                    level_str="2 m above ground",
                    level_kind="height_agl",
                    asset_uri=f"s3://grids/gfs/{fhr:03d}/{var}.nc",
                    min_val=0.0, max_val=100.0, mean_val=40.0,
                )
            )
    store.write(assets, source="gfs", date="2026-09-01", run_id="gfs.2026-09-01.00z")


def _build_forecast(data_root: Path) -> str:
    import numpy as np

    from aqf_delhi.ml.publish import write_forecast

    stations = _stations()
    n_lead, n_stn = 72, len(stations)
    rng = np.random.default_rng(3)
    base = 100 + 60 * np.sin(np.linspace(0, 8, n_lead))
    obs = base[:, None] + 10.0 * np.arange(n_stn)[None, :] + rng.normal(0, 4, (n_lead, n_stn))
    engine = obs * 0.92 + 6.0
    lo = engine - 25.0
    hi = engine + 25.0
    raw = obs * 0.7 + 12.0
    times = np.arange(n_lead) * np.timedelta64(1, "h") + np.datetime64("2026-10-01T00:00")
    run_id = "fc.20261001.000000"
    write_forecast(
        data_root / "forecasts", run_id,
        times=times,
        station_ids=[s["station_id"] for s in stations],
        lat=np.array([float(s["lat"]) for s in stations]),
        lon=np.array([float(s["lon"]) for s in stations]),
        obs_pm25=obs, pm25_raw=raw, engine_pm25=engine, engine_lo=lo, engine_hi=hi,
        report={
            "stations": n_stn,
            "quick": True,
            "raw": {"label": "raw-model", "mbe": 30.0, "mae": 30.0, "rmse": 40.0, "spe": None},
            "engine": {"label": "engine", "mbe": 5.0, "mae": 12.0, "rmse": 16.0, "spe": None},
            "interval_halfwidth": 25.0,
            "interval_coverage_90": 0.93,
            "rmse_improvement_pct": 60.0,
            "mbe_abs_after": 5.0,
        },
    )
    return run_id


def test_health_and_registry_endpoints(tmp_path):
    from fastapi.testclient import TestClient

    from aqf_delhi.api.app import create_app
    from aqf_delhi.config import load_domain

    _build_ingest(tmp_path)
    run_id = _build_forecast(tmp_path)
    client = TestClient(create_app(root=tmp_path, cache_ttl=30))

    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["cache"] == "memory"
    assert body["sources"]["cpcb"]["completed_partitions"] == 1
    assert body["sources"]["firms"]["completed_partitions"] == 1
    assert body["forecasts"]["run_id"] == run_id

    dom = load_domain()
    r = client.get("/domain")
    assert r.status_code == 200
    d = r.json()
    assert d["name"] == dom.name
    assert d["nx"] > 1 and d["ny"] > 1
    assert d["lat_min"] == dom.lat_min

    r = client.get("/domain/grid")
    cells = r.json()
    assert len(cells) == d["nx"] * d["ny"]
    assert cells[0]["i"] == 0 and cells[0]["j"] == 0

    r = client.get("/stations")
    stations = r.json()
    assert len(stations) >= 3
    assert all(s["station_id"] and s["lat"] > 0 for s in stations)


def test_observation_endpoints(tmp_path):
    from fastapi.testclient import TestClient

    from aqf_delhi.api.app import create_app

    _build_ingest(tmp_path)
    client = TestClient(create_app(root=tmp_path))

    r = client.get("/observations/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 6                       # 3 stations × 2 metrics
    assert body["partitions"] == ["2026-09-01/cpcb.2026-09-01.120000"]

    r = client.get("/observations/latest", params={"metric": "PM2.5"})
    assert r.json()["count"] == 3
    r = client.get("/observations/latest", params={"station_id": "DL-001"})
    assert r.json()["count"] == 2

    r = client.get(
        "/observations/timeseries",
        params={"station_id": "DL-001", "metric": "PM2.5"},
    )
    assert r.status_code == 200
    ts = r.json()
    assert ts["count"] == 3
    assert ts["unit"] == "ug/m3"
    vals = [p["value"] for p in ts["points"]]
    assert sorted(vals) == vals                       # ascending time ordering

    r = client.get(
        "/observations/timeseries",
        params={"station_id": "UNKNOWN", "metric": "PM2.5"},
    )
    assert r.status_code == 404


def test_fires_and_gfs_endpoints(tmp_path):
    from fastapi.testclient import TestClient

    from aqf_delhi.api.app import create_app

    _build_ingest(tmp_path)
    client = TestClient(create_app(root=tmp_path))

    r = client.get("/fires/recent")
    assert r.status_code == 200
    assert len(r.json()) == 3

    r = client.get(
        "/fires/recent",
        params={"min_lon": 76.8, "min_lat": 28.2, "max_lon": 77.6, "max_lat": 29.0},
    )
    assert len(r.json()) == 2                        # outside-bbox fire filtered

    r = client.get("/gfs/runs")
    assert "gfs.2026-09-01.00z" in r.json()

    r = client.get("/gfs/assets", params={"run_id": "gfs.2026-09-01.00z"})
    assert len(r.json()) == 4
    r = client.get("/gfs/assets", params={"variable": "TMP"})
    assert len(r.json()) == 2
    a = r.json()[0]
    assert a["valid_utc"] and a["lead_h"] in (0, 6)


def test_forecast_endpoints(tmp_path):
    from fastapi.testclient import TestClient

    from aqf_delhi.api.app import create_app

    _build_ingest(tmp_path)
    run_id = _build_forecast(tmp_path)
    client = TestClient(create_app(root=tmp_path))

    r = client.get("/forecasts/runs")
    assert r.json() == [run_id]

    r = client.get("/forecasts/latest")
    assert r.status_code == 200
    s = r.json()
    assert s["run_id"] == run_id
    assert s["stations"] == 3
    assert s["engine"]["rmse"] < s["raw"]["rmse"]
    assert s["rmse_improvement_pct"] > 0

    r = client.get(f"/forecasts/{run_id}/series")
    assert r.status_code == 200
    assert r.json()["count"] == 3 * 72
    assert r.json()["start_utc"].startswith("2026-10-01")

    r = client.get(f"/forecasts/{run_id}/series", params={"station_id": "DL-001"})
    assert r.json()["count"] == 72

    r = client.get(f"/forecasts/{run_id}/map", params={"lead": 12})
    assert r.status_code == 200
    field = r.json()
    assert field["lead"] == 12
    assert field["count"] == 3
    assert len(field["cells"]) == 3

    r = client.get(f"/forecasts/{run_id}/map", params={"lead": 999})
    assert r.status_code == 404
    r = client.get("/forecasts/unknown/series")
    assert r.status_code == 404


def test_cache_and_ttl(tmp_path):
    from aqf_delhi.api.app import create_app
    from aqf_delhi.api.cache import MemoryCache
    from aqf_delhi.api.store import ApiStore

    _build_ingest(tmp_path)
    cache = MemoryCache(ttl=30)
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    cache.set("k2", {"b": 2}, ttl=1)
    assert cache.get("k2") == {"b": 2}
    assert cache.entries() >= 2

    store = ApiStore(tmp_path)
    from fastapi.testclient import TestClient

    client = TestClient(create_app(root=tmp_path, store=store, cache=cache))
    client.get("/observations/latest")
    reads_after_first = store.reads
    assert reads_after_first > 0                    # store hit the parquet
    client.get("/observations/latest")
    assert store.reads == reads_after_first         # second hit served from cache

    store.observations_latest()                     # uncached direct call re-reads
    assert store.reads == reads_after_first + 1