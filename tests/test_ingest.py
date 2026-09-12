"""Unit tests for the Phase-1 ingestion pipeline (offline — no network)."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

CPCB_CSV = (
    "stn_code,sampling_date,state,city,location,agency,type,SO2,NO2,PM2.5,PM10,NH3,CO,OZONE,AQI,"
    "DATE_TIME\n"
    "VI01,12-09-2026,Delhi,Delhi,Anand Vihar,IMD,Continuous,8.0,52.0,182.4,301.0,18.0,1.23,41.2,228,\n"
    "VI02,12-09-2026,Delhi,Delhi,Ashok Vihar,CPCB,Continuous,7.0,48.0,,280.0,15.0,1.10,38.0,210,\n"
    "XX01,12-09-2026,Haryana,Haryana,Unknown Town,CPCB,Continuous,9.0,30.0,90.0,150.0,12.0,0.90,30.0,90,\n"
)

FIRMS_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,"
    "version,bright_ti5,frp,daynight\n"
    "28.7000,77.0500,340.2,0.6,0.5,2026-09-12,1230,NPP,VIIRS,78,2.0,330.0,45.2,D\n"
    "28.7144,77.0600,331.0,0.4,0.4,2026-09-12,1235,NPP,VIIRS,50,2.0,322.0,12.1,N\n"
    "28.9000,77.5000,342.0,0.8,0.7,2026-09-12,1240,Aqua,MODIS,85,6.1,334.0,200.0,D\n"
)


def _stations():
    from aqf_delhi.config import load_stations

    return load_stations()


def test_domain_grid():
    from aqf_delhi.config import load_domain
    from aqf_delhi.domain import build_grid

    dom = load_domain()
    grid = build_grid(dom)
    assert grid.contains(28.6, 77.2)
    assert not grid.contains(28.1, 77.2)
    assert 15 <= grid.nx <= 25 and 18 <= grid.ny <= 26
    i, j = grid.index_of(28.6461, 77.3161)
    assert 0 <= i < grid.nx and 0 <= j < grid.ny


def test_firms_csv_parse():
    from aqf_delhi.config import load_ingest, load_ops
    from aqf_delhi.sources.firms import FIRMSConnector

    conn = FIRMSConnector(load_ingest().firms, load_ops())
    dets = conn.parse_csv(FIRMS_CSV)
    assert len(dets) == 3
    assert dets[0].satellite == "VIIRS_SNPP"
    assert dets[0].frp_mw == 45.2
    assert dets[2].satellite == "MODIS_AQUA"
    assert dets[1].day_night == "N"
    assert dets[0].observed_at_utc.tzinfo is not None


def test_cpcb_csv_parse(tmp_path):
    from aqf_delhi.config import load_ingest, load_ops
    from aqf_delhi.sources.cpcb import CPCBConnector

    conn = CPCBConnector(load_ingest().cpcb, load_ops())
    from aqf_delhi.sources.base import FetchResult

    result = FetchResult(200, "https://mirror", 1, CPCB_CSV.encode(), {}, "utf-8")
    rows = conn.parse_csv_digest(result, _stations())
    names = {r.station_name for r in rows}
    assert "Anand Vihar" in names
    assert "Unknown Town" not in names  # not in registry → dropped
    pm25 = [r for r in rows if r.metric == "PM2.5" and r.station_name == "Anand Vihar"]
    assert pm25 and pm25[0].value == 182.4


def test_validate_firms():
    from aqf_delhi.config import load_ingest, load_ops
    from aqf_delhi.pipeline.validate import validate_firms
    from aqf_delhi.sources.firms import FIRMSConnector

    conn = FIRMSConnector(load_ingest().firms, load_ops())
    cfg = load_ingest()
    dets = conn.parse_csv(FIRMS_CSV)
    kept, report = validate_firms(dets, cfg.quality)
    assert report.kept == 3  # all within range/confidence
    # low confidence dropped
    dets[1].confidence_percent = 5.0
    kept, report = validate_firms(dets, cfg.quality)
    assert report.kept == 2 and report.reasons.get("low_confidence", 0) == 1


def test_gfs_manifest():
    from aqf_delhi.config import load_ingest, load_ops
    from aqf_delhi.sources.gfs import GFSConnector

    cfg = load_ingest()
    cfg.sources["gfs"]["fhr_start"] = 0
    cfg.sources["gfs"]["fhr_end"] = 6
    cfg.sources["gfs"]["fhr_step"] = 3
    conn = GFSConnector(cfg.gfs, load_ops())
    init = datetime(2026, 9, 12, 0, tzinfo=timezone.utc)
    availability = [
        {"fhr": 0, "available": True, "endpoint": "https://s3", "size_bytes": 10},
        {"fhr": 3, "available": False, "endpoint": None, "size_bytes": 0},
        {"fhr": 6, "available": True, "endpoint": "https://s3", "size_bytes": 10},
    ]
    assets = conn.build_manifest(init, availability)
    assert len(assets) == len(cfg.gfs.variables) * 3
    assert all(a.source == "gfs" for a in assets)
    miss = [a for a in assets if a.lead_h == 3]
    assert miss and all(a.quality_flag.value == "MISSING" for a in miss)
    ok = [a for a in assets if a.lead_h == 0]
    assert ok and ok[0].asset_uri.startswith("gfs/")
    assert ok[0].h_key == ok[0].h_key


def test_store_roundtrip_idempotent():
    from aqf_delhi.config import load_ingest, load_ops
    from aqf_delhi.sources.firms import FIRMSConnector
    from aqf_delhi.storage.local import ParquetStore

    conn = FIRMSConnector(load_ingest().firms, load_ops())
    dets = conn.parse_csv(FIRMS_CSV)
    store = ParquetStore("data/test-store")
    part = store.write(dets, source="firms", date="2026-09-12", run_id="t1")
    assert store.is_complete("firms", "2026-09-12", "t1")
    df = store.read("firms", "2026-09-12", "t1")
    assert len(df) == 3
    # idempotent rewrite
    store.write(dets, source="firms", date="2026-09-12", run_id="t1")
    assert len(store.read("firms", "2026-09-12", "t1")) == 3
    import shutil

    shutil.rmtree("data/test-store", ignore_errors=True)  # cleanup


def test_orchestrator_cpcb_fails_without_network(monkeypatch):
    from aqf_delhi.pipeline.orchestrate import Orchestrator
    from aqf_delhi.sources.base import FetchError, requests

    def _boom(*args, **kwargs):
        raise requests.ConnectionError("forced offline failure")

    monkeypatch.setattr("requests.request", _boom)
    orch = Orchestrator()
    orch.ops.retries = 0  # fail fast: no backoff sleep across 38 stations
    orch.cfg.sources["cpcb"]["endpoints"] = ["https://must-not-be-hit.invalid"]
    with pytest.raises(FetchError):
        orch.run("cpcb", asof=datetime(2026, 9, 12, 3, 0, tzinfo=timezone.utc))