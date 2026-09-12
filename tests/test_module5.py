"""Module-5: 3D WebGIS dashboard — offline tests.

Builds a small coupled artifact via the Module-2 writer into a temp store,
creates the API app over it, then asserts the dashboard state aggregation and
static assets are served correctly.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aqf_delhi.api.app import create_app
from aqf_delhi.wrf.coupling import coupled_run


@pytest.fixture(scope="module")
def dash_client(tmp_path_factory):
    root = tmp_path_factory.mktemp("dash")
    out_root = root / "coupled"
    info = coupled_run(
        init=datetime(2026, 1, 5, 0),
        fhrs=[0, 3, 6],
        real=False,
        out_root=out_root,
    )
    run_id = info["run_id"]
    assert (out_root / run_id / "_DONE").is_file(), "coupled artifact not written"

    app = create_app(root=root)
    return TestClient(app), root / "coupled" / run_id


def test_index_served(dash_client):
    client, _ = dash_client
    r = client.get("/dashboard/")
    assert r.status_code == 200
    html = r.text
    assert "maplibre" in html.lower()
    assert "app.js" in html

    bare = client.get("/dashboard")
    assert bare.status_code == 200


def test_assets_served(dash_client):
    client, _ = dash_client
    for path, ctype in (("/dashboard/app.js", "javascript"),
                        ("/dashboard/styles.css", "css")):
        r = client.get(path)
        assert r.status_code == 200, path
        assert ctype in r.headers["content-type"], path
    assert client.get("/dashboard/missing.js").status_code == 404


def test_state_aggregation(dash_client):
    client, run_dir = dash_client
    r = client.get("/dashboard/state")
    assert r.status_code == 200
    body = r.json()

    assert body["domain"]["name"]
    assert body["coupled"]["run_id"] == run_dir.name
    assert len(body["coupled"]["times"]) == 3
    ny, nx = body["coupled"]["ny"], body["coupled"]["nx"]
    assert len(body["coupled"]["hours"]) == 3

    for h in body["coupled"]["hours"]:
        assert len(h["pm25_raw"]) == ny * nx
        assert len(h["pm25_obs_demo"]) == ny * nx
        assert all(isinstance(v, (int, float)) for v in h["pm25_raw"])

    assert len(body["coupled"]["lat_nodes"]) == ny
    assert len(body["coupled"]["lon_nodes"]) == nx
    assert body["coupled"]["times"][0].startswith("2026-01-05")

    for key in ("t2m_c", "ws10_ms", "wd_deg", "pblh_m"):
        assert len(body["coupled"]["met_last"][key]) == ny * nx

    assert isinstance(body["stations"], list)
    assert len(body["stations"]) > 0
    fields = body["stations"][0]
    for k in ("station_id", "lat", "lon", "pm25_raw", "pm25_obs_demo"):
        assert k in fields

    assert isinstance(body["fires"], list)
    assert "forecasts" in body
    assert "server_time" in body


def test_no_duplicate_dashboard_prefix(dash_client):
    client, _ = dash_client
    r = client.get("/dashboard/state")
    assert r.status_code == 200
    assert r.json()["coupled"]["run_id"]  # non-empty coupled block


def test_static_assets_reference_valid_paths(dash_client):
    client, _ = dash_client
    html = client.get("/dashboard/").text
    js = client.get("/dashboard/app.js").text
    css = client.get("/dashboard/styles.css").text
    assert html and js and css
    assert "/dashboard/app.js" in html
    assert "/dashboard/styles.css" in html
    assert "maplibregl" in js
    assert "#map {" in css
    assert "flush" not in js  # modal error text must not exist