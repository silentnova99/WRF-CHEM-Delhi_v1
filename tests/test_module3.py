"""Module-3 unit tests (offline, deterministic, no network)."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def _stations(subset: list[int] | None = None):
    from aqf_delhi.config import load_stations

    st = load_stations()
    return [st[i] for i in (subset if subset else [0, 2, 11, 19, 28])]


def test_haversine_sanity():
    from aqf_delhi.ml.graph import haversine_km

    d = haversine_km(28.6461, 77.3161, 28.6934, 77.1740)  # Anand Vihar → Ashok Vihar
    assert 10 < d < 20
    assert haversine_km(0, 0, 0, 0) == 0.0


def test_graph_build():
    from aqf_delhi.ml.graph import StationGraphBuilder

    g = StationGraphBuilder(k=4, wind_bearing_deg=300, wind_season_months=(10, 11)).build(
        _stations(), month=10
    )
    a = g.adjacency()
    n = len(g)
    assert a.shape == (n, n)
    assert (a.T == a).all()                       # symmetric
    # every node has at least k-nearest neighbors
    assert (a.sum(axis=1) > 0).all()
    # normalized adjacency rows exist
    an = g.adjacency(normalized=True)
    assert an.shape == (n, n)


def test_wind_edges_only_in_season():
    from aqf_delhi.ml.graph import StationGraphBuilder

    winter = StationGraphBuilder(k=4, use_wind_edges=True).build(_stations(), month=10)
    summer = StationGraphBuilder(k=4, use_wind_edges=True).build(_stations(), month=6)
    assert winter.adjacency().sum() > summer.adjacency().sum()


def test_aqi_known_values():
    from aqf_delhi.ml.aqi import aqi_from_concentrations, grap_stage, sub_index

    assert sub_index("PM2.5", 60.0) == 100
    assert sub_index("PM2.5", 90.0) == 200
    assert sub_index("PM2.5", 120.0) == 300
    score = aqi_from_concentrations({"PM2.5": 182.4, "PM10": 301.0, "O3": 41.2})
    assert 300 <= score.value <= 400             # PM2.5 dominates (severe band)
    assert score.dominant_pollutant == "PM2.5"
    assert grap_stage(250) == 1
    assert grap_stage(320) == 2
    assert grap_stage(420) == 3
    assert grap_stage(460) == 4
    assert grap_stage(120) == 0


def test_feature_spec_shapes():
    import numpy as np

    from aqf_delhi.ml.features import FeatureSpec

    spec = FeatureSpec()
    t, n = 48, 5
    from aqf_delhi.ml.features import RAW_KEYS

    rng = np.random.default_rng(0)
    x = {}
    for key in RAW_KEYS:
        x[key] = rng.normal(size=(t, n))
    arr = spec.from_raw(x, hours=np.arange(t), doys=np.repeat(260, t))
    assert arr.shape == (t, n, spec.n_features)
    assert spec.n_features > len(RAW_KEYS)        # engineered terms added


def test_synthetic_experiment_shapes_and_bias():
    import numpy as np

    from aqf_delhi.ml.datagen import SyntheticDelhiExperiment

    exp = SyntheticDelhiExperiment(_stations(), n_days=3, seed=1).generate()
    raw = exp.raw
    assert raw["pm25_raw"].shape == (72, 5)
    assert exp.target.shape == (72, 5)
    # model bias is real: winter inversion hours → model under-predicts
    truth = exp.raw["pm25_raw"]  # biased model
    obs = exp.target
    assert np.abs(np.mean(truth - obs)) > 2.0    # non-trivial systematic bias


def test_gnn_forward_shape():
    import numpy as np
    import torch

    from aqf_delhi.ml.features import FeatureSpec
    from aqf_delhi.ml.gnn import TemporalGCN
    from aqf_delhi.ml.graph import StationGraphBuilder

    stations = _stations()
    g = StationGraphBuilder(k=3).build(stations, month=10)
    model = TemporalGCN(n_features=8, hidden=8, out_dim=1, adj=g.adjacency(normalized=True))
    x = torch.randn(12, len(g), 8)
    out = model(x)
    assert out.shape == (len(g), 1)


def test_hybrid_loss_reduces():
    import numpy as np
    import torch

    from aqf_delhi.ml.gnn import hybrid_loss

    y = torch.from_numpy(np.random.default_rng(0).normal(size=(8, 5)).astype(np.float32))
    p_bad = torch.zeros_like(y)
    p_good = y * 1.02
    w = torch.ones(5)
    assert hybrid_loss(p_good, y, w).item() < hybrid_loss(p_bad, y, w).item()


def test_full_pipeline_quick():
    """End-to-end: engine must beat the raw biased model on held-out 72h."""
    from aqf_delhi.ml.train import run_pipeline

    report = run_pipeline(seeded=True, torch_seed=3, quick=True)
    assert report["engine"]["rmse"] < report["raw"]["rmse"]
    assert abs(report["engine"]["mbe"]) < abs(report["raw"]["mbe"])
    assert report["rmse_improvement_pct"] > 0
    assert 0.0 <= report["interval_coverage_90"] <= 1.0