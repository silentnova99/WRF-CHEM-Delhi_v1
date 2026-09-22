"""Pytest configuration: put the repo `src/` on sys.path (mirrors tests_v2)."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_spec = importlib.util.spec_from_file_location(
    "v3helpers", Path(__file__).resolve().parent / "_v3_helpers.py"
)
_v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3)
make_cfg_lake = _v3.make_cfg_lake


@pytest.fixture
def cfg_lake_fixture(tmp_path):
    cfg, tile = make_cfg_lake(tmp_path)
    yield cfg, tile