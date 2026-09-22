"""Shared fixtures for VAYU-SETU tests (tests_v3)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


def make_cfg_lake(tile: Path | None = None):
    """Build a VayuConfig that resolves paths under an isolated temp root.

    `VayuConfig.resolve()` honours AQF_DATA_ROOT per call, so a fresh tile
    per test keeps all lake writes outside the real repo data/ dir.
    """
    from vayu_setu.config import VayuConfig, load_vayu_config
    if tile is None:
        tile = Path(tempfile.mkdtemp(prefix="vayu_test_"))
    os.environ["AQF_DATA_ROOT"] = str(tile)
    cfg = load_vayu_config()
    return cfg, tile


@pytest.fixture
def cfg_lake_fixture(tmp_path):
    cfg, tile = make_cfg_lake(tmp_path)
    yield cfg, tile