"""Configuration loader for VAYU-SETU.

Reads `configs/vayu_setu.yaml` plus repository defaults (configs/domain.yaml,
configs/ingest.yaml) and exposes a flat, typed runtime namespace. All paths
here are project-relative to the repository root.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "vayu_setu.yaml"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "domain.yaml"
INGEST_CONFIG_PATH = REPO_ROOT / "configs" / "ingest.yaml"


class VayuConfig:
    """Typed-ish accessor over the merged VAYU-SETU configuration."""

    def __init__(self, data: Dict[str, Any]) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, item: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if item in data:
            return data[item]
        raise AttributeError(f"VayuConfig has no section '{item}'")

    @property
    def raw(self) -> Dict[str, Any]:
        return object.__getattribute__(self, "_data")

    def resolve(self, *parts: str) -> Path:
        """Resolve a path relative to the repository root (also via env override)."""
        path = Path(*parts)
        if not path.is_absolute():
            root = os.environ.get("AQF_DATA_ROOT") or str(REPO_ROOT)
            path = Path(root) / path
        else:
            root = os.environ.get("AQF_DATA_ROOT") or str(REPO_ROOT)
            path = Path(root) / path
        return path

    def section(self, name: str) -> Dict[str, Any]:
        return dict(self._data.get(name, {}))


def compute_grid(domain_cfg: Dict[str, Any]) -> Dict[str, int]:
    """Cosine-corrected grid dimensions for the Delhi domain (mirrors
    aqf_delhi.domain; kept here for standalone use)."""
    import math

    lat_min = domain_cfg["lat_min"]
    lat_max = domain_cfg["lat_max"]
    lon_min = domain_cfg["lon_min"]
    lon_max = domain_cfg["lon_max"]
    dy_km = domain_cfg["dy_km"]
    dx_km = domain_cfg["dx_km"]
    lat_c = (lat_min + lat_max) / 2.0
    dlat = lat_max - lat_min
    dlon = lon_max - lon_min
    ny = int(round(dlat * 111.0 / dy_km))
    nx = int(round(dlon * 111.0 * math.cos(math.radians(lat_c)) / dx_km))
    return {"ny": ny, "nx": nx}


def load_vayu_config(path: Path | None = None) -> VayuConfig:
    path = path or CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    # Pull repository domain grid defaults into the vayu domain section.
    if DEFAULT_CONFIG_PATH.exists():
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
            domain_yml = yaml.safe_load(fh) or {}
    else:
        domain_yml = {"domain": data.get("domain", {})}
    data["domain"] = {**data.get("domain", {}), **domain_yml.get("domain", {})}
    data["grid"] = compute_grid(data["domain"])
    data["grid_lookup"] = data["grid"]
    return VayuConfig(data)


CONFIG = load_vayu_config()