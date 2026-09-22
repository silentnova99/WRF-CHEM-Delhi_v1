"""Machine-readable data-source registry.

One record per logical source (CPCB, GFS, FIRMS, AOD, EMISSIONS) carrying
provider, Kaggle id, authoritative URL, local path, download timestamp,
version, temporal/spatial extent, variables, units, license, checksum and
runtime status. Persisted as YAML + JSON.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from .config import VayuConfig
from .lake import sha256_file


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


REGISTRY_FIELDS = [
    "source_name",
    "provider",
    "kaggle_dataset_id",
    "authoritative_url",
    "local_path",
    "download_timestamp",
    "dataset_version",
    "temporal_start",
    "temporal_end",
    "spatial_extent",
    "variables",
    "units",
    "license",
    "checksum",
    "status",
]


class SourceRegistry:
    """Holds registry records for the five required logical sources."""

    def __init__(self, cfg: VayuConfig, records: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.cfg = cfg
        self.records: Dict[str, Dict[str, Any]] = records or {}

    def add(self, source: str, **fields: Any) -> None:
        rec = {k: fields.get(k) for k in REGISTRY_FIELDS}
        rec["source_name"] = source
        self.records[source] = rec

    def update(self, source: str, **fields: Any) -> None:
        rec = self.records.setdefault(source, {"source_name": source})
        for k, v in fields.items():
            rec[k] = v

    def get(self, source: str) -> Dict[str, Any]:
        return self.records.get(source, {})

    def mark(self, source: str, status: str) -> None:
        self.update(source, status=status)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump({"generated_at": _now_utc(), "sources": self.records}, fh, sort_keys=False)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"generated_at": _now_utc(), "sources": self.records}, fh, indent=2, default=str)

    @classmethod
    def from_yaml(cls, path: Path) -> "SourceRegistry":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(VayuConfig({}), data.get("sources", {}))


def probe_local_file(path: Path) -> Dict[str, Any]:
    """Checksum + first-line probe for a raw candidate file (no full read)."""
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def scan_local_csv(path: Path, sample: int = 2000) -> Dict[str, Any]:
    """Schema + coarse temporal/lat-lon probe of a CSV without loading it all."""
    info: Dict[str, Any] = {"path": str(path), "rows_estimated": None}
    try:
        probe = pd.read_csv(path, nrows=sample, low_memory=False)
        info["columns"] = list(probe.columns)
        info["dtypes"] = {c: str(t) for c, t in probe.dtypes.items()}
        for c in ("latitude", "lat", "Latitude"):
            if c in probe.columns:
                info["lat_range"] = [float(probe[c].min()), float(probe[c].max())]
        for c in ("longitude", "lon", "Longitude"):
            if c in probe.columns:
                info["lon_range"] = [float(probe[c].min()), float(probe[c].max())]
        for c in ("acq_date", "Datetime", "timestamp", "date"):
            if c in probe.columns:
                try:
                    t = pd.to_datetime(probe[c], errors="coerce", utc=True)
                    info["time_range_sample"] = [str(t.min()), str(t.max())]
                except Exception:
                    pass
        info["sample_rows"] = len(probe)
    except Exception as exc:  # pragma: no cover
        info["error"] = str(exc)
    return info


def ingest_manifest_row(
    source: str,
    provider: str,
    kaggle_id: Optional[str],
    authoritative_url: str,
    local_path: Path,
    license: str,
    dataset_version: str,
    variables: List[str],
    temporal: Optional[tuple] = None,
    spatial: Optional[str] = None,
    status: str = "available",
) -> Dict[str, Any]:
    return {
        "source_name": source,
        "provider": provider,
        "kaggle_dataset_id": kaggle_id,
        "authoritative_url": authoritative_url,
        "local_path": str(local_path),
        "download_timestamp": _now_utc(),
        "dataset_version": dataset_version,
        "temporal_start": temporal[0] if temporal else None,
        "temporal_end": temporal[1] if temporal else None,
        "spatial_extent": spatial,
        "variables": variables,
        "units": "-",
        "license": license,
        "checksum": sha256_file(local_path) if local_path.exists() else None,
        "status": status,
    }