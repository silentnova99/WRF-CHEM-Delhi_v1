"""Layered data lake helpers.

Layout (relative to AQF_DATA_ROOT / repo root):

    data/raw/{source}/                        untouched original files
    data/interim/cleaned/{source}/            cleaned originals
    data/interim/aligned/                     canonical hourly UTC, common coords
    data/interim/quality_controlled/          QC-passed aligned tables
    data/processed/{station,grid,plume,physics}/
    data/features/{train,validation,test}/

Every write is idempotent and ends in a `_DONE` marker. Checksums are
computed on raw files so ingestion can skip unchanged inputs.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .config import VayuConfig


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class DataLake:
    """Path helpers and atomic-ish parquet writes over the layered lake."""

    def __init__(self, cfg: VayuConfig, root: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.root = Path(root) if root else cfg.resolve("data")
        layout = cfg.section("lake").get("layout", {})
        self.layout = {k: (self.root / str(v).replace("data/", "", 1)) for k, v in layout.items()}
        self.layout["raw"] = self.root / "raw"
        for source in ("cpcb", "gfs", "firms", "aod", "emissions"):
            self.layout.setdefault(f"raw_{source}", self.root / "raw" / source)

    def raw_dir(self, source: str) -> Path:
        return self.root / "raw" / source

    def interim(self, stage: str, *parts: str) -> Path:
        return self.root / "interim" / stage / Path(*parts)

    def processed(self, kind: str, *parts: str) -> Path:
        return self.root / "processed" / kind / Path(*parts)

    def features(self, split: str, *parts: str) -> Path:
        return self.root / "features" / split / Path(*parts)

    def ensure(self, *parts: str) -> Path:
        p = Path(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write_parquet(self, df: pd.DataFrame, path: Path, done: bool = True) -> Path:
        comp = self.cfg.section("lake").get("parquet_compression", "zstd")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression=comp)
        if done:
            (path.parent / "_DONE").write_text(_now_utc(), encoding="utf-8")
        return path

    def write_json(self, payload: Any, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path


def compute_storage_sizes(cfg: VayuConfig, root: Optional[Path] = None) -> Dict[str, Any]:
    """Dataset size report: bytes, rows, counts per lake tier (no estimates)."""
    lake = DataLake(cfg, root)
    results: Dict[str, Any] = {"generated_at": _now_utc(), "tiers": {}}

    def tier(name: str, path: Path) -> None:
        files = list(path.rglob("*")) if path.exists() else []
        files = [f for f in files if f.is_file() and f.name != "_DONE"]
        nbytes = sum(f.stat().st_size for f in files)
        results["tiers"][name] = {
            "path": str(path),
            "files": len(files),
            "bytes": nbytes,
        }

    tier("raw", lake.root / "raw")
    tier("interim", lake.root / "interim")
    tier("processed", lake.root / "processed")
    tier("features", lake.root / "features")

    # Counts computed from actual parquet tables.
    counts: Dict[str, int] = {}
    for p in list((lake.root / "processed").rglob("*.parquet")) + list(
        (lake.root / "features").rglob("*.parquet")
    ):
        try:
            df = pd.read_parquet(p, columns=None)
            counts.setdefault(p.parent.name, 0)
            counts[p.stem] = len(df)
        except Exception:
            pass
    results["row_counts"] = counts
    results["bytes_total"] = sum(t["bytes"] for t in results["tiers"].values())
    return results


def normalize_timestamps(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)