"""Dataset manifest + size report generation (sections 41-42 of the spec).

Manifest rows: source, dataset_id, file, size, rows, columns, date_min,
date_max, lat_min, lat_max, lon_min, lon_max, missing_percentage, checksum,
download_date, license, provenance.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from .config import VayuConfig
from .lake import DataLake, sha256_file
from .registry import probe_local_file


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_manifest(cfg: VayuConfig, lake: DataLake) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []
    for source in ("cpcb", "gfs", "firms", "aod", "emissions"):
        raw_dir = lake.raw_dir(source)
        if not raw_dir.exists():
            continue
        for f in sorted(raw_dir.rglob("*")):
            if not f.is_file() or f.name == "_DONE":
                continue
            manifest.append(probe_row(cfg, source, f))
    return manifest


def probe_row(cfg: VayuConfig, source: str, path: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "source": source,
        "dataset_id": f"{source}::{path.name}",
        "file": str(path.relative_to(cfg.resolve("data"))),
        "size": path.stat().st_size,
        "checksum": sha256_file(path),
        "download_date": _now_utc(),
        "license": cfg.section("sources").get(source, {}).get("license"),
        "provenance": cfg.section("sources").get(source, {}).get("primary"),
    }
    if path.suffix.lower() == ".csv" or path.name.endswith(".parquet"):
        try:
            if path.name.endswith(".parquet"):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, nrows=3000, low_memory=False)
            row["rows"] = len(df)
            row["columns"] = list(df.columns)
            ts_col = next((c for c in ("observed_at_utc", "Datetime", "datetime", "timestamp", "acq_date") if c in df.columns), None)
            if ts_col:
                t = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
                if t.notna().any():
                    row["date_min"] = str(t.min())
                    row["date_max"] = str(t.max())
            lat_col = next((c for c in ("Latitude", "latitude", "location_lat", "lat") if c in df.columns), None)
            lon_col = next((c for c in ("Longitude", "longitude", "location_lon", "lon") if c in df.columns), None)
            if lat_col and lon_col:
                row["lat_min"] = round(float(df[lat_col].min()), 4)
                row["lat_max"] = round(float(df[lat_col].max()), 4)
                row["lon_min"] = round(float(df[lon_col].min()), 4)
                row["lon_max"] = round(float(df[lon_col].max()), 4)
            missing = int(df.isna().sum().sum())
            row["missing_percentage"] = round(
                100.0 * missing / max(df.shape[0] * df.shape[1], 1), 3
            )
        except Exception as exc:  # pragma: no cover
            row["probe_error"] = str(exc)
    return row


def write_manifest(cfg: VayuConfig, lake: DataLake, out: Path) -> Path:
    rows = compute_manifest(cfg, lake)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"generated_at": _now_utc(), "datasets": rows}, fh, sort_keys=False)
    return out


def write_human_report(cfg: VayuConfig, lake: DataLake, out: Path) -> Path:
    import io

    rows = compute_manifest(cfg, lake)
    buf = io.StringIO()
    buf.write("# VAYU-SETU dataset manifest (human-readable)\n\n")
    buf.write(f"Generated: {_now_utc()}\n\n")
    buf.write("| source | file | rows | columns | date_min | date_max | missing % | size |\n")
    buf.write("|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        buf.write(
            f"| {r.get('source')} | {r.get('file')} | {r.get('rows','-')} | "
            f"{len(r.get('columns') or [])} | {r.get('date_min','-')} | {r.get('date_max','-')} | "
            f"{r.get('missing_percentage','-')} | {r.get('size')} |\n"
        )
    buf.write("\n## Provenance / license\n\n")
    for r in rows:
        buf.write(f"- `{r['dataset_id']}` license: {r.get('license')}; provenance: {r.get('provenance')} checksum `{r['checksum'][:16]}...`\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(buf.getvalue(), encoding="utf-8")
    return out