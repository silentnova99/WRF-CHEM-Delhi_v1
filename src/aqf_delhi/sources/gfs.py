"""NOAA GFS 0.25° 72-hour forecast connector (AWS S3 / NOMADS / NCEP FTP).

Phase-1 responsibilities:
  * deterministic URL/path construction for each forecast hour;
  * reachability probe (HEAD) across the failover chain;
  * manifest emission (:class:`GfsFieldAsset` metadata records) pointing at
    the object-store assets, so downstream WRF-Chem feeding and COG
    conversion consume a uniform catalog.

Heavy grid pull / GRIB→NetCDF conversion is a worker task (Phase-2).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from aqf_delhi.config import GFSEndpointConfig, OpsConfig
from aqf_delhi.schemas.common import QualityFlag
from aqf_delhi.schemas.gfs import GfsFieldAsset, LevelKind
from aqf_delhi.sources.base import FetchResult, fetch_with_failover, getstream

logger = logging.getLogger(__name__)

# Global 0.25° product dimensions: 1440 (lon) × 721 (lat) — used for
# manifest metadata until the actual grid is decoded in Phase-2.
GRID_ROWS = 721
GRID_COLS = 1440

# Map GFS grib message abbreviations to conceptual level_kind
SURFACE_HINTS = ("surface", "above ground", "mean sea level")
PRESSURE_HINTS = ("mb", "hpa")


class GFSConnector:
    def __init__(self, cfg: GFSEndpointConfig, ops: OpsConfig) -> None:
        self.cfg = cfg
        self.ops = ops

    # ------------------------------------------------------------------ #
    def build_path_explicit(self, init_utc: datetime, fhr: int) -> str:
        return (
            f"gfs.{init_utc:%Y%m%d}/{init_utc:%H}/atmos/"
            f"gfs.t{init_utc:%H}z.pgrb2.0p25.f{fhr:03d}"
        )

    # ------------------------------------------------------------------ #
    def probe(
        self, init_utc: datetime, fhrs: list[int] | None = None
    ) -> list[dict]:
        """HEAD each forecast hour across the failover chain.

        Returns ``[{fhr, available, endpoint, size_bytes}]``. A field is
        ``available`` iff at least one endpoint accepted the request.
        """
        fhrs = fhrs if fhrs is not None else self.cfg.forecast_hours
        results: list[dict] = []
        for fhr in fhrs:
            path = self.build_path_explicit(init_utc, fhr)
            try:
                result: FetchResult = fetch_with_failover(
                    self.cfg.endpoints,
                    path,
                    ops=self.ops,
                    method="HEAD",
                )
                results.append(
                    {
                        "fhr": fhr,
                        "available": True,
                        "endpoint": result.used_endpoint,
                        "size_bytes": int(result.headers.get("Content-Length", 0) or 0),
                    }
                )
            except Exception as exc:  # all endpoints failed for this hour
                logger.warning("GFS fhr=%03d unavailable: %s", fhr, exc)
                results.append(
                    {"fhr": fhr, "available": False, "endpoint": None, "size_bytes": 0}
                )
        return results

    # ------------------------------------------------------------------ #
    def build_manifest(
        self, init_utc: datetime, availability: list[dict]
    ) -> list[GfsFieldAsset]:
        """Emit canonical metadata records for each configured variable/hour."""
        init_utc = _utc(init_utc)
        ingest_ts = datetime.now(timezone.utc)
        avail = {a["fhr"]: a for a in availability}
        assets: list[GfsFieldAsset] = []
        for fhr in self.cfg.forecast_hours:
            info = avail.get(fhr, {"available": False})
            for spec in self.cfg.variables:
                asset = self._asset_for(
                    spec, init_utc, fhr, info, ingest_ts, used_endpoint=info.get("endpoint")
                )
                assets.append(asset)
        return assets

    # ------------------------------------------------------------------ #
    def download(self, init_utc: datetime, fhr: int, local_path: Path) -> str:
        """Stream a forecast-hour grid file to ``local_path`` (failover)."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        path = self.build_path_explicit(init_utc, fhr)
        used = getstream(
            self.cfg.endpoints,
            path,
            ops=self.ops,
            write_to=lambda resp: _stream_to(resp, local_path),
        )
        return used

    # ------------------------------------------------------------------ #
    def _asset_for(
        self, spec: str, init_utc, fhr: int, info: dict, ingest_ts, used_endpoint
    ) -> GfsFieldAsset:
        variable, _, level_str = spec.partition(":")
        variable = variable.strip().upper()
        level_str = (level_str or "").strip() or "surface"
        valid = _add_hours(init_utc, fhr)
        return GfsFieldAsset(
            source="gfs",
            run_id=f"gfs.{init_utc:%Y%m%d}.{init_utc:%H}z",
            init_utc=init_utc,
            lead_h=fhr,
            valid_utc=valid,
            variable=variable,
            level_str=level_str,
            level_kind=_level_kind(level_str),
            grid="0p25",
            crs="EPSG:4326",
            asset_uri=f"gfs/{init_utc:%Y%m%d}/{init_utc:%H}/{variable}:{level_str}.f{fhr:03d}",
            rows=GRID_ROWS,
            cols=GRID_COLS,
            missing_fraction=0.0 if info["available"] else 1.0,
            observed_at_utc=valid,
            ingest_ts=ingest_ts,
            quality_flag=QualityFlag.OK if info["available"] else QualityFlag.MISSING,
        )


# --------------------------------------------------------------------------- #
def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _add_hours(dt: datetime, hours: int) -> datetime:
    from datetime import timedelta

    return _utc(dt) + timedelta(hours=hours)


def _level_kind(level_str: str) -> LevelKind:
    low = level_str.lower()
    if any(h in low for h in SURFACE_HINTS):
        return "surface"
    if any(h in low for h in PRESSURE_HINTS):
        return "pressure"
    return "model_level"


def _stream_to(resp, dest: Path) -> None:
    with dest.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                fh.write(chunk)