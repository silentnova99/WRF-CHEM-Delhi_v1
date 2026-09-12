"""NASA FIRMS active-fire / FRP connector (VIIRS 375 m primary).

Polls the FIRMS area CSV API for fire detections inside the Delhi NCR
bounding box and maps rows into canonical :class:`FireDetection` records,
assigning each detection to its 4 km model grid cell along the way.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from aqf_delhi.config import FirmsConfig, OpsConfig
from aqf_delhi.schemas.common import QualityFlag
from aqf_delhi.schemas.firms import FireDetection
from aqf_delhi.sources.base import FetchError, decode_csv, fetch_with_failover

logger = logging.getLogger(__name__)

SATELLITE_MAP = {
    "NPP": "VIIRS_SNPP",
    "NOAA-20": "VIIRS_NOAA20",
    "NOAA-21": "VIIRS_NOAA20",
    "AQUA": "MODIS_AQUA",
    "TERRA": "MODIS_TERRA",
}


class FIRMSConnector:
    def __init__(self, cfg: FirmsConfig, ops: OpsConfig) -> None:
        self.cfg = cfg
        self.ops = ops

    def map_key(self) -> str:
        key = os.environ.get(self.cfg.map_key_env)
        if not key:
            raise FetchError(
                f"env var {self.cfg.map_key_env!r} not set; "
                "register at https://firms.modaps.eosdis.nasa.gov (free) "
                "and export your map key"
            )
        return key.strip()

    def fetch(
        self,
        asof: datetime | None = None,
        collection: str | None = None,
        assign_grid=None,
    ) -> list[FireDetection]:
        """Fetch detections for ``day_window`` days ending ``asof`` (UTC).

        The FIRMS area API for NRT sources accepts only a day-range query
        ``{key}/{collection}/{day}/{bbox}`` with day in 1..5.
        """
        asof = asof or datetime.now(timezone.utc)
        collection = collection or self.cfg.default_collection
        day = max(1, int(self.cfg.day_window))
        path = (
            f"{self.map_key()}/{collection}/{day}/{self.cfg.bbox}"
        )
        result = fetch_with_failover(
            [self.cfg.api_base], path, ops=self.ops, timeout=self.ops.request_timeout_seconds
        )
        detections = self.parse_csv(result.text)
        for det in detections:
            if assign_grid is not None:
                try:
                    gi, gj = assign_grid(det.lat, det.lon)
                    det.grid_i, det.grid_j, det.assigned_grid_id = gi, gj, f"{gi}:{gj}"
                except ValueError:
                    continue
        return detections

    def parse_csv(self, payload: str) -> list[FireDetection]:
        import csv
        import io

        ingest_ts = datetime.now(timezone.utc)
        seen: set[str] = set()
        rows: list[FireDetection] = []
        reader = csv.DictReader(io.StringIO(payload))
        for row in reader:
            satellite = SATELLITE_MAP.get((row.get("satellite") or "").strip().upper(), "")
            if not satellite:
                continue
            acq = self._acq_datetime(row.get("acq_date"), row.get("acq_time"))
            if acq is None:
                continue
            fire_id = (
                f"{satellite}|{row.get('acq_date')}|{row.get('acq_time')}"
                f"|{row.get('latitude')}|{row.get('longitude')}|{row.get('scan')}|{row.get('track')}"
            )
            if fire_id in seen:  # duplicate suppression across collections
                continue
            seen.add(fire_id)
            rows.append(
                FireDetection(
                    source="firms",
                    fire_id=fire_id,
                    satellite=satellite,
                    lat=float(row["latitude"]),
                    lon=float(row["longitude"]),
                    acq_datetime=acq,
                    observed_at_utc=acq,
                    frp_mw=max(0.0, self._f(row.get("frp"), 0.0)),
                    brightness_kelvin=self._f(row.get("bright_ti4"), 350.0),
                    confidence_percent=float(str(row.get("confidence")).replace("%", "")),
                    scan_km=self._f(row.get("scan"), 0.0),
                    track_km=self._f(row.get("track"), 0.0),
                    day_night=self._dn(row.get("daynight", "D")),
                    ingest_ts=ingest_ts,
                    quality_flag=self._flag(row.get("confidence")),
                )
            )
        return rows

    @staticmethod
    def _acq_datetime(date_str: str | None, time_str: str | None) -> datetime | None:
        if not date_str:
            return None
        hh, mm = divmod(int(str(time_str or "0").zfill(4)), 100)
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(
                hour=hh, minute=mm, tzinfo=timezone.utc
            )
        except ValueError:
            return None

    @staticmethod
    def _f(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _dn(value: str) -> str:
        v = (value or "D").strip().upper()
        return "N" if v.startswith("N") else "D"

    @staticmethod
    def _flag(confidence) -> QualityFlag:
        try:
            val = float(str(confidence).replace("%", ""))
        except (TypeError, ValueError):
            return QualityFlag.SUSPECT
        return QualityFlag.OK if val >= 20 else QualityFlag.SUSPECT