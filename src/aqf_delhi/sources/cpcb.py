"""CPCB ground-station observation connector.

Two strategy tiers with automatic failover:

  Tier-1 ``json_strategy`` — per-station JSON observations via the CPCB
     CAAQM data-link REST API (each registry station queried).
  Tier-2 ``csv_strategy``  — national AQI CSV digest, filtered to the
     Delhi NCR registry; used when the structured API is unavailable.

Both emit the canonical :class:`~aqf_delhi.schemas.cpcb.CpcbObservation`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from aqf_delhi.config import CPcbConfig, OpsConfig
from aqf_delhi.schemas.cpcb import CpcbObservation
from aqf_delhi.schemas.common import QualityFlag
from aqf_delhi.sources.base import FetchError, FetchResult, decode_csv, fetch_with_failover

logger = logging.getLogger(__name__)

METRIC_COLUMNS = {
    "SO2": "SO2",
    "NO2": "NO2",
    "PM2.5": "PM2.5",
    "PM10": "PM10",
    "CO": "CO",
    "O3": "OZONE",
    "NH3": "NH3",
}
# gaseous species reported in ppb by CPCB; particulates in ug/m3
GAS_UNITS = {"SO2": "ppb", "NO2": "ppb", "O3": "ppb", "CO": "mg/m3", "NH3": "ppb"}


class CPCBConnector:
    def __init__(self, cfg: CPcbConfig, ops: OpsConfig) -> None:
        self.cfg = cfg
        self.ops = ops

    # ------------------------------------------------------------------ #
    def fetch_summary(
        self, stations: list[dict], asof: datetime | None = None
    ) -> list[CpcbObservation]:
        """Tiered fetch. Returns canonical observations for in-domain stations."""
        asof = asof or datetime.now(timezone.utc)
        try:
            rows = self._json_strategy(stations, asof)
            logger.info("CPCB json tier returned %d observation rows", len(rows))
            return rows
        except FetchError as exc:
            logger.warning("CPCB json tier failed (%s); falling back to CSV", exc)
            return self._csv_strategy(stations, asof)

    # ------------------------------------------------------------------ #
    def _json_strategy(
        self, stations: list[dict], asof: datetime
    ) -> list[CpcbObservation]:
        rows: list[CpcbObservation] = []
        ingest_ts = datetime.now(timezone.utc)
        failures = 0
        for stn in stations:
            payload = {
                "city": stn.get("district") or stn.get("state") or "Delhi",
                "station": stn["name"],
                "startdate": (asof - self._window()).strftime("%Y-%m-%d"),
                "enddate": asof.strftime("%Y-%m-%d"),
            }
            try:
                result = fetch_with_failover(
                    self.cfg.endpoints,
                    "CAAQM/AqiDataLink/GetData",
                    ops=self.ops,
                    method="POST",
                    json_body=payload,
                )
                # Some mirrors expect a prefixed body key; tolerate both.
                parsed = self._decode_json(result)
                # NOTE: real CPCB responses vary; a rich parser replaces this
                # marker block at Phase-2 to normalize per-metric fields.
                rows += self._parse_json_response(parsed, stn, asof, ingest_ts)
            except (FetchError, ValueError) as exc:
                failures += 1
                logger.warning("station=%s json fetch failed: %s", stn["name"], exc)

        coverage = 1.0 - failures / max(len(stations), 1)
        if coverage < self.cfg.min_station_coverage:
            raise FetchError(
                f"station coverage {coverage:.0%} below minimum "
                f"{self.cfg.min_station_coverage:.0%}"
            )
        return rows

    def _window(self) -> object:
        from datetime import timedelta

        return timedelta(hours=12)

    @staticmethod
    def _decode_json(result: FetchResult) -> object | None:
        import json

        try:
            return json.loads(result.text)
        except (ValueError, UnicodeDecodeError):
            return None

    def _parse_json_response(self, parsed, stn, asof, ingest_ts) -> list[CpcbObservation]:
        """Placeholder tolerant parser — metrics normalize here at Phase-2."""
        if not isinstance(parsed, dict) or not parsed.get("success"):
            return []
        bucket = parsed.get("data") or {}
        rows = []
        for series in bucket.values():
            if not isinstance(series, list):
                continue
            for item in series:
                if not isinstance(item, dict):
                    continue
                rows.append(
                    self._obs_from_pairs(
                        stn, item, asof=asof, ingest_ts=ingest_ts, source="cpcb-json"
                    )
                )
        return rows

    # ------------------------------------------------------------------ #
    def _csv_strategy(self, stations, asof: datetime) -> list[CpcbObservation]:
        _ = asof
        result = fetch_with_failover(self.cfg.endpoints, "AQI_Data/AQI_Data.csv", ops=self.ops)
        return self.parse_csv_digest(result, stations)

    def parse_csv_digest(
        self, result: FetchResult, stations: list[dict]
    ) -> list[CpcbObservation]:
        names = {s["name"].strip().lower(): s for s in stations}
        ingest_ts = datetime.now(timezone.utc)
        rows: list[CpcbObservation] = []
        for row in decode_csv(result):
            location = (row.get("location") or "").strip().lower()
            station = names.get(location)
            if station is None:
                continue  # outside registry
            observed_at = self._parse_date(row.get("sampling_date"))
            if observed_at is None:
                continue  # cannot timestamp the observation — drop row
            for metric, col in METRIC_COLUMNS.items():
                raw = row.get(col)
                value = self._to_float(raw)
                if value is None:
                    continue
                rows.append(
                    CpcbObservation(
                        source="cpcb",
                        station_id=station["station_id"],
                        station_name=station["name"],
                        lat=float(station["lat"]),
                        lon=float(station["lon"]),
                        metric=metric,
                        value=value,
                        unit=GAS_UNITS.get(metric, "ug/m3"),
                        observed_at_utc=observed_at,
                        ingest_ts=ingest_ts,
                        quality_flag=QualityFlag.OK,
                        source_api_ver="aqi_data_csv",
                        raw_payload={"row": row},
                    )
                )
        return rows

    # ------------------------------------------------------------------ #
    def _obs_from_pairs(self, stn, item: dict, *, asof, ingest_ts, source: str):
        observed_at = self._parse_date(item.get("updated_at")) or asof
        for metric, col in METRIC_COLUMNS.items():
            value = self._to_float(item.get(col))
            if value is None:
                continue
            return CpcbObservation(
                source="cpcb",
                station_id=stn["station_id"],
                station_name=stn["name"],
                lat=float(stn["lat"]),
                lon=float(stn["lon"]),
                metric=metric,
                value=value,
                unit=GAS_UNITS.get(metric, "ug/m3"),
                observed_at_utc=observed_at,
                ingest_ts=ingest_ts,
                quality_flag=QualityFlag.OK,
                source_api_ver=source,
                raw_payload=item,
            )
        return None  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_float(value) -> float | None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if value != value:  # NaN
            return None
        return value

    @staticmethod
    def _parse_date(value) -> datetime | None:
        if value in (None, "", "-"):
            return None
        from dateutil import parser as du

        try:
            parsed = du.parse(str(value))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except (ValueError, OverflowError, TypeError):
            return None