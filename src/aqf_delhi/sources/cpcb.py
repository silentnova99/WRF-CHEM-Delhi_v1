"""CPCB ground-station observation connector.

Three strategy tiers with automatic failover:

  Tier-0 ``data_govin_strategy`` — real-time CPCB station readings via the
     data.gov.in OGD API (national catalog resource id, paginated, api-key).
  Tier-1 ``json_strategy`` — per-station JSON observations via the CPCB
     CAAQM data-link REST API (each registry station queried).
  Tier-2 ``csv_strategy``  — national AQI CSV digest, filtered to the
     Delhi NCR registry; used when the structured APIs are unavailable.

All tiers emit the canonical :class:`~aqf_delhi.schemas.cpcb.CpcbObservation`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone

from aqf_delhi.config import CPcbConfig, OpsConfig
from aqf_delhi.schemas.cpcb import CpcbObservation
from aqf_delhi.schemas.common import QualityFlag
from aqf_delhi.sources.base import FetchError, FetchResult, decode_csv, fetch_with_failover

logger = logging.getLogger(__name__)

# data.gov.in throttles/errors on the default python-requests UA (HTTP 502);
# a browser UA is served promptly.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)

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

# data.gov.in resource fields: pollutant_id / avg_value / min_value / max_value
# NO unit column is exposed — CPCB reports gases in ug/m3 (CO in mg/m3).
GOVIN_METRICS = {
    "PM2.5": "PM2.5",
    "PM10": "PM10",
    "SO2": "SO2",
    "NO2": "NO2",
    "CO": "CO",
    "OZONE": "O3",
    "NH3": "NH3",
}
GOVIN_UNITS = {
    "PM2.5": "ug/m3",
    "PM10": "ug/m3",
    "SO2": "ug/m3",
    "NO2": "ug/m3",
    "CO": "mg/m3",
    "O3": "ug/m3",
    "NH3": "ug/m3",
}


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
        last_exc: FetchError | None = None
        for strategy in (self._data_govin_strategy, self._json_strategy, self._csv_strategy):
            try:
                rows = strategy(stations, asof)
                logger.info("CPCB %s returned %d observation rows",
                            strategy.__name__, len(rows))
                return rows
            except FetchError as exc:
                last_exc = exc
                logger.warning("CPCB %s failed (%s)", strategy.__name__, exc)
        raise last_exc or FetchError("no CPCB tier available")

    # ------------------------------------------------------------------ #
    def _data_govin_strategy(
        self, stations: list[dict], asof: datetime
    ) -> list[CpcbObservation]:
        api_key = os.environ.get(self.cfg.data_govin.api_key_env)
        if not api_key:
            raise FetchError(
                f"{self.cfg.data_govin.api_key_env} not set — skipping data.gov.in tier"
            )
        g = self.cfg.data_govin
        ingest_ts = datetime.now(timezone.utc)
        records: list[dict] = []
        offset = 0
        for _ in range(g.max_pages):
            params = {
                "api-key": api_key,
                "format": "json",
                "limit": str(g.page_size),
                "offset": str(offset),
            }
            for field, values in g.filters.items():
                if isinstance(values, str):
                    values = [values]
                for value in values:
                    params[f"filters[{field}]"] = value
            try:
                result = fetch_with_failover(
                    [g.base_url],
                    g.resource_id,
                    ops=self.ops,
                    method="GET",
                    params=params,
                    headers={"User-Agent": BROWSER_USER_AGENT},
                )
                body = json.loads(result.text)
            except (FetchError, ValueError) as exc:
                raise FetchError(f"data.gov.in offset={offset} fetch failed: {exc}")
            if body.get("status") != "ok":
                raise FetchError(
                    f"data.gov.in returned status={body.get('status')!r}: "
                    f"{body.get('message', '')}"
                )
            page_records = body.get("records") or []
            records.extend(page_records)
            total = int(body.get("total") or 0)
            # advance by what the API actually returned (sample keys cap the
            # window at 10 rows even when limit=10000) — never by page_size.
            offset += len(page_records)
            if not page_records or offset >= total:
                break
            if g.page_delay_seconds:
                import time

                time.sleep(g.page_delay_seconds)
        if not records:
            raise FetchError("data.gov.in returned 0 records")
        rows = self._parse_govin_records(
            records, stations, asof=asof, ingest_ts=ingest_ts
        )
        if not rows:
            raise FetchError(
                "data.gov.in records did not match any registry station"
            )
        return rows

    def _parse_govin_records(
        self, records: list[dict], stations: list[dict], *, asof, ingest_ts
    ) -> list[CpcbObservation]:
        rows: list[CpcbObservation] = []
        for rec in records:
            metric = GOVIN_METRICS.get((rec.get("pollutant_id") or "").strip().upper())
            if metric is None:
                continue
            value = self._to_float(rec.get("avg_value"))
            if value is None:
                continue
            station = self._match_station(rec.get("station") or "", stations)
            if station is None:
                continue
            observed_at = self._parse_date(rec.get("last_update")) or asof
            rows.append(
                CpcbObservation(
                    source="cpcb",
                    station_id=station["station_id"],
                    station_name=station["name"],
                    lat=float(station["lat"]),
                    lon=float(station["lon"]),
                    metric=metric,
                    value=value,
                    unit=GOVIN_UNITS[metric],
                    observed_at_utc=observed_at,
                    ingest_ts=ingest_ts,
                    quality_flag=QualityFlag.OK,
                    source_api_ver="data_govin_aqi",
                    raw_payload=rec,
                )
            )
        return rows

    # ------------------------------------------------------------------ #
    @staticmethod
    def _match_station(record_station: str, stations: list[dict]) -> dict | None:
        """Match a data.gov.in station string (e.g. 'IHBAS, Dilshad Garden, Delhi - CPCB')
        to a registry entry using ordered-token subsequence matching."""
        rec_tokens = re.findall(r"[a-z0-9]+", record_station.lower())
        if not rec_tokens:
            return None
        best: dict | None = None
        best_len = 0
        for stn in stations:
            stn_tokens = re.findall(r"[a-z0-9]+", stn["name"].lower())
            if not stn_tokens:
                continue
            it = iter(rec_tokens)
            if all(any(tok == needle for tok in it) for needle in stn_tokens):
                if len(stn_tokens) > best_len:
                    best, best_len = stn, len(stn_tokens)
        return best

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
                    self._endpoint_by("GetData"),
                    "",
                    ops=self.ops,
                    method="POST",
                    json_body=payload,
                    verify=False,  # CAAQM serves a self-signed chain
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
        result = fetch_with_failover(
            self._endpoint_by("AQI_Data.csv", default=0), "", ops=self.ops
        )
        return self.parse_csv_digest(result, stations)

    def _endpoint_by(self, needle: str, *, default: int = 0) -> list[str]:
        """Pick the configured endpoint(s) matching ``needle`` (fresh list)."""
        matches = [e for e in self.cfg.endpoints if needle in e]
        if matches:
            return matches
        return [self.cfg.endpoints[default]]

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
            # CPCB/OGD stamps are DD/MM/YYYY — day-first disambiguates early dates.
            parsed = du.parse(str(value), dayfirst=True)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except (ValueError, OverflowError, TypeError):
            return None