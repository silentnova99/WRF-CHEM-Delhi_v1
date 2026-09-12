"""Orchestrator — one idempotent run per (source, run_id).

Execution contract per source:

  1. compute ``run_id`` (cycle/poll slot);
  2. if the partition is already complete → skip (idempotency);
  3. fetch → validate → store → mark done.

Any exception aborts the partition (no ``_DONE`` is written), so a retry
re-runs cleanly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable

from aqf_delhi.config import load_domain, load_ingest, load_ops, load_stations
from aqf_delhi.pipeline.validate import VALIDATORS, QCReport
from aqf_delhi.storage.local import ParquetStore
from aqf_delhi.sources.base import FetchError

logger = logging.getLogger(__name__)


class IngestionResult:
    def __init__(self, source: str, run_id: str, skipped: bool = False):
        self.source = source
        self.run_id = run_id
        self.skipped = skipped
        self.records_written = 0
        self.report: QCReport | None = None

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<IngestionResult {self.source}/{self.run_id} "
            f"skipped={self.skipped} written={self.records_written}>"
        )


class Orchestrator:
    def __init__(self, store: ParquetStore | None = None) -> None:
        self.cfg = load_ingest()
        self.domain = load_domain()
        self.ops = load_ops()
        self.stations = load_stations()
        self.store = store or ParquetStore(self.cfg.storage.base_dir)

    # ------------------------------------------------------------------ #
    def run(self, source: str, asof: datetime | None = None) -> IngestionResult:
        asof = asof or datetime.now(timezone.utc)
        if asof.tzinfo is None:
            asof = asof.replace(tzinfo=timezone.utc)
        connector, run_id = self._connect(source, asof)
        result = IngestionResult(source, run_id)
        date = asof.date().isoformat()  # partition date from wall clock (UTC)

        if self.store.is_complete(source, date, run_id):
            logger.info("partition %s/%s already complete — skipping", source, run_id)
            result.skipped = True
            return result

        records = connector()
        validator = VALIDATORS[source]
        kept, report = validator(records, self.cfg.quality)
        result.report = report
        logger.info(
            "%s: kept=%d dropped=%d (%s)",
            source, report.kept, report.dropped, report.reasons,
        )
        if not kept:
            # gfs may legitimately yield all-MISSING manifests; firms an
            # empty detection window (e.g. pre-sowing season). Only a
            # non-empty fetch that fails QC is a hard error.
            if records and source != "gfs":
                raise FetchError(f"all {source} records failed validation")
            logger.info("%s: valid empty result — nothing persisted", source)
            return result

        partition = self.store.write(
            kept, source=source, date=date, run_id=run_id,
            compression=self.cfg.storage.parquet_compression,
        )
        result.records_written = len(kept)
        logger.info("persisted %s -> %s", source, partition)
        return result

    # ------------------------------------------------------------------ #
    def _connect(self, source: str, asof: datetime) -> tuple[Callable, str]:
        utc = asof.astimezone(timezone.utc) if asof.tzinfo else asof.replace(tzinfo=timezone.utc)
        if source == "cpcb":
            from aqf_delhi.sources.cpcb import CPCBConnector

            conn = CPCBConnector(self.cfg.cpcb, self.ops)
            run_id = f"{utc:%Y-%m-%d}T{utc:%H%M}Z-cpcb-poll"
            return (lambda: conn.fetch_summary(self.stations, asof=utc)), run_id

        if source == "firms":
            from aqf_delhi.sources.firms import FIRMSConnector
            from aqf_delhi.domain import build_grid

            conn = FIRMSConnector(self.cfg.firms, self.ops)
            grid = build_grid(self.domain)
            run_id = f"{utc:%Y-%m-%d}T{utc:%H%M}Z-firms-poll"
            return (lambda: conn.fetch(asof=utc, assign_grid=grid.index_of)), run_id

        if source == "gfs":
            from aqf_delhi.sources.gfs import GFSConnector

            conn = GFSConnector(self.cfg.gfs, self.ops)
            init = utc.replace(minute=0, second=0, microsecond=0)
            init = init.replace(hour=(init.hour // 6) * 6)
            run_id = f"gfs.{init:%Y-%m-%d}.{init:%H}z"
            availability = conn.probe(init)

            def _manifest():
                return conn.build_manifest(init, availability)

            return _manifest, run_id

        raise KeyError(f"unknown source: {source!r} (expected cpcb|firms|gfs)")