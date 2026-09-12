"""Parquet partition store with idempotent done-markers.

Layout mirrors the canonical object store (Module-1 contract):

    {base_dir}/{source}/{YYYY-MM-DD}/{run_id}/records.parquet
    {base_dir}/{source}/{YYYY-MM-DD}/{run_id}/_DONE

Writing a partition is atomic: data first, then ``_DONE``. Consumers only
consider a partition complete when ``_DONE`` exists — retries re-write the
same run_id harmlessly (idempotency by run_id).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DONE_MARKER = "_DONE"
RECORDS_FILE = "records.parquet"


class ParquetStore:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)

    # ------------------------------------------------------------------ #
    def partition_path(self, source: str, date: str, run_id: str) -> Path:
        return self.base_dir / source / date / run_id

    # ------------------------------------------------------------------ #
    def write(
        self,
        records: list,
        *,
        source: str,
        date: str,
        run_id: str,
        compression: str = "zstd",
    ) -> Path:
        """Write canonical records as one parquet partition (idempotent)."""
        if not records:
            logger.warning("no records to persist for %s/%s/%s", source, date, run_id)
            return self.partition_path(source, date, run_id)
        df = pd.DataFrame(
            [r.model_dump(mode="json") if hasattr(r, "model_dump") else dict(r) for r in records]
        )
        part = self.partition_path(source, date, run_id)
        part.mkdir(parents=True, exist_ok=True)
        tmp = part / f".tmp-{run_id}"
        df.to_parquet(tmp, index=False, compression=compression)
        (part / RECORDS_FILE).write_bytes(tmp.read_bytes())
        tmp.unlink(missing_ok=True)
        (part / DONE_MARKER).write_text(
            f"written_at={datetime.now(timezone.utc).isoformat()}\ncount={len(df)}\n"
        )
        logger.info("wrote %d records -> %s", len(df), part.relative_to(self.base_dir))
        return part

    # ------------------------------------------------------------------ #
    def is_complete(self, source: str, date: str, run_id: str) -> bool:
        part = self.partition_path(source, date, run_id)
        return (part / DONE_MARKER).is_file() and (part / RECORDS_FILE).is_file()

    def read(self, source: str, date: str, run_id: str) -> pd.DataFrame:
        if not self.is_complete(source, date, run_id):
            raise FileNotFoundError(
                f"partition incomplete: {source}/{date}/{run_id}"
            )
        return pd.read_parquet(self.partition_path(source, date, run_id) / RECORDS_FILE)

    def list_partitions(self, source: str) -> list[Path]:
        root = self.base_dir / source
        return sorted(p for p in root.rglob(DONE_MARKER) if p.name == DONE_MARKER)