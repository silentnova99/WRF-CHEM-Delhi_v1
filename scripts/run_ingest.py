"""Phase-1 ingestion CLI.

Usage:
    python scripts/run_ingest.py cpcb [--asof 2026-09-12T12:00:00Z]
    python scripts/run_ingest.py firms
    python scripts/run_ingest.py gfs [--init 2026-09-12T00:00:00Z]

Writes idempotent parquet partitions under configs/ingest.yaml → storage.base_dir.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

from aqf_delhi.config import load_ops
from aqf_delhi.pipeline.orchestrate import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)


def _parse_dt(text: str | None, *, cycle: bool = False) -> datetime | None:
    if not text:
        return None
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    logging.getLogger().setLevel(getattr(logging, load_ops().log_level.upper()))

    orch = Orchestrator()
    for source in args.source:
        asof = _parse_dt(args.asof) if args.asof else None
        try:
            result = orch.run(source, asof=asof)
            print(f"[{source}] run_id={result.run_id} skipped={result.skipped} "
                  f"written={result.records_written}")
        except Exception as exc:
            print(f"[{source}] FAILED: {exc}", file=sys.stderr)
            return 1
    return 0


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", nargs="+", choices=["cpcb", "firms", "gfs"])
    p.add_argument("--asof", help="ISO timestamp override (UTC), e.g. 2026-09-12T03:00:00Z")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())