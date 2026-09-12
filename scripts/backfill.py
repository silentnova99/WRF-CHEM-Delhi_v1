"""Historical backfill to current (Module-1 store).

    python scripts/backfill.py gfs    --days 8   # last N days, 4 cycles/day
    python scripts/backfill.py firms  --days 7   # needs FIRMS_MAP_KEY
    python scripts/backfill.py cpcb   --days 3   # needs DATA_GOVIN_API_KEY

Run blocks are idempotent: completed partitions are skipped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)


def _days_ago(days: int, hour: int = 12) -> datetime:
    now = datetime.now(timezone.utc)
    d = (now - timedelta(days=days)).replace(hour=hour, minute=0, second=0, microsecond=0)
    return d


def backfill_gfs(days: int) -> None:
    from aqf_delhi.pipeline.orchestrate import Orchestrator

    orch = Orchestrator()
    done = skipped = 0
    for d in range(days, -1, -1):
        for cyc in (0, 6, 12, 18):
            asof = _days_ago(d, hour=(cyc + 3) % 24)
            res = orch.run("gfs", asof=asof)
            done += res.records_written
            skipped += res.skipped
            print(f"gfs {asof:%Y-%m-%d %H}z -> run_id={res.run_id} "
                  f"written={res.records_written} skipped={res.skipped}")
    print(f"[gfs] backfill done: {done} records written, {skipped} partitions skipped")


def backfill_firms(days: int) -> None:
    import os

    from aqf_delhi.pipeline.orchestrate import Orchestrator

    if not os.environ.get("FIRMS_MAP_KEY"):
        print("[firms] BLOCKED: FIRMS_MAP_KEY not set — collect a free key at "
              "https://firms.modaps.eosdis.nasa.gov and re-run.")
        return
    orch = Orchestrator()
    done = skipped = 0
    for d in range(days, -1, -1):
        asof = _days_ago(d, hour=23)
        res = orch.run("firms", asof=asof)
        done += res.records_written
        skipped += res.skipped
        print(f"firms {asof:%Y-%m-%d} -> written={res.records_written} skipped={res.skipped}")
    print(f"[firms] backfill done: {done} records written, {skipped} partitions skipped")


def backfill_cpcb(days: int) -> None:
    import os

    import requests

    from aqf_delhi.config import load_ingest
    from aqf_delhi.pipeline.orchestrate import Orchestrator

    cfg = load_ingest()
    key = os.environ.get(cfg.cpcb.data_govin.api_key_env)
    probe = f"{cfg.cpcb.data_govin.base_url}/{cfg.cpcb.data_govin.resource_id}"
    if not key:
        print(f"[cpcb] BLOCKED: {cfg.cpcb.data_govin.api_key_env} not set — get a key "
              "from https://data.gov.in (My Account) and re-run.")
        return
    try:
        requests.get(probe, params={"api-key": key, "format": "json", "limit": "1"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    except Exception as exc:
        print(f"[cpcb] BLOCKED: data.gov.in unreachable ({probe}): "
              f"{type(exc).__name__}: {exc}")
        print("[cpcb] the HTTPS-API may be rate-limiting the sample key (HTTP 429); wait and retry.")
        return
    orch = Orchestrator()
    done = skipped = 0
    for d in range(days, -1, -1):
        asof = _days_ago(d, hour=12)
        res = orch.run("cpcb", asof=asof)
        done += res.records_written
        skipped += res.skipped
        print(f"cpcb {asof:%Y-%m-%d} -> written={res.records_written} skipped={res.skipped}")
    print(f"[cpcb] backfill done: {done} records written, {skipped} partitions skipped")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", choices=["gfs", "firms", "cpcb"], help="source to backfill")
    ap.add_argument("--days", type=int, default=8, help="historical days to cover")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    {"gfs": backfill_gfs, "firms": backfill_firms, "cpcb": backfill_cpcb}[args.source](args.days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())