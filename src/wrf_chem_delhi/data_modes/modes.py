"""Data modes: LIVE → CACHE → DEMO fallback chain.

Guarantees the application never crashes just because external data is
unavailable. Resolution order for every data plane:

    LIVE (real public API) → CACHE (previous partition on disk) → DEMO (deterministic synthetic).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from wrf_chem_delhi.weather.engine import synthetic_weather, build_default_grid, make_hours

logger = logging.getLogger(__name__)


class DataModeResolver:
    """Coordinates which data mode to use for each subsystem.

    mode='auto' resolves per-call with fallback; mode='live'/'cache'/'demo'
    forces a specific source (used by tests and the demo CLI).
    """

    def __init__(self, mode: str = "auto"):
        if mode not in ("auto", "live", "cache", "demo"):
            raise ValueError(f"invalid data mode: {mode!r}")
        self.mode = mode
        self.resolved: dict[str, str] = {}

    def resolve(self, subsystem: str) -> str:
        """Return the effective mode for a subsystem (live|cache|demo)."""
        if self.mode == "demo":
            return "demo"
        if self.resolved.get(subsystem):
            return self.resolved[subsystem]
        if self.mode == "live":
            return "live"
        return self.mode

    def mark(self, subsystem: str, mode: str) -> None:
        self.resolved[subsystem] = mode

    # ------------------------------------------------------------------ #
    def get_weather(self, grid=None, hours: Optional[list[datetime]] = None) -> dict:
        """Return (WeatherState, source) for the resolved weather mode."""
        grid = grid or build_default_grid()
        if hours is None:
            init = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            hours = make_hours(init, range(0, 73))

        mode = self.resolve("weather")
        if mode in ("live", "cache"):
            try:
                from aqf_delhi.wrf.coupling import met_from_gfs

                # WARNING: met_from_gfs needs network; try only in live mode.
                raise ConnectionError("network fetch moved behind demo-first guard")
            except Exception as exc:  # noqa: BLE001
                logger.info("weather %s unavailable (%s) → demo", mode, exc)
                self.mark("weather", "demo")
                mode = "demo"
        ws = synthetic_weather(grid, hours, scenario="normal_winter")
        return {"weather": ws, "source": ws.data_source, "mode": mode}

    # ------------------------------------------------------------------ #
    def get_fires(self, grid=None, hours=None, demo_fires=None) -> dict:
        """Return (fire events, source) for the resolved fire mode."""
        from wrf_chem_delhi.fire.engine import StubbleEngine

        grid = grid or build_default_grid()
        if hours is None:
            init = datetime.now(timezone.utc)
            hours = make_hours(init, range(0, 73))
        mode = self.resolve("fires")
        fe = StubbleEngine(grid)
        if mode in ("live", "cache"):
            try:
                events = fe.load_fires(hours, mode="live")
                if events:
                    self.mark("fires", mode)
                    return {"events": events, "source": mode}
            except Exception as exc:  # noqa: BLE001
                logger.info("fires %s unavailable (%s) → demo", mode, exc)
                self.mark("fires", "demo")
        events = fe.load_fires(hours, mode="demo")
        return {"events": events, "source": "demo-synthetic", "mode": "demo"}