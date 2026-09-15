"""Stubble-burning engine: FIRMS-real or deterministic synthetic fire events.

Data sources (all clearly labelled):
  * LIVE   - NASA FIRMS area CSV (requires ``FIRMS_MAP_KEY`` env var) via legacy
             ``aqf_delhi.sources.firms`` connector.
  * CACHE  - previously cached FIRMS partition.
  * DEMO   - deterministic synthetic fire clusters across Punjab/Haryana/UP.

The engine computes per fire:
  * fire location / timestamp / FRP / confidence
  * distance to each grid cell
  * wind-relative upwind/downwind influence on a cell
  * fire-density contribution (FRP-weighted spatial kernel)

Emissions are estimated from FRP using ``configs/emission_factors.yaml``
(Wooster FRP→fuel + literature emission factors) → returns per-hour, per-cell
emissions for PM2.5 / PM10 / CO / NOx / VOC in g/h.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import yaml

from aqf_delhi.config import PROJECT_ROOT

FRP_TO_FUEL = 0.368  # kg fuel per MJ FRP (Wooster et al. 2005)


@dataclass
class FireEvent:
    fire_id: str
    lat: float
    lon: float
    acq: datetime
    frp_mw: float
    confidence_pct: float
    satellite: str = "synthetic"
    data_source: str = "demo-synthetic"


@dataclass
class FireState:
    events: list[FireEvent]
    # per grid point, per hour: influence 0..1 (downwind-weighted)
    influence: np.ndarray  # [nt, ny, nx]
    upwind: np.ndarray  # [nt, ny, nx] 0..1
    downwind: np.ndarray  # [nt, ny, nx] 0..1
    density: np.ndarray  # [nt, ny, nx] FRP-weighted density (MW per unit)
    hours: list
    lat_nodes: np.ndarray
    lon_nodes: np.ndarray
    data_source: str = "demo-synthetic"

    @property
    def nt(self):
        return len(self.hours)

    def json(self, hi: int) -> dict:
        active = [e for e in self.events if _within_window(e.acq, self.hours[hi], 6)]
        return {
            "hour": self.hours[hi].isoformat(),
            "n_fires": len(self.events),
            "n_active": len(active),
            "total_frp_mw": float(sum(e.frp_mw for e in active)),
            "mean_influence": float(np.nanmean(self.influence[hi])),
            "data_source": self.data_source,
        }


def _within_window(acq: datetime, at: datetime, hours: float) -> bool:
    if acq.tzinfo is None:
        acq = acq.replace(tzinfo=timezone.utc)
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return 0 <= (at - acq).total_seconds() / 3600.0 < hours


def _synthetic_active(ev: FireEvent) -> bool:
    """Synthetic demo fires burn for the whole forecast episode.

    Represents a *sustained* stubble-burning event (individual VIIRS/FRP
    detections would repeat every few hours for days) rather than a single
    6-hour detection — used so the DEMO scenario shows a continuous plume.
    """
    return getattr(ev, "data_source", "") == "demo-synthetic-fires"


class StubbleEngine:
    """Fire detection + influence engine.

    Parameters
    ----------
    grid : aqf_delhi.domain.Grid
    emission_factors_path : path to configs/emission_factors.yaml
    seed : RNG seed for deterministic synthetic fires.
    """

    def __init__(self, grid, emission_factors_path=PROJECT_ROOT / "configs" / "emission_factors.yaml", seed: int = 11):
        self.grid = grid
        self.seed = seed
        with open(emission_factors_path, "r", encoding="utf-8") as fh:
            self.ef = yaml.safe_load(fh)
        self._km_per_deg = 111.13

    # ------------------------------------------------------------------ #
    # data acquisition
    # ------------------------------------------------------------------ #
    def load_fires(self, hours: list[datetime], *, mode: str = "demo") -> list[FireEvent]:
        if mode == "live":
            try:
                events = self._load_live_firms()
            except Exception:
                events = self._load_cache_firms() or []
            if events:
                return events
            # fall through to deterministic demo if live/cache empty
        return self._synthetic_fires(hours)

    def _load_live_firms(self) -> list[FireEvent]:
        from aqf_delhi.config import load_ingest
        from aqf_delhi.sources.firms import FIRMSConnector

        cfg = load_ingest()
        ops = cfg.ops
        conn = FIRMSConnector(cfg.firms, ops)
        records = conn.fetch(asof=datetime.now(timezone.utc), assign_grid=None)
        events = []
        for rec in records:
            events.append(
                FireEvent(
                    fire_id=rec.fire_id,
                    lat=rec.lat,
                    lon=rec.lon,
                    acq=rec.observed_at_utc or rec.ingest_ts,
                    frp_mw=rec.frp_mw,
                    confidence_pct=rec.confidence_percent,
                    satellite=rec.satellite,
                    data_source="live-firms",
                )
            )
        return events

    def _load_cache_firms(self):
        """Attempt to read cached FIRMS partition (best-effort)."""
        try:
            from aqf_delhi.storage.local import ParquetStore
            from aqf_delhi.config import load_ingest

            store = ParquetStore(load_ingest().storage.base_dir)
            parts = store.list_partitions("firms")
            if not parts:
                return []
            records = store.read("firms", parts[-1][0], parts[-1][1])
            return [
                FireEvent(
                    fire_id=getattr(r, "fire_id", f"cached-{i}"),
                    lat=float(r.lat),
                    lon=float(r.lon),
                    acq=getattr(r, "observed_at_utc", None),
                    frp_mw=float(r.frp_mw),
                    confidence_pct=float(getattr(r, "confidence_percent", 100)),
                    satellite=getattr(r, "satellite", "cached"),
                    data_source="cache-firms",
                )
                for i, r in enumerate(records)
            ]
        except Exception:
            return []

    def _synthetic_fires(self, hours: list[datetime]) -> list[FireEvent]:
        """Deterministic stubble-burning clusters at realistic NCR locations.

        Clusters around Punjab, Haryana, western UP. FRP / number scaled by
        seasonal factor so the demo shows the late-Oct..Nov stubble peak.
        """
        rng = np.random.RandomState(self.seed)
        clusters = [
            # (lat, lon, n_fires, peak_frp_mw) — Punjab belt (Sangrur, Patiala)
            (30.25, 75.80, 6, 90),
            (29.95, 75.45, 5, 80),
            (30.60, 75.10, 4, 70),
            # Haryana (Kurukshetra, Karnal, Rohtak)
            (29.95, 76.85, 4, 65),
            (29.68, 76.98, 3, 55),
            (28.90, 76.55, 3, 50),
            # Western UP
            (28.85, 77.90, 2, 45),
            (28.98, 78.15, 2, 40),
            # NCR fringe smoke sources
            (28.78, 77.01, 2, 35),
            (28.45, 77.15, 2, 30),
        ]
        doy = hours[0].timetuple().tm_yday if hours else 300
        season = _seasonal_factor(doy)  # 0..1 stubble season weight
        events = []
        n = 0
        for lat, lon, n_f, peak_frp in clusters:
            for _ in range(n_f):
                # jitter positions slightly to spread a cluster
                llat = lat + rng.uniform(-0.12, 0.12)
                llon = lon + rng.uniform(-0.12, 0.12)
                frp = max(10.0, peak_frp * season * rng.uniform(0.55, 1.0))
                conf = rng.randint(45, 100)
                # fire "active" at a random hour in the last 24 h
                acq = hours[-1] - timedelta(hours=float(rng.randint(0, 24)))
                n += 1
                events.append(
                    FireEvent(
                        fire_id=f"SYN-{n:04d}-{lat:.2f}-{lon:.2f}",
                        lat=round(llat, 4),
                        lon=round(llon, 4),
                        acq=acq,
                        frp_mw=round(frp, 1),
                        confidence_pct=conf,
                        satellite="synthetic",
                        data_source="demo-synthetic-fires",
                    )
                )
        return events

    # ------------------------------------------------------------------ #
    # influence / density
    # ------------------------------------------------------------------ #
    def compute_state(
        self, hours: list[datetime], events: list[FireEvent], ws10: np.ndarray, wd_deg: np.ndarray
    ) -> FireState:
        nt, ny, nx = ws10.shape
        influence = np.zeros((nt, ny, nx))
        upwind = np.zeros((nt, ny, nx))
        downwind = np.zeros((nt, ny, nx))
        density = np.zeros((nt, ny, nx))

        lats = np.asarray(self.grid.lat_nodes)[:, None]
        lons = np.asarray(self.grid.lon_nodes)[None, :]

        for ev in events:
            if ev.frp_mw < float(self.ef.get("fire_min_frp_mw", 10.0)):
                continue
            dlat = lats - ev.lat
            dlon = (lons - ev.lon) * np.cos(np.radians(ev.lat))
            dist_km = np.hypot(dlat * self._km_per_deg, dlon * self._km_per_deg)

            # spatial kernel: Gaussian 40 km e-folding in km
            kernel = np.exp(-(dist_km / 40.0) ** 2)
            density += kernel * ev.frp_mw

            # wind-relative component per hour
            for hi in range(nt):
                if not _synthetic_active(ev) and not _within_window(
                        ev.acq, hours[hi], float(self.ef.get("fire_lifetime_h", 6))):
                    continue
                wd = wd_deg[hi]
                winds = np.stack([np.cos(np.radians(wd)), np.sin(np.radians(wd))], axis=-1)  # [ny,nx,2]
                # unit vector from fire to cell in km
                dlat_2d = dlat + np.zeros((ny, nx))
                dlon_2d = dlon + np.zeros((ny, nx))
                vec = np.stack([dlon_2d * self._km_per_deg, dlat_2d * self._km_per_deg], axis=-1)
                dot = np.sum(vec * winds, axis=-1)
                # downwind (dot > 0) gets extra weight
                down = np.clip(dot / (dist_km + 1.0), 0, 1)  # 0..1 downwind alignment
                up = np.clip(-dot / (dist_km + 1.0), 0, 1)
                inf = kernel * (1.0 + 1.6 * down)
                influence[hi] += inf * ev.frp_mw
                upwind[hi] += up * ev.frp_mw
                downwind[hi] += down * ev.frp_mw

        # normalize influence to 0..1 scale (FRP-weighted softmax-like)
        maxi = influence.max() if influence.max() > 0 else 1.0
        influence = influence / maxi
        upwind = upwind / (upwind.max() if upwind.max() > 0 else 1.0)
        downwind = downwind / (downwind.max() if downwind.max() > 0 else 1.0)

        return FireState(
            events=events, influence=influence, upwind=upwind, downwind=downwind,
            density=density, hours=hours,
            lat_nodes=np.asarray(self.grid.lat_nodes), lon_nodes=np.asarray(self.grid.lon_nodes),
            data_source="unknown",
        )

    # ------------------------------------------------------------------ #
    # emissions (FRP → mass flux per cell per hour, g/h)
    # ------------------------------------------------------------------ #
    def emissions_grid(
        self, hours: list[datetime], events: list[FireEvent], influence=None
    ) -> dict[str, np.ndarray]:
        """Return per-hour per-cell net emissions (g/h) for PM2.5/PM10/CO/NOx/VOC.

        Uses the spatial kernel of each active fire to distribute FRP-derived
        mass flux over the grid.
        """
        nt, ny, nx = len(hours), self.grid.ny, self.grid.nx
        out = {sp: np.zeros((nt, ny, nx)) for sp in ("pm25", "pm10", "co", "nox", "voc", "so2", "bc", "oc")}

        fuel_kg_per_hour_per_mw = FRP_TO_FUEL * 3600.0  # kg/h per MW
        ef = self.ef["emission_factors"]

        lats = np.asarray(self.grid.lat_nodes)[:, None]
        lons = np.asarray(self.grid.lon_nodes)[None, :]

        for ev in events:
            if ev.frp_mw < float(self.ef.get("fire_min_frp_mw", 10.0)):
                continue
            dlat = lats - ev.lat
            dlon = (lons - ev.lon) * np.cos(np.radians(ev.lat))
            dist_km = np.hypot(dlat * self._km_per_deg, dlon * self._km_per_deg)
            kernel = np.exp(-(dist_km / 40.0) ** 2)
            kernel = kernel / (kernel.sum() if kernel.sum() > 0 else 1.0)

            mass_total_gph = ev.frp_mw * fuel_kg_per_hour_per_mw * 1000.0  # g/h

            for hi in range(nt):
                if not _synthetic_active(ev) and not _within_window(
                        ev.acq, hours[hi], float(self.ef.get("fire_lifetime_h", 6))):
                    continue
                for sp in out:
                    out[sp][hi] += mass_total_gph * ef[sp] * kernel

        return out

    def emissions_point(self, ev: FireEvent) -> dict:
        """Per-fire g/h emission estimate (for API reporting)."""
        ef = self.ef["emission_factors"]
        fuel_gph = ev.frp_mw * FRP_TO_FUEL * 3600.0 * 1000.0
        return {sp: round(fuel_gph * v, 3) for sp, v in ef.items()}


def _seasonal_factor(doy: int) -> float:
    """Crop-residue burning seasonal weight: peak late Oct–Nov, off-season low."""
    import math

    # cosine-bump centered on Nov 5 (doy 309), width ~45 days
    return float(max(0.0, math.cos((doy - 309) / 45.0 * math.pi) ** 2))