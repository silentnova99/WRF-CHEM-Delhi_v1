"""Delhi NCR domain geometry + 4 km grid utilities (Module-2 grid spec)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from aqf_delhi.config import DomainConfig

KM_PER_DEG_LAT = 111.13


@dataclass(frozen=True)
class Grid:
    """Regular lat/lon grid over the study domain."""

    domain: DomainConfig
    nx: int          # columns (along lon)
    ny: int          # rows (along lat)

    @property
    def lon_nodes(self) -> list[float]:
        return [
            self.domain.lon_min + i * self.domain.lon_span / (self.nx - 1)
            for i in range(self.nx)
        ]

    @property
    def lat_nodes(self) -> list[float]:
        return [
            self.domain.lat_min + j * self.domain.lat_span / (self.ny - 1)
            for j in range(self.ny)
        ]

    def index_of(self, lat: float, lon: float) -> tuple[int, int]:
        """Nearest-node (i, j) indices for a point inside the domain."""
        if not self.contains(lat, lon):
            raise ValueError(f"point ({lat:.4f}, {lon:.4f}) outside domain")
        i = round((lon - self.domain.lon_min) / self.domain.lon_span * (self.nx - 1))
        j = round((lat - self.domain.lat_min) / self.domain.lat_span * (self.ny - 1))
        return min(max(i, 0), self.nx - 1), min(max(j, 0), self.ny - 1)

    def contains(self, lat: float, lon: float) -> bool:
        d = self.domain
        return d.lat_min <= lat <= d.lat_max and d.lon_min <= lon <= d.lon_max

    def describe(self) -> str:
        return (
            f"grid[name={self.domain.name}, {self.nx}x{self.ny} cells, "
            f"dx≈{self.lon_res_km():.2f} km x dy≈{self.lat_res_km():.2f} km]"
        )

    def lon_res_km(self) -> float:
        d = self.domain
        mid_lat = math.radians((d.lat_min + d.lat_max) / 2.0)
        return d.lon_span / (self.nx - 1) * KM_PER_DEG_LAT * math.cos(mid_lat)

    def lat_res_km(self) -> float:
        d = self.domain
        return d.lat_span / (self.ny - 1) * KM_PER_DEG_LAT


def build_grid(domain: DomainConfig) -> Grid:
    """Derive grid dimensions so cell edges ≈ dx_km × dy_km."""
    mid_lat = math.radians((domain.lat_min + domain.lat_max) / 2.0)
    nx = max(2, round(domain.lon_span * KM_PER_DEG_LAT * math.cos(mid_lat) / domain.dx_km) + 1)
    ny = max(2, round(domain.lat_span * KM_PER_DEG_LAT / domain.dy_km) + 1)
    return Grid(domain=domain, nx=nx, ny=ny)


def make_default_grid() -> Grid:
    from aqf_delhi.config import load_domain

    return build_grid(load_domain())