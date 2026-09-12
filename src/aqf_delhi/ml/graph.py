"""Spatiotemporal graph construction (Module-3 section 3.2).

Nodes  = CPCB registry stations (+ optional coarse pseudo-nodes pinned to
         the 4 km domain grid for downscaling).
Edges  = (a) k-nearest by geodesic distance, (b) wind-shifted transport
         edges activated for the stubble-burning season, (c) optional
         historical cross-correlation edges.

Adjacency is returned as a symmetric-normalized matrix Â = D^-1/2 A D^-1/2,
consumed directly by the T-GCN message-passing layer.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass
class Node:
    """A graph node: a CPCB station or a grid pseudo-node."""

    node_id: str
    lat: float
    lon: float
    kind: str = "station"  # station | grid
    attrs: dict = field(default_factory=dict)


class StationGraph:
    """Immutable graph container; adjacency is a dense numpy matrix."""

    def __init__(self, nodes: list[Node], adjacency: np.ndarray) -> None:
        if len(nodes) != adjacency.shape[0]:
            raise ValueError("node count must match adjacency size")
        self.nodes = nodes
        self._adj = adjacency
        self._norm: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def ids(self) -> list[str]:
        return [n.node_id for n in self.nodes]

    def index(self, node_id: str) -> int:
        return self.ids.index(node_id)

    def adjacency(self, normalized: bool = False) -> np.ndarray:
        if not normalized:
            return self._adj
        if self._norm is None:
            deg = self._adj.sum(axis=1)
            deg_inv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
            d = np.diag(deg_inv)
            self._norm = d @ self._adj @ d
        return self._norm


class StationGraphBuilder:
    def __init__(
        self,
        *,
        k: int = 6,
        transport_radius_km: float = 60.0,
        wind_bearing_deg: float = 300.0,  # WNW: prevailing stubble flow
        wind_season_months: tuple[int, int] = (10, 11),
        corr_threshold: float = 0.75,
        use_wind_edges: bool = True,
        use_corr_edges: bool = False,
    ) -> None:
        self.k = k
        self.radius = transport_radius_km
        self.wind_bearing = wind_bearing_deg
        self.wind_season = wind_season_months
        self.corr_threshold = corr_threshold
        self.use_wind_edges = use_wind_edges
        self.use_corr_edges = use_corr_edges

    def build(
        self,
        stations: list[dict],
        *,
        grid_nodes: list[Node] | None = None,
        month: int = 10,
    ) -> StationGraph:
        """Build the graph. ``month`` controls wind-edge activation."""
        nodes = [
            Node(
                node_id=s["station_id"],
                lat=float(s["lat"]),
                lon=float(s["lon"]),
                kind="station",
                attrs={"name": s["name"]},
            )
            for s in stations
        ]
        if grid_nodes:
            nodes = nodes + list(grid_nodes)

        n = len(nodes)
        adj = np.zeros((n, n), dtype=float)
        lat = np.array([nd.lat for nd in nodes])
        lon = np.array([nd.lon for nd in nodes])

        # (a) k-nearest geodesic edges (spatial proximity)
        dist = self._distance_matrix(lat, lon)  # km
        for i in range(n):
            order = np.argsort(dist[i])[1 : self.k + 1]
            for j in order:
                adj[i, j] = adj[i, j] + 1.0
                adj[j, i] = adj[j, i] + 1.0

        # (b) wind-transport edges (directed alignment → symmetric weight)
        if self.use_wind_edges and self._in_window(month):
            for i, j in itertools.combinations(range(n), 2):
                d = dist[i, j]
                if d > self.radius:
                    continue
                bearing = self._bearing(lat[i], lon[i], lat[j], lon[j])
                alignment = abs(math.cos(math.radians(bearing - self.wind_bearing)))
                w = max(0.0, alignment)
                adj[i, j] += w
                adj[j, i] += w

        # (c) correlation edges are injected by the dataset builder later as
        # an adjacency overlay (weights 0..1); callers use :meth:`add_corr_weight`.
        return StationGraph(nodes=nodes, adjacency=adj)

    def add_corr_weight(self, graph: StationGraph, w: np.ndarray) -> StationGraph:
        """Overlay correlation-derived edge weights (masked to 0..1)."""
        adj = graph.adjacency().copy()
        corr = np.clip(w, 0.0, 1.0)
        adj = np.maximum(adj, corr)
        return StationGraph(graph.nodes, adj)

    @staticmethod
    def _distance_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        n = len(lat)
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = d[j, i] = haversine_km(lat[i], lon[i], lat[j], lon[j])
        return d

    @staticmethod
    def _bearing(lat1, lon1, lat2, lon2) -> float:
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        y = math.sin(dlon) * math.cos(p2)
        x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
        return (math.degrees(math.atan2(y, x)) + 360) % 360

    def _in_window(self, month: int) -> bool:
        lo, hi = self.wind_season
        return lo <= month <= hi if lo <= hi else (month >= lo or month <= hi)