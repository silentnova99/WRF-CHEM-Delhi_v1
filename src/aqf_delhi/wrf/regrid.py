"""Bilinear regridding from the GFS subregion grid onto the Delhi domain.

The Delhi cell grid is regular in lat/lon (see :mod:`aqf_delhi.domain`), which
matches GFS's regular lat/lon 0.25 deg grid, so a pure numpy bilinear
interpolant is exact and dependency-free here.
"""

from __future__ import annotations

import numpy as np


def bilinear(
    s_lat: np.ndarray,          # [nj] source rows, ascending
    s_lon: np.ndarray,          # [ni] source cols, ascending
    data: np.ndarray,           # [nj, ni]
    t_lat: np.ndarray,          # target latitudes (any shape, flattened)
    t_lon: np.ndarray,          # target longitudes
) -> np.ndarray:
    """Bilinear interpolation of a (nj, ni) field at target points."""
    t_lat = np.asarray(t_lat, dtype=float).ravel()
    t_lon = np.asarray(t_lon, dtype=float).ravel()
    s_lat = np.asarray(s_lat, dtype=float)
    s_lon = np.asarray(s_lon, dtype=float)
    data = np.asarray(data, dtype=float)

    if s_lat[0] > s_lat[-1]:
        s_lat = s_lat[::-1]
        data = data[::-1]
    t_lat = np.clip(t_lat, s_lat[0], s_lat[-1])
    t_lon = np.clip(t_lon, s_lon[0], s_lon[-1])

    i0 = np.clip(np.searchsorted(s_lat, t_lat, side="right") - 1, 0, len(s_lat) - 2)
    j0 = np.clip(np.searchsorted(s_lon, t_lon, side="right") - 1, 0, len(s_lon) - 2)
    f_r = (t_lat - s_lat[i0]) / np.maximum(s_lat[i0 + 1] - s_lat[i0], 1e-12)
    f_c = (t_lon - s_lon[j0]) / np.maximum(s_lon[j0 + 1] - s_lon[j0], 1e-12)

    v = (
        data[i0, j0] * (1 - f_r) * (1 - f_c)
        + data[i0 + 1, j0] * f_r * (1 - f_c)
        + data[i0, j0 + 1] * (1 - f_r) * f_c
        + data[i0 + 1, j0 + 1] * f_r * f_c
    )
    return v


def subset_to_grid(
    s_lat: np.ndarray,
    s_lon: np.ndarray,
    data: np.ndarray,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> np.ndarray:
    """Regrid a decoded GFS field (rows north->south) onto [ny, nx] cell nodes."""
    plat, plon = np.meshgrid(grid_lat, grid_lon, indexing="ij")
    v = bilinear(s_lat, s_lon, data, plat.ravel(), plon.ravel())
    return v.reshape(len(grid_lat), len(grid_lon))


EP = 1e-12


def nearest(
    s_lat: np.ndarray,
    s_lon: np.ndarray,
    data: np.ndarray,
    t_lat: np.ndarray,
    t_lon: np.ndarray,
) -> np.ndarray:
    """Nearest-source-cell value at target points (edge fallback)."""
    t_lat = np.asarray(t_lat, dtype=float).ravel()
    t_lon = np.asarray(t_lon, dtype=float).ravel()
    i = np.abs(s_lat[:, None] - t_lat[None, :]).argmin(axis=0)
    j = np.abs(s_lon[:, None] - t_lon[None, :]).argmin(axis=0)
    return data[i, j]