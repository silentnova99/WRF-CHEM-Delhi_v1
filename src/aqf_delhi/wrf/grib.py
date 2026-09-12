"""GFS 0.25 deg subset retrieval + GRIB2 decode.

Fetches tiny subset GRIB2 files for the Delhi region from the NOMADS GFS
0.25 deg filter service (only the levels/variables needed by the emulator),
then decodes them with eccodes into numpy arrays keyed ``var:leveltype:level``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import requests
import numpy as np

from aqf_delhi.wrf.config import Module2Config


class GribFetchError(RuntimeError):
    pass


@dataclass
class GfsSubset:
    """Decoded GRIB2 subset fields on the native GFS subregion grid."""

    init: datetime
    fhr: int
    lat: np.ndarray          # [nj] rows, north -> south
    lon: np.ndarray          # [ni] cols, ascending
    fields: dict[str, np.ndarray] = field(default_factory=dict)   # [nj, ni]


def filter_url(cfg, init: datetime, fhr: int) -> str:
    """Build the NOMADS `filter_gfs_0p25.pl` query for one forecast hour.

    NOMADS requires ``dir`` URL-encoded **with a leading slash** and the
    ``subregion=`` trigger parameter for geo-subsetting to be applied; without
    them the CGI returns the full global field or 404s respectively.
    """
    dir_ = f"%2Fgfs.{init:%Y%m%d}%2F{init:%H}%2Fatmos"
    file_ = f"gfs.t{init:%H}z.pgrb2.0p25.f{fhr:03d}"
    sr = cfg.subregion
    args = ["subregion=", f"dir={dir_}", f"file={file_}"]
    for lev, vars_ in cfg.levels.items():
        args.append(f"lev_{lev}=on")
        for v in vars_:
            args.append(f"var_{v}=on")
    args += [
        f"toplat={sr.top_lat:.2f}",
        f"bottomlat={sr.bottom_lat:.2f}",
        f"leftlon={sr.left_lon:.2f}",
        f"rightlon={sr.right_lon:.2f}",
    ]
    return cfg.filter_base + "?" + "&".join(args)


def fetch_subset(
    cfg,
    init: datetime,
    fhr: int,
    dest: Path | None = None,
    timeout: float = 60.0,
) -> bytes:
    """Download one subset GRIB2 file; optionally persist it."""
    url = filter_url(cfg, init, fhr)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise GribFetchError(f"GFS subset fetch failed [{fhr:03d}]: {exc}") from exc
    if not resp.content.startswith(b"GRIB"):
        raise GribFetchError(f"Response is not GRIB2 [{fhr:03d}]: {resp.content[:40]!r}")
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
    return resp.content


def _msg_key(eccodes, msg) -> str:
    var = eccodes.codes_get(msg, "shortName", None) or "?"
    tol = eccodes.codes_get(msg, "typeOfLevel", None) or "?"
    lev = eccodes.codes_get(msg, "level", None) or 0
    return f"{var}:{tol}:{lev}"


def decode_grib(path: Path) -> GfsSubset:
    """Decode a subset GRIB2 file into a :class:`GfsSubset` using eccodes."""
    import eccodes

    init = None
    fhr = None
    lat = lon = None
    fields: dict[str, np.ndarray] = {}
    with open(path, "rb") as fh:
        while True:
            msg = eccodes.codes_grib_new_from_file(fh)
            if msg is None:
                break
            try:
                vals = np.asarray(eccodes.codes_get_values(msg), dtype=float)
                ni = eccodes.codes_get(msg, "Ni")
                nj = eccodes.codes_get(msg, "Nj")
                vals = vals.reshape(nj, ni)
                if not fields:
                    la1 = eccodes.codes_get(msg, "latitudeOfFirstGridPointInDegrees")
                    la2 = eccodes.codes_get(msg, "latitudeOfLastGridPointInDegrees")
                    lo1 = eccodes.codes_get(msg, "longitudeOfFirstGridPointInDegrees")
                    lo2 = eccodes.codes_get(msg, "longitudeOfLastGridPointInDegrees")
                    lat = np.linspace(la1, la2, nj)
                    lon = np.linspace(lo1, lo2, ni)
                    y = eccodes.codes_get(msg, "year", None)
                    mo = eccodes.codes_get(msg, "month", None)
                    da = eccodes.codes_get(msg, "day", None)
                    ho = eccodes.codes_get(msg, "hour", None)
                    init = datetime(y, mo, da, ho)
                    fhr = eccodes.codes_get(msg, "step", None) or 0
                fields[_msg_key(eccodes, msg)] = vals
            finally:
                eccodes.codes_release(msg)
    if lat is None or lon is None:
        raise GribFetchError(f"no messages decoded from {path}")
    if lat[0] < lat[-1]:                 # force north -> south rows
        lat = lat[::-1]
        fields = {k: v[::-1] for k, v in fields.items()}
    return GfsSubset(init=init, fhr=fhr, lat=lat, lon=lon, fields=fields)