"""Reduced-order chemistry engine.

Implements a deliberately simplified gas + aerosol chemistry module that
produces the species array consumed by the coupled forecast engine:
PM2.5, PM10, O3, NOx, SO2, CO.

The chemistry is explicitly labelled reduced-order — it is NOT a full
gas-phase mechanism (no CBMZ/RACM, no condensed-phase microphysics), but it
captures the leading coupling terms that matter for an SIH demonstration:

  * NOx + O3 titration:  NO + O3 -> NO2  (nighttime O3 depression)
  * photochemical O3 production driven by solar radiation (J-independent proxy)
  * secondary-organic-aerosol (SOA) yield from VOC in sunlit, aged plumes
  * secondary inorganic aerosol from NOx + humidity
  * hygroscopic particle growth at high RH (multiplier on PM)
  * temperature dependence of reaction rates (Arrhenius-lite)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from wrf_chem_delhi.weather.engine import WeatherState

PROD_PHOTO_UG_M3 = 22.0    # max photochemical O3 production (ug/m3 per hr)
TITRATE_UG_M3 = 18.0       # max nighttime titration (ug/m3)
TEMP_SCALE_K = 8.0         # temperature half-span for photolysis coefficient
SOA_YIELD = 0.12           # VOC -> SOA mass yield (primary)
SECONDARY_INORG_RATE = 0.04  # h^-1 NOx + humidity -> nitrate/aerosol
HUMIDITY_GROWTH_MAX = 1.45  # hygroscopic growth factor at RH=100%


@dataclass
class ChemistryResult:
    hours: list[datetime]
    pm25: np.ndarray
    pm10: np.ndarray
    o3: np.ndarray
    nox: np.ndarray
    so2: np.ndarray
    co: np.ndarray
    secondary_pm: np.ndarray
    aerosol_growth: np.ndarray  # [nt,ny,nx] multiplier >= 1

    def json(self, hi: int) -> dict:
        return {
            "hour": self.hours[hi].isoformat(),
            "pm25_avg": float(np.nanmean(self.pm25[hi])),
            "pm10_avg": float(np.nanmean(self.pm10[hi])),
            "o3_avg": float(np.nanmean(self.o3[hi])),
            "nox_avg": float(np.nanmean(self.nox[hi])),
            "so2_avg": float(np.nanmean(self.so2[hi])),
            "co_avg": float(np.nanmean(self.co[hi])),
            "secondary_pm_avg": float(np.nanmean(self.secondary_pm[hi])),
            "aerosol_growth_avg": float(np.nanmean(self.aerosol_growth[hi])),
        }


def calculate_o3(
    day: np.ndarray, t2m_c: np.ndarray, nox: np.ndarray, radiation: np.ndarray,
    southern_valley: bool = True,
) -> np.ndarray:
    """Photochemical O3 measured from solar radiation & NOx titration.

    O3 = bg + prod * radiation * temp_factor - titr * (1-radiation)
    where titration scales with NOx/NO2 mass (proxy [NOx]).
    """
    tempf = np.clip((t2m_c - 15.0) / TEMP_SCALE_K, 0, 1)  # warm = faster chemistry
    radiation_f = np.clip(radiation, 0, 1) * day
    photo = PROD_PHOTO_UG_M3 * radiation_f * tempf
    titr = np.zeros_like(photo)
    if southern_valley:
        # NOx titration strongest at night in the urban valley
        titr = TITRATE_UG_M3 * (1.0 - radiation_f) * np.clip(nox / 150.0, 0, 1)
    o3 = 38 + photo - titr
    return np.clip(o3, 5.0, None)


def calculate_secondary_pm(voc_ug: np.ndarray, nox_ug: np.ndarray, day: np.ndarray, dt_h: float = 1.0) -> np.ndarray:
    """Secondary PM formation from VOC oxidation and NOx->nitrate."""
    soa = SOA_YIELD * voc_ug * day * dt_h
    inorganic = SECONDARY_INORG_RATE * nox_ug * dt_h
    return np.clip(soa + inorganic, 0, None)


def calculate_nox_effect(o3_base: np.ndarray, nox_ug: np.ndarray, radiation: np.ndarray) -> np.ndarray:
    """Net NOx effect on O3: titration during night, photochemical source by day."""
    return (o3_base - TITRATE_UG_M3 * (1.0 - radiation) * np.clip(nox_ug / 150.0, 0, 1)) - o3_base


def calculate_aerosol_growth(rh_pct: np.ndarray, t2m_c: np.ndarray) -> np.ndarray:
    """Hygroscopic growth multiplier for particles (kölbler-lite)."""
    rhf = np.clip((rh_pct - 40.0) / 60.0, 0, 1)
    growth = 1.0 + (HUMIDITY_GROWTH_MAX - 1.0) * rhf
    return growth


def run_chemistry(
    ws: WeatherState,
    transported_pm25: np.ndarray,
    transported_pm10: np.ndarray,
    transported_o3: np.ndarray,
    transported_nox: np.ndarray,
    voc_emission: np.ndarray,
    so2_emission: np.ndarray,
    co_emission: np.ndarray,
) -> ChemistryResult:
    """Apply reduced chemistry to transported species.

    Parameters
    ----------
    ws          : WeatherState (gives t2m, rh, dayness)
    transported_* : [nt,ny,nx] after transport & emissions
    voc_/so2_/co_emission : [nt,ny,nx] primary emissions (µg/m³ added)

    Returns ChemistryResult with post-chemistry fields.
    """
    nt, ny, nx = ws.nt, ws.ny, ws.nx
    day = dayness_flat(ws)

    # O3 chemistry using transported NOx, radiation proxy = day
    o3 = calculate_o3(day, ws.t2m_c, np.maximum(transported_nox, 0), day, southern_valley=True)

    # secondary PM from VOC & NOx
    sec = calculate_secondary_pm(np.maximum(voc_emission, 0), np.maximum(transported_nox, 0), day, dt_h=1.0)

    # aerosol growth (hygroscopic): applies to PM2.5, PM10, partly O3? (not O3)
    growth = calculate_aerosol_growth(ws.rh_pct, ws.t2m_c)
    pm25 = np.clip(transported_pm25, 0, None) * growth + 0.4 * sec
    pm10 = np.clip(transported_pm10, 0, None) * growth + 0.5 * sec * 1.6
    nox = np.clip(transported_nox + 0.2 * sec, 0, None)
    so2 = np.clip(so2_emission, 0, None)
    co = np.clip(co_emission, 0, None)

    return ChemistryResult(
        hours=ws.hours, pm25=pm25, pm10=pm10, o3=o3, nox=nox, so2=so2,
        co=co, secondary_pm=sec, aerosol_growth=growth,
    )


def dayness_flat(ws: WeatherState) -> np.ndarray:
    from wrf_chem_delhi.weather.engine import dayness

    if not ws.hours:
        return np.zeros((0, ws.ny, ws.nx))
    d = dayness(ws.hours, float(ws.lon_nodes.mean()), float(ws.lat_nodes.mean()))
    return d[:, None, None] * np.ones((ws.nt, ws.ny, ws.nx))