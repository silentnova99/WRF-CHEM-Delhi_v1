"""Coupled forecast engine: the main orchestrator.

Chains: Weather → Inversion → Fire → Plume → Transport → Chemistry → Feedback
→ T+1...T+72.

The feedback loop closes at each forecast step: aerosol-radiation feedback
modifies the PBL height for the *next* hour, which changes transport mixing,
which changes PM2.5 → next feedback. This is the coupled signature.

SCENARIOS
---------
  normal_winter    | moderate inversion, active winds
  strong_inversion | deep inversion, stagnant air
  stubble_plume    | fire-driven smoke transport into NCR

DATA SOURCES
------------
DEMO mode uses deterministic synthetic weather + fires (offline, CPU only).
HISTORICAL / LIVE modes are wired through ``wrf_chem_delhi.data_modes`` when
a legacy ``MetArrays``/CachedState is supplied; everything downstream is
identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from wrf_chem_delhi.weather.engine import (
    build_default_grid, make_hours, synthetic_weather,
)
from wrf_chem_delhi.inversion.engine import compute_from_weather
from wrf_chem_delhi.fire.engine import StubbleEngine
from wrf_chem_delhi.plume.engine import PlumeModel
from wrf_chem_delhi.transport.engine import TransportModel, SPECIES
from wrf_chem_delhi.chemistry.engine import run_chemistry
from wrf_chem_delhi.feedback.engine import compute_feedback

logger = logging.getLogger(__name__)

MODEL_VERSION = "2.0.0"
MODEL_NAME = "WRF-Chem-inspired reduced-order coupled atmospheric model"


@dataclass
class ForecastSnapshot:
    time: datetime
    pm25: np.ndarray
    pm10: np.ndarray
    o3: np.ndarray
    nox: np.ndarray
    so2: np.ndarray
    co: np.ndarray
    pblh_m: np.ndarray
    inversion_strength: np.ndarray
    trapping_idx: np.ndarray
    vent_idx: np.ndarray
    fire_influence: np.ndarray
    plume_concentration: np.ndarray
    weather: dict
    feedback_strength: np.ndarray
    secondary_pm: np.ndarray


@dataclass
class ForecastRun:
    run_id: str
    init: datetime
    snapshots: list[ForecastSnapshot]
    lat_nodes: np.ndarray
    lon_nodes: np.ndarray
    ny: int
    nx: int
    data_source: str
    model_name: str = MODEL_NAME
    model_version: str = MODEL_VERSION
    scenario: str = "normal_winter"

    def json(self) -> dict:
        return {
            "run_id": self.run_id,
            "init": self.init.isoformat(),
            "n_hours": len(self.snapshots),
            "ny": self.ny,
            "nx": self.nx,
            "lat_nodes": [round(float(v), 3) for v in self.lat_nodes],
            "lon_nodes": [round(float(v), 3) for v in self.lon_nodes],
            "data_source": self.data_source,
            "model_version": self.model_version,
            "scenario": self.scenario,
            "model_name": self.model_name,
            "label": "Reduced-order physics engine (NOT operational WRF-Chem).",
        }

    def field(self, key: str, hi: int) -> np.ndarray:
        return np.asarray(getattr(self.snapshots[hi], key), dtype=float)

    def station_timeseries(self, station_lat: float, station_lon: float) -> dict:
        j = int(np.argmin(np.abs(self.lat_nodes - float(station_lat))))
        i = int(np.argmin(np.abs(self.lon_nodes - float(station_lon))))
        s = {}
        for key in ("pm25", "pm10", "o3", "nox", "so2", "co", "pblh_m",
                    "inversion_strength", "trapping_idx", "fire_influence",
                    "feedback_strength", "secondary_pm"):
            s[key] = [float(np.asarray(getattr(sn, key))[j, i]) for sn in self.snapshots]
        s["hours"] = [sn.time.isoformat() for sn in self.snapshots]
        s["station_lat"] = float(self.lat_nodes[j])
        s["station_lon"] = float(self.lon_nodes[i])
        return s

    def domain_mean_series(self, key: str) -> list[float]:
        return [float(np.nanmean(np.asarray(getattr(sn, key), dtype=float))) for sn in self.snapshots]

    def lead_map(self, key: str, lead_h: int) -> dict:
        """Concentration field at a requested lead hour, nearest-index match."""
        idx = min(int(np.argmin([abs((sn.time - self.init).total_seconds() / 3600 - lead_h)
                                  for sn in self.snapshots])), len(self.snapshots) - 1)
        arr = np.asarray(getattr(self.snapshots[idx], key), dtype=float)
        return {
            "lead_h": lead_h,
            "time": self.snapshots[idx].time.isoformat(),
            "ny": self.ny,
            "nx": self.nx,
            "values": [round(float(v), 2) for v in arr.ravel()][: self.nx * self.ny],
        }


class CoupledForecastEngine:
    """Runs the coupled chain over an hourly sequence.

    Parameters
    ----------
    grid : aqf_delhi.domain.Grid (defaults to Delhi NCR 4 km)
    scenario : str (normal_winter | strong_inversion | stubble_plume)
    init : datetime UTC
    fhrs : forecast lead hours (default 0..72 step 1 → cheap on CPU)
    seed : RNG seed
    use_feedback : bool (aerosol-met feedback coupling on/off) — ablation knob
    """

    def __init__(
        self,
        grid=None,
        *,
        scenario: str = "normal_winter",
        init: datetime = None,
        fhrs: list[int] = None,
        seed: int = 11,
        use_feedback: bool = True,
    ):
        self.grid = grid or build_default_grid()
        self.scenario = scenario
        self.init = init or datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
        self.fhrs = fhrs or list(range(0, 73))
        self.seed = seed
        self.use_feedback = use_feedback

    # ------------------------------------------------------------------ #
    def run(self, weather_state: Optional[object] = None,
            legacy_met=None) -> ForecastRun:
        hours = make_hours(self.init, self.fhrs)
        run_id = f"v2.{self.init:%Y%m%d.%H%M}.{self.scenario[:4]}"

        # weather (DEMO default; LIVE/CACHE swaps weather_state in)
        ws = weather_state or synthetic_weather(
            self.grid, hours, seed=self.seed, scenario=self.scenario
        )

        # inversion
        inv = compute_from_weather(ws)

        # fire + plume
        fe = StubbleEngine(self.grid, seed=self.seed)
        fires = (
            fe.load_fires(hours, mode="demo")
            if weather_state is None
            else fe.load_fires(hours, mode="live")
        )
        fs = fe.compute_state(hours, fires, ws.ws10_ms, ws.wd_deg)
        pl = PlumeModel(self.grid)
        ps = pl.compute_state(hours, fires, ws)

        # emission masks (g/h arrays per hour across grid)
        emis_fire = fe.emissions_grid(hours, fires, fs.influence)
        ef = fe.ef
        urban_pm25_gph = ef.get("urbgum_base_kg_km2_day", 180.0) * 1000.0 / 24.0 * (
            self.grid.lon_res_km() * self.grid.lat_res_km()
        )
        urban = np.full((len(hours), self.grid.ny, self.grid.nx), urban_pm25_gph)
        for t in range(len(hours)):
            hr = hours[t].hour
            diurnal = np.clip(0.4 + 1.0 * np.sin((hr - 6) / 24.0 * 2 * np.pi), 0.35, 1.4)
            urban[t] *= diurnal

        # Scenario-dependent fire intensity: SC3's defining feature is strong
        # fire transport; SC2's is stagnation (fires secondary). The absolute
        # scale is tuned so DEMO peak PM2.5 lands in observed Delhi NCR bands
        # (SC1 150–250, SC2 350–600, SC3 400–700).
        fire_factor = {"normal_winter": 0.22,
                       "strong_inversion": 0.16,
                       "stubble_plume": 0.30}[self.scenario]
        emis_fire = {k: np.asarray(v, dtype=float) * fire_factor for k, v in emis_fire.items()}

        emits = {
            "pm25": urban + emis_fire["pm25"],
            "pm10": urban * 1.55 + emis_fire["pm10"],
            "nox": emis_fire["nox"] + urban * 0.05,
            "so2": emis_fire["so2"] + urban * 0.012,
            "co": emis_fire["co"] + urban * 0.8,
            "o3": np.zeros((len(hours), self.grid.ny, self.grid.nx)),
        }
        # Grid-representativeness: a 1-h emission impulse is mixed over a few
        # cells + the full column before it becomes an hourly box concentration.
        # Without this factor the cell-average of the FRP-derived mass sits ~4x
        # above the observed Delhi NCR winter bands.
        DILUTION_FACTOR = 0.25
        for _sp in ("pm25", "pm10", "nox", "so2", "co"):
            emits[_sp] = np.asarray(emits[_sp], dtype=float) * DILUTION_FACTOR

        # transport model seeded once
        tm = TransportModel(self.grid, seed=self.seed)

        # initial concentration (previous state) = urban + fire at hour 0
        from wrf_chem_delhi.transport.engine import MIX_DEPTH_FLOOR as _MIX_FLOOR
        init_depth = max(float(ws.pblh_m[0].max()), _MIX_FLOOR)
        c_prev = {
            "pm25": emits["pm25"][0] / init_depth / (self.grid.lon_res_km() * self.grid.lat_res_km() * 1e6) * 1e6,
            "pm10": emits["pm10"][0] / init_depth / (self.grid.lon_res_km() * self.grid.lat_res_km() * 1e6) * 1e6,
            "nox": emits["nox"][0] / init_depth / (self.grid.lon_res_km() * self.grid.lat_res_km() * 1e6) * 1e6,
            "so2": emits["so2"][0] / init_depth / (self.grid.lon_res_km() * self.grid.lat_res_km() * 1e6) * 1e6,
            "co": emits["co"][0] / init_depth / (self.grid.lon_res_km() * self.grid.lat_res_km() * 1e6) * 1e6,
            "o3": np.full((self.grid.ny, self.grid.nx), 38.0),
        }

        pblh = ws.pblh_m.copy()
        snapshots: list[ForecastSnapshot] = []

        for t in range(len(hours)):
            # transport one step
            c_new = tm.step(
                c_prev, ws, t, emits,
                pblh_m=pblh[t] if pblh.ndim == 3 else pblh,
                trapping_idx=inv.trapping_idx[t] if inv.trapping_idx.ndim == 3 else inv.trapping_idx,
            )
            # chemistry prep: emissions-driven secondary inputs
            from wrf_chem_delhi.chemistry.engine import (
                calculate_o3, calculate_secondary_pm, calculate_aerosol_growth,
            )
            # reduced chemistry application
            day = _day_arr(ws, t)
            # O3: relax toward photochemical balance in the transported box
            nox_arr = np.maximum(c_new["nox"], 0)
            # secondary PM from VOC & NOx emissions (g/h → µg/m³ over the PBL box)
            cell_m2 = self.grid.lon_res_km() * self.grid.lat_res_km() * 1e6
            voc_conc = np.asarray(emis_fire["voc"][t], dtype=float) / np.maximum(
                pblh[t] * cell_m2, 1.0) * 1e6
            sec = calculate_secondary_pm(voc_conc, nox_arr, day, dt_h=1.0)
            growth = calculate_aerosol_growth(ws.rh_pct[t], ws.t2m_c[t])
            pm25_c = np.clip(c_new["pm25"], 0, None) * growth + 0.35 * sec
            pm10_c = np.clip(c_new["pm10"], 0, None) * growth + 0.45 * sec * 1.6
            o3_c = calculate_o3(day, ws.t2m_c[t], nox_arr, day, True)
            nox_c = np.clip(nox_arr + 0.2 * sec, 0, None)
            so2_c = np.clip(c_new["so2"], 0, None)
            co_c = np.clip(c_new["co"], 0, None)

            # aerosol-met feedback
            if self.use_feedback:
                fb = compute_feedback(pm25_c, pblh[t])
                pblh_adj = fb.pblh_adjusted
                pm25_c = fb.pm25_coupled
                fb_strength = fb.feedback_strength
            else:
                pblh_adj = pblh[t]
                fb_strength = np.zeros_like(pm25_c)
            pblh[t] = pblh_adj

            snapshots.append(ForecastSnapshot(
                time=hours[t],
                pm25=pm25_c,
                pm10=pm10_c,
                o3=o3_c,
                nox=nox_c,
                so2=so2_c,
                co=co_c,
                pblh_m=pblh_adj,
                inversion_strength=inv.strength_k[t] if inv.strength_k.ndim == 3 else inv.strength_k,
                trapping_idx=inv.trapping_idx[t] if inv.trapping_idx.ndim == 3 else inv.trapping_idx,
                vent_idx=ws.ventilation_idx[t] if ws.ventilation_idx is not None else np.zeros_like(pm25_c),
                fire_influence=fs.influence[t] if fs.influence.ndim == 3 and t < fs.influence.shape[0] else np.zeros_like(pm25_c),
                plume_concentration=ps.concentration[t] if ps.concentration.ndim == 3 and t < ps.concentration.shape[0] else np.zeros_like(pm25_c),
                weather=ws.to_json(t),
                feedback_strength=fb_strength,
                secondary_pm=sec,
            ))
            c_prev = {"pm25": pm25_c, "pm10": pm10_c, "o3": o3_c, "nox": nox_c,
                      "so2": so2_c, "co": co_c}

        return ForecastRun(
            run_id=run_id,
            init=self.init,
            snapshots=snapshots,
            lat_nodes=ws.lat_nodes,
            lon_nodes=ws.lon_nodes,
            ny=ws.ny,
            nx=ws.nx,
            data_source=ws.data_source,
            scenario=self.scenario,
        )


def _day_arr(ws, t: int) -> np.ndarray:
    from wrf_chem_delhi.weather.engine import dayness

    d = dayness([ws.hours[t]], float(ws.lon_nodes.mean()), float(ws.lat_nodes.mean()))
    return np.full((ws.ny, ws.nx), d[0])