"""V2 configuration: emission factors + forecast defaults.

Emission factors live in ``configs/emission_factors.yaml`` (already created);
here we expose typed access plus the scenario/profile defaults used by the
engines. Remains 100% offline-capable.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path

import yaml

from aqf_delhi.config import PROJECT_ROOT

DEFAULT_EF_PATH = PROJECT_ROOT / "configs" / "emission_factors.yaml"

MODEL_VERSION = "2.0.0"
MODEL_NAME = "WRF-Chem-inspired reduced-order atmospheric model"
ALLOWED_SCENARIOS = ("normal_winter", "strong_inversion", "stubble_plume")
SPECIES = ["pm25", "pm10", "o3", "nox", "so2", "co"]


class EmissionFactors:
    def __init__(self, data: dict | None = None, path: Path | None = None):
        self.path = Path(path or DEFAULT_EF_PATH)
        if data is None:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        self.data = data

    @property
    def factors(self) -> dict:
        return self.data.get("emission_factors", {})

    def factor(self, species: str, default: float = 0.0) -> float:
        return float(self.factors.get(species, default))

    @property
    def frp_to_fuel_kg_per_mj(self) -> float:
        return float(self.data.get("frp_to_fuel_kg_per_mj", 0.368))

    @property
    def secondary_soa_yield(self) -> float:
        return float(self.data.get("secondary_organic_aerosol_yield", 0.15))


@dataclass
class ForecastProfile:
    scenario: str = "normal_winter"
    fhrs: list[int] = dc_field(default_factory=lambda: list(range(0, 73)))
    seed: int = 11
    min_pbl_m: float = 120.0


def load_emission_factors(path: Path | None = None) -> EmissionFactors:
    return EmissionFactors(path=path)


def validate_scenario(scenario: str) -> str:
    if scenario not in ALLOWED_SCENARIOS:
        raise ValueError(f"scenario must be one of {ALLOWED_SCENARIOS}; got {scenario!r}")
    return scenario