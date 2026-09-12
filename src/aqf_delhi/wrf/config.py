"""Module-2 configuration (typed, from ``configs/module2.yaml``)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from aqf_delhi.config import PROJECT_ROOT

DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


class GfsSubregion(BaseModel):
    top_lat: float
    bottom_lat: float
    left_lon: float
    right_lon: float


class GfsFilterConfig(BaseModel):
    filter_base: str
    subregion: GfsSubregion
    fhrs: list[int] = Field(default_factory=lambda: list(range(0, 73, 3)))
    levels: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def variables(self) -> list[str]:
        out: list[str] = []
        for _, vars_ in self.levels.items():
            out += [v for v in vars_ if v not in out]
        return out


class EmissionsConfig(BaseModel):
    pm25_ef: float = Field(gt=0.0)
    pm10_ratio: float = Field(gt=1.0)
    urban_base_kg_km2_day: float = Field(ge=0.0)
    fire_min_frp_mw: float = Field(ge=0.0)
    fire_lifetime_h: float = Field(gt=0.0)


class PlumeConfig(BaseModel):
    model: str = "freitas07"
    entrainment_coef: float = Field(gt=0.0, le=1.0)
    sensible_fraction: float = Field(gt=0.0, le=1.0)
    dr_dz: float = Field(gt=0.0)
    z_step: float = Field(gt=0.0)
    max_top_m: float = Field(gt=0.0)
    n_layers: int = Field(gt=0)


class EmulatorConfig(BaseModel):
    seed: int
    decay_rate_h: float = Field(ge=0.0)
    wet_removal_per_rh01: float = Field(ge=0.0)
    diffusion_sigma_km: float = Field(gt=0.0)
    advect_cap_h: int = Field(gt=0)
    inversion_bias_mult: float = Field(ge=0.1, le=1.5)
    urban_offset_ug_m3: float = Field(ge=0.0)
    obs_noise_ug_m3: float = Field(ge=0.0)


class ChemistryConfig(BaseModel):
    species: list[str]
    cp_air: float = Field(gt=0.0)
    g_const: float = Field(gt=0.0)
    mair: float = Field(gt=0.0)


class ArtifactConfig(BaseModel):
    out_root: str = "data/coupled"


class Module2Config(BaseModel):
    gfs: GfsFilterConfig
    emissions: EmissionsConfig
    plume: PlumeConfig
    emulator: EmulatorConfig
    chemistry: ChemistryConfig
    artifact: ArtifactConfig = ArtifactConfig()

    @model_validator(mode="after")
    def _defaults(self) -> "Module2Config":
        if not self.emulator.seed:
            self.emulator.seed = 11
        return self

    @property
    def species(self) -> list[str]:
        return self.chemistry.species

    @classmethod
    def from_yaml_default(cls, config_dir: Path = DEFAULT_CONFIG_DIR) -> "Module2Config":
        return load_module2(config_dir)


def load_module2(config_dir: Path = DEFAULT_CONFIG_DIR) -> Module2Config:
    path = config_dir / "module2.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Module-2 config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file must contain a mapping: {path}")
    return Module2Config(**data)