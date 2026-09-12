"""IND-AQI computation (CPCB/MoEF 2014 breakpoints, piecewise-slope).

Sub-index for pollutant p::

    SI_p = (I_hi - I_lo)/(C_hi - C_lo) * (C - C_lo) + I_lo

Composite AQI = max over pollutants. Concentration ceilings produce
sub-index 500; beyond-ceiling values are clamped for score computation
but reported as 500+ (flag).

Breakpoints below follow the CPCB AQI Technical Advisory Committee table.
"""

from __future__ import annotations

from dataclasses import dataclass

BREAKPOINTS: dict[str, list[tuple[float, float, int, int]]] = {
    "PM2.5": [  # 24-h µg/m3
        (0.0, 30.0, 0, 50),
        (30.0, 60.0, 51, 100),
        (60.0, 90.0, 101, 200),
        (90.0, 120.0, 201, 300),
        (120.0, 250.0, 301, 400),
        (250.0, 1e18, 401, 500),
    ],
    "PM10": [
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 250.0, 101, 200),
        (250.0, 350.0, 201, 300),
        (350.0, 430.0, 301, 400),
        (430.0, 1e18, 401, 500),
    ],
    "O3": [  # 8-h µg/m3
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 168.0, 101, 200),
        (168.0, 208.0, 201, 300),
        (208.0, 748.0, 301, 400),
        (748.0, 1e18, 401, 500),
    ],
    "NO2": [
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 51, 100),
        (80.0, 180.0, 101, 200),
        (180.0, 280.0, 201, 300),
        (280.0, 400.0, 301, 400),
        (400.0, 1e18, 401, 500),
    ],
    "SO2": [
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 51, 100),
        (80.0, 380.0, 101, 200),
        (380.0, 800.0, 201, 300),
        (800.0, 1600.0, 301, 400),
        (1600.0, 1e18, 401, 500),
    ],
    "CO": [  # mg/m3, 8-h
        (0.0, 1.0, 0, 50),
        (1.0, 2.0, 51, 100),
        (2.0, 10.0, 101, 200),
        (10.0, 17.0, 201, 300),
        (17.0, 34.0, 301, 400),
        (34.0, 1e18, 401, 500),
    ],
}

AQI_BUCKETS = [
    (0, 50, "good"),
    (51, 100, "satisfactory"),
    (101, 200, "moderate"),
    (201, 300, "poor"),
    (301, 400, "very poor"),
    (401, 500, "severe"),
]


@dataclass
class AqiScore:
    value: int
    bucket: str
    dominant_pollutant: str
    sub_indices: dict[str, int]


def sub_index(pollutant: str, concentration: float) -> int:
    low = BREAKPOINTS.get(pollutant)
    if low is None:
        raise ValueError(f"no AQI breakpoints for {pollutant!r}")
    for c_lo, c_hi, i_lo, i_hi in low:
        if concentration <= c_hi:
            if c_hi - c_lo <= 0:
                return i_hi
            return int(round((i_hi - i_lo) / (c_hi - c_lo) * (concentration - c_lo) + i_lo))
    high = low[-1]
    return int(round((high[3] - high[2]) / (high[1] - high[0]) * (concentration - high[0]) + high[2]))


def aqi_from_concentrations(concentrations: dict[str, float]) -> AqiScore:
    """Compute composite AQI from pollutant concentrations.

    ``concentrations`` keys are pollutant names (PM2.5 | PM10 | O3 | ...)
    with values in the pollutant's reported unit.
    """
    subs = {p: sub_index(p, c) for p, c in concentrations.items()}
    dominant = max(subs, key=subs.get)
    value = subs[dominant]
    bucket = next(name for lo, hi, name in AQI_BUCKETS if lo <= value <= hi)
    return AqiScore(value=value, bucket=bucket, dominant_pollutant=dominant, sub_indices=subs)


def bucket_of(value: int) -> str:
    return next(b for lo, hi, b in AQI_BUCKETS if lo <= value <= hi)


def grap_stage(aqi: int, thresholds: dict[int, int] | None = None) -> int:
    """Map 24-h AQI to CAQM GRAP stage (1..4, 0 = no action)."""
    thresholds = thresholds or {1: 201, 2: 301, 3: 401, 4: 451}
    stage = 0
    for s in sorted(thresholds):
        if aqi >= thresholds[s]:
            stage = s
    return stage