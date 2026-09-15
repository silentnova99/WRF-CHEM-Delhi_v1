"""Indian AQI calculator (CPCB / MoEF&CC 2014 breakpoints).

Implements the official piecewise-linear AQI sub-index for PM2.5, PM10,
O3, NO2, SO2, CO, aggregates the overall AQI as the max sub-index, and
labels category + dominant pollutant + health message (Indian AQI 6 bands).
"""

from __future__ import annotations

from dataclasses import dataclass

# (C_low, C_high, I_low, I_high) per pollutant, 6 bands (CPCB 2014)
BREAKPOINTS = {
    "PM2.5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
              (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    "PM10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
             (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
    "O3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
           (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
    "NO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
            (181, 280, 201, 300), (281, 400, 301, 400), (401, 1000, 401, 500)],
    "SO2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
            (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2000, 401, 500)],
    "CO": [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10.0, 101, 200),
           (10.1, 17.0, 201, 300), (17.1, 34.0, 301, 400), (34.1, 100.0, 401, 500)],
}

AQI_BUCKETS = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]

HEALTH_MESSAGES = {
    "Good": "No health risk. Safe for all outdoor activity.",
    "Satisfactory": "Minor discomfort for hypersensitive individuals.",
    "Moderate": "Sensitive people should limit prolonged outdoor exertion.",
    "Poor": "General public may experience mild to moderate symptoms.",
    "Very Poor": "Most people may experience health effects; reduce outdoor activity.",
    "Severe": "Health alert: everyone may experience serious health effects.",
}


@dataclass
class AqiScore:
    value: float
    category: str
    dominant_pollutant: str
    sub_indices: dict[str, int] | None = None
    health_message: str | None = None


def sub_index(pollutant: str, concentration: float) -> float:
    """Piecewise-linear AQI sub-index for one pollutant.

    Concentration units follow the CPCB formula: µg/m³ for PM2.5/PM10/O3/NO2/
    SO2 and **mg/m³** for CO. Callers pass µg/m³ everywhere (model convention);
    CO is normalised automatically.
    """
    if concentration < 0:
        concentration = 0.0
    if pollutant == "CO":
        concentration = concentration / 1000.0  # µg/m³ → mg/m³
    bp = BREAKPOINTS[pollutant]
    for (clo, chi, ilo, ihi) in bp:
        if concentration <= chi:
            if chi == clo:
                return float(ilo)
            ratio = (concentration - clo) / (chi - clo)
            return float(ilo + ratio * (ihi - ilo))
    # beyond last band: clamp at 500 (ceiling by CPCB formula)
    return 500.0


def aqi_from_concentrations(concentrations: dict) -> AqiScore:
    """Compute Indian AQI from a dict {pollutant: concentration}."""
    sub = {}
    for pol, conc in concentrations.items():
        if pol in BREAKPOINTS and conc is not None:
            sub[pol] = round(sub_index(pol, float(conc)))
    if not sub:
        return AqiScore(value=0.0, category="Good", dominant_pollutant="", sub_indices={})

    overall = max(sub.values())
    dominant = max(sub, key=sub.get)
    category = bucket_of(overall)
    return AqiScore(
        value=float(overall),
        category=category,
        dominant_pollutant=dominant,
        sub_indices=sub,
        health_message=HEALTH_MESSAGES[category],
    )


def bucket_of(value: float) -> str:
    for lo, hi, name in AQI_BUCKETS:
        if lo <= value <= hi:
            return name
    return "Severe"


def map_metrics_to_pollutants(concentrations: dict) -> dict:
    """Alias keys PM25→PM2.5, NOX→NO2, SO2, CO, O3, PM10."""
    aliases = {"PM25": "PM2.5", "pm25": "PM2.5", "pm10": "PM10", "o3": "O3",
               "nox": "NO2", "so2": "SO2", "co": "CO", "NOX": "NO2"}
    out = {}
    for k, v in concentrations.items():
        key = aliases.get(k, k)
        out[key] = v
    return out