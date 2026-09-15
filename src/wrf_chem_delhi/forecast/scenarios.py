"""Three demo scenarios for the SIH presentation.

SCENARIO 1:  Normal winter in Delhi — moderate pollution, active NCR winds,
              occasional mild inversion at night.

SCENARIO 2:  Strong multi-day inversion — stagnant air, shallow PBL,
              trapping layer prevents ventilation, PM builds progressively.

SCENARIO 3:  Stubble-burning plume approaching Delhi — active Punjab/Haryana
              fires, NW winds transport a visible plume southward into the NCR.
"""

SCENARIOS = {
    "normal_winter": {
        "id": "normal_winter",
        "label": "Scenario 1: Normal winter",
        "description": (
            "A typical December day in Delhi with moderate regional pollution, "
            "variable N-NW winds, and a shallow nighttime inversion that "
            "dissipates by late morning. Traffic + urban emission baseline."
        ),
        "weather": "normal_winter",
        "forecast": "normal_winter",
        "expected_peak_pm25": [150, 250],
    },
    "strong_inversion": {
        "id": "strong_inversion",
        "label": "Scenario 2: Strong inversion + stagnant air",
        "description": (
            "A persistent temperature inversion traps pollutants near the "
            "surface. PBL stays below 300 m throughout the day; winds below "
            "1.5 m/s. PM2.5 builds over 72 h with no ventilation."
        ),
        "weather": "strong_inversion",
        "forecast": "strong_inversion",
        "expected_peak_pm25": [350, 600],
    },
    "stubble_plume": {
        "id": "stubble_plume",
        "label": "Scenario 3: Stubble-burning plume approaching Delhi",
        "description": (
            "Active crop-residue burning across Punjab and Haryana. A steady "
            "NW wind channels FRP-heated plumes southward over Delhi NCR. "
            "Sharp PM2.5 rise coincides with peak plume arrival."
        ),
        "weather": "stubble_plume",
        "forecast": "stubble_plume",
        "expected_peak_pm25": [400, 700],
    },
}


def get_scenario(scenario_id: str) -> dict:
    if scenario_id not in SCENARIOS:
        raise ValueError(f"unknown scenario '{scenario_id}'; available: {list(SCENARIOS.keys())}")
    return SCENARIOS[scenario_id]


def list_scenarios() -> list[dict]:
    return list(SCENARIOS.values())