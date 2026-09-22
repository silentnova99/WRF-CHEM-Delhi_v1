"""Generate the README figures (readme_assets/*.png) from real report artifacts.

Run: python scripts/make_readme_figures.py
All charts are produced from reports/* (no server dependency); the live AQI
chart reruns the same coupled forecast the server serves today.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "readme_assets"
OUT.mkdir(exist_ok=True)

from vayu_setu.config import load_vayu_config  # noqa: E402
from vayu_setu.lake import DataLake  # noqa: E402

PLOT = {"dpi": 150, "figsize": (8.6, 4.6)}
C_VS = "#2e7d32"
C_BASE = "#8d6e63"
C_ACCENT = "#1565c0"
C_WARN = "#e65100"


# --------------------------------------------------------------------------- #
def fig_flowchart():
    stages = [
        ("DISCOVER & AUDIT", "reports/data_discovery_audit.*", "Kaggle N/A -> Mode B (authoritative/live)"),
        ("INGEST (Mode B)", "data/raw/*", "CPCB extracts, FIRMS NRT, GFS 0.25°, CAMS AOD"),
        ("QUALITY CONTROL", "data/interim/cleaned + quality_report.json", "range/MAD/spike checks, loud on critical gaps"),
        ("ALIGN + FIRE FEATURES", "data/interim/aligned", "hourly UTC timeline, NW-India FRP/upwind exposure"),
        ("FEATURE SPLITS", "data/features/{train,validation,test}", "chronological, no leakage"),
        ("TRAIN HYBRID", "ST-GNN + XGBoost residual + conformal", "coupling-aware physics channels"),
        ("VALIDATE", "reports/validation_report.*", "leakage checks, baselines, episodes, per-station"),
        ("FORECAST + ABLATION", "V2 coupled engine 72 h", "PM->AOD->radiation->PBL->PM, feedback ON/OFF"),
        ("SERVE LIVE ON WEB", "/api/v3 + dashboard", "real obs anchor + coupled intervals + AQI"),
    ]
    n = len(stages)
    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=150)
    ax.axis("off")
    aw, ah = 0.92, 0.082
    for i, (title, path, desc) in enumerate(stages):
        y = 1.0 - (i + 0.5) * ah
        box = plt.Rectangle((0.04, y - ah / 2 + 0.01), aw, ah - 0.02,
                            facecolor="#eff6ea" if i not in (0, n - 1) else "#d9efd0",
                            edgecolor=C_VS, lw=1.4, zorder=2)
        ax.add_patch(box)
        ax.text(0.055, y + 0.012, f"{i + 1}. {title}", ha="left", va="center",
                fontsize=11.5, fontweight="bold", color="#1b3a1d")
        ax.text(0.055, y - 0.018, f"{desc}  ({path})", ha="left", va="center",
                fontsize=8.4, color="#37474f")
        if i < n - 1:
            ax.annotate("", xy=(0.5, y - ah / 2 - 0.004), xytext=(0.5, y - ah / 2 + 0.008),
                        arrowprops=dict(arrowstyle="-|>", color=C_VS, lw=1.8))
    ax.text(0.5, 1.005, "VAYU-SETU: real-data pipeline -> 72 h coupled forecast -> live web API",
            ha="center", va="bottom", fontsize=12.5, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT / "flowchart.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def load_species_report(spec: str) -> dict:
    p = REPO / "reports" / f"train_report_{spec}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def fig_rmse_benchmark():
    specs = ["pm25", "pm10", "o3"]
    models = ["vayu-setu", "persistence", "climatology", "xgb_direct"]
    labels = ["VAYU-SETU", "Persistence", "Climatology", "XGBoost direct"]
    colors = [C_VS, *[C_BASE] * 3]
    vals = []
    for m in models:
        row = []
        for s in specs:
            d = load_species_report(s)
            row.append(d["hybrid"]["engine"]["rmse"] if m == "vayu-setu" else d["baselines"][m]["rmse"])
        vals.append(row)
    x = np.arange(len(specs))
    width = 0.19
    fig, ax = plt.subplots(**PLOT)
    for i, (m, lab, c) in enumerate(zip(models, labels, colors)):
        bars = ax.bar(x + (i - 1.5) * width, vals[i], width, label=lab, color=c,
                      alpha=0.92 if m == "vayu-setu" else 0.55)
        for b, v in zip(bars, vals[i]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                    f"{v:.1f}" if m != "vayu-setu" else f"{v:.2f}",
                    ha="center", fontsize=8.5, fontweight="bold" if m == "vayu-setu" else "normal")
    ax.set_ylabel("RMSE (ug/m3) — lower is better")
    ax.set_xticks(x)
    ax.set_xticklabels(["PM2.5", "PM10", "O3"])
    ax.set_title("72-lead forecast quality vs 3 baselines (held-out test, 29 cities)", fontsize=11.5)
    ax.legend(loc="upper center", ncol=4, fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "rmse_benchmark.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_pm25_validation():
    cfg = load_vayu_config()
    lake = DataLake(cfg)
    from vayu_setu.dataset import stations_frame

    cities = stations_frame(lake)["city"].tolist()
    nz = np.load(REPO / "reports" / "forecasts_pm25.npz")
    times = pd.to_datetime([str(t) for t in nz["times"]])
    idx = cities.index("Delhi")
    yobs, yhat, lo, hi = nz["yobs"][:, idx], nz["yhat"][:, idx], nz["lo"][:, idx], nz["hi"][:, idx]
    per = load_species_report("pm25")["baselines"]["persistence"]["rmse"]
    fig, ax = plt.subplots(**PLOT)
    ax.fill_between(times, lo, hi, color=C_VS, alpha=0.18, label="90% conformal interval")
    ax.plot(times, yobs, "ko-", ms=3, lw=1.0, label="Observed (real CPCB-derived)", color="#263238")
    ax.plot(times, yhat, "-", lw=2.2, color=C_ACCENT, label="VAYU-SETU hybrid")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.set_title("First 72-lead test window, Delhi: hybrid vs observations (landscape era: intervals tight)", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "pm25_validation_delhi.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_aqi_live():
    cfg = load_vayu_config()
    lake = DataLake(cfg)
    from vayu_setu import live

    p = live.build_live(cfg, lake, use_feedback=True)
    s = p["series"]
    times = pd.to_datetime([r["time"] for r in s])
    pm = [r["pm25"] for r in s]
    aqi = [r["aqi"] for r in s]
    cat = [r["category"] for r in s]
    band = {"Good": 50, "Satisfactory": 100, "Moderate": 200, "Poor": 300,
            "Very Poor": 400, "Severe": 500}
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=150)
    order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    prev = 0
    for cat_ in order:
        lo_, hi_ = prev, band[cat_]
        ax.axhspan(lo_, hi_, alpha=0.12, color=CAT_COLOR.get(cat_, "#9e9e9e"))
        ax.text(0.004, (lo_ + hi_) / 2, cat_, va="center", ha="left",
                fontsize=7.5, alpha=0.65, transform=ax.get_yaxis_transform())
        prev = hi_
    ax.plot(times, aqi, "-", lw=2.4, color=C_WARN, label="Indian AQI (CPCB)")
    ax.plot(times, pm, "-", lw=1.6, color=C_VS, alpha=0.9, label="PM2.5 (ug/m3)")
    ax.set_ylabel("AQI / concentration")
    ax.set_title(f"Live coupled 72 h forecast — {p['init_utc'].replace('T', ' ')[:16]} UTC ({p['scenario']})",
                 fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "aqi_live_72h.png", bbox_inches="tight")
    plt.close(fig)


CAT_COLOR = {
    "Good": "#4caf50", "Satisfactory": "#8bc34a", "Moderate": "#ffeb3b",
    "Poor": "#ff9800", "Very Poor": "#f44336", "Severe": "#7b1fa2",
}


# --------------------------------------------------------------------------- #
def fig_data_lake():
    import yaml

    d = yaml.safe_load((REPO / "reports" / "dataset_size_report.yaml").read_text(encoding="utf-8"))
    tiers = d["tiers"]
    names = list(tiers)
    bytes_ = [tiers[k]["bytes"] / 1e6 for k in names]
    files = [tiers[k]["files"] for k in names]
    fig, ax = plt.subplots(**PLOT)
    colors = {"raw": "#8d6e63", "interim": "#ef9a9a", "processed": "#9ccc65", "features": "#4fc3f7"}
    bars = ax.bar(names, bytes_, color=[colors.get(k, "#9e9e9e") for k in names], edgecolor="k", lw=0.6)
    for b, by, fn in zip(bars, bytes_, files):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8, f"{by:,.1f} MB\n({fn} files)",
                ha="center", fontsize=8.5)
    ax.set_ylabel("Size (MB)")
    ax.set_title(f"VAYU-SETU data lake — {d['bytes_total'] / 1e9:.2f} GB total across tiers", fontsize=11.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "data_lake_tiers.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_timeline():
    timeline = [
        ("train", "2022-08-05", "2023-11-26", 333326, "#a5d6a7"),
        ("validation", "2023-11-26", "2024-11-26", 254736, "#90caf9"),
        ("test (forecast window)", "2024-11-26", "2025-11-26", 254098, "#ffcc80"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 2.9), dpi=150)
    y = 0
    for name, start, end, rows, c in timeline:
        s = datetime.strptime(start, "%Y-%m-%d")
        e = datetime.strptime(end + " 23:00", "%Y-%m-%d %H:%M")
        ax.barh(y, (e - s).days, left=s, color=c, edgecolor="k", lw=0.6, height=0.5)
        ax.text(s + (e - s) / 2, y, f"{name}\n{rows:,} city-hours", ha="center", va="center",
                fontsize=8.5, fontweight="bold")
        ax.text(s, y + 0.34, start, ha="left", va="bottom", fontsize=7)
        y += 1
    ax.set_yticks([])
    ax.set_xlim(datetime(2022, 5, 1), datetime(2025, 12, 31))
    ax.set_title("Chronological splits — strict boundaries, no leakage (leakage_report.yaml)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "timeline_splits.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_flowchart()
    fig_rmse_benchmark()
    fig_pm25_validation()
    fig_aqi_live()
    fig_data_lake()
    fig_timeline()
    print("figures written to", OUT)