# Delhi NCR Air-Quality–Weather Coupled Forecasting System

End-to-end pipeline that turns **real operational meteorology** (GFS 0.25° GRIB2)
and **satellite fire detections** (NASA FIRMS) into a gridded PM2.5 forecast for
the Delhi NCR 4 km domain — with plume-rise physics, an ML bias-correction
stage, a geospatial API and a 3D WebGIS dashboard.

Target: a Phase-1 offline implementation of the MoES / NCMRWF-style WRF-Chem
coupled forecasting workflow, structured so the offline PM2.5 *emulator* can be
replaced by a full WRF-Chem solver without changing the surrounding pipeline.

```
GFS GRIB2 ──┐                                             ┌─ Module-5 3D WebGIS
FIRMS ──────┼─ Module-1 ingest ─┬─ Module-2 coupler ─┐    │   dashboard (MapLibre)
CPCB* ──────┘      (parquet)     │  (met + plume +   ├─Module-3 bias-correction─┼─ Module-4 API
                                 │   emissions)      └─ (ST-GNN + XGBoost) ─────┘
                                 └─────────────────────────────────────────▶ data/{coupled|forecasts}/
```

> \* CPCB live ground-truth is a known TODO — the dashboard/API mount points are
> ready; the public endpoint is currently unavailable (see [Roadmap](#roadmap)).

## Modules

| # | Name | Delivers |
|---|------|----------|
| 1 | Ingestion | Resilient connectors (failover + retry), validation, idempotent parquet store for CPCB / FIRMS / GFS |
| 2 | Coupled emulator | Real GFS GRIB2 subsets → Delhi grid; Freitas plume rise; FRP→PM2.5 emissions; two-reservoir transport-chemistry; WRF namelist scaffolding |
| 3 | Bias correction | ST-GNN + XGBoost ensemble over model field → per-station corrected forecasts |
| 4 | Geospatial API | FastAPI + Redis-cached endpoints over all artifacts |
| 5 | 3D WebGIS | MapLibre GL dashboard: animated PM2.5 field, 3D extrusions, met overlays, fire & station layers |

## Install

Requires **Python ≥ 3.11** (developed on 3.13, Windows).

```bash
pip install -e ".[dev]"          # core + test tooling
pip install -e ".[module2]"      # + eccodes / cfgrib / xarray / scipy (GRIB)
pip install -e ".[api]"          # + FastAPI / uvicorn / httpx (API + dashboard)
```

GRIB decoding on Windows/macOS uses the `eccodes` PyPI wheel (bundled DLLs);
on Linux install the system library first (`libeccodes-dev`).

## Quickstart

```bash
# 1) full offline test suite (41 tests, no network)
pytest

# 2) Module-2: coupled emulator
python scripts/run_module2.py run --init 2026-09-12 00 --real   # real GFS
python scripts/run_module2.py run --init 2026-09-12 00 --demo   # synthetic met

# 3) Module-3: bias correction
python scripts/run_module3.py demo --quick

# 4) Module-4/5: API + dashboard
python scripts/run_module4.py serve --port 8000
# open http://localhost:8000/dashboard/
```

The dashboard needs an internet connection for the MapLibre CDN and OSM base
map; the state aggregation endpoint (`/dashboard/state`) is fully local.

## Ingestion

```bash
python scripts/run_ingest.py cpcb      # live CPCB poll (endpoint currently dead)
python scripts/run_ingest.py firms     # FIRMS (export FIRMS_MAP_KEY)
python scripts/run_ingest.py gfs       # GFS availability probe + manifest
```

```text
data/ingest/{source}/{YYYY-MM-DD}/{run_id}/records.parquet
data/ingest/{source}/{YYYY-MM-DD}/{run_id}/_DONE
```

## Artifacts

All runtime data lives under `data/` (gitignored).

### Module-2 coupled run
```text
data/coupled/{run_id}/meta.json               run metadata + config
data/coupled/{run_id}/grid.parquet            (time × i × j) met + PM2.5
data/coupled/{run_id}/stations.parquet        per-station raw/obs series
data/coupled/{run_id}/emissions_fire.parquet  fire detections used
data/coupled/{run_id}/_DONE
```

### Module-3 forecast → Module-4 hand-off
```text
data/forecasts/{run_id}/forecast.parquet      (time × station, long form)
data/forecasts/{run_id}/report.json
data/forecasts/{run_id}/_DONE
```

## Module-2 details

The emulator fetches tiny Delhi-window GRIB2 subsets for each forecast hour
from the NOMADS GFS 0.25° filter service (~4 KB/hour → ~100 KB for a full 72 h
window), decodes them with `eccodes`, bilinearly regrids onto the 23×21
Delhi grid, derives PBL/inversion/ventilation, then runs a Freitas-style 1-D
integral plume-rise model, FRP emission coupling and a multi-species
transport-chemistry kernel: **PM2.5 + PM10** primary reservoirs (shared
advection/decay/wet-loss physics, PM10 = coarse ratio × PM2.5) plus an **O3**
photochemical proxy (solar-geometry production, temperature dependence, NOx
night titration, inversion suppression). It is fully **deterministic** (seeded
RNG) so offline tests are reproducible.

```bash
python scripts/run_module2.py grib  --init 2026-09-12 00 --fhr 0 3 6   # subsets only
python scripts/run_module2.py namelist --out data/coupled/namelist     # WRF-Chem input
python scripts/run_module2.py plume --frp 320                          # plume physics demo
```

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Service + store health |
| `GET /domain`, `/domain/grid` | Study domain + grid cell centres |
| `GET /stations` | Station registry |
| `GET /observations/latest`, `/observations/timeseries` | CPCB ground data |
| `GET /fires/recent` | Active fires (`?bbox`) |
| `GET /gfs/runs`, `/gfs/assets` | GFS run manifests |
| `GET /forecasts/runs`, `/forecasts/latest` | Published forecasts + metrics |
| `GET /forecasts/{run_id}/series`, `/map?lead=N` | Per-station forecast |
| `GET /dashboard/state` | 3D dashboard aggregation (coupled + met + fires + stations + forecast) |
| `GET /dashboard/` | Dashboard SPA |

Cache backend defaults to an in-process TTL store; set `AQF_REDIS_URL` to use
Redis (falls back to memory if unavailable). Overrides: `AQF_DATA_ROOT`,
`AQF_CACHE_TTL`.

## Testing

```bash
pytest                        # offline, deterministic (network never required)
pytest tests/test_module2.py  # per-module
```

41 tests: 7 ingest · 9 Module-3 · 5 Module-4 · 15 Module-2 · 5 Module-5.
Disable plugin autoload if `pytest-html` breaks collection:

```bash
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest tests/
```

## Configuration

Typed YAML under `configs/` (pydantic-backed):

- `domain.yaml` — Delhi NCR domain, grid resolution, operational knobs
- `ingest.yaml` — source URLs, failovers, schedules
- `module2.yaml` — GFS filter window, emissions EF, plume + emulator settings

## Repository layout

```text
src/aqf_delhi/
  api/         Module-4 FastAPI (app, routes, schemas, store, cache)
  dashboard/   Module-5 3D WebGIS (state aggregation + static SPA)
  ml/          Module-3 ST-GNN + XGBoost (features, graph, train, ensemble, metrics)
  pipeline/    Module-1 validation + orchestration
  schemas/     canonical record models (CPCB, FIRMS, GFS)
  sources/     resilient data connectors
  storage/     idempotent parquet partition store
  wrf/         Module-2 coupler (grib, regrid, plume, emissions, emulator, coupling)
  config.py    typed YAML configuration
  domain.py    Delhi 4 km grid derivation
configs/       domain.yaml · ingest.yaml · module2.yaml · stations.csv
scripts/       run_ingest · run_module2 · run_module3 · run_module4
tests/         offline pytest suite
```

## Roadmap

- [x] Module-1 ingestion & pipeline
- [x] Module-2 coupled emulator with real GFS GRIB input
- [x] Module-3 ST-GNN + XGBoost bias correction
- [x] Module-4 geospatial API
- [x] Module-5 3D WebGIS dashboard
- [ ] CPCB live ground-truth ingestion (public endpoint currently unavailable)
- [ ] Replace PM2.5 emulator kernel with a full WRF-Chem solver run
- [ ] Operational scheduler (GFS cycle-triggered forecast + publish)

## License

Not yet licensed — contact the maintainers before reusing this code.