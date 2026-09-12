# Contributing

Thanks for considering a contribution to the Delhi NCR AQF coupled forecasting
system.

## Development workflow

```bash
pip install -e ".[dev]"            # + ".[module2]" for GRIB work, ".[api]" for API work
pytest                             # full offline suite — must stay network-free
```

## Conventions

- Pure-python, dependency-light core (`numpy`/`pandas`/`pydantic`/`pyarrow`).
  Heavy optional dependencies (eccodes, scipy, fastapi) live in **extras**,
  not the core dependency list.
- Typed, validated configuration in YAML (`configs/`) loaded via pydantic —
  no hardcoded endpoints outside config.
- New behaviour ships with **offline, deterministic tests**. Tests must never
  require network access; live behaviour is exercised via the CLI and the
  `scripts/` smoke tests instead.
- Artifact writing is idempotent and ends with a `_DONE` marker file.

## Testing

- Keep tests fast and repeatable (seeded RNG, tmp dirs via `tmp_path_factory`).
- If `pytest-html` breaks collection on your platform:
  `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"; python -m pytest tests/`.

## Pull requests

1. Branch from `main` (`git checkout -b fix/your-change`).
2. Run the suite locally before pushing.
3. Describe the change, why, and how you validated it (offline tests + any
   live smoke run).

## Reporting issues

Describe the expected vs observed behaviour, the command that failed, and the
relevant log output. Data artifacts live under `data/` and are gitignored —
do not commit them.