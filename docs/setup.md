# setup.md

## Requirements

- Python 3.11+
- An LLM API key (for `llm/extract.py`, `data/entity_resolve.py`, `explain/narrate.py`)
- ~2 GB disk for raw + processed data

## Install

```bash
git clone <repo> && cd worldcup-predictor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

`pyproject.toml` dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`, `pyyaml`, `matplotlib`, `requests`, plus your LLM SDK. Dev extras: `pytest`.

## Configuration

Edit the files in `config/`:

- `tournament.yaml` — group structure, advancement rules, tiebreakers (from the published 2026 format).
- `teams.yaml` — the 48 qualified teams and their group assignments.
- `settings.yaml` — `n_runs` (default 25000), ET λ-scaling factor, adjustment clip caps, decay constants, and data paths.

Secrets go in `.env` (gitignored):

```
LLM_API_KEY=...
```

## Data

See `docs/data_sources.md` for where each source comes from and `docs/data_exp.md` for schemas. Place downloads in `data/raw/`. `data/ingest.py` is the only module that touches the network; everything downstream reads from `data/processed/`.

## Run the pipeline

Each stage is a CLI subcommand:

```bash
wcpredict build-data      # ingest + join → data/processed/matches.parquet, features
wcpredict fit             # train the match model on structured history
wcpredict simulate        # run 50k tournaments → outputs/probabilities
wcpredict evaluate        # de-vig market + score calibration (with vs without adjustment)
```

During the tournament, after each match-day:

```bash
wcpredict update --results <match-day-results>   # fix realized results, re-simulate the rest
```

The `update` step is also where the adjustment layer applies confirmed lineups (supersession) before re-simulating.

## Tests

```bash
pytest                    # all
pytest tests/test_sampler.py tests/test_bracket.py tests/test_adjust.py   # correctness-critical
pytest tests/test_no_leakage.py   # asserts odds never reach the model
```

The leakage test must pass before any model run is trusted.

## Order of operations (first build)

`build-data` → confirm the match table and feature schemas match `docs/data_exp.md` → `fit` → check residuals across the strength range (`notebooks/exploration.ipynb`) → `simulate` → `evaluate` against the futures market. Wire the adjustment layer last and prove it earns its place with the A/B in `evaluate`.
