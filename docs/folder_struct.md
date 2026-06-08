# folder_struct.md

The target layout. `★` marks correctness-critical files (concentrate review and tests here). `‡` marks non-predictive modules (must stay out of training and eval).

```
worldcup-predictor/
├── CLAUDE.md                      Orientation + invariants (read first)
├── README.md                      One-paragraph summary + quickstart
├── pyproject.toml                 Deps + package metadata
├── .gitignore                     Ignore data/raw, data/processed, caches, .env
│
├── docs/
│   ├── ai_rules.md                Hard rules + conventions
│   ├── arch.md                    Full architecture + seam contract
│   ├── folder_struct.md           This file
│   ├── setup.md                   Environment + run commands
│   ├── windsurf-prompts.md        Ordered build prompts
│   ├── data_sources.md            Where each source comes from
│   └── data_exp.md                Schemas + field meanings
│
├── config/
│   ├── tournament.yaml            48-team format: groups, advancement, tiebreakers
│   ├── teams.yaml                 The 48 teams + group assignments
│   └── settings.yaml              n_runs, ET λ-scaling, clip caps, decay constants, paths
│
├── data/
│   ├── raw/                       Downloaded sources (gitignored)
│   └── processed/                 Clean match table + features (gitignored)
│
├── src/wcpredict/
│   ├── __init__.py
│   ├── data/
│   │   ├── ingest.py              ONLY module with network access; pulls all sources
│   │   ├── build_table.py         Join → per-match table + importance weights (pure)
│   │   └── entity_resolve.py      ‡ LLM entity resolution for messy joins (build-time)
│   ├── features/
│   │   └── engineer.py            Strength, squad quality (rank not level), context
│   ├── model/
│   │   ├── match_model.py         Bivariate Poisson + Dixon–Coles: fit() + predict()
│   │   └── shootout.py            Shrunk-logistic / coinflip shootout sub-model
│   ├── llm/
│   │   ├── schema.py              Typed extraction objects (structure only, no numbers)
│   │   └── extract.py             Raw news → typed availability / lineup signals
│   ├── adjust/
│   │   ├── magnitude.py           ★ status + importance + recency → signed delta + confidence
│   │   ├── layer.py               ★ pre-match layer; 2 channels; clip/decay/supersession/no-op
│   │   └── provenance.py          Audit record per adjustment
│   ├── sim/
│   │   ├── sampler.py             ★ THE SEAM — sample scoreline, dependence, λ-mixture, ET
│   │   ├── bracket.py             ★ 48-team structure, standings, tiebreakers, seeding
│   │   └── simulator.py           Monte Carlo run loop, stage tallies, live re-sim
│   ├── eval/
│   │   └── calibration.py         De-vig, Brier/log-loss, reliability, A/B, by-source ablation
│   ├── explain/
│   │   └── narrate.py             ‡ LLM narrative grounded in outputs + attributions (output-time)
│   └── cli.py                     Entrypoints: build-data · fit · simulate · evaluate · update
│
├── tests/
│   ├── test_sampler.py            Sample-not-average, dependence, λ-mixture spread, ET scaling
│   ├── test_bracket.py            Advancement + tiebreakers vs published rules
│   ├── test_adjust.py             Clip, decay, no-op, supersession, channel separation
│   ├── test_magnitude.py          Club-prior mapping, shrinkage, clip bounds
│   ├── test_calibration.py        De-vig sums to 1, scoring correctness
│   └── test_no_leakage.py         Asserts no model/feature/sim module imports market or odds
│
└── notebooks/
    └── exploration.ipynb          Residual checks across strength range; scratch
```

## Boundary encoded in the layout

- **Prediction path:** `data/` → `features/` → `model/` + `llm/` (structure) → `adjust/` (magnitude) → `sim/` → outputs.
- **Structure vs magnitude:** `llm/` only ever emits typed objects; `adjust/` owns every number. They are separate packages on purpose.
- **Non-predictive (‡):** `data/entity_resolve.py` (build-time) and `explain/narrate.py` (output-time). Neither is importable from training or eval code.
- **Market isolation:** odds appear only inside `eval/calibration.py`. `tests/test_no_leakage.py` enforces it.
