# windsurf-prompts.md

The build, in 7 phases. Run them in order — each phase assumes the previous one is done and green. Paste one phase at a time. Open every phase with:

> *Follow `CLAUDE.md` and `docs/ai_rules.md`. Build only what this phase asks. Stop at the acceptance criteria and report status before continuing.*

The two correctness-critical files (`sim/sampler.py`, `sim/bracket.py`) and the adjustment layer get their tests written first, in-phase. The Claude API (the `anthropic` SDK, Messages API, key from `ANTHROPIC_API_KEY`) is used only in Phases 4 and 6.

---

## Phase 1 — Foundation & data pipeline

> Scaffold the repo exactly as `docs/folder_struct.md`. `README.md`, `pyproject.toml`, `.gitignore`, and the three `config/*.yaml` files are **already provided** — place them as-is, do not regenerate them. Create the `src/wcpredict/` package tree with empty `__init__.py` files.
>
> Then build the data layer. `data/ingest.py` is the ONLY module allowed network access: one function per source in `docs/data_sources.md`, each returning a raw frame matching the source schema in `docs/data_exp.md`, written to `data/raw/`. `data/build_table.py` is a pure transform (no network): join the raw sources into the per-match table in `docs/data_exp.md`, including the importance weight (friendlies down-weighted per `config/settings.yaml`). Write to `data/processed/`.

**Acceptance:** `build_table` output matches the `matches` schema and weight ranges; a test asserts schema + weights; no module here imports any odds/market source.

---

## Phase 2 — Structured match model

> Build `features/engineer.py` (pure transform): produce the feature vector in `docs/data_exp.md` — Elo-difference strength, rank-signal squad quality (not raw level), recency-weighted form, and context (host/neutral/rest/travel/importance). No LLM or market fields (ai_rules.md #2, #3).
>
> Build `model/match_model.py`: a goals-based bivariate Poisson with a Dixon–Coles low-score correction. `fit()` trains on the weighted historical table using only structured features. `predict(teamA, teamB, context)` returns goals-distribution params `(lambda_a, lambda_b, dependence)` sufficient to *sample* a scoreline. It must not import `llm/`, `adjust/`, or any market module.
>
> Build `model/shootout.py`: a deliberately simple resolver — near-coinflip with a small, heavily shrunk strength lean (config-driven), deterministic given a seed.

**Acceptance:** `predict()` returns sampleable params (not a point result); a notebook cell shows match-model residuals across the full strength range including mismatches; W/D/L derives by integrating the scoreline grid.

---

## Phase 3 — Simulation engine (correctness-critical)

> Build `sim/sampler.py` against the seam contract in `docs/arch.md` and ai_rules.md #6–9, and write `tests/test_sampler.py` FIRST. It consumes a goals distribution plus an optional channel-2 dispersion term and returns a sampled scoreline. Tested requirements: (a) it samples, never averages; (b) it honors the bivariate / Dixon–Coles dependence, never independent marginals; (c) given a dispersion term it draws λ from a small uncertainty-conditioned mixture before sampling the scoreline; (d) the ET path samples at the configured scaled λ then hands off to the shootout when still level.
>
> Build `sim/bracket.py` and write `tests/test_bracket.py` FIRST. Encode the 48-team structure from `config/tournament.yaml`: group standings, the tiebreaker order, ranking the 12 third-placed teams (top 8 advance), R32 seeding. **Blocker:** `round_of_32_slots` and `third_place_allocation` in `tournament.yaml` are empty and must be filled from FIFA's official 2026 bracket (the deterministic best-third combination chart) before this runs — source them, do not invent. Tests assert advancement, tiebreakers, and third-place allocation against the official chart.
>
> Build `sim/simulator.py`: the Monte Carlo loop wrapping sampler + bracket. Per run, play group matches (sample), apply standings, seed and resolve the knockout (sample → ET → shootout), tally stage reached per team. Recompute path-dependent context inside the loop (ai_rules.md #8); vectorize or cache λ for the ~3.2M-call budget. Add `resimulate(realized_results)`.

**Acceptance:** `test_sampler.py` and `test_bracket.py` pass; `simulate` produces per-team stage / finalist / champion probabilities for 50k runs.

---

## Phase 4 — Claude-powered prediction layer (role 1)

> Build `llm/schema.py` and `llm/extract.py`. `schema.py` defines the typed extraction object in `docs/data_exp.md` — categorical/enum fields and evidence only, NO numbers. `extract.py` calls the **Claude API** (`anthropic` SDK, Messages API, `ANTHROPIC_API_KEY`, model from `config/settings.yaml`, `temperature: 0`) to read raw team-news / injury / lineup text and return objects validating against the schema. It runs on raw information only — never market commentary (ai_rules.md #4). Reject any LLM output containing a magnitude.
>
> Build `adjust/magnitude.py`, `adjust/layer.py`, `adjust/provenance.py` (ai_rules.md #10–15). `magnitude.py` maps `(status, importance tier, recency)` to a signed delta + confidence from the club-football prior, shrunk and hard-clipped per config. `layer.py` is the two-channel pre-match layer: channel 1 (availability) → signed mean-λ delta; channel 2 (lineup certainty) → a dispersion term for the sampler's λ-mixture, never a mean delta. Implement clip, decay (news-age + kickoff-proximity), no-op default, and supersession. `provenance.py` logs every adjustment. Write `tests/test_adjust.py` and `tests/test_magnitude.py`.

**Acceptance:** extraction validates to schema and carries no numbers; `test_adjust.py` proves channel 2 never moves the mean, plus clip bounds, decay-to-zero, no-op on weak confidence, supersession collapse; magnitude stays within clip after shrinkage.

---

## Phase 5 — Evaluation & leakage guard

> Build `eval/calibration.py` — the ONLY module that may read market data. De-vig odds (Shin and proportional) so they sum to 1. Score the model with Brier, log-loss, and a reliability diagram against the futures market (champion/finalist) and per-match closing lines. Implement the A/B: score with vs without the adjustment layer, and ablate by `source_type` (official / beat_report / rumor).
>
> Build `tests/test_no_leakage.py`: statically assert that no module under `data/`, `features/`, `model/`, `llm/`, `adjust/`, or `sim/` imports any market/odds module or reads an odds file.

**Acceptance:** de-vigged probabilities sum to 1; the A/B and source ablation run and produce Brier/log-loss + a reliability diagram; `test_no_leakage.py` passes.

---

## Phase 6 — Non-predictive Claude modules (roles 2 & 3)

> Build `data/entity_resolve.py` (build-time only): use the **Claude API** (cheap model from `config/settings.yaml`) to reconcile team/player/club name mismatches across sources into the canonical names in `config/teams.yaml`; validate the resolved joins. Build `explain/narrate.py` (output-time only): use the Claude API to turn produced probabilities and feature attributions into a readable rationale, grounded strictly in the actual outputs — it may not invent facts.
>
> Neither module may be imported by any training or eval code (ai_rules.md #5). Add an import-guard test if convenient.

**Acceptance:** entity resolution produces validated joins; `narrate.py` references only numbers the system produced; neither is reachable from training/eval.

---

## Phase 7 — Integration & first run

> Build `cli.py` with subcommands `build-data`, `fit`, `simulate`, `evaluate`, `update`, matching `docs/setup.md` — thin wrappers over the modules, all parameters from `config/`. `update` fixes realized results, applies confirmed lineups (supersession), and re-simulates the remainder.
>
> Then run the full first-build order from `docs/setup.md`: `build-data` → confirm schemas → `fit` → check residuals → `simulate` → `evaluate` against the futures market. Report the A/B result.

**Acceptance:** an end-to-end run emits champion / finalist / per-stage probabilities, and `evaluate` shows model-with-adjustment vs model-without vs the de-vigged close. **The gate:** if the adjustment layer does not improve calibration against the close, it comes out — that is the contract, not a failure.

---

### Order summary
Foundation → match model → simulation engine → Claude prediction layer → eval & leakage guard → non-predictive Claude modules → integration & run. Tests for the sampler, bracket, and adjustment layer are written before anything depends on them. The Claude API appears only in Phases 4 and 6, and only ever emits structure.
