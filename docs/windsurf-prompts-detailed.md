# windsurf-prompts-detailed.md

Detailed build prompts, one per phase. Paste a phase, let the agent finish and report, then move to the next. Each prompt restates the load-bearing rules inline so the agent stays correct even if it doesn't re-read every doc. The Claude API (`anthropic` SDK, Messages API, key from `ANTHROPIC_API_KEY`) is used only in Phases 4 and 6, and only ever emits structure — never numbers.

Open every phase by pasting its block verbatim. Always prefix with:

> Follow `CLAUDE.md` and `docs/ai_rules.md`. Build only what this phase specifies. Stop at the acceptance criteria and report status before continuing. Do not start later phases.

---

## Phase 1 — Foundation & data pipeline

> **What you're building.** The repo skeleton and the data-in layer. Two kinds of data exist: (a) a one-time historical bulk the user has already downloaded by hand into `data/raw/` (match results, Elo, FIFA ranking, Transfermarkt player/appearance/valuation CSVs), and (b) live feeds (news, odds) fetched via API later — not in this phase. You build the local loaders and the join that produces the training table.
>
> **Files to create:**
> - The package tree under `src/wcpredict/` with empty `__init__.py` files, exactly as `docs/folder_struct.md`.
> - `data/ingest.py` — the ONLY module allowed network access.
> - `data/build_table.py` — a pure transform, no network.
> - `tests/test_build_table.py`.
>
> `pyproject.toml`, `README.md`, `.gitignore`, `.env.example`, and `config/tournament.yaml`, `config/teams.yaml`, `config/settings.yaml` are **already provided** — place them as-is, never regenerate them.
>
> **`ingest.py` spec.** For each historical source, write `load_local(name)` that reads from `data/raw/` and, if the file is missing, raises a clear error naming the file and its download URL (from `docs/data_sources.md`). Expected raw files: `results.csv`, `shootouts.csv`, `elo_ratings.csv`, `fifa_ranking.csv`, `players.csv`, `appearances.csv`, `player_valuations.csv` (pandas reads `.csv` and `.csv.gz` natively). For the live sources (news, odds), define `fetch_*()` with real signatures reading keys from env, but they aren't exercised until Phases 4–5 — leave them as thin, correct stubs.
>
> **`build_table.py` spec.** A pure function that joins raw results with Elo (as-of-date join: each match gets the ratings current at its date), FIFA ranking, and computes `importance_weight` from `config/settings.yaml` (friendly < qualifier < tournament). Output must match the `matches` schema in `docs/data_exp.md` §1, written to `data/processed/matches.parquet`. For team-name joins, do a best-effort normalization for now (lowercase/strip plus an alias map seeded from `config/teams.yaml`) and leave a clearly-marked hook where `data/entity_resolve.py` (Phase 6) will replace it.
>
> **Rules that apply here.** `ingest.py` is the only networked module. `build_table.py` must be pure and testable offline. NO odds/market data anywhere in this phase (ai_rules #3).
>
> **Tests.** `test_build_table` asserts the output schema matches `data_exp.md` §1, importance weights fall in the configured ranges, join keys are non-null, and row count is sane (tens of thousands).
>
> **Acceptance.** `build-data`'s table step produces `data/processed/matches.parquet` matching the schema; the test passes; no module imports any market source.
>
> **Do NOT.** Do not fetch historical data over the network (it's a manual download). Do not invent rows for missing data — error instead. Do not touch odds.

---

## Phase 2 — Structured match model

> **What you're building.** The statistical core. A goals-based model that, for any matchup plus context, returns a goals *distribution* you can sample from — not a single predicted result. It trains ONLY on structured historical features; it must never see LLM-extracted or market data. W/D/L is derived from the goals model, not modeled separately.
>
> **Files to create:** `features/engineer.py`, `model/match_model.py`, `model/shootout.py`, `tests/test_features.py`.
>
> **`engineer.py` spec.** A pure transform producing the feature vector in `docs/data_exp.md` §2: `elo_diff` (the workhorse), `fifa_rank_diff`/`spi_diff`, `squad_quality_rank_a/_b`, `form_a/_b` (recency-weighted over `form_window_months` from config), `host_flag`, `neutral_flag`, `rest_days_a/_b`, `travel_km_a/_b`, `importance_weight`. For squad quality, aggregate the squad's Transfermarkt market values into a **rank-style** signal — use its ranking, not the raw euro level (raw level overstates one-star teams and understates balanced ones). Include NO `llm/` or market fields.
>
> **`match_model.py` spec.** A bivariate Poisson with a Dixon–Coles low-score correction. `fit(matches, features)` trains by MLE on the importance-weighted history using only structured features. `predict(team_a, team_b, context) -> GoalsDist` returns `(lambda_a, lambda_b, dependence)` — enough to *sample* a full scoreline. Provide `derive_wdl(goals_dist)` that integrates the scoreline grid up to `max_goals` (config). This module must not import `llm/`, `adjust/`, or anything market.
>
> **`shootout.py` spec.** `resolve_shootout(team_a, team_b, strength_diff, rng) -> winner`: near-coinflip with a small, heavily-shrunk strength lean (coefficient from config). Deterministic given the rng/seed. Keep it simple — do not build a "deep" shootout model; sim sensitivity to it is low and complexity here manufactures fake signal.
>
> **Rules that apply here.** ai_rules #2 (no LLM features in training), #3 (no market), and the simple-shootout rule.
>
> **Tests.** Feature schema matches `data_exp.md`; `predict` returns sampleable params (not a point); `derive_wdl` sums to ≈1. Add a notebook cell checking match-model residuals across the full strength range, including blowout mismatches.
>
> **Acceptance.** `predict` returns a distribution; W/D/L integrates correctly; no forbidden imports.
>
> **Do NOT.** Do not build a W/D/L (outcome-only) head as the base — derive W/D/L from goals. Do not pass raw squad value as a level. Do not over-engineer the shootout.

---

## Phase 3 — Simulation engine (correctness-critical)

> **What you're building.** The heart of the system: it turns the match model into tournament probabilities by playing the 48-team World Cup 50,000 times, sampling every match. This is where most "looks sophisticated, still wrong" bugs live, so write the sampler's and bracket's tests FIRST. Every failure mode here is *bias*, not variance — more runs make a wrong number more precise, not more correct, and only Phase 5's calibration can detect it.
>
> **Files to create (tests first for the first two):** `sim/sampler.py` + `tests/test_sampler.py`; `sim/bracket.py` + `tests/test_bracket.py`; then `sim/simulator.py`.
>
> **`sampler.py` spec — the seam. Four hard requirements:**
> 1. `sample_match(goals_dist, dispersion=None, rng) -> (goals_a, goals_b)` draws an actual scoreline from the bivariate-Poisson/Dixon–Coles distribution every call. NEVER resolve by comparing `lambda_a` to `lambda_b`.
> 2. Honor the dependence term — joint sampling, never two independent marginals (independence underrates draws, which matters because draws route into ET/penalties).
> 3. If `dispersion` is given (the channel-2 term from Phase 4), first draw a λ from a small uncertainty-conditioned mixture centered on the dist's λ with spread set by `dispersion`, then sample the scoreline from that draw. High dispersion widens outcome variance; it must not shift the mean.
> 4. `sample_knockout_match(...)` resolves 90′; if level, plays ET at `et_lambda_scale * λ` (config); if still level, calls the shootout sub-model.
>
> **`bracket.py` spec.** Load the format from `config/tournament.yaml`: group round-robin standings using the tiebreaker order (points → GD → goals → conduct → FIFA rank), rank the 12 third-placed teams and advance the top 8, and seed the round of 32 from `round_of_32_slots` + `third_place_allocation`. **Blocker:** those two config keys are intentionally empty and MUST be filled from FIFA's official 2026 bracket chart (the deterministic best-third combination table) before this can run — source them, never invent them.
>
> **`simulator.py` spec.** `run(n_runs, seed)` from config: per run, play group matches via the sampler, compute standings, seed and resolve the knockout (sample → ET → shootout), and record the stage each team reached; aggregate across runs to per-team stage/finalist/champion probabilities. Recompute path-dependent context (rest/travel/host) inside the loop for each simulated matchup — NO precomputed fixed strength matrix. The match model is called on the order of 50k × ~64 ≈ 3.2M times, so vectorize it or cache λ on `(team_a, team_b, context-bucket)`. Add `resimulate(realized_results)` that fixes completed matches and runs the remainder.
>
> **Rules that apply here.** ai_rules #6 (sample, never average), #7 (dependence), #8 (path-dependent context + the call budget), #9 (ET/shootout handoff).
>
> **Tests.** `test_sampler`: a fixed-seed test that the *sampled* outcome distribution matches the analytic distribution (not just the mean); a test that draws reflect the dependence (not independent); a test that higher `dispersion` widens outcome variance while leaving the mean unchanged; an ET test that the scaled λ is used. `test_bracket`: standings, tiebreakers, third-place ranking, and allocation asserted against the official chart with constructed fixtures.
>
> **Acceptance.** Both test files pass; `simulate` emits per-team probabilities over 50k runs; `resimulate` works.
>
> **Do NOT.** Do not resolve matches by comparing expected goals. Do not sample marginals independently. Do not precompute a fixed strength matrix. Do not fabricate the round-of-32 allocation.

---

## Phase 4 — Claude prediction layer (role 1)

> **What you're building.** The only place an LLM touches prediction. Claude reads RAW news (injuries, team news, lineups) and emits STRUCTURE — typed objects with no numbers. A separate adjustment layer turns that structure into a bounded modification of the match model's *output*, applied at inference, never in training. Two channels, acting on two different moments of the distribution: availability (a signed delta on mean λ) and lineup certainty (a dispersion term for the sampler's λ-mixture — never a mean shift). All magnitude comes from a club-football prior, shrunk and hard-clipped.
>
> **Files to create:** `llm/schema.py`, `llm/extract.py`, `adjust/magnitude.py`, `adjust/layer.py`, `adjust/provenance.py`, `tests/test_adjust.py`, `tests/test_magnitude.py`.
>
> **`schema.py` spec.** The typed extraction object from `docs/data_exp.md` §4: `match_id`, `team`, `channel` (`availability` | `lineup_certainty`), `player_id`, `status` (`ruled_out` | `doubtful` | `fit` | `unconfirmed`), `role` (`starter` | `rotation`), `source_type` (`official` | `beat_report` | `rumor`), `evidence_span`, `as_of`. NO numeric fields. Validate strictly.
>
> **`extract.py` spec.** Calls the **Claude API** (`anthropic` SDK, Messages API, `ANTHROPIC_API_KEY`, `model` and `temperature: 0` from `config/settings.yaml`). Input: raw news text plus match/team context. Output: a list of validated schema objects. Prompt Claude to return ONLY JSON matching the schema; parse, validate, and reject (raise) if any field contains a number or magnitude. Runs on raw information only — never market commentary or odds-derived punditry (ai_rules #4).
>
> **`magnitude.py` spec.** `magnitude(status, importance_tier, recency) -> (signed_delta, confidence)`. This is the ONLY place numbers originate. Use the club-absence prior (`prior_source` in config) as the base effect, apply `shrinkage`, scale by the `importance_tiers` multiplier (tier derived from the squad-value rank in features), then hard-clip to the configured caps. The LLM never sets magnitude.
>
> **`layer.py` spec.** The pre-match adjustment layer. For a match, gather its extraction objects and compute per-channel adjustments. Channel 1 (availability): sum signed mean-λ deltas, clipped to `availability_max_abs_delta`. Channel 2 (lineup_certainty): produce a dispersion term (clipped to `lineup_dispersion_max`) for the sampler — NOT a mean delta. Apply `confidence_threshold` (below it, no-op → zero). Apply decay (news half-life + kickoff proximity). Apply supersession: if an official lineup is present, set channel 2 to 0 and drive channel 1 from confirmed absences only. Output what the simulator consumes: adjusted `(lambda_a, lambda_b)` plus a per-team dispersion.
>
> **`provenance.py` spec.** Log one record per adjustment per `data_exp.md` §5: `match_id`, `team`, `channel`, `status`, `source_type`, `raw_magnitude`, `applied_value`, `confidence`, `decay_factor`, `superseded`, `rule_version`. This is what Phase 5's source-tier ablation reads.
>
> **Rules that apply here.** ai_rules #10–15 (signed delta + confidence, hard clip, two channels on two moments, club prior, no-op default, decay + supersession, provenance), #4 (raw news only), #2 (inference-time, applied to the match model's output, never training).
>
> **Tests.** `test_adjust`: channel 2 NEVER moves the mean (assert mean λ unchanged when only lineup_certainty is present); clip bounds hold; confidence below threshold → zero adjustment; decay reduces magnitude with age and distance from kickoff; supersession collapses channel 2 to zero and restricts channel 1 to confirmed absences. `test_magnitude`: output stays within clip after shrinkage; importance tiers scale as configured.
>
> **Acceptance.** Extraction validates to schema with no numbers; all adjust/magnitude tests pass; provenance records are written.
>
> **Do NOT.** Do not let Claude output a probability, coefficient, or delta. Do not make channel 2 a mean shift. Do not feed market text to extraction. Do not bake any of this into match-model training.

---

## Phase 5 — Evaluation & leakage guard

> **What you're building.** The validator — the ONLY place market data is allowed. It de-vigs the closing line, scores the model's calibration, and runs the A/B that justifies the entire LLM layer: model WITH vs WITHOUT the adjustment, ablated by source tier. A static test enforces that market data never reaches prediction.
>
> **Files to create:** `eval/calibration.py`, `tests/test_calibration.py`, `tests/test_no_leakage.py`.
>
> **`calibration.py` spec.** The only module that reads odds. Load/fetch market data (pre-tournament futures + per-match closing lines). De-vig with both the proportional method and Shin's method so implied probabilities sum to 1 (futures carry a larger overround — de-vig matters more there). `score(model_probs, market_probs)` returns Brier and log-loss. `reliability_diagram(...)` uses `reliability_bins` from config. `ab_test()` scores the model with and without the adjustment layer against the close. `ablate_by_source()` slices the provenance log by `source_type` and reports calibration per tier. Reading guidance (from `arch.md`): on the diagonal but wider → the market has info you don't; systematically off-diagonal → model quality.
>
> **`test_no_leakage.py` spec.** Statically scan imports and assert that no module under `data/`, `features/`, `model/`, `llm/`, `adjust/`, or `sim/` imports any market/odds module or reads an odds file/path.
>
> **Rules that apply here.** ai_rules #3 (market only in eval).
>
> **Tests.** De-vigged probabilities sum to 1 for both methods; Brier/log-loss correct on toy inputs; `test_no_leakage` passes.
>
> **Acceptance.** The A/B and source ablation produce Brier/log-loss plus a reliability diagram against the de-vigged close; `test_no_leakage` passes.
>
> **Do NOT.** Do not let odds reach any prediction module. Do not compare against raw (non-de-vigged) implied probabilities.

---

## Phase 6 — Non-predictive Claude modules (roles 2 & 3)

> **What you're building.** Two Claude-powered helpers that sit OUTSIDE prediction and eval. Role 2 (build-time): entity resolution that reconciles messy names across sources into the canonical names in `config/teams.yaml` — this also replaces the placeholder normalization in Phase 1's `build_table.py`. Role 3 (output-time): narration that turns produced probabilities and attributions into readable prose, grounded strictly in real outputs.
>
> **Files to create:** `data/entity_resolve.py`, `explain/narrate.py`, and optionally `tests/test_side_modules_isolation.py`.
>
> **`entity_resolve.py` spec.** Build-time only. Uses the **Claude API** (the cheap model in `config/settings.yaml`) to map source-specific team/player/club names to the canonical `config/teams.yaml` names; validate the resolved joins and report anything unmatched. Wire `build_table.py` (Phase 1) to call this instead of its placeholder normalization. It may not be imported by any training or eval code.
>
> **`narrate.py` spec.** Output-time only. Uses the Claude API to write a rationale from the simulator's probabilities and the match model's feature attributions. It MUST be grounded strictly in the produced numbers — prompt it with the actual outputs and forbid invented facts. It may not be imported by any training or eval code.
>
> **Rules that apply here.** ai_rules #5 (roles 2 and 3 stay out of training and eval); narration grounded only in real outputs.
>
> **Tests.** Isolation — neither module is importable from training/eval code; a spot check that `narrate` references only numbers it was given.
>
> **Acceptance.** Entity resolution produces validated joins and `build_table` uses it; `narrate` produces grounded prose; isolation holds.
>
> **Do NOT.** Do not let `narrate` invent stats. Do not import these modules anywhere under `model/`, `sim/`, or `eval/`.

---

## Phase 7 — Integration & first run

> **What you're building.** The CLI that wires everything together, plus the first end-to-end run. The run ends on the A/B gate: if the LLM adjustment doesn't improve calibration against the close, it comes out — that's the contract, not a failure.
>
> **Files to create:** `cli.py` (no new modules; this is integration).
>
> **`cli.py` spec.** Subcommands matching `docs/setup.md`: `build-data` (ingest + build_table), `fit` (train the match model), `simulate` (run the simulator → outputs), `evaluate` (calibration + A/B + source ablation), `update --results <match-day>` (fix realized results, apply confirmed lineups via supersession, re-simulate the remainder). Thin wrappers over the modules; all parameters from `config/`; explicit seed everywhere stochastic.
>
> **First-run order.** `build-data` → confirm the produced schemas against `docs/data_exp.md` → `fit` → check residuals across the strength range → `simulate` → `evaluate` against the futures market. Report the A/B result.
>
> **Rules that apply here.** The A/B gate (cut the adjustment layer if it doesn't beat the close); supersession inside `update`.
>
> **Acceptance.** An end-to-end run emits champion / finalist / per-stage probabilities; `evaluate` shows model-with-adjustment vs model-without vs the de-vigged close; `update` re-simulates from realized results.
>
> **Do NOT.** Do not keep the adjustment layer if the A/B shows it doesn't help. Do not hardcode anything that belongs in `config/`.

---

### Sequencing note
Tests for the sampler, bracket, and adjustment layer are written before anything depends on them, because those three files concentrate the system's correctness. The round-of-32 allocation table (Phase 3) is the one external dependency that blocks a real run — source it from FIFA's official bracket before `simulate`.
