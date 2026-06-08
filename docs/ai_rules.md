# ai_rules.md

Hard rules. Where a rule says "never," treat a violation as a defect even if the code runs.

## Boundary rules (the ones that keep the system honest)

1. **LLM emits structure, never magnitude.** Any LLM call returns a typed object with categorical/enum fields and evidence — no probabilities, no coefficients, no λ deltas. If you find yourself prompting an LLM for a number, stop: that number belongs in `adjust/magnitude.py` or a small auxiliary model.

2. **LLM-extracted signal is inference-time only — never a training feature.** `model/match_model.py` trains exclusively on structured historical data (results, Elo, squad value, context). It must not import anything from `llm/` or `adjust/`. The adjustment is applied to the model's *output*, between `match_model.predict()` and the sampler.

3. **Market/odds data lives only in `eval/`.** No module under `data/`, `features/`, `model/`, `llm/`, `adjust/`, or `sim/` may read odds. Odds are the calibration benchmark, never an input. Enforce with an import check in tests.

4. **Extraction runs on raw information only.** `llm/extract.py` consumes injury reports, team news, and lineup announcements — never market commentary, punditry, or odds-derived analysis. Reading market-flavored text launders the closing line into the adjustment and makes the calibration check circular.

5. **Roles 2 and 3 are non-predictive.** `data/entity_resolve.py` (entity resolution) is build-time only. `explain/narrate.py` (explanation) is output-time only and grounded strictly in produced outputs and attributions — it may not invent facts. Neither touches training or eval.

## The sampler / seam (correctness-critical — see arch.md §Seam)

6. **Sample, never average.** Every simulated match draws an actual scoreline from the match model's distribution. Never resolve by comparing `λ_A` to `λ_B`.

7. **Honor goal dependence.** The sampler respects the bivariate dependence / Dixon–Coles correction the model fit. Never sample the two marginals independently.

8. **Recompute path-dependent context inside the loop.** Rest, travel, and host terms depend on the simulated knockout path. No precomputed fixed strength matrix. Vectorize or cache λ on `(teamA, teamB, context-bucket)` — the model is called ~50k × ~64 ≈ 3.2M times per full simulation.

9. **ET and shootout handoff.** A 90-minute draw goes to extra time at a *scaled-down* goal rate, then to the shootout sub-model if still level. Never leave ET at full λ; never enter the shootout with stale state.

## The adjustment layer

10. **Output is signed delta + confidence, hard-clipped.** Every adjustment is `(delta, confidence)`, and `|delta|` is clipped to a configured cap so noisy text cannot yank the model around.

11. **Two channels on two moments. Do not blend them into one scalar.**
    - **Availability channel** → signed delta on mean λ (a confirmed/likely absence lowers attacking output).
    - **Lineup-certainty channel** → a *dispersion* term, not a location delta. It sets the spread of a small λ-mixture the sampler draws over each run. High certainty collapses the mixture to a point; low certainty widens it.

12. **Magnitude comes from a club-football prior.** Calibrate absence effects from club-level data (team xG with/without a player), shrink aggressively by importance tier, then clip. International absence data is too sparse to fit on its own. Magnitude logic lives only in `adjust/magnitude.py`.

13. **No-op default.** When extraction confidence is weak or absent, the layer returns zero adjustment. Silence is the default, not a guess.

14. **Decay and supersession.** Adjustments decay with both news age and proximity to kickoff. When an official lineup is released, observed availability supersedes all inference: the lineup-certainty channel collapses to ~zero and the availability channel is driven by confirmed absences, not predictions.

15. **Provenance is mandatory.** Every adjustment writes a record (`adjust/provenance.py`) tracing it to channel, status, source_type, evidence span, magnitude-rule version, and final clipped value. This enables ablation by channel *and* by source tier.

## Code conventions

- Python 3.11+. Type-hint public functions. Prefer pure functions; isolate I/O.
- Config-driven: tournament format, teams, run count, clip caps, decay constants, and ET λ-scaling live in `config/*.yaml`, never hard-coded.
- `data/ingest.py` is the only module allowed network access. `data/build_table.py` and all feature/model/sim code must be pure transforms, testable offline.
- Determinism: every stochastic component takes an explicit seed.
- Tests are required for `sim/sampler.py`, `sim/bracket.py`, `adjust/layer.py`, and `adjust/magnitude.py` before they're considered done. Bracket logic is unit-tested against the published 2026 rules, not historical tournaments.
- No silent fallbacks that fabricate data. Missing inputs → no-op or explicit error, never invented values.
