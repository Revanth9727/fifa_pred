# data_exp.md

The schemas at each stage and what every field means. These are contracts — modules must produce and consume exactly these shapes. Types are indicative; adapt to your storage (parquet/dataframe).

## 1. Per-match table (`data/processed/matches.parquet`)

The training backbone after joining sources. One row per historical international match.

| Field | Type | Meaning |
|---|---|---|
| `match_id` | str | Unique id |
| `date` | date | Match date |
| `team_home`, `team_away` | str | Canonical team names (post entity-resolution) |
| `goals_home`, `goals_away` | int | Final-90 goals |
| `tournament` | str | Competition name |
| `importance_weight` | float | Training weight; friendlies low, tournament high (from `config`) |
| `neutral` | bool | Neutral venue |
| `venue_lat`, `venue_lon` | float | For travel/rest computation |

## 2. Feature vector (`features/engineer.py` output)

Per matchup, the structured input to the match model. **No LLM or market fields here** (ai_rules.md #2, #3).

| Field | Type | Meaning |
|---|---|---|
| `elo_diff` | float | Home/A minus away/B Elo — the workhorse strength signal |
| `fifa_rank_diff`, `spi_diff` | float | Secondary strength signals |
| `squad_quality_rank_a/_b` | float | Rank-signal squad quality (not raw market value level) |
| `form_a`, `form_b` | float | Recency-weighted recent results, last 6–12 months |
| `host_flag` | int | Host nation involved (US/MEX/CAN) |
| `neutral_flag` | int | Neutral venue |
| `rest_days_a`, `rest_days_b` | float | Days since last match (path-dependent in sim) |
| `travel_km_a`, `travel_km_b` | float | Travel since last match |
| `importance_weight` | float | Match importance |

## 3. Match-model output (`match_model.predict()`)

Goals-distribution parameters sufficient to *sample* a scoreline — not a point result.

| Field | Type | Meaning |
|---|---|---|
| `lambda_a`, `lambda_b` | float | Expected goals for each side |
| `dependence` | float | Bivariate / Dixon–Coles low-score correction term |

## 4. LLM extraction object (`llm/schema.py`) — STRUCTURE ONLY

Emitted by `llm/extract.py`. **No numbers, no probabilities, no magnitudes.** If a field would be numeric strength, it does not belong here.

| Field | Type | Meaning |
|---|---|---|
| `match_id` | str | Match this pertains to |
| `team` | str | Affected team |
| `channel` | enum | `availability` \| `lineup_certainty` |
| `player_id` | str | Affected player (for availability) |
| `status` | enum | `ruled_out` \| `doubtful` \| `fit` \| `unconfirmed` |
| `role` | enum | `starter` \| `rotation` |
| `source_type` | enum | `official` \| `beat_report` \| `rumor` |
| `evidence_span` | str | Quoted/located text the inference came from |
| `as_of` | datetime | Timestamp of the underlying news |

## 5. Adjustment record (`adjust/provenance.py`)

One per applied adjustment — the audit trail and the basis for the A/B and source-tier ablation.

| Field | Type | Meaning |
|---|---|---|
| `match_id`, `team` | str | Target |
| `channel` | enum | `availability` (mean-λ delta) \| `lineup_certainty` (dispersion) |
| `status`, `source_type` | enum | Carried from extraction |
| `raw_magnitude` | float | Pre-clip value from `magnitude.py` (club prior, shrunk) |
| `applied_value` | float | Post-clip value actually used |
| `confidence` | float | Layer confidence (drives no-op when low) |
| `decay_factor` | float | From news-age and kickoff-proximity |
| `superseded` | bool | True once the official lineup overrides inference |
| `rule_version` | str | Version of the magnitude mapping used |

**Channel semantics (ai_rules.md #11):** `availability` writes a signed delta to mean λ. `lineup_certainty` writes a *dispersion* term that sets the spread of the λ-mixture the sampler draws over — it must never move the mean. `applied_value` is interpreted accordingly per channel.

## 6. Simulator output (`sim/simulator.py`)

| Field | Type | Meaning |
|---|---|---|
| `team` | str | Team |
| `p_group_advance` … `p_champion` | float | Probability of reaching each stage (group exit → R32 → … → title) |
| `p_finalist` | float | Reaches the final |
| `as_of` | datetime | Snapshot time (re-emitted after each match-day) |

Plus a per-match table of single-game probabilities (for eval and live odds).

## 7. Market / eval data (`eval/calibration.py` only)

| Field | Type | Meaning |
|---|---|---|
| `market_type` | enum | `futures` \| `match_close` |
| `team` / `match_id` | str | Subject |
| `implied_raw` | float | Raw implied probability (overround > 1) |
| `implied_devig` | float | After proportional / Shin de-vig (sums to 1) |
| `captured_at` | datetime | When the line was taken (use the close) |

Scored against model outputs with Brier, log-loss, and a reliability diagram. The A/B compares model-with-adjustment vs model-without; the ablation slices by `source_type`. This is the only place market data is permitted.
