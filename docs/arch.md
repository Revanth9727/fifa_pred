# arch.md

The full architecture. Two things compute outcomes — the match model and the simulator. Everything else prepares inputs, adjusts outputs, or validates.

## Data flow

```
  DATA SOURCES  results · Elo · squads · fixtures · (market: eval only)
        |
        v
  FEATURE ENGINEERING  strength · squad quality (rank, not level) · context
        |
        v
  MATCH MODEL  bivariate Poisson + Dixon–Coles
  trains on structured history only · emits goals-distribution params
        |
        v
  ADJUSTMENT LAYER  <-- LLM-extracted late info enters here, at inference
  channel 1: availability  -> signed delta on mean λ
  channel 2: lineup certainty -> dispersion of a λ-mixture
  clip · decay · supersession · no-op default · provenance
        |
        |  <-- THE SEAM: sampler contract -->
        v
  SIMULATOR  samples scorelines · ET · shootout · 48-team bracket · 50k runs
        |
        v
  OUTPUTS  per-stage / finalist / champion probabilities; live-updated
        |
        v
  EVAL (validator, not in the prediction path)
  de-vig market · Brier / log-loss / reliability · A/B with vs without adjustment

  side modules (non-predictive):
    entity resolution (build-time)  ·  explanation (output-time)
```

## Match model

Goals-based. Bivariate Poisson with a Dixon–Coles low-score correction. Train goals first; derive W/D/L by integrating the scoreline grid. Outcome-only models are not simpler here — they defer the scoreline problem to the simulator, where it is harder.

- **Input:** the structured feature vector (relative strength via Elo, rank-signal squad quality, recent form, context: host/neutral/rest/travel/importance). Friendlies are down-weighted in training.
- **Output:** goals-distribution parameters `(λ_A, λ_B, dependence terms)` sufficient to *sample* a scoreline.
- **Watch:** the strength→λ mapping must hold across the full strength range, including lopsided mismatches under-represented in the data. Check residuals at the extremes.

## The adjustment layer

This is where LLM-extracted late information enters — at inference, never in training. It sits between `match_model.predict()` and the sampler and applies a bounded modification to the predicted distribution.

The split into two channels is load-bearing: **availability changes the expected strength; lineup uncertainty changes confidence in that expectation.** Different statistical objects, different parts of the distribution.

- **Channel 1 — availability.** Confirmed or likely absence of a key player → signed delta on mean λ. Magnitude from a club-prior (below), shrunk and clipped.
- **Channel 2 — lineup certainty.** Uncertainty about the XI → a dispersion term that sets the spread of a small λ-mixture the simulator draws over per run. High certainty collapses the mixture to a point estimate; low certainty widens it. This is nearly free because the simulator already samples — it adds one draw over lineup-conditioned λ inside the loop.

Contract for every adjustment: signed delta + confidence, hard-clipped, decaying with news-age and kickoff-proximity, no-op when confidence is weak, provenance-logged. **Supersession:** once the official lineup drops, observed availability overrides inference — channel 2 collapses to ~zero, channel 1 is driven by confirmed absences.

### Magnitude calibration

International matches with a star absent are too sparse and confounded to fit. Calibrate from **club football**: team xG delta with vs without a given player over league play, shrunk aggressively by importance tier, used as the prior for the international adjustment, then hard-clipped. The LLM never sets magnitude; it only supplies the structured status that selects which prior applies.

## The seam (sampler contract)

The highest-risk interface. All four requirements are sources of *bias*, not variance — Monte Carlo gives an arbitrarily precise estimate of the wrong number, and more runs only sharpen the error. Only `eval/` can detect a rotten seam.

1. **Sample, never average** — draw a real scoreline every run.
2. **Preserve goal dependence** — honor the bivariate/Dixon–Coles structure; never sample marginals independently.
3. **Recompute path-dependent context in the loop** — knockout opponents are random; rest/travel/host depend on the path. Vectorize or cache for the ~3.2M-call budget.
4. **ET / shootout handoff** — draw → ET at scaled λ → shootout, conditioning carried through.

Channel 2's dispersion term plugs in here: the sampler draws λ from the uncertainty-conditioned mixture, then draws the scoreline.

## Simulator

- **Structure:** 12 groups of 4 → 32 advance into a knockout round of 32 → single-elimination to the final. Group tiebreakers and advancement encoded from published rules and unit-tested against them (no historical precedent for this format).
- **Loop per run:** play group matches (sample), apply standings + tiebreakers, seed knockout, resolve each knockout match (sample → ET → shootout), record stage reached per team.
- **Runs:** 50k. Standard error at p≈0.2 is ~0.0018 — convergence is not the risk, bias is.
- **Shootout sub-model:** deliberately simple — near-coinflip with a small, shrunk strength lean. A "deep" shootout model manufactures fake signal; sim sensitivity to it is low.
- **Live re-simulation:** after each match-day, fix realized results and re-simulate the remainder. This is also where supersession matters — by kickoff the adjustment layer is confirmed-absence-only.

## Eval

A validator. De-vig the market first (proportional or Shin's method) so probabilities sum to 1. Score with Brier and log-loss plus a reliability diagram against two benchmarks: the outright/futures market (champion/finalist probabilities, available pre-tournament) and per-match closing lines (single-game probabilities, collected during).

Reading it: unbiased-but-wider (on the diagonal, just noisier) means the market has information you don't — go get data. Systematically off-diagonal (favorite–longshot bias, draw underrate) means model quality — fix the model.

**The adjustment A/B is the point.** Score with and without the LLM adjustment, and ablate by source tier (official / beat report / rumor). If the adjustment doesn't improve calibration versus the close, it is variance with a confident voice — cut it.

## Outputs

Per-team stage probabilities (group exit → R32 → … → champion), finalist and final-matchup probabilities, per-match single-game probabilities (for eval and live odds), all re-emitted after each match-day.

## Honest limits

High variance is intrinsic (even the top team is ~20–25% to win it all). The market may simply know more. Squad and LLM-extracted features are soft, not precision instruments. The 48-team format has no precedent, so bracket logic rests on rules, not fit. Mismatch calibration is the weak spot.
