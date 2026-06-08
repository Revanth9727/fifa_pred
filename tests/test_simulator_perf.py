"""
test_simulator_perf.py
Phase 3 validation: simulator performance + architecture invariants.

Checks:
  - 50k-run full tournament completes in < 120 s (wall-clock) on a laptop.
  - The predict cache is populated (not rebuilt per match).
  - Path-dependent context (rest/travel) is recomputed inside the knockout loop.
  - λ is not a precomputed constant — it changes when context changes.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from wcpredict.model.match_model import MatchModel, GoalsDist
from wcpredict.sim.sampler import _flat_grid_cached, _vectorized_draw
from wcpredict.sim.simulator import (
    TournamentSimulator,
    _build_context,
    _cached_predict,
    _ctx_key,
)


# ---------------------------------------------------------------------------
# Minimal synthetic model fixture
# ---------------------------------------------------------------------------

_SETTINGS = {
    "simulation": {
        "n_runs": 100,
        "random_seed": 42,
        "et_lambda_scale": 0.33,
        "shootout_strength_coef": 0.10,
    },
    "match_model": {"max_goals": 10},
}

_SETTINGS_50K = {
    "simulation": {
        "n_runs": 50000,
        "random_seed": 42,
        "et_lambda_scale": 0.33,
        "shootout_strength_coef": 0.10,
    },
    "match_model": {"max_goals": 10},
}


def _make_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for letter in "ABCDEFGHIJKL":
        groups[letter] = [f"T{letter}{i}" for i in range(1, 5)]
    return groups


def _fit_model() -> MatchModel:
    """Fit a MatchModel on 300-row synthetic data (fast, just needs to converge)."""
    rng = np.random.default_rng(0)
    n = 300
    elo = rng.normal(0, 200, n)
    matches = pd.DataFrame({
        "goals_home":       rng.poisson(np.clip(1.2 + elo / 400, 0.3, 3.5)),
        "goals_away":       rng.poisson(1.0),
        "importance_weight": np.ones(n),
    })
    features = pd.DataFrame({
        "elo_diff":             elo,
        "form_a":               rng.uniform(0.3, 0.7, n),
        "form_b":               rng.uniform(0.3, 0.7, n),
        "squad_quality_rank_a": rng.uniform(0, 1, n),
        "squad_quality_rank_b": rng.uniform(0, 1, n),
        "rest_days_a":          rng.uniform(3, 10, n),
        "rest_days_b":          rng.uniform(3, 10, n),
        "travel_km_a":          rng.uniform(0, 5000, n),
        "travel_km_b":          rng.uniform(0, 5000, n),
        "neutral_flag":         rng.integers(0, 2, n).astype(float),
        "host_flag":            np.zeros(n),
    })
    model = MatchModel(settings=_SETTINGS)
    model.fit(matches, features)
    return model


# ---------------------------------------------------------------------------
# Predict-cache invariant
# ---------------------------------------------------------------------------

def test_predict_cache_avoids_duplicate_compute():
    """
    When the same context is presented twice, _cached_predict should return
    the same GoalsDist object and the cache should have exactly 1 entry.
    """
    model = _fit_model()
    cache: dict = {}
    ctx = _build_context("TA1", "TB1", neutral=True, settings=_SETTINGS, model=model)

    d1 = _cached_predict(model, ctx, cache)
    d2 = _cached_predict(model, ctx, cache)

    assert len(cache) == 1, "Identical context should produce exactly one cache entry"
    assert d1 is d2, "Same context must return cached GoalsDist (identity)"


def test_predict_cache_different_contexts_different_entries():
    """Different contexts must produce different cache entries."""
    model = _fit_model()
    cache: dict = {}
    ctx_home = _build_context("TA1", "TB1", neutral=False, settings=_SETTINGS, model=model)
    ctx_neut = _build_context("TA1", "TB1", neutral=True,  settings=_SETTINGS, model=model)

    _cached_predict(model, ctx_home, cache)
    _cached_predict(model, ctx_neut, cache)

    assert len(cache) == 2, "Home vs neutral contexts must be separate cache entries"


# ---------------------------------------------------------------------------
# Path-dependent context: rest/travel recomputed per call
# ---------------------------------------------------------------------------

def test_context_varies_with_rest_days():
    """
    _build_context with different rest_days_home must produce different contexts.
    Ensures the simulator can't precompute a single context for all KO rounds.
    """
    model = _fit_model()
    ctx_3 = _build_context("TA1", "TB1", neutral=True, settings=_SETTINGS, model=model,
                            rest_days_home=3.0)
    ctx_7 = _build_context("TA1", "TB1", neutral=True, settings=_SETTINGS, model=model,
                            rest_days_home=7.0)

    assert _ctx_key(ctx_3) != _ctx_key(ctx_7), (
        "rest_days difference must produce distinct context keys"
    )


def test_context_varies_with_travel():
    model = _fit_model()
    ctx_a = _build_context("TA1", "TB1", neutral=True, settings=_SETTINGS, model=model,
                            travel_km_home=500.0)
    ctx_b = _build_context("TA1", "TB1", neutral=True, settings=_SETTINGS, model=model,
                            travel_km_home=4000.0)
    assert _ctx_key(ctx_a) != _ctx_key(ctx_b)


def test_path_dependent_lambda_differs():
    """
    predict() called with different rest/travel contexts must return different λ.
    """
    model = _fit_model()
    ctx_near = _build_context("TA1", "TB1", neutral=True, settings=_SETTINGS, model=model,
                               rest_days_home=2.0, travel_km_home=100.0)
    ctx_far  = _build_context("TA1", "TB1", neutral=True, settings=_SETTINGS, model=model,
                               rest_days_home=10.0, travel_km_home=8000.0)
    d_near = model.predict(ctx_near)
    d_far  = model.predict(ctx_far)
    # Different contexts → different λ
    assert d_near.lambda_a != d_far.lambda_a or d_near.lambda_b != d_far.lambda_b, (
        "Path-dependent context must produce different GoalsDist"
    )


# ---------------------------------------------------------------------------
# Task 1 — cache-key correctness: grids must differ by team strength
# ---------------------------------------------------------------------------

def test_grids_differ_for_different_strength_matchups():
    """
    Strong-vs-weak (elo_diff=+580) and weak-vs-weak (elo_diff=+60) matchups
    at neutral context must produce DIFFERENT scoreline grids.

    Before the elo_diff fix all matchups shared elo_diff=0.0, producing a
    single shared grid regardless of teams.  After the fix, distinct elo_diff
    values flow into model.predict() → distinct (λa, λb) → distinct grids.
    """
    model = _fit_model()
    cache: dict = {}

    ctx_strong = _build_context(
        "Argentina", "Jordan", neutral=True, settings=_SETTINGS, model=model,
        elo_diff=1960 - 1380,   # +580
    )
    ctx_weak = _build_context(
        "Qatar", "Jordan", neutral=True, settings=_SETTINGS, model=model,
        elo_diff=1440 - 1380,   # +60
    )

    d_strong = _cached_predict(model, ctx_strong, cache)
    d_weak   = _cached_predict(model, ctx_weak,   cache)

    assert len(cache) == 2, "Two distinct elo_diffs must produce 2 cache entries"

    grid_strong = _flat_grid_cached(
        float(d_strong.lambda_a), float(d_strong.lambda_b),
        float(d_strong.dependence), 10,
    )
    grid_weak = _flat_grid_cached(
        float(d_weak.lambda_a), float(d_weak.lambda_b),
        float(d_weak.dependence), 10,
    )

    assert not np.allclose(grid_strong, grid_weak), (
        "Grids for strong-vs-weak and weak-vs-weak must differ. "
        "elo_diff must flow into model.predict() — check _build_context."
    )


# ---------------------------------------------------------------------------
# Task 3 — vectorized draw: each row sampled from its own distribution
# ---------------------------------------------------------------------------

def test_vectorized_draw_distributions_differ():
    """
    When _vectorized_draw is given a matrix whose rows are two VERY different
    grids (strong vs weak team as home side), sampling 2000 rows each should
    yield visibly different home-goal averages.

    This confirms each row is drawn from its OWN PMF — not a single shared
    probability vector.
    """
    model = _fit_model()
    cache: dict = {}

    ctx_strong = _build_context(
        "Argentina", "Jordan", neutral=True, settings=_SETTINGS, model=model,
        elo_diff=1960 - 1380,
    )
    ctx_weak = _build_context(
        "Qatar", "Jordan", neutral=True, settings=_SETTINGS, model=model,
        elo_diff=1440 - 1380,
    )

    d_strong = _cached_predict(model, ctx_strong, cache)
    d_weak   = _cached_predict(model, ctx_weak,   cache)

    grid_s = _flat_grid_cached(
        float(d_strong.lambda_a), float(d_strong.lambda_b),
        float(d_strong.dependence), 10,
    )
    grid_w = _flat_grid_cached(
        float(d_weak.lambda_a), float(d_weak.lambda_b),
        float(d_weak.dependence), 10,
    )

    rng = np.random.default_rng(123)
    n_samples = 2000

    # Stack n_samples copies of each grid, then draw in one call
    grids_s = np.tile(grid_s, (n_samples, 1))  # (2000, 121)
    grids_w = np.tile(grid_w, (n_samples, 1))
    goals_s = _vectorized_draw(grids_s, rng, 10)  # (2000, 2)
    goals_w = _vectorized_draw(grids_w, rng, 10)

    mean_home_strong = goals_s[:, 0].mean()
    mean_home_weak   = goals_w[:, 0].mean()

    assert mean_home_strong > mean_home_weak + 0.05, (
        f"Strong home avg={mean_home_strong:.3f} should exceed "
        f"weak home avg={mean_home_weak:.3f} by at least 0.05 goals. "
        "Each row must be sampled from its own PMF."
    )


# ---------------------------------------------------------------------------
# 50k-run timing test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_50k_run_wall_clock():
    """
    Full 50k-run simulation must complete in < 120 s on a development laptop.
    Wall-clock time is printed so the user can track it across commits.

    Mark: slow (skip by default with -m 'not slow'; include with -m slow or plain pytest).
    """
    model = _fit_model()
    groups = _make_groups()
    sim = TournamentSimulator(groups=groups, model=model, settings=_SETTINGS_50K)

    t0 = time.perf_counter()
    results = sim.run()
    elapsed = time.perf_counter() - t0

    print(f"\n[perf] 50k-run wall-clock: {elapsed:.2f}s")

    # Every team should appear in results
    all_teams = {t for g in groups.values() for t in g}
    assert set(results.keys()) == all_teams, "All 48 teams must appear in results"

    # Sanity: champion count sums to n_runs
    champion_total = sum(v.get("champion", 0) for v in results.values())
    assert champion_total == 50000, f"Expected 50000 champions, got {champion_total}"

    assert elapsed < 120.0, (
        f"50k runs took {elapsed:.1f}s — exceeds 120s budget. "
        "Vectorise _build_grid or batch predict() before Phase 7."
    )
