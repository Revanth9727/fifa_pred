"""
check_cache_key.py
Phase 3 — Task 1: cache-key correctness.

Retrieves the scoreline grid for:
  (a) a strong-vs-weak matchup (elo_diff = +580, e.g. Argentina vs Jordan)
  (b) a weak-vs-weak matchup  (elo_diff =  +60, e.g. Qatar vs Jordan)

at neutral context and reports whether the grids differ.
They MUST differ; identical grids would mean elo_diff is not flowing into
the predict call and the cache is effectively strength-blind.

Usage: python3.11 scripts/check_cache_key.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve project root regardless of cwd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpredict.model.match_model import MatchModel
from wcpredict.sim.simulator import _build_context, _cached_predict
from wcpredict.sim.sampler import _flat_grid_cached

# ---------------------------------------------------------------------------
# Synthetic model (timing-only fit — context flows through correctly)
# ---------------------------------------------------------------------------
SETTINGS = {
    "simulation": {"n_runs": 100, "random_seed": 42,
                   "et_lambda_scale": 0.33, "shootout_strength_coef": 0.10},
    "match_model": {"max_goals": 10},
}

rng = np.random.default_rng(7)
n = 300
elo = rng.normal(0, 200, n)
matches = pd.DataFrame({
    "goals_home":        rng.poisson(np.clip(1.2 + elo / 400, 0.3, 3.5)),
    "goals_away":        rng.poisson(np.clip(1.0 - elo / 400, 0.3, 3.0)),
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
model = MatchModel(settings=SETTINGS)
model.fit(matches, features)

# ---------------------------------------------------------------------------
# Retrieve grids for two very different matchups at neutral context
# ---------------------------------------------------------------------------
MAX_GOALS = 10

# (a) strong vs weak: Argentina (1960) vs Jordan (1380) → elo_diff = +580
ctx_strong = _build_context(
    "Argentina", "Jordan", neutral=True, settings=SETTINGS, model=model,
    elo_diff=1960 - 1380,
)
# (b) weak vs weak: Qatar (1440) vs Jordan (1380) → elo_diff = +60
ctx_weak = _build_context(
    "Qatar", "Jordan", neutral=True, settings=SETTINGS, model=model,
    elo_diff=1440 - 1380,
)

cache: dict = {}
d_strong = _cached_predict(model, ctx_strong, cache)
d_weak   = _cached_predict(model, ctx_weak,   cache)

grid_strong = _flat_grid_cached(
    float(d_strong.lambda_a), float(d_strong.lambda_b), float(d_strong.dependence), MAX_GOALS
)
grid_weak = _flat_grid_cached(
    float(d_weak.lambda_a), float(d_weak.lambda_b), float(d_weak.dependence), MAX_GOALS
)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("=== Cache-key correctness check ===")
print()
print(f"  Strong-vs-weak (Argentina vs Jordan, elo_diff=+580):")
print(f"    λa={d_strong.lambda_a:.4f}  λb={d_strong.lambda_b:.4f}")
print()
print(f"  Weak-vs-weak   (Qatar vs Jordan, elo_diff=+60):")
print(f"    λa={d_weak.lambda_a:.4f}  λb={d_weak.lambda_b:.4f}")
print()

grids_differ = not np.allclose(grid_strong, grid_weak)
lambdas_differ = (
    abs(d_strong.lambda_a - d_weak.lambda_a) > 1e-6
    or abs(d_strong.lambda_b - d_weak.lambda_b) > 1e-6
)

print(f"  λ values differ:   {lambdas_differ}")
print(f"  Grids differ:      {grids_differ}")
print()

if grids_differ and lambdas_differ:
    print("PASS — elo_diff flows into predict(); distinct matchups produce distinct grids.")
    print("       Cache key (λa, λb, ρ, max_goals) is correct: same λ → same grid, "
          "different λ → different grid.")
    sys.exit(0)
else:
    print("FAIL — grids are IDENTICAL. elo_diff is not reaching predict().")
    print("       Fix: pass elo_diff=strengths[home]-strengths[away] in _build_context().")
    sys.exit(1)
