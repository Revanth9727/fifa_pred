"""
time_50k.py
Phase 3 — Task 3: 50k-run wall-clock measurement.

NOTE: timing-only.  The model is fit on synthetic data so champion
probabilities are MEANINGLESS and are NOT printed.  Only the wall-clock
and structural sanity checks (all 48 teams present, sum = n_runs) are
reported.

Usage: python3.11 scripts/time_50k.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpredict.model.match_model import MatchModel
from wcpredict.sim.simulator import TournamentSimulator

# ---------------------------------------------------------------------------
# Synthetic model fit
# ---------------------------------------------------------------------------
SETTINGS = {
    "simulation": {
        "n_runs": 50000,
        "random_seed": 42,
        "et_lambda_scale": 0.33,
        "shootout_strength_coef": 0.10,
    },
    "match_model": {"max_goals": 10},
}

rng = np.random.default_rng(0)
n = 300
elo = rng.normal(0, 200, n)
matches = pd.DataFrame({
    "goals_home":        rng.poisson(np.clip(1.2 + elo / 400, 0.3, 3.5)),
    "goals_away":        rng.poisson(1.0),
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
# Build 12-group tournament with synthetic team names
# ---------------------------------------------------------------------------
groups: dict[str, list[str]] = {
    letter: [f"T{letter}{i}" for i in range(1, 5)]
    for letter in "ABCDEFGHIJKL"
}

sim = TournamentSimulator(groups=groups, model=model, settings=SETTINGS)

# ---------------------------------------------------------------------------
# Time 50k runs
# ---------------------------------------------------------------------------
print("Running 50 000 simulations (synthetic model — timing only)...")
t0 = time.perf_counter()
results = sim.run()
elapsed = time.perf_counter() - t0

# Sanity checks
all_teams = {t for g in groups.values() for t in g}
missing = all_teams - set(results.keys())
champion_total = sum(v.get("champion", 0) for v in results.values())

print()
print(f"  Wall-clock:        {elapsed:.2f}s")
print(f"  Teams in results:  {len(results)} / {len(all_teams)}")
print(f"  Champion sum:      {champion_total} / 50000")
if missing:
    print(f"  Missing teams:     {missing}")
print()

ok = (not missing) and (champion_total == 50000) and (elapsed < 120.0)
if ok:
    print("PASS — 50k runs complete, all teams present, champion sum correct.")
    sys.exit(0)
else:
    if elapsed >= 120.0:
        print(f"FAIL — wall-clock {elapsed:.1f}s exceeds 120s budget.")
    if missing:
        print(f"FAIL — teams missing from results: {missing}")
    if champion_total != 50000:
        print(f"FAIL — champion_total={champion_total}, expected 50000.")
    sys.exit(1)
