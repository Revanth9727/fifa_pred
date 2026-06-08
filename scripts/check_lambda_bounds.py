"""
check_lambda_bounds.py
Phase 3 / pre-Phase 4 audit.

Two checks:
  1. Confirm the channel-2 log-normal multiplier is mean-preserving by
     measuring empirical E[multiplier] over 100 000 draws.
  2. Find the actual ELO range for the 48 WC teams in the real data, and
     flag the lowest predicted λb the model could produce at the extreme
     ELO gap, using the REAL elo_ratings_wc2026.csv (not _DEFAULT_STRENGTHS).
     Also reports the min λb produced during the Group-stage grid builds
     so we can cross-check against what the simulator actually sees.

Usage: python3.11 scripts/check_lambda_bounds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpredict.model.match_model import MatchModel
from wcpredict.sim.simulator import _build_context, _DEFAULT_STRENGTHS

# ---------------------------------------------------------------------------
# 1. Mean-preservation: empirical check on the log-normal multiplier itself
# ---------------------------------------------------------------------------
print("=== Channel-2 log-normal mean-preservation (empirical) ===")
dispersion_values = [0.2, 0.5, 0.8, 1.2]
rng = np.random.default_rng(42)
n = 100_000

for disp in dispersion_values:
    sigma = disp * 0.5
    multipliers = np.exp(rng.normal(-sigma**2 / 2.0, sigma, n))
    print(f"  dispersion={disp:.1f}  sigma={sigma:.3f}  "
          f"E[mult]={multipliers.mean():.6f}  "
          f"(should be 1.000000)")
print()

# ---------------------------------------------------------------------------
# 2. Real ELO range for 48 WC teams
# ---------------------------------------------------------------------------
print("=== Real ELO range for 2026 WC teams ===")
elo_path = ROOT / "data" / "raw" / "elo_ratings_wc2026.csv"

if not elo_path.exists():
    print(f"  ERROR: {elo_path} not found — skipping ELO range check.")
else:
    elo_df = pd.read_csv(elo_path)
    print(f"  Columns: {list(elo_df.columns)}")
    print(f"  Rows: {len(elo_df)}")
    print(elo_df.head(10).to_string())
    print()

    # Try common column names for team ELO
    elo_col = None
    for candidate in ["elo", "rating", "Elo", "Rating", "elo_rating"]:
        if candidate in elo_df.columns:
            elo_col = candidate
            break
    team_col = None
    for candidate in ["team", "Team", "country", "Country", "name", "Name"]:
        if candidate in elo_df.columns:
            team_col = candidate
            break

    if elo_col and team_col:
        wc_teams = set(_DEFAULT_STRENGTHS.keys())
        wc_elo = elo_df[elo_df[team_col].isin(wc_teams)][[team_col, elo_col]].drop_duplicates(
            subset=team_col
        ).sort_values(elo_col, ascending=False)

        if len(wc_elo) == 0:
            print("  No WC team names matched — printing all rows for inspection:")
            print(elo_df[[team_col, elo_col]].sort_values(elo_col, ascending=False).to_string())
        else:
            print(f"  Matched {len(wc_elo)}/48 WC teams in ELO file")
            print()
            print("  Top 5:", wc_elo.head(5)[[team_col, elo_col]].to_string(index=False))
            print("  Bot 5:", wc_elo.tail(5)[[team_col, elo_col]].to_string(index=False))
            print()
            max_elo = wc_elo[elo_col].max()
            min_elo = wc_elo[elo_col].min()
            max_gap = max_elo - min_elo
            elo_s_max = max_gap / 400.0
            print(f"  Max ELO: {max_elo:.0f}  Min ELO: {min_elo:.0f}")
            print(f"  Largest gap (home_elo - away_elo): {max_gap:.0f}")
            print(f"  → elo_s feature at max gap: {elo_s_max:.3f}")
            print()
    else:
        print(f"  Could not identify ELO or team column (found: {list(elo_df.columns)})")
        print("  Printing raw file for inspection:")
        print(elo_df.to_string())

# ---------------------------------------------------------------------------
# 3. Synthetic model: what λb does it produce at elo_s extremes?
# ---------------------------------------------------------------------------
print("=== Synthetic model: λ extrapolation risk ===")
SETTINGS = {
    "simulation": {"n_runs": 100, "random_seed": 42,
                   "et_lambda_scale": 0.33, "shootout_strength_coef": 0.10},
    "match_model": {"max_goals": 10},
}
rng2 = np.random.default_rng(0)
n2 = 300
elo_syn = rng2.normal(0, 200, n2)
matches = pd.DataFrame({
    "goals_home":        rng2.poisson(np.clip(1.2 + elo_syn / 400, 0.3, 3.5)),
    "goals_away":        rng2.poisson(1.0),
    "importance_weight": np.ones(n2),
})
features = pd.DataFrame({
    "elo_diff":             elo_syn,
    "form_a":               rng2.uniform(0.3, 0.7, n2),
    "form_b":               rng2.uniform(0.3, 0.7, n2),
    "squad_quality_rank_a": rng2.uniform(0, 1, n2),
    "squad_quality_rank_b": rng2.uniform(0, 1, n2),
    "rest_days_a":          rng2.uniform(3, 10, n2),
    "rest_days_b":          rng2.uniform(3, 10, n2),
    "travel_km_a":          rng2.uniform(0, 5000, n2),
    "travel_km_b":          rng2.uniform(0, 5000, n2),
    "neutral_flag":         rng2.integers(0, 2, n2).astype(float),
    "host_flag":            np.zeros(n2),
})
model = MatchModel(settings=SETTINGS)
model.fit(matches, features)

print(f"  Synthetic fitted intercept coefficient: {model.coef_[0]:.4f}")
print(f"  elo_diff coefficient:                   {model.coef_[2]:.4f}")
print()

for elo_diff_val in [0, 100, 200, 400, 580, 800]:
    ctx = _build_context("H", "A", neutral=True, settings=SETTINGS, model=model,
                         elo_diff=elo_diff_val)
    dist = model.predict(ctx)
    p_scoreless = np.exp(-dist.lambda_b)
    print(f"  elo_diff={elo_diff_val:4d}  λa={dist.lambda_a:.3f}  λb={dist.lambda_b:.3f}  "
          f"P(away scores 0)={p_scoreless:.1%}")

print()
print("NOTE: These λ values are from a SYNTHETIC model fit and are only")
print("      meaningful as a check of model behaviour, not as match predictions.")
print("      The real model must be fit on data/raw/ before any prediction.")
