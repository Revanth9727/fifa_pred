"""
tests/test_calibration.py — Phase 2 calibration gate.

Buckets matches by elo_diff (including blowout extremes) and compares
predicted vs actual mean goals and W/D/L rates.  Prints the full table
to stdout (run with -s to see it).  Fails if any bucket with ≥10 matches
shows large systematic bias (goals >0.50, WDL rate >0.12).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wcpredict.features.engineer import build_features
from wcpredict.model.calibration import calibration_table, print_calibration_table
from wcpredict.model.match_model import MatchModel


_SETTINGS = {
    "match_model": {
        "max_goals": 10,
        "form_window_months": 9,
        "friendly_weight": 0.3,
        "qualifier_weight": 0.7,
        "tournament_weight": 1.0,
    },
}


def _synthetic_matches(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """
    500-match synthetic dataset with elo as the ground-truth signal.
    All games neutral so the model only needs to capture elo → goals.
    Teams span elo 1400-3250 giving a realistic elo_diff range including
    blowout mismatches (±800+).
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(20)]
    elos = {t: 1400 + 95 * i for i, t in enumerate(teams)}

    rows = []
    for k in range(n):
        i, j = rng.choice(len(teams), size=2, replace=False)
        home, away = teams[i], teams[j]
        eh, ea = elos[home], elos[away]
        lh = np.exp(0.30 + 0.55 * (eh - ea) / 400.0)
        la = np.exp(0.30 - 0.55 * (eh - ea) / 400.0)
        rows.append({
            "date": pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(k * 3)),
            "team_home": home, "team_away": away,
            "goals_home": int(rng.poisson(lh)),
            "goals_away": int(rng.poisson(la)),
            "tournament": "FIFA World Cup",
            "neutral": True,
            "importance_weight": 1.0,
            "home_elo_pre": float(eh),
            "away_elo_pre": float(ea),
            "home_fifa_rank": i + 1,
            "away_fifa_rank": j + 1,
            "venue_lat": None, "venue_lon": None,
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="module")
def fitted_model_and_data():
    m = _synthetic_matches(n=500)
    feats = build_features(m, settings=_SETTINGS)
    model = MatchModel(settings=_SETTINGS)
    model.fit(m, feats)
    return model, m, feats


class TestCalibration:
    def test_calibration_table_shape(self, fitted_model_and_data):
        model, m, feats = fitted_model_and_data
        table = calibration_table(m, feats, model)
        assert len(table) == 5  # five elo_diff buckets
        assert "bucket" in table.columns
        assert "n" in table.columns
        assert "large_bias" in table.columns

    def test_calibration_table_n_sums_to_match_count(self, fitted_model_and_data):
        model, m, feats = fitted_model_and_data
        table = calibration_table(m, feats, model)
        assert table["n"].sum() == len(m)

    def test_calibration_table_no_large_bias(self, fitted_model_and_data):
        """
        No bucket with ≥10 matches should show large systematic bias.
        This is the calibration gate: if it fails the strength→λ mapping
        is broken at the reported bucket.
        """
        model, m, feats = fitted_model_and_data
        table = calibration_table(m, feats, model)

        print("\n\n=== Calibration Table (predicted vs actual, bucketed by elo_diff) ===")
        print_calibration_table(table)
        print()

        flagged = table[table["large_bias"] & (table["n"] >= 10)]
        assert flagged.empty, (
            f"Large systematic bias detected in {len(flagged)} bucket(s):\n"
            + flagged[["bucket", "n",
                        "pred_goals_a", "actual_goals_a",
                        "pred_goals_b", "actual_goals_b",
                        "pred_W", "actual_W",
                        "pred_D", "actual_D",
                        "pred_L", "actual_L"]].to_string(index=False)
        )

    def test_extreme_buckets_populated(self, fitted_model_and_data):
        """The blowout-mismatch buckets (≤-500, ≥+500) must have data."""
        model, m, feats = fitted_model_and_data
        table = calibration_table(m, feats, model)
        low = table.loc[table["bucket"] == "≤-500", "n"].values[0]
        high = table.loc[table["bucket"] == "≥+500", "n"].values[0]
        assert low >= 5, f"Too few matches in ≤-500 bucket: {low}"
        assert high >= 5, f"Too few matches in ≥+500 bucket: {high}"

    def test_strong_team_lambda_exceeds_weak_in_each_bucket(self, fitted_model_and_data):
        """
        Within each bucket where elo_diff > 0, mean pred_goals_a > pred_goals_b.
        This guards against the strength→lambda mapping inverting at extremes.
        """
        model, m, feats = fitted_model_and_data
        table = calibration_table(m, feats, model)
        positive_buckets = table[table["bucket"].isin(["+200..+500", "≥+500"])]
        for _, row in positive_buckets.iterrows():
            if row["n"] < 5:
                continue
            assert row["pred_goals_a"] > row["pred_goals_b"], (
                f"Bucket {row['bucket']}: pred_goals_a={row['pred_goals_a']:.3f} "
                f"<= pred_goals_b={row['pred_goals_b']:.3f} — mapping inverted"
            )
