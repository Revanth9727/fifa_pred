"""
tests/test_features.py

Covers:
  - Feature schema matches data_exp.md §2
  - predict() returns sampleable GoalsDist (not a point)
  - derive_wdl() sums to ≈1 across the full strength range
  - No forbidden imports in features/, model/ (ai_rules.md #2, #3)
  - Shootout is near-coinflip and deterministic given seed

All tests use synthetic in-memory fixtures — no data/raw/ files required.
"""
from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wcpredict.features.engineer import (
    FEATURE_COLS,
    build_features,
    compute_squad_quality_ranks,
    haversine_km,
)
from wcpredict.model.match_model import GoalsDist, MatchModel, derive_wdl
from wcpredict.model.shootout import resolve_shootout


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_SETTINGS = {
    "match_model": {
        "max_goals": 10,
        "form_window_months": 9,
        "friendly_weight": 0.3,
        "qualifier_weight": 0.7,
        "tournament_weight": 1.0,
    },
    "simulation": {
        "shootout_strength_coef": 0.10,
    },
}


def _matches(n: int = 6) -> pd.DataFrame:
    """Synthetic match table matching data_exp.md §1."""
    rows = [
        ("2023-01-15", "Brazil",    "Argentina",  2, 1, "FIFA World Cup",              False, 1.0,  None,     None),
        ("2023-03-20", "Germany",   "France",      0, 1, "Friendly",                   False, 0.3,  None,     None),
        ("2023-06-10", "Spain",     "Portugal",    1, 1, "FIFA World Cup qualification",False, 0.7,  None,     None),
        ("2023-09-05", "Brazil",    "Germany",     3, 0, "FIFA World Cup",              True,  1.0,  33.7553,  -84.4006),
        ("2023-11-14", "Argentina", "Spain",       2, 2, "Friendly",                   True,  0.3,  19.3029,  -99.1505),
        ("2024-01-20", "France",    "Portugal",    1, 0, "UEFA Euro",                   False, 1.0,  48.8566,  2.3522),
    ]
    df = pd.DataFrame(rows, columns=[
        "date", "team_home", "team_away",
        "goals_home", "goals_away",
        "tournament", "neutral", "importance_weight",
        "venue_lat", "venue_lon",
    ])
    df["date"] = pd.to_datetime(df["date"])
    df["home_elo_pre"] = [2050.0, 1920.0, 1900.0, 2060.0, 2100.0, 1980.0]
    df["away_elo_pre"] = [2100.0, 1980.0, 1950.0, 1920.0, 1900.0, 1950.0]
    df["home_fifa_rank"] = [5, 16, 8, 5, 1, 2]
    df["away_fifa_rank"] = [1, 2, 10, 16, 8, 10]
    return df.head(n)


def _squad_values() -> pd.DataFrame:
    return pd.DataFrame({
        "team": ["Brazil", "Argentina", "Germany", "France", "Spain", "Portugal"],
        "total_value": [900e6, 850e6, 700e6, 950e6, 800e6, 750e6],
    })


# ---------------------------------------------------------------------------
# engineer.py — feature schema
# ---------------------------------------------------------------------------

class TestFeatureSchema:
    def test_all_feature_cols_present(self):
        feats = build_features(_matches(), settings=_SETTINGS)
        for col in FEATURE_COLS:
            assert col in feats.columns, f"Missing feature column: {col}"

    def test_no_extra_columns(self):
        feats = build_features(_matches(), settings=_SETTINGS)
        assert set(feats.columns) == set(FEATURE_COLS)

    def test_row_count_aligned(self):
        m = _matches()
        feats = build_features(m, settings=_SETTINGS)
        assert len(feats) == len(m)

    def test_elo_diff_sign(self):
        """elo_diff = home_elo_pre - away_elo_pre."""
        m = _matches(1)
        feats = build_features(m, settings=_SETTINGS)
        expected = m["home_elo_pre"].iloc[0] - m["away_elo_pre"].iloc[0]
        assert feats["elo_diff"].iloc[0] == pytest.approx(expected)

    def test_fifa_rank_diff_sign(self):
        """fifa_rank_diff = away_rank - home_rank (lower rank = better team)."""
        m = _matches(1)
        feats = build_features(m, settings=_SETTINGS)
        expected = m["away_fifa_rank"].iloc[0] - m["home_fifa_rank"].iloc[0]
        assert feats["fifa_rank_diff"].iloc[0] == pytest.approx(expected)

    def test_neutral_flag_bool(self):
        feats = build_features(_matches(), settings=_SETTINGS)
        assert feats["neutral_flag"].isin([0, 1]).all()

    def test_host_flag_detects_host_nations(self):
        m = _matches()
        m.loc[0, "team_home"] = "United States"
        feats = build_features(m, settings=_SETTINGS)
        assert feats["host_flag"].iloc[0] == 1

    def test_importance_weight_passthrough(self):
        m = _matches()
        feats = build_features(m, settings=_SETTINGS)
        assert (feats["importance_weight"] == m["importance_weight"].values).all()

    def test_spi_diff_nan(self):
        feats = build_features(_matches(), settings=_SETTINGS)
        assert feats["spi_diff"].isna().all()

    def test_form_within_range(self):
        feats = build_features(_matches(), settings=_SETTINGS)
        valid = feats["form_a"].dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all()

    def test_form_nan_for_first_match(self):
        """First match in the dataset has no prior history → form is NaN."""
        feats = build_features(_matches(), settings=_SETTINGS)
        assert pd.isna(feats["form_a"].iloc[0])

    def test_squad_quality_rank_in_unit_interval(self):
        feats = build_features(_matches(), squad_values=_squad_values(), settings=_SETTINGS)
        valid_a = feats["squad_quality_rank_a"].dropna()
        valid_b = feats["squad_quality_rank_b"].dropna()
        assert (valid_a >= 0.0).all() and (valid_a <= 1.0).all()
        assert (valid_b >= 0.0).all() and (valid_b <= 1.0).all()

    def test_squad_quality_rank_nan_without_data(self):
        feats = build_features(_matches(), squad_values=None, settings=_SETTINGS)
        assert feats["squad_quality_rank_a"].isna().all()
        assert feats["squad_quality_rank_b"].isna().all()

    def test_rest_days_non_negative(self):
        feats = build_features(_matches(), settings=_SETTINGS)
        non_nan = feats["rest_days_a"].dropna()
        assert (non_nan >= 0).all()

    def test_travel_km_non_negative(self):
        feats = build_features(_matches(), settings=_SETTINGS)
        non_nan = feats["travel_km_a"].dropna()
        assert (non_nan >= 0).all()


# ---------------------------------------------------------------------------
# compute_squad_quality_ranks
# ---------------------------------------------------------------------------

class TestSquadQualityRanks:
    def test_best_squad_gets_rank_one(self):
        sv = pd.DataFrame({
            "team": ["A", "B", "C"],
            "total_value": [1000.0, 500.0, 250.0],
        })
        ranks = compute_squad_quality_ranks(sv)
        assert ranks["A"] == pytest.approx(1.0)

    def test_worst_squad_gets_rank_zero(self):
        sv = pd.DataFrame({
            "team": ["A", "B", "C"],
            "total_value": [1000.0, 500.0, 250.0],
        })
        ranks = compute_squad_quality_ranks(sv)
        assert ranks["C"] == pytest.approx(0.0)

    def test_rank_is_not_raw_value(self):
        """A has 10000x the raw value of B — rank signal must not reflect that."""
        sv = pd.DataFrame({
            "team": ["A", "B", "C"],
            "total_value": [10_000e6, 1e6, 5_000e6],
        })
        ranks = compute_squad_quality_ranks(sv)
        # Rank values are bounded to [0, 1] regardless of raw value spread
        assert all(0.0 <= float(r) <= 1.0 for r in ranks.values)
        # A is the richest squad — it gets the best rank
        assert ranks["A"] > ranks["B"]
        # The rank gap between A and B is at most 1.0, not ~10000x the raw ratio
        assert ranks.max() - ranks.min() <= 1.0

    def test_single_team_gets_rank_one(self):
        sv = pd.DataFrame({"team": ["Brazil"], "total_value": [900e6]})
        ranks = compute_squad_quality_ranks(sv)
        assert ranks["Brazil"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# haversine_km sanity checks
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine_km(40.0, -74.0, 40.0, -74.0) == pytest.approx(0.0)

    def test_new_york_london_approx(self):
        d = haversine_km(40.7128, -74.0060, 51.5074, -0.1278)
        assert 5500 < d < 5600


# ---------------------------------------------------------------------------
# match_model.py — GoalsDist is a distribution, not a point
# ---------------------------------------------------------------------------

def _large_synthetic_matches(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic dataset large enough for reliable MLE convergence.
    Goals generated from a Poisson where elo is the dominant positive signal,
    so the optimizer reliably recovers a positive elo coefficient.
    """
    rng = np.random.default_rng(seed)
    teams = [f"Team{i}" for i in range(16)]
    elos = {t: 1400 + 150 * i for i, t in enumerate(teams)}

    rows = []
    for k in range(n):
        i, j = rng.choice(len(teams), size=2, replace=False)
        home, away = teams[i], teams[j]
        eh, ea = elos[home], elos[away]
        lh = np.exp(0.30 + 0.55 * (eh - ea) / 400.0)
        la = np.exp(0.30 - 0.55 * (eh - ea) / 400.0)
        rows.append({
            "date": pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(k * 4)),
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
            "venue_lat": None,
            "venue_lon": None,
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _fitted_model(n_matches: int = 6) -> tuple[MatchModel, pd.DataFrame]:
    m = _matches(n_matches)
    from wcpredict.features.engineer import build_features as bf
    feats = bf(m, settings=_SETTINGS)
    model = MatchModel(settings=_SETTINGS)
    model.fit(m, feats)
    return model, feats


def _fitted_model_large() -> MatchModel:
    """Model fitted on 300-match synthetic data with a reliable positive elo coef."""
    from wcpredict.features.engineer import build_features as bf
    m = _large_synthetic_matches()
    feats = bf(m, settings=_SETTINGS)
    model = MatchModel(settings=_SETTINGS)
    model.fit(m, feats)
    return model


class TestMatchModel:
    def test_predict_returns_goals_dist(self):
        model, feats = _fitted_model()
        ctx = feats.iloc[0].to_dict()
        dist = model.predict(ctx)
        assert isinstance(dist, GoalsDist)

    def test_predict_lambda_a_positive(self):
        model, feats = _fitted_model()
        dist = model.predict(feats.iloc[0].to_dict())
        assert dist.lambda_a > 0

    def test_predict_lambda_b_positive(self):
        model, feats = _fitted_model()
        dist = model.predict(feats.iloc[0].to_dict())
        assert dist.lambda_b > 0

    def test_predict_has_dependence_field(self):
        """dist.dependence must exist and be finite — required for the sampler."""
        model, feats = _fitted_model()
        dist = model.predict(feats.iloc[0].to_dict())
        assert math.isfinite(dist.dependence)

    def test_higher_elo_diff_raises_lambda_a(self):
        """A stronger team_a (higher elo_diff) should predict more goals for a."""
        model = _fitted_model_large()
        ctx_weak = {"elo_diff": -400.0, "neutral_flag": 1.0}
        ctx_strong = {"elo_diff": 400.0, "neutral_flag": 1.0}
        dist_weak = model.predict(ctx_weak)
        dist_strong = model.predict(ctx_strong)
        assert dist_strong.lambda_a > dist_weak.lambda_a

    def test_predict_before_fit_raises(self):
        model = MatchModel(settings=_SETTINGS)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict({"elo_diff": 0.0})

    def test_fit_returns_self(self):
        m = _matches()
        from wcpredict.features.engineer import build_features as bf
        feats = bf(m, settings=_SETTINGS)
        model = MatchModel(settings=_SETTINGS)
        result = model.fit(m, feats)
        assert result is model


# ---------------------------------------------------------------------------
# derive_wdl — sums to ≈1 across the strength range
# ---------------------------------------------------------------------------

class TestDeriveWdl:
    @pytest.mark.parametrize("elo_diff", [-600, -200, 0, 200, 600])
    def test_wdl_sums_to_one(self, elo_diff: int):
        """W+D+L must sum to 1.0 for any strength level, including blowout mismatches."""
        model, feats = _fitted_model()
        ctx = {"elo_diff": float(elo_diff), "neutral_flag": 1.0}
        dist = model.predict(ctx)
        w, d, l = model.derive_wdl(dist)
        assert w + d + l == pytest.approx(1.0, abs=1e-6), (
            f"elo_diff={elo_diff}: W+D+L = {w + d + l}"
        )

    @pytest.mark.parametrize("elo_diff", [-600, -200, 0, 200, 600])
    def test_wdl_all_non_negative(self, elo_diff: int):
        model, feats = _fitted_model()
        ctx = {"elo_diff": float(elo_diff), "neutral_flag": 1.0}
        dist = model.predict(ctx)
        w, d, l = model.derive_wdl(dist)
        assert w >= 0 and d >= 0 and l >= 0

    def test_strong_team_wins_more_often(self):
        model = _fitted_model_large()
        ctx_fav = {"elo_diff": 500.0, "neutral_flag": 1.0}
        ctx_dog = {"elo_diff": -500.0, "neutral_flag": 1.0}
        w_fav, _, _ = model.derive_wdl(model.predict(ctx_fav))
        w_dog, _, _ = model.derive_wdl(model.predict(ctx_dog))
        assert w_fav > w_dog

    def test_module_level_derive_wdl(self):
        dist = GoalsDist(lambda_a=1.4, lambda_b=1.2, dependence=0.05)
        w, d, l = derive_wdl(dist, max_goals=10)
        assert w + d + l == pytest.approx(1.0, abs=1e-6)

    def test_draw_probability_nonzero(self):
        """Draws must be possible — not modelled away."""
        model, feats = _fitted_model()
        dist = model.predict({"elo_diff": 0.0, "neutral_flag": 1.0})
        _, d, _ = model.derive_wdl(dist)
        assert d > 0.05

    def test_residual_strength_range(self):
        """
        Residual check: lambda_a / lambda_b ratio should increase monotonically
        with elo_diff across the full strength range (including blowout mismatches).
        This guards against the strength→lambda mapping inverting at extremes.
        """
        model = _fitted_model_large()
        elo_diffs = [-800, -400, -200, 0, 200, 400, 800]
        ratios = []
        for ed in elo_diffs:
            dist = model.predict({"elo_diff": float(ed), "neutral_flag": 1.0})
            ratios.append(dist.lambda_a / dist.lambda_b)
        for i in range(len(ratios) - 1):
            assert ratios[i] < ratios[i + 1], (
                f"lambda ratio not monotone at elo_diff={elo_diffs[i]}: "
                f"{ratios[i]:.3f} >= {ratios[i+1]:.3f}"
            )


# ---------------------------------------------------------------------------
# Task 1 — Form temporal safety
# ---------------------------------------------------------------------------

class TestFormTemporalSafety:
    def test_same_date_result_excluded(self):
        """A result on exactly the match date must NOT count towards form."""
        from wcpredict.features.engineer import _compute_form
        target = pd.Timestamp("2023-06-01")
        history = pd.DataFrame([
            # strictly before → SHOULD count (win, pts=1)
            {"date": "2023-02-10", "team_home": "A", "team_away": "X",
             "goals_home": 2, "goals_away": 0},
            # same day → must NOT count (loss, pts=0)
            {"date": "2023-06-01", "team_home": "X", "team_away": "A",
             "goals_home": 5, "goals_away": 0},
        ])
        history["date"] = pd.to_datetime(history["date"])
        form = _compute_form("A", target, history, window_months=9)
        # Only the February win contributes → form == 1.0
        assert form == pytest.approx(1.0, abs=1e-9), (
            f"Same-date result leaked into form: got {form}"
        )

    def test_future_result_excluded(self):
        """A result strictly after the match date must NOT count towards form."""
        from wcpredict.features.engineer import _compute_form
        target = pd.Timestamp("2023-06-01")
        history = pd.DataFrame([
            # before → counts (win)
            {"date": "2023-03-01", "team_home": "A", "team_away": "X",
             "goals_home": 1, "goals_away": 0},
            # after → must NOT count (loss)
            {"date": "2023-09-15", "team_home": "X", "team_away": "A",
             "goals_home": 5, "goals_away": 0},
        ])
        history["date"] = pd.to_datetime(history["date"])
        form = _compute_form("A", target, history, window_months=9)
        assert form == pytest.approx(1.0, abs=1e-9), (
            f"Future result leaked into form: got {form}"
        )

    def test_adding_same_date_result_does_not_change_form(self):
        """
        build_features form is identical with or without a same-day result appended.
        Uses Brazil at row-3 (2023-09-05) which has a prior result on 2023-01-15.
        """
        from wcpredict.features.engineer import _compute_form

        target = pd.Timestamp("2023-09-05")
        # Prior win strictly before target
        prior = pd.DataFrame([{
            "date": "2023-01-15", "team_home": "Brazil", "team_away": "X",
            "goals_home": 2, "goals_away": 0,
        }])
        # Blowout loss ON the target date — must not affect form
        same_day_contaminant = pd.DataFrame([{
            "date": "2023-09-05", "team_home": "Foe", "team_away": "Brazil",
            "goals_home": 9, "goals_away": 0,
        }])
        prior["date"] = pd.to_datetime(prior["date"])
        same_day_contaminant["date"] = pd.to_datetime(same_day_contaminant["date"])

        history_clean = prior
        history_contaminated = pd.concat([prior, same_day_contaminant], ignore_index=True)

        form_clean = _compute_form("Brazil", target, history_clean, window_months=9)
        form_dirty = _compute_form("Brazil", target, history_contaminated, window_months=9)

        assert not pd.isna(form_clean), "Expected non-NaN form (prior result exists)"
        assert form_clean == pytest.approx(form_dirty, abs=1e-9), (
            f"Same-date contaminant changed form: clean={form_clean:.6f}, dirty={form_dirty:.6f}"
        )

    def test_window_cutoff_excludes_old_results(self):
        """Results older than form_window_months must not count."""
        from wcpredict.features.engineer import _compute_form
        target = pd.Timestamp("2023-06-01")
        history = pd.DataFrame([
            # 10 months ago — outside the 9-month window
            {"date": "2022-08-01", "team_home": "A", "team_away": "X",
             "goals_home": 3, "goals_away": 0},
            # 3 months ago — inside the window
            {"date": "2023-03-01", "team_home": "X", "team_away": "A",
             "goals_home": 0, "goals_away": 0},
        ])
        history["date"] = pd.to_datetime(history["date"])
        form = _compute_form("A", target, history, window_months=9)
        # Only the draw 3 months ago counts → form == 0.5
        assert form == pytest.approx(0.5, abs=1e-9), (
            f"Old result outside window leaked: got {form}"
        )


# ---------------------------------------------------------------------------
# Task 2 — Home advantage via flags, not a blanket constant
# ---------------------------------------------------------------------------

class TestNeutralVenueSymmetry:
    def test_neutral_venue_equal_teams_symmetric_lambda(self):
        """At neutral venue with equal features, lambda_a must equal lambda_b."""
        model, _ = _fitted_model()
        ctx = {
            "neutral_flag": 1.0, "elo_diff": 0.0,
            "form_a": 0.5, "form_b": 0.5,
            "squad_quality_rank_a": 0.5, "squad_quality_rank_b": 0.5,
            "rest_days_a": 7.0, "rest_days_b": 7.0,
            "travel_km_a": 500.0, "travel_km_b": 500.0,
            "host_flag": 0.0,
        }
        dist = model.predict(ctx)
        assert dist.lambda_a == pytest.approx(dist.lambda_b, rel=1e-10), (
            f"Spurious home edge at neutral venue: λ_a={dist.lambda_a:.4f}, λ_b={dist.lambda_b:.4f}"
        )

    def test_home_advantage_disappears_at_neutral(self):
        """Setting neutral_flag=1 must eliminate whatever home boost existed."""
        model, _ = _fitted_model()
        model.coef_[1] = 0.30  # force a non-zero home_adv coef
        equal = {"elo_diff": 0.0, "form_a": 0.5, "form_b": 0.5,
                 "squad_quality_rank_a": 0.5, "squad_quality_rank_b": 0.5,
                 "rest_days_a": 7.0, "rest_days_b": 7.0,
                 "travel_km_a": 500.0, "travel_km_b": 500.0, "host_flag": 0.0}
        dist_home = model.predict({**equal, "neutral_flag": 0.0})
        dist_neutral = model.predict({**equal, "neutral_flag": 1.0})
        # Neutral → symmetric
        assert dist_neutral.lambda_a == pytest.approx(dist_neutral.lambda_b, rel=1e-10)
        # Non-neutral → asymmetric (home team boosted)
        assert dist_home.lambda_a > dist_home.lambda_b

    def test_neutral_flag_is_the_switch_not_team_identity(self):
        """The home/away label alone never drives asymmetry; neutral_flag does."""
        model, _ = _fitted_model()
        model.coef_[1] = 0.25
        equal = {"elo_diff": 0.0, "form_a": 0.5, "form_b": 0.5,
                 "squad_quality_rank_a": 0.5, "squad_quality_rank_b": 0.5,
                 "rest_days_a": 7.0, "rest_days_b": 7.0,
                 "travel_km_a": 500.0, "travel_km_b": 500.0, "host_flag": 0.0}
        dist_n = model.predict({**equal, "neutral_flag": 1.0})
        dist_h = model.predict({**equal, "neutral_flag": 0.0})
        # The magnitude of asymmetry at non-neutral is coef[1]*2 in log-space
        import math
        expected_ratio = math.exp(2 * model.coef_[1])
        actual_ratio = dist_h.lambda_a / dist_h.lambda_b
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-6)
        # At neutral the ratio is exactly 1
        assert dist_n.lambda_a / dist_n.lambda_b == pytest.approx(1.0, rel=1e-10)


# ---------------------------------------------------------------------------
# shootout.py
# ---------------------------------------------------------------------------

class TestShootout:
    def test_returns_one_of_the_two_teams(self):
        rng = np.random.default_rng(0)
        winner = resolve_shootout("A", "B", 0.0, rng, coef=0.1)
        assert winner in ("A", "B")

    def test_deterministic_given_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        w1 = resolve_shootout("X", "Y", 0.2, rng1, coef=0.1)
        w2 = resolve_shootout("X", "Y", 0.2, rng2, coef=0.1)
        assert w1 == w2

    def test_near_coinflip_at_zero_diff(self):
        """At strength_diff=0, both outcomes must occur over many trials."""
        rng = np.random.default_rng(7)
        winners = [resolve_shootout("A", "B", 0.0, rng, coef=0.1) for _ in range(500)]
        p_a = winners.count("A") / 500
        assert 0.35 < p_a < 0.65, f"Not near-coinflip: P(A)={p_a}"

    def test_stronger_team_wins_more(self):
        """A strong positive strength_diff should favour team_a."""
        rng = np.random.default_rng(99)
        winners = [resolve_shootout("A", "B", 1.0, rng, coef=0.1) for _ in range(500)]
        p_a = winners.count("A") / 500
        assert p_a > 0.55

    def test_clip_prevents_certain_outcome(self):
        """Even an extreme strength_diff must not produce a certain winner."""
        rng = np.random.default_rng(1)
        results = [resolve_shootout("A", "B", 1000.0, rng, coef=0.1) for _ in range(100)]
        assert "B" in results, "Extreme diff produced a certain outcome (clip failed)"


# ---------------------------------------------------------------------------
# No forbidden imports in features/ and model/ (ai_rules.md #2, #3)
# ---------------------------------------------------------------------------

_FORBIDDEN = ("llm", "adjust", "odds", "market", "calibration", "betfair", "pinnacle")

_MODULES_TO_CHECK = (
    "wcpredict.features.engineer",
    "wcpredict.model.match_model",
    "wcpredict.model.shootout",
)


@pytest.mark.parametrize("mod_name", _MODULES_TO_CHECK)
def test_no_forbidden_imports(mod_name: str):
    spec = importlib.util.find_spec(mod_name)
    assert spec is not None, f"Module {mod_name} not found"
    source = Path(spec.origin).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [alias.name for alias in node.names]
            )
            for name in names:
                for kw in _FORBIDDEN:
                    assert kw not in (name or "").lower(), (
                        f"{mod_name} contains forbidden import touching {kw!r}: {name!r}"
                    )
