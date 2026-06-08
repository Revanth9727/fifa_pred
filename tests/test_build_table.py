"""
tests/test_build_table.py

Asserts that build_table() produces a DataFrame matching the schema in
docs/data_exp.md §1, with correct importance weights, non-null join keys,
and no market/odds imports.

All tests use synthetic in-memory fixtures — no data/raw/ files required.
Integration note: against the full historical results.csv (~45k rows) the
row count should be in the tens of thousands.
"""
from __future__ import annotations

import pandas as pd
import pytest

from wcpredict.data.build_table import build_table, _classify_importance


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SETTINGS: dict = {
    "match_model": {
        "friendly_weight": 0.3,
        "qualifier_weight": 0.7,
        "tournament_weight": 1.0,
    },
    "paths": {
        "raw": "data/raw",
        "processed": "data/processed",
        "match_table": "data/processed/matches.parquet",
    },
}

_TEAMS: list[dict] = [
    {"name": "Brazil",      "group": "C", "confederation": "CONMEBOL"},
    {"name": "Argentina",   "group": "J", "confederation": "CONMEBOL"},
    {"name": "Germany",     "group": "E", "confederation": "UEFA"},
    {"name": "France",      "group": "I", "confederation": "UEFA"},
    {"name": "South Korea", "group": "A", "confederation": "AFC"},
]


def _results() -> pd.DataFrame:
    """Three rows covering all three importance tiers and a name alias."""
    return pd.DataFrame({
        "date":       ["2024-01-15",  "2024-03-20",                     "2024-06-01"],
        "home_team":  ["Brazil",      "Korea Republic",                  "Germany"],
        "away_team":  ["Argentina",   "France",                          "Brazil"],
        "home_score": [2,             0,                                 1],
        "away_score": [1,             1,                                 1],
        "tournament": ["FIFA World Cup", "Friendly",                     "FIFA World Cup qualification"],
        "city":       ["São Paulo",   "Seoul",                           "Berlin"],
        "country":    ["Brazil",      "South Korea",                     "Germany"],
        "neutral":    [False,         False,                             False],
    })


def _elo() -> pd.DataFrame:
    return pd.DataFrame({
        "date": [
            "2024-01-01", "2024-01-01",
            "2024-03-01", "2024-03-01",
            "2024-05-01", "2024-05-01",
        ],
        "team": ["Brazil", "Argentina", "Korea Republic", "France", "Germany", "Brazil"],
        "elo":  [2050.0,   2100.0,      1750.0,           1980.0,   1920.0,    2060.0],
    })


def _fifa() -> pd.DataFrame:
    return pd.DataFrame({
        "rank_date":    [
            "2024-01-01", "2024-01-01",
            "2024-03-01", "2024-03-01",
            "2024-05-01", "2024-05-01",
        ],
        "country_full": ["Brazil", "Argentina", "Korea Republic", "France", "Germany", "Brazil"],
        "rank":         [5, 1, 23, 2, 16, 5],
    })


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

REQUIRED_COLS = [
    "match_id",
    "date",
    "team_home",
    "team_away",
    "goals_home",
    "goals_away",
    "tournament",
    "importance_weight",
    "neutral",
    "venue_lat",
    "venue_lon",
]


def test_schema_has_all_required_columns():
    df = build_table(_results(), _elo(), _fifa(), _SETTINGS, teams=_TEAMS)
    for col in REQUIRED_COLS:
        assert col in df.columns, f"Missing required column: {col}"


def test_row_count_matches_input():
    df = build_table(_results(), settings=_SETTINGS, teams=_TEAMS)
    assert len(df) == 3


# ---------------------------------------------------------------------------
# Importance weights
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tournament, expected", [
    ("FIFA World Cup",               1.0),
    ("UEFA Euro",                    1.0),
    ("Copa América",                 1.0),
    ("Africa Cup of Nations",        1.0),
    ("AFC Asian Cup",                1.0),
    ("Friendly",                     0.3),
    ("International Friendly",       0.3),
    ("FIFA World Cup qualification", 0.7),
    ("UEFA Euro qualification",      0.7),
    ("AFC Asian Cup qualification",  0.7),
    ("UEFA Nations League",          0.7),
])
def test_classify_importance(tournament: str, expected: float):
    assert _classify_importance(tournament, _SETTINGS) == pytest.approx(expected)


def test_importance_weights_in_range():
    df = build_table(_results(), settings=_SETTINGS, teams=_TEAMS)
    assert df["importance_weight"].between(0.0, 1.0).all()


def test_all_three_weight_tiers_present():
    df = build_table(_results(), settings=_SETTINGS, teams=_TEAMS)
    weights = set(df["importance_weight"].round(3))
    assert 0.3 in weights, "No friendly rows"
    assert 0.7 in weights, "No qualifier rows"
    assert 1.0 in weights, "No tournament rows"


# ---------------------------------------------------------------------------
# Join keys / nullability
# ---------------------------------------------------------------------------

def test_join_keys_non_null():
    df = build_table(_results(), _elo(), _fifa(), _SETTINGS, teams=_TEAMS)
    for col in ("match_id", "team_home", "team_away"):
        assert df[col].notna().all(), f"Column {col!r} has unexpected NaN"


def test_match_id_unique():
    df = build_table(_results(), settings=_SETTINGS, teams=_TEAMS)
    assert df["match_id"].is_unique, "Duplicate match_id values"


# ---------------------------------------------------------------------------
# Column dtypes
# ---------------------------------------------------------------------------

def test_goals_integer_dtype():
    df = build_table(_results(), settings=_SETTINGS, teams=_TEAMS)
    for col in ("goals_home", "goals_away"):
        assert pd.api.types.is_integer_dtype(df[col]), (
            f"{col} dtype is {df[col].dtype}; expected integer"
        )


def test_neutral_bool_dtype():
    df = build_table(_results(), settings=_SETTINGS, teams=_TEAMS)
    assert pd.api.types.is_bool_dtype(df["neutral"]), (
        f"neutral dtype is {df['neutral'].dtype}; expected bool"
    )


def test_date_datetime_dtype():
    df = build_table(_results(), settings=_SETTINGS, teams=_TEAMS)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


# ---------------------------------------------------------------------------
# Team-name normalisation
# ---------------------------------------------------------------------------

def test_alias_korea_republic_normalised():
    """'Korea Republic' in source data must become 'South Korea'."""
    df = build_table(_results(), settings=_SETTINGS, teams=_TEAMS)
    assert "South Korea" in df["team_home"].values


# ---------------------------------------------------------------------------
# Elo as-of-date join
# ---------------------------------------------------------------------------

def test_elo_columns_present_when_elo_provided():
    df = build_table(_results(), elo_df=_elo(), settings=_SETTINGS, teams=_TEAMS)
    assert "home_elo_pre" in df.columns
    assert "away_elo_pre" in df.columns


def test_elo_pre_match_rating_non_null_for_known_team():
    """Brazil is in elo_df; its pre-match rating must be non-null."""
    df = build_table(_results(), elo_df=_elo(), settings=_SETTINGS, teams=_TEAMS)
    brazil_rows = df[df["team_home"] == "Brazil"]
    assert brazil_rows["home_elo_pre"].notna().all()


def test_elo_is_nan_when_no_elo_df():
    df = build_table(_results(), elo_df=None, settings=_SETTINGS, teams=_TEAMS)
    assert df["home_elo_pre"].isna().all()
    assert df["away_elo_pre"].isna().all()


# ---------------------------------------------------------------------------
# FIFA rank as-of-date join
# ---------------------------------------------------------------------------

def test_fifa_rank_columns_present_when_fifa_provided():
    df = build_table(_results(), fifa_df=_fifa(), settings=_SETTINGS, teams=_TEAMS)
    assert "home_fifa_rank" in df.columns
    assert "away_fifa_rank" in df.columns


def test_fifa_rank_is_nan_when_no_fifa_df():
    df = build_table(_results(), fifa_df=None, settings=_SETTINGS, teams=_TEAMS)
    assert df["home_fifa_rank"].isna().all()
    assert df["away_fifa_rank"].isna().all()


# ---------------------------------------------------------------------------
# Temporal leakage — allow_exact_matches=False (Task 1)
# ---------------------------------------------------------------------------

def test_elo_join_strictly_pre_match():
    """
    A rating stamped ON the match date (the post-match update) must NOT be
    used.  allow_exact_matches=False ensures the join is strictly backward.
    """
    results = pd.DataFrame({
        "date":       ["2024-06-15"],
        "home_team":  ["Brazil"],
        "away_team":  ["Argentina"],
        "home_score": [1],
        "away_score": [0],
        "tournament": ["FIFA World Cup"],
        "neutral":    [False],
    })
    elo = pd.DataFrame({
        "date": ["2024-06-14", "2024-06-15"],
        "team": ["Brazil",     "Brazil"],
        "elo":  [2000.0,        2050.0],   # 2050 is the same-date post-match update
    })
    df = build_table(results, elo_df=elo, settings=_SETTINGS, teams=_TEAMS)
    row = df[df["team_home"] == "Brazil"].iloc[0]
    assert row["home_elo_pre"] == pytest.approx(2000.0), (
        f"Post-match Elo (2050, dated on match day) leaked into table; "
        f"expected pre-match value 2000.0, got {row['home_elo_pre']}"
    )


def test_fifa_join_strictly_pre_match():
    """
    A FIFA rank stamped ON the match date must not be used.
    """
    results = pd.DataFrame({
        "date":       ["2024-06-15"],
        "home_team":  ["Brazil"],
        "away_team":  ["Argentina"],
        "home_score": [1],
        "away_score": [0],
        "tournament": ["FIFA World Cup"],
        "neutral":    [False],
    })
    fifa = pd.DataFrame({
        "rank_date":    ["2024-06-14", "2024-06-15"],
        "country_full": ["Brazil",     "Brazil"],
        "rank":         [5,            4],   # rank 4 is the same-date post-match update
    })
    df = build_table(results, fifa_df=fifa, settings=_SETTINGS, teams=_TEAMS)
    row = df[df["team_home"] == "Brazil"].iloc[0]
    assert row["home_fifa_rank"] == pytest.approx(5.0), (
        f"Post-match FIFA rank (4, dated on match day) leaked into table; "
        f"expected pre-match value 5, got {row['home_fifa_rank']}"
    )


# ---------------------------------------------------------------------------
# Venue coordinates (Task 2)
# ---------------------------------------------------------------------------

def _venues_fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "city":    ["Atlanta",       "Mexico City",   "Vancouver"],
        "country": ["United States", "Mexico",        "Canada"],
        "lat":     [33.7553,          19.3029,          49.2779],
        "lon":     [-84.4006,         -99.1505,         -123.1121],
        "stadium": ["Mercedes-Benz Stadium", "Estadio Azteca", "BC Place"],
    })


def test_venue_coords_non_null_for_known_venue():
    """Matches at a host-city venue must have non-null lat/lon after the join."""
    results = pd.DataFrame({
        "date":       ["2026-06-15"],
        "home_team":  ["Brazil"],
        "away_team":  ["Germany"],
        "home_score": [2],
        "away_score": [1],
        "tournament": ["FIFA World Cup"],
        "city":       ["Atlanta"],
        "country":    ["United States"],
        "neutral":    [True],
    })
    df = build_table(results, settings=_SETTINGS, teams=_TEAMS, venues_df=_venues_fixture())
    assert df["venue_lat"].notna().all(), "venue_lat is NaN for a known host city"
    assert df["venue_lon"].notna().all(), "venue_lon is NaN for a known host city"
    assert df["venue_lat"].iloc[0] == pytest.approx(33.7553)
    assert df["venue_lon"].iloc[0] == pytest.approx(-84.4006)


def test_venue_coords_nan_for_unknown_venue():
    """Matches at venues not in the lookup must retain NaN coords."""
    results = pd.DataFrame({
        "date":       ["2024-01-15"],
        "home_team":  ["Brazil"],
        "away_team":  ["Argentina"],
        "home_score": [2],
        "away_score": [1],
        "tournament": ["FIFA World Cup"],
        "city":       ["São Paulo"],
        "country":    ["Brazil"],
        "neutral":    [False],
    })
    df = build_table(results, settings=_SETTINGS, teams=_TEAMS, venues_df=_venues_fixture())
    assert df["venue_lat"].isna().all(), "venue_lat should be NaN for unlisted venue"
    assert df["venue_lon"].isna().all(), "venue_lon should be NaN for unlisted venue"


# ---------------------------------------------------------------------------
# No market / odds imports (ai_rules.md #3)
# ---------------------------------------------------------------------------

def test_no_market_imports_in_build_table_and_ingest():
    """
    build_table and ingest must not import any odds/market/calibration module.
    Enforced via AST scan of each source file.
    """
    import ast
    import importlib.util

    forbidden_keywords = ("odds", "market", "calibration", "betfair", "pinnacle")
    modules_to_check = ("wcpredict.data.build_table", "wcpredict.data.ingest")

    for mod_name in modules_to_check:
        spec = importlib.util.find_spec(mod_name)
        assert spec is not None, f"Module {mod_name} not found"
        assert spec.origin is not None
        source = __import__("pathlib").Path(spec.origin).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported_names = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [alias.name for alias in node.names]
                )
                for name in imported_names:
                    for kw in forbidden_keywords:
                        assert kw not in (name or "").lower(), (
                            f"{mod_name} imports a forbidden market/odds module: {name!r}"
                        )
