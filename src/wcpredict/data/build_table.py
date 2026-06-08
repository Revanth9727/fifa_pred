"""
data/build_table.py — pure transform, no network access.

Joins raw historical sources into the per-match training table
(data/processed/matches.parquet) matching the schema in docs/data_exp.md §1.

Also attaches pre-match Elo and FIFA rank columns (home_elo_pre, away_elo_pre,
home_fifa_rank, away_fifa_rank) so that features/engineer.py can compute
difference signals without re-loading the raw files.

Entity-resolution hook: team-name normalization is delegated to
data/entity_resolve.py (build-time only). Claude resolution is available there
for manual build-time use, but the default path remains deterministic/offline.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Project-root discovery
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Walk up from this file until we find the directory containing pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]  # fallback: src/wcpredict/data/ → 3 levels up


def _load_settings(settings_path: Path | str | None = None) -> dict[str, Any]:
    root = _project_root()
    path = Path(settings_path) if settings_path else root / "config" / "settings.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _load_teams_config(teams_path: Path | str | None = None) -> list[dict[str, Any]]:
    root = _project_root()
    path = Path(teams_path) if teams_path else root / "config" / "teams.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("teams", [])


# ---------------------------------------------------------------------------
# Team-name normalisation
# Phase 6 hook: replace _make_normalizer() body with entity_resolve.py call.
# ---------------------------------------------------------------------------

_STATIC_ALIASES: dict[str, str] = {
    # United States
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.": "United States",
    "united states of america": "United States",
    # South Korea
    "korea republic": "South Korea",
    "republic of korea": "South Korea",
    # Czechia
    "czech republic": "Czechia",
    # Côte d'Ivoire
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "cote divoire": "Côte d'Ivoire",
    # Cabo Verde
    "cape verde": "Cabo Verde",
    "cape verde islands": "Cabo Verde",
    # DR Congo
    "drc": "DR Congo",
    "congo dr": "DR Congo",
    "dr congo": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "democratic republic of congo": "DR Congo",
    # Bosnia and Herzegovina
    "bosnia": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    # Türkiye
    "turkey": "Türkiye",
    "turkiye": "Türkiye",
    # Curaçao
    "curacao": "Curaçao",
    # China
    "china pr": "China",
    "china, pr": "China",
    "china people's republic": "China",
}


def _build_alias_map(teams: list[dict[str, Any]]) -> dict[str, str]:
    """Build a normalised-key → canonical-name map from teams.yaml + static aliases."""
    alias_map: dict[str, str] = {}
    for team in teams:
        canonical = team["name"]
        alias_map[canonical.lower().strip()] = canonical
    alias_map.update(_STATIC_ALIASES)
    return alias_map


def _make_normalizer(alias_map: dict[str, str]):
    """Return a callable: raw team string → canonical name (best-effort)."""
    teams = sorted(set(alias_map.values()))
    from wcpredict.data.entity_resolve import make_resolver

    normalize, _ = make_resolver(teams, alias_map)
    return normalize


# ---------------------------------------------------------------------------
# Importance-weight classification
# ---------------------------------------------------------------------------

def _classify_importance(tournament: str, settings: dict[str, Any]) -> float:
    """Map a tournament string to a training importance weight from config."""
    mm = settings.get("match_model", {})
    friendly_w: float = mm.get("friendly_weight", 0.3)
    qualifier_w: float = mm.get("qualifier_weight", 0.7)
    tournament_w: float = mm.get("tournament_weight", 1.0)

    t = str(tournament).lower()
    if "friendly" in t:
        return friendly_w
    if any(kw in t for kw in ("qualification", "qualifying", "qualifier", "nations league")):
        return qualifier_w
    return tournament_w


# ---------------------------------------------------------------------------
# Elo normalisation helper
# ---------------------------------------------------------------------------

def _normalize_elo_df(elo_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw Elo DataFrame to long format: (date, team, elo_rating)."""
    df = elo_df.copy()
    cl = {c.lower(): c for c in df.columns}

    date_col = next((cl[k] for k in ("date", "match_date") if k in cl), None)
    if date_col is None:
        raise ValueError(
            f"Elo file has no recognisable date column. "
            f"Found: {list(df.columns)}. Expected 'date'."
        )

    team_col = next(
        (cl[k] for k in ("team", "country", "name", "country_full") if k in cl), None
    )
    if team_col is None:
        raise ValueError(
            f"Elo file has no recognisable team column. "
            f"Found: {list(df.columns)}. Expected 'team', 'country', or 'name'."
        )

    rating_col = next(
        (cl[k] for k in ("elo", "rating", "elo_rating", "total_points", "points", "rank_points") if k in cl),
        None,
    )
    if rating_col is None:
        raise ValueError(
            f"Elo file has no recognisable rating column. "
            f"Found: {list(df.columns)}. Expected 'elo', 'rating', 'total_points', or 'points'."
        )

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "team": df[team_col].astype(str).str.strip(),
        "elo_rating": pd.to_numeric(df[rating_col], errors="coerce"),
    })
    return (
        out.dropna(subset=["date", "elo_rating"])
        .sort_values(["team", "date"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# FIFA ranking normalisation helper
# ---------------------------------------------------------------------------

def _normalize_fifa_df(fifa_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw FIFA ranking DataFrame to long format: (date, team, fifa_rank)."""
    df = fifa_df.copy()
    cl = {c.lower(): c for c in df.columns}

    date_col = next((cl[k] for k in ("rank_date", "date", "ranking_date") if k in cl), None)
    if date_col is None:
        raise ValueError(
            f"FIFA ranking file has no recognisable date column. "
            f"Found: {list(df.columns)}. Expected 'rank_date' or 'date'."
        )

    team_col = next(
        (cl[k] for k in ("country_full", "team", "country", "name") if k in cl), None
    )
    if team_col is None:
        raise ValueError(
            f"FIFA ranking file has no recognisable team column. "
            f"Found: {list(df.columns)}. Expected 'country_full', 'team', or 'country'."
        )

    rank_col = next((cl[k] for k in ("rank", "ranking", "fifa_rank") if k in cl), None)
    if rank_col is None:
        raise ValueError(
            f"FIFA ranking file has no recognisable rank column. "
            f"Found: {list(df.columns)}. Expected 'rank' or 'ranking'."
        )

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "team": df[team_col].astype(str).str.strip(),
        "fifa_rank": pd.to_numeric(df[rank_col], errors="coerce"),
    })
    return (
        out.dropna(subset=["date", "fifa_rank"])
        .sort_values(["team", "date"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# As-of-date join helpers
# ---------------------------------------------------------------------------

def _asof_join_elo(matches: pd.DataFrame, elo_long: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the most-recent pre-match Elo for each team using merge_asof.

    Both DataFrames must have a 'date' column; elo_long must have 'team' and
    'elo_rating'. Missing team/date combinations remain NaN.
    """
    elo_sorted = elo_long.sort_values("date")

    home_lkp = elo_sorted.rename(columns={"team": "team_home", "elo_rating": "home_elo_pre"})
    out = pd.merge_asof(
        matches.sort_values("date"),
        home_lkp[["date", "team_home", "home_elo_pre"]],
        on="date",
        by="team_home",
        direction="backward",
        allow_exact_matches=False,
    )

    away_lkp = elo_sorted.rename(columns={"team": "team_away", "elo_rating": "away_elo_pre"})
    out = pd.merge_asof(
        out.sort_values("date"),
        away_lkp[["date", "team_away", "away_elo_pre"]],
        on="date",
        by="team_away",
        direction="backward",
        allow_exact_matches=False,
    )
    return out


def _asof_join_fifa(matches: pd.DataFrame, fifa_long: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the most-recent pre-match FIFA rank for each team using merge_asof.
    """
    fifa_sorted = fifa_long.sort_values("date")

    home_lkp = fifa_sorted.rename(columns={"team": "team_home", "fifa_rank": "home_fifa_rank"})
    out = pd.merge_asof(
        matches.sort_values("date"),
        home_lkp[["date", "team_home", "home_fifa_rank"]],
        on="date",
        by="team_home",
        direction="backward",
        allow_exact_matches=False,
    )

    away_lkp = fifa_sorted.rename(columns={"team": "team_away", "fifa_rank": "away_fifa_rank"})
    out = pd.merge_asof(
        out.sort_values("date"),
        away_lkp[["date", "team_away", "away_fifa_rank"]],
        on="date",
        by="team_away",
        direction="backward",
        allow_exact_matches=False,
    )
    return out


# ---------------------------------------------------------------------------
# Venue coordinate lookup
# ---------------------------------------------------------------------------

def _load_venues(venues_path: Path) -> pd.DataFrame:
    """Load venue lat/lon from config/venues.csv. Non-fatal if absent."""
    if not venues_path.exists():
        return pd.DataFrame(columns=["city", "country", "lat", "lon", "stadium"])
    return pd.read_csv(venues_path)


def _join_venue_coords(df: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join venue lat/lon onto the match table by normalised (city, country).

    Rows without a matching venue in the lookup retain NaN venue_lat/venue_lon.
    city and country columns must exist in df; if absent this is a no-op.
    """
    if venues.empty or not {"city", "country"}.issubset(df.columns):
        return df

    v = venues[["city", "country", "lat", "lon"]].copy()
    v["_city_key"] = v["city"].str.lower().str.strip()
    v["_country_key"] = v["country"].str.lower().str.strip()
    v = (
        v[["_city_key", "_country_key", "lat", "lon"]]
        .drop_duplicates(subset=["_city_key", "_country_key"])
    )

    result = df.copy()
    result["_city_key"] = df["city"].fillna("").str.lower().str.strip()
    result["_country_key"] = df["country"].fillna("").str.lower().str.strip()

    result = result.merge(v, on=["_city_key", "_country_key"], how="left")

    matched = result["lat"].notna()
    result.loc[matched, "venue_lat"] = result.loc[matched, "lat"]
    result.loc[matched, "venue_lon"] = result.loc[matched, "lon"]

    result = result.drop(columns=["_city_key", "_country_key", "lat", "lon"], errors="ignore")
    return result


# ---------------------------------------------------------------------------
# Main pure transform
# ---------------------------------------------------------------------------

_RESULTS_COL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "date":       ("date",),
    "home_team":  ("home_team", "home team", "hometeam"),
    "away_team":  ("away_team", "away team", "awayteam"),
    "home_score": ("home_score", "home score", "score_home", "goals_home"),
    "away_score": ("away_score", "away score", "score_away", "goals_away"),
    "tournament": ("tournament", "competition", "competition_name"),
    "neutral":    ("neutral",),
}


def build_table(
    results_df: pd.DataFrame,
    elo_df: pd.DataFrame | None = None,
    fifa_df: pd.DataFrame | None = None,
    settings: dict[str, Any] | None = None,
    output_path: Path | str | None = None,
    teams: list[dict[str, Any]] | None = None,
    venues_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Produce the per-match training table matching docs/data_exp.md §1.

    Parameters
    ----------
    results_df:
        Raw match results (results.csv). Expected columns:
        date, home_team, away_team, home_score, away_score, tournament,
        city, country, neutral.
    elo_df:
        Raw Elo time-series (elo_ratings.csv). Normalised internally to
        (date, team, elo_rating). If None, Elo columns are NaN.
    fifa_df:
        Raw FIFA ranking time-series (fifa_ranking.csv). If None, fifa_rank
        columns are NaN.
    settings:
        Parsed config/settings.yaml. Loaded from disk if None.
    output_path:
        If given, the result is written to this parquet path.
    teams:
        Parsed config/teams.yaml teams list. Loaded from disk if None.
    venues_df:
        Pre-loaded venue coordinates DataFrame (city, country, lat, lon).
        If None, loads config/venues.csv from disk. Pass a DataFrame directly
        in tests to keep the function a pure transform.

    Returns
    -------
    pd.DataFrame with columns:
        match_id, date, team_home, team_away, goals_home, goals_away,
        tournament, importance_weight, neutral, venue_lat, venue_lon,
        home_elo_pre, away_elo_pre, home_fifa_rank, away_fifa_rank.

    Notes
    -----
    - NO market / odds data (ai_rules.md #3).
    - NO network access — pure transform.
    - venue_lat / venue_lon are filled from config/venues.csv for the 16 2026
      host-city venues; all other historical venues remain NaN (sufficient for
      features/engineer.py's travel computation on 2026 matches).
    """
    if settings is None:
        settings = _load_settings()
    if teams is None:
        teams = _load_teams_config()
    if venues_df is None:
        _venues = _load_venues(_project_root() / "config" / "venues.csv")
    else:
        _venues = venues_df

    alias_map = _build_alias_map(teams)
    normalize = _make_normalizer(alias_map)

    # ------------------------------------------------------------------
    # 1. Normalise results columns
    # ------------------------------------------------------------------
    df = results_df.copy()
    cl = {c.lower(): c for c in df.columns}

    col_map: dict[str, str | None] = {}
    for target, candidates in _RESULTS_COL_CANDIDATES.items():
        col_map[target] = next((cl[c] for c in candidates if c in cl), None)

    required = [k for k in _RESULTS_COL_CANDIDATES if k != "neutral"]
    missing = [k for k in required if col_map[k] is None]
    if missing:
        raise ValueError(
            f"results.csv is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    rename = {v: k for k, v in col_map.items() if v is not None and v != k}
    df = df.rename(columns=rename)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])

    df["team_home"] = df["home_team"].map(normalize)
    df["team_away"] = df["away_team"].map(normalize)

    df["goals_home"] = pd.to_numeric(df["home_score"], errors="coerce").astype("Int64")
    df["goals_away"] = pd.to_numeric(df["away_score"], errors="coerce").astype("Int64")

    if "neutral" in df.columns:
        n = df["neutral"]
        if pd.api.types.is_string_dtype(n):
            df["neutral"] = n.str.lower().isin(["true", "1", "yes"])
        else:
            df["neutral"] = n.astype(bool)
    else:
        df["neutral"] = False

    df["tournament"] = df["tournament"].fillna("Unknown").astype(str)
    df["importance_weight"] = df["tournament"].map(
        lambda t: _classify_importance(t, settings)
    )

    df["match_id"] = (
        df["date"].dt.strftime("%Y%m%d")
        + "_"
        + df["team_home"].str.lower().str.replace(" ", "_", regex=False)
        + "_"
        + df["team_away"].str.lower().str.replace(" ", "_", regex=False)
    )

    # venue_lat / venue_lon: fill from config/venues.csv for known host cities
    df["venue_lat"] = float("nan")
    df["venue_lon"] = float("nan")
    df = _join_venue_coords(df, _venues)

    # ------------------------------------------------------------------
    # 2. As-of-date Elo join
    # ------------------------------------------------------------------
    if elo_df is not None:
        elo_long = _normalize_elo_df(elo_df)
        elo_long["team"] = elo_long["team"].map(normalize)
        df = _asof_join_elo(df, elo_long)
    else:
        df["home_elo_pre"] = float("nan")
        df["away_elo_pre"] = float("nan")

    # ------------------------------------------------------------------
    # 3. As-of-date FIFA ranking join
    # ------------------------------------------------------------------
    if fifa_df is not None:
        fifa_long = _normalize_fifa_df(fifa_df)
        fifa_long["team"] = fifa_long["team"].map(normalize)
        df = _asof_join_fifa(df, fifa_long)
    else:
        df["home_fifa_rank"] = float("nan")
        df["away_fifa_rank"] = float("nan")

    # ------------------------------------------------------------------
    # 4. Assemble final schema
    # ------------------------------------------------------------------
    output_cols = [
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
        "home_elo_pre",
        "away_elo_pre",
        "home_fifa_rank",
        "away_fifa_rank",
    ]
    result = df[[c for c in output_cols if c in df.columns]].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Optional write to parquet
    # ------------------------------------------------------------------
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out, index=False)

    return result


# ---------------------------------------------------------------------------
# Entry-point wrapper (used by cli.py build-data)
# ---------------------------------------------------------------------------

def run(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
    settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Load raw files from data/raw/, build the training table, write to
    data/processed/matches.parquet.

    Raises FileNotFoundError (via ingest.load_local) if any required file is
    absent, with a clear message naming the file and its download URL.
    """
    from wcpredict.data.ingest import load_results, load_elo, load_fifa_ranking

    if settings is None:
        settings = _load_settings()

    root = _project_root()
    raw = Path(raw_dir) if raw_dir else root / settings["paths"]["raw"]
    proc = Path(processed_dir) if processed_dir else root / settings["paths"]["processed"]
    output_path = proc / "matches.parquet"

    results_df = load_results(raw)
    elo_df = load_elo(raw)
    fifa_df = load_fifa_ranking(raw)

    return build_table(
        results_df=results_df,
        elo_df=elo_df,
        fifa_df=fifa_df,
        settings=settings,
        output_path=output_path,
    )
