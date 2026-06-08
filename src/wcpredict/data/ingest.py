"""
data/ingest.py — the ONLY module with network access.

load_local(name) reads historical source files from data/raw/.
fetch_*() are stubs for live sources exercised in Phases 4-5.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Download URLs — used only in FileNotFoundError messages
# ---------------------------------------------------------------------------
_DOWNLOAD_URLS: dict[str, str] = {
    "results.csv": (
        "https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017"
    ),
    "shootouts.csv": (
        "https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017"
    ),
    "elo_ratings.csv": "https://www.eloratings.net/downloads",
    "fifa_ranking.csv": (
        "https://www.kaggle.com/datasets/cashncarry/fifaworldranking"
    ),
    "players.csv": (
        "https://www.kaggle.com/datasets/davidcariboo/player-scores"
        "  (Transfermarkt player data via Kaggle)"
    ),
    "appearances.csv": (
        "https://www.kaggle.com/datasets/davidcariboo/player-scores"
        "  (Transfermarkt appearance data via Kaggle)"
    ),
    "player_valuations.csv": (
        "https://www.kaggle.com/datasets/davidcariboo/player-scores"
        "  (Transfermarkt market valuations via Kaggle)"
    ),
}


def _get_raw_dir() -> Path:
    """Resolve data/raw/. Override with WCPREDICT_RAW_DIR env var."""
    env = os.environ.get("WCPREDICT_RAW_DIR")
    if env:
        return Path(env)
    # src/wcpredict/data/ingest.py  →  parents[3] = project root
    return Path(__file__).resolve().parents[3] / "data" / "raw"


def load_local(name: str, raw_dir: Path | str | None = None) -> pd.DataFrame:
    """
    Load a raw CSV (or .csv.gz) from data/raw/ by filename.

    Raises FileNotFoundError with the download URL when the file is absent.
    pandas reads .csv and .csv.gz natively — no decompression step needed.
    """
    base = Path(raw_dir) if raw_dir is not None else _get_raw_dir()
    path = base / name
    if not path.exists():
        url = _DOWNLOAD_URLS.get(name, "(no URL listed)")
        raise FileNotFoundError(
            f"Raw file not found: {path}\n"
            f"  Download from : {url}\n"
            f"  Place file at : {path}\n"
            "  All historical data is a manual download; "
            "ingest.py never fetches it automatically."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def load_results(raw_dir: Path | str | None = None) -> pd.DataFrame:
    """International match results 1872-present (Kaggle / Mart Jürisoo)."""
    return load_local("results.csv", raw_dir)


def load_shootouts(raw_dir: Path | str | None = None) -> pd.DataFrame:
    """Historical shootout outcomes (Kaggle / Mart Jürisoo)."""
    return load_local("shootouts.csv", raw_dir)


def load_elo(raw_dir: Path | str | None = None) -> pd.DataFrame:
    """World Football Elo ratings time-series (eloratings.net)."""
    return load_local("elo_ratings.csv", raw_dir)


def load_fifa_ranking(raw_dir: Path | str | None = None) -> pd.DataFrame:
    """FIFA world ranking time-series."""
    return load_local("fifa_ranking.csv", raw_dir)


def load_players(raw_dir: Path | str | None = None) -> pd.DataFrame:
    """Transfermarkt player metadata (used in Phase 2 squad-quality features)."""
    return load_local("players.csv", raw_dir)


def load_appearances(raw_dir: Path | str | None = None) -> pd.DataFrame:
    """Transfermarkt player appearance history (used in Phase 2 form features)."""
    return load_local("appearances.csv", raw_dir)


def load_player_valuations(raw_dir: Path | str | None = None) -> pd.DataFrame:
    """Transfermarkt market valuations by player and date (used in Phase 2)."""
    return load_local("player_valuations.csv", raw_dir)


# ---------------------------------------------------------------------------
# Live source stubs — implemented in Phases 4-5
# ---------------------------------------------------------------------------

def fetch_news(
    api_key: str | None = None,
    query: str = "FIFA World Cup injury team news",
    max_articles: int = 100,
) -> list[dict[str, Any]]:
    """
    Fetch raw team-news / injury text from a news API.

    Phase 4 implementation: calls the news API, returns raw article dicts.
    Each dict feeds into llm/extract.py for availability / lineup-certainty
    extraction (ai_rules.md #4 — raw information only, never market text).
    Key from env var NEWS_API_KEY.
    """
    _key = api_key or os.environ.get("NEWS_API_KEY")  # noqa: F841
    raise NotImplementedError(
        "fetch_news() is a Phase 4 stub. "
        "Implement with a news aggregator (e.g. GDELT, NewsAPI). "
        "Set NEWS_API_KEY in .env."
    )


def fetch_lineups(
    api_key: str | None = None,
    match_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch official pre-match lineup releases (~1 hr before kickoff).

    Phase 4 implementation: confirmed XI triggers supersession in the
    adjustment layer (ai_rules.md #14).
    Key from env var LINEUP_API_KEY.
    """
    _key = api_key or os.environ.get("LINEUP_API_KEY")  # noqa: F841
    raise NotImplementedError(
        "fetch_lineups() is a Phase 4 stub. "
        "Implement with an official lineup data source. "
        "Set LINEUP_API_KEY in .env."
    )


def fetch_odds(
    api_key: str | None = None,
    market: str = "h2h",
    sport: str = "soccer_fifa_world_cup",
) -> list[dict[str, Any]]:
    """
    Fetch closing-line match odds via an odds API.

    FOR eval/ ONLY — never feed to the model (ai_rules.md #3).
    Phase 5 implementation. Key from env var ODDS_API_KEY.
    """
    _key = api_key or os.environ.get("ODDS_API_KEY")  # noqa: F841
    raise NotImplementedError(
        "fetch_odds() is a Phase 5 stub. For eval/ only — NEVER pass to the model. "
        "Set ODDS_API_KEY in .env."
    )
