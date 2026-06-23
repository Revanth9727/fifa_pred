#!/usr/bin/env python3
"""
Normalize local player appearances into player-season-competition stats.

Output:
  data/processed/player_competition_stats.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _settings() -> dict[str, Any]:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def _paths(settings: dict[str, Any]) -> dict[str, Path]:
    cfg = settings.get("paths", {})
    return {
        "raw": ROOT / cfg.get("raw", "data/raw"),
        "processed": ROOT / cfg.get("processed", "data/processed"),
    }


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, compression="gzip", **kwargs)


def _starts_by_player_game(lineups_path: Path) -> pd.DataFrame:
    if not lineups_path.exists():
        return pd.DataFrame(columns=["game_id", "player_id", "start"])
    lineups = _read_csv(lineups_path, usecols=["game_id", "player_id", "type"])
    lineups["player_id"] = pd.to_numeric(lineups["player_id"], errors="coerce")
    lineups["game_id"] = pd.to_numeric(lineups["game_id"], errors="coerce")
    lineups = lineups.dropna(subset=["game_id", "player_id"])
    lineups["game_id"] = lineups["game_id"].astype(int)
    lineups["player_id"] = lineups["player_id"].astype(int)
    lineups["start"] = lineups["type"].astype(str).str.lower().str.contains("start").astype(int)
    return lineups.groupby(["game_id", "player_id"], as_index=False)["start"].max()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    settings = _settings()
    paths = _paths(settings)
    raw = paths["raw"]
    processed = paths["processed"]
    processed.mkdir(parents=True, exist_ok=True)

    appearances = _read_csv(raw / "appearances.csv")
    games = _read_csv(raw / "games.csv")
    competitions = _read_csv(raw / "competitions.csv")
    clubs = _read_csv(raw / "clubs.csv")
    players = _read_csv(raw / "players.csv")

    app = appearances.copy()
    app["player_id"] = pd.to_numeric(app["player_id"], errors="coerce")
    app["game_id"] = pd.to_numeric(app["game_id"], errors="coerce")
    app["player_club_id"] = pd.to_numeric(app["player_club_id"], errors="coerce")
    app["date"] = pd.to_datetime(app["date"], errors="coerce")
    for col in ["minutes_played", "goals", "assists", "yellow_cards", "red_cards"]:
        app[col] = pd.to_numeric(app.get(col, 0.0), errors="coerce").fillna(0.0)
    app = app.dropna(subset=["player_id", "game_id", "date"])
    app["player_id"] = app["player_id"].astype(int)
    app["game_id"] = app["game_id"].astype(int)

    game_cols = [
        "game_id", "competition_id", "season", "round", "home_club_id",
        "away_club_id", "home_club_manager_name", "away_club_manager_name",
    ]
    game_cols = [col for col in game_cols if col in games.columns]
    games_small = games[game_cols].copy()
    games_small["game_id"] = pd.to_numeric(games_small["game_id"], errors="coerce")
    games_small = games_small.dropna(subset=["game_id"])
    games_small["game_id"] = games_small["game_id"].astype(int)

    comp_cols = [
        "competition_id", "name", "sub_type", "type", "country_name",
        "domestic_league_code", "confederation",
    ]
    comp_cols = [col for col in comp_cols if col in competitions.columns]
    comps = competitions[comp_cols].copy().rename(columns={
        "name": "competition_name",
        "type": "competition_type",
        "sub_type": "competition_sub_type",
        "country_name": "competition_country",
    })

    club_cols = ["club_id", "name", "domestic_competition_id"]
    club_cols = [col for col in club_cols if col in clubs.columns]
    club_lookup = clubs[club_cols].copy().rename(columns={
        "club_id": "player_club_id",
        "name": "club_name",
        "domestic_competition_id": "club_domestic_competition_id",
    })
    club_lookup["player_club_id"] = pd.to_numeric(club_lookup["player_club_id"], errors="coerce")
    club_lookup = club_lookup.dropna(subset=["player_club_id"])
    club_lookup["player_club_id"] = club_lookup["player_club_id"].astype(int)

    player_cols = ["player_id", "name", "country_of_citizenship", "position", "sub_position"]
    player_cols = [col for col in player_cols if col in players.columns]
    player_lookup = players[player_cols].copy().rename(columns={
        "name": "tm_player_name",
        "country_of_citizenship": "player_country",
    })
    player_lookup["player_id"] = pd.to_numeric(player_lookup["player_id"], errors="coerce")
    player_lookup = player_lookup.dropna(subset=["player_id"])
    player_lookup["player_id"] = player_lookup["player_id"].astype(int)

    starts = _starts_by_player_game(raw / "game_lineups.csv")
    merged = (
        app.merge(games_small, on="game_id", how="left", suffixes=("", "_game"))
        .merge(comps, on="competition_id", how="left")
        .merge(club_lookup, on="player_club_id", how="left")
        .merge(player_lookup, on="player_id", how="left")
        .merge(starts, on=["game_id", "player_id"], how="left")
    )
    if "player_name" not in merged.columns and "tm_player_name" in merged.columns:
        merged["player_name"] = merged["tm_player_name"]
    merged["start"] = merged["start"].fillna(0).astype(int)
    if "season" not in merged.columns:
        merged["season"] = merged["date"].dt.year
    merged["season"] = pd.to_numeric(merged["season"], errors="coerce").fillna(merged["date"].dt.year)
    merged["cards"] = merged["yellow_cards"] + merged["red_cards"]

    group_cols = [
        "player_id", "player_name", "player_country", "position", "sub_position",
        "player_club_id", "club_name", "club_domestic_competition_id",
        "competition_id", "competition_name", "competition_type",
        "competition_sub_type", "competition_country", "confederation", "season",
    ]
    group_cols = [col for col in group_cols if col in merged.columns]

    out = (
        merged.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            matches=("game_id", "nunique"),
            starts=("start", "sum"),
            minutes=("minutes_played", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
            cards=("cards", "sum"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .sort_values(["season", "competition_id", "player_name"], ascending=[False, True, True])
    )

    output = Path(args.output) if args.output else processed / "player_competition_stats.csv"
    out.to_csv(output, index=False)
    print(f"Wrote {output} ({len(out):,} rows)")
    print(f"Players: {out['player_id'].nunique():,}")
    print(f"Competitions: {out['competition_id'].nunique():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
