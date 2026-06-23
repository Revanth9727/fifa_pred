#!/usr/bin/env python3
"""
Audit 2026 World Cup squad coverage against local player data.

Outputs:
  outputs/squad_coverage_<cutoff>.csv
  outputs/squad_coverage_summary_<cutoff>.json
"""
from __future__ import annotations

import argparse
import json
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
        "outputs": ROOT / cfg.get("outputs", "outputs"),
    }


def _cutoff(settings: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    return str(settings.get("snapshot", {}).get("cutoff"))


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, compression="gzip", **kwargs)


def _latest_values(
    valuations: pd.DataFrame,
    squad_ids: set[int],
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    val = valuations.copy()
    val["player_id"] = pd.to_numeric(val["player_id"], errors="coerce")
    val["date"] = pd.to_datetime(val["date"], errors="coerce")
    val["market_value_in_eur"] = pd.to_numeric(val["market_value_in_eur"], errors="coerce")
    val = val.dropna(subset=["player_id", "date"])
    val["player_id"] = val["player_id"].astype(int)
    val = val[val["player_id"].isin(squad_ids) & (val["date"] <= cutoff)]
    if val.empty:
        return pd.DataFrame(columns=["player_id", "latest_value_date", "latest_market_value_in_eur"])
    return (
        val.sort_values("date")
        .groupby("player_id", as_index=False)
        .tail(1)[["player_id", "date", "market_value_in_eur"]]
        .rename(columns={
            "date": "latest_value_date",
            "market_value_in_eur": "latest_market_value_in_eur",
        })
    )


def _recent_appearances(
    appearances: pd.DataFrame,
    squad_ids: set[int],
    cutoff: pd.Timestamp,
    days: int,
) -> pd.DataFrame:
    app = appearances.copy()
    app["player_id"] = pd.to_numeric(app["player_id"], errors="coerce")
    app["date"] = pd.to_datetime(app["date"], errors="coerce")
    app["minutes_played"] = pd.to_numeric(app.get("minutes_played", 0.0), errors="coerce").fillna(0.0)
    app["goals"] = pd.to_numeric(app.get("goals", 0.0), errors="coerce").fillna(0.0)
    app["assists"] = pd.to_numeric(app.get("assists", 0.0), errors="coerce").fillna(0.0)
    app["yellow_cards"] = pd.to_numeric(app.get("yellow_cards", 0.0), errors="coerce").fillna(0.0)
    app["red_cards"] = pd.to_numeric(app.get("red_cards", 0.0), errors="coerce").fillna(0.0)
    app = app.dropna(subset=["player_id", "date"])
    app["player_id"] = app["player_id"].astype(int)
    app = app[
        app["player_id"].isin(squad_ids)
        & (app["date"] <= cutoff)
        & (app["date"] >= cutoff - pd.Timedelta(days=days))
    ]
    if app.empty:
        return pd.DataFrame(columns=[
            "player_id", f"appearances_{days}d", f"minutes_{days}d",
            f"goals_{days}d", f"assists_{days}d", f"cards_{days}d",
            f"competitions_{days}d",
        ])
    return (
        app.groupby("player_id", as_index=False)
        .agg(
            **{
                f"appearances_{days}d": ("game_id", "nunique"),
                f"minutes_{days}d": ("minutes_played", "sum"),
                f"goals_{days}d": ("goals", "sum"),
                f"assists_{days}d": ("assists", "sum"),
                f"cards_{days}d": ("yellow_cards", "sum"),
                f"red_cards_{days}d": ("red_cards", "sum"),
                f"competitions_{days}d": ("competition_id", "nunique"),
            }
        )
    )


def _recent_starts(
    lineups: pd.DataFrame,
    squad_ids: set[int],
    cutoff: pd.Timestamp,
    days: int,
) -> pd.DataFrame:
    if lineups.empty or "type" not in lineups.columns:
        return pd.DataFrame(columns=["player_id", f"starts_{days}d"])
    lu = lineups.copy()
    lu["player_id"] = pd.to_numeric(lu["player_id"], errors="coerce")
    lu["date"] = pd.to_datetime(lu["date"], errors="coerce")
    lu = lu.dropna(subset=["player_id", "date"])
    lu["player_id"] = lu["player_id"].astype(int)
    lu = lu[
        lu["player_id"].isin(squad_ids)
        & (lu["date"] <= cutoff)
        & (lu["date"] >= cutoff - pd.Timedelta(days=days))
    ]
    lu = lu[lu["type"].astype(str).str.lower().str.contains("start")]
    if lu.empty:
        return pd.DataFrame(columns=["player_id", f"starts_{days}d"])
    return lu.groupby("player_id", as_index=False).size().rename(columns={"size": f"starts_{days}d"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default=None)
    parser.add_argument("--recent-days", type=int, default=180)
    args = parser.parse_args(argv)

    settings = _settings()
    cutoff_text = _cutoff(settings, args.cutoff)
    cutoff = pd.Timestamp(cutoff_text)
    paths = _paths(settings)
    raw = paths["raw"]
    out_dir = paths["outputs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    squad_path = raw / f"squads_2026_{cutoff_text}.csv"
    unmatched_path = out_dir / f"squad_unmatched_{cutoff_text}.csv"
    squads = _read_csv(squad_path)
    squads["player_id"] = pd.to_numeric(squads["player_id"], errors="coerce")
    squads = squads.dropna(subset=["player_id"]).copy()
    squads["player_id"] = squads["player_id"].astype(int)
    squad_ids = set(squads["player_id"])

    valuations = _read_csv(raw / "player_valuations.csv")
    appearances = _read_csv(raw / "appearances.csv")
    lineups_path = raw / "game_lineups.csv"
    lineups = _read_csv(lineups_path) if lineups_path.exists() else pd.DataFrame()

    coverage = squads[["team", "name", "position", "club", "player_id", "market_value_in_eur"]].copy()
    coverage = coverage.merge(_latest_values(valuations, squad_ids, cutoff), on="player_id", how="left")
    coverage = coverage.merge(
        _recent_appearances(appearances, squad_ids, cutoff, args.recent_days),
        on="player_id",
        how="left",
    )
    coverage = coverage.merge(
        _recent_starts(lineups, squad_ids, cutoff, args.recent_days),
        on="player_id",
        how="left",
    )

    for col in [
        f"appearances_{args.recent_days}d", f"minutes_{args.recent_days}d",
        f"goals_{args.recent_days}d", f"assists_{args.recent_days}d",
        f"cards_{args.recent_days}d", f"red_cards_{args.recent_days}d",
        f"competitions_{args.recent_days}d", f"starts_{args.recent_days}d",
    ]:
        if col in coverage.columns:
            coverage[col] = coverage[col].fillna(0)

    value = pd.to_numeric(coverage["latest_market_value_in_eur"], errors="coerce")
    fallback_value = pd.to_numeric(coverage["market_value_in_eur"], errors="coerce")
    effective_value = value.fillna(fallback_value).fillna(0.0)
    minutes = coverage[f"minutes_{args.recent_days}d"]
    coverage["missing_market_value"] = effective_value <= 0
    coverage["missing_recent_minutes"] = minutes <= 0
    coverage["high_value_flag"] = effective_value >= 20_000_000
    coverage["review_priority"] = "low"
    coverage.loc[coverage["missing_market_value"] | coverage["missing_recent_minutes"], "review_priority"] = "medium"
    coverage.loc[
        coverage["high_value_flag"] & (coverage["missing_market_value"] | coverage["missing_recent_minutes"]),
        "review_priority",
    ] = "high"
    coverage = coverage.sort_values(["review_priority", "team", "name"], ascending=[False, True, True])

    coverage_path = out_dir / f"squad_coverage_{cutoff_text}.csv"
    coverage.to_csv(coverage_path, index=False)

    unmatched_rows = 0
    high_value_unmatched = 0
    if unmatched_path.exists():
        unmatched = _read_csv(unmatched_path)
        unmatched_rows = int(len(unmatched))
        if "high_value_flag" in unmatched.columns:
            high_value_unmatched = int(unmatched["high_value_flag"].fillna(False).sum())

    summary = {
        "cutoff": cutoff_text,
        "squad_rows": int(len(coverage)),
        "teams": int(coverage["team"].nunique()),
        "players_missing_market_value": int(coverage["missing_market_value"].sum()),
        f"players_missing_minutes_{args.recent_days}d": int(coverage["missing_recent_minutes"].sum()),
        "high_value_players_needing_review": int((coverage["review_priority"] == "high").sum()),
        "unmatched_rows": unmatched_rows,
        "high_value_unmatched_candidates": high_value_unmatched,
        "coverage_csv": str(coverage_path),
    }
    summary_path = out_dir / f"squad_coverage_summary_{cutoff_text}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
