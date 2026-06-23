"""
scripts/build_2026_training_rows.py

Extracts completed 2026 World Cup matches from the live tournament state and
writes them as training rows in the matches.parquet schema, so the model can
be refitted to include observed tournament performance.

Key invariants preserved:
- Pre-match Elo is the pre-tournament snapshot (no post-match Elo leakage).
- All WC matches are neutral=True.
- importance_weight = 1.0 (tournament weight).
- Squad features come from the frozen 2026 squad snapshot.
- A `source` column ("wc2026") tags these rows so the eval boundary can
  exclude them from calibration scoring if needed.

Usage:
    python scripts/build_2026_training_rows.py
    # or from Python:
    from scripts.build_2026_training_rows import build_rows
    df = build_rows()
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[1]


def _load_settings() -> dict[str, Any]:
    root = _project_root()
    with open(root / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def _load_elo_snapshot(raw_dir: Path) -> dict[str, float]:
    """Load the latest pre-tournament Elo rating for each team."""
    path = raw_dir / "elo_ratings_wc2026.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "snapshot_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"snapshot_date": "date"})
    team_col = "team" if "team" in df.columns else "country"
    rating_col = "elo" if "elo" in df.columns else "rating"
    if team_col not in df.columns or rating_col not in df.columns:
        return {}
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
    latest = (
        df.dropna(subset=[team_col, rating_col])
        .groupby(team_col)
        .tail(1)
    )
    return {str(row[team_col]): float(row[rating_col]) for _, row in latest.iterrows()}


def _load_squad_features(processed_dir: Path, settings: dict[str, Any]) -> dict[str, dict]:
    """Load the 2026 squad team features keyed by team name."""
    cutoff = str(settings.get("snapshot", {}).get("cutoff", ""))
    path = processed_dir / f"squad_team_features_{cutoff}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "team" not in df.columns:
        return {}
    return df.set_index("team").to_dict(orient="index")


def build_rows(
    live_state_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of completed 2026 WC match rows in the matches.parquet schema.

    Parameters
    ----------
    live_state_path : path to live_tournament_state.json (default: auto-detected)
    output_path     : if given, write the result to this parquet path

    Returns
    -------
    DataFrame with the matches.parquet schema, tagged with source='wc2026'.
    Empty DataFrame if no completed matches are found.
    """
    root = _project_root()
    settings = _load_settings()
    raw_dir = root / settings["paths"]["raw"]
    processed_dir = root / settings["paths"]["processed"]

    if live_state_path is None:
        live_state_path = processed_dir / "live_tournament_state.json"

    if not live_state_path.exists():
        return pd.DataFrame()

    with open(live_state_path) as f:
        state = json.load(f)

    completed = [
        m for m in state.get("matches", [])
        if m.get("locked") and m.get("home_score") is not None and m.get("away_score") is not None
    ]
    if not completed:
        return pd.DataFrame()

    elo = _load_elo_snapshot(raw_dir)
    squad_feat = _load_squad_features(processed_dir, settings)

    rows: list[dict] = []
    for m in completed:
        home = m["home"]
        away = m["away"]
        date_str = m.get("date", "2026-06-11")  # fallback to tournament start date
        try:
            match_date = pd.Timestamp(date_str)
        except Exception:
            match_date = pd.Timestamp("2026-06-11")

        home_elo = elo.get(home, float("nan"))
        away_elo = elo.get(away, float("nan"))

        row: dict[str, Any] = {
            "match_id": m.get("match_id", f"wc2026_{home}_{away}").lower().replace(" ", "_"),
            "date": match_date,
            "team_home": home,
            "team_away": away,
            "goals_home": int(m["home_score"]),
            "goals_away": int(m["away_score"]),
            "tournament": "FIFA World Cup",
            "importance_weight": 1.0,
            "neutral": True,
            "venue_lat": float("nan"),
            "venue_lon": float("nan"),
            "home_elo_pre": home_elo,
            "away_elo_pre": away_elo,
            "home_fifa_rank": float("nan"),
            "away_fifa_rank": float("nan"),
            "source": "wc2026",
        }

        # Attach squad features for elo_diff computation and feature engineering
        hf = squad_feat.get(home, {})
        af = squad_feat.get(away, {})
        for feat in [
            "squad_quality_rank", "squad_depth_rank", "star_power_rank",
            "star_concentration", "attack_value_rank", "midfield_value_rank",
            "defense_value_rank", "club_minutes_rank", "club_form_rank",
            "gk_rating", "national_form",
        ]:
            row[f"{feat}_home"] = hf.get(feat, float("nan"))
            row[f"{feat}_away"] = af.get(feat, float("nan"))

        rows.append(row)

    df = pd.DataFrame(rows)

    # Drop rows where we have no Elo for either team (can't compute elo_diff feature)
    df = df.dropna(subset=["home_elo_pre", "away_elo_pre"]).reset_index(drop=True)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"Wrote {len(df)} rows → {out}")

    return df


def build_and_merge(
    live_state_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build 2026 rows and concatenate with the existing historical matches.parquet.

    Returns (merged_matches, rows_2026) — merged_matches is what the model
    should be retrained on in production mode.
    """
    root = _project_root()
    settings = _load_settings()
    match_table_path = root / settings["paths"]["match_table"]

    rows_2026 = build_rows(live_state_path)

    if not match_table_path.exists():
        return rows_2026, rows_2026

    historical = pd.read_parquet(match_table_path)
    historical["date"] = pd.to_datetime(historical["date"])

    if rows_2026.empty:
        return historical, rows_2026

    # Drop columns only in rows_2026 (squad feature columns) before concat
    # so the schema stays aligned with what match_model.fit() expects.
    common_cols = [c for c in historical.columns if c in rows_2026.columns]
    merged = pd.concat(
        [historical, rows_2026[common_cols]],
        ignore_index=True,
    ).sort_values("date").reset_index(drop=True)

    out_path = root / settings["paths"]["processed"] / "matches_2026.parquet"
    rows_2026.to_parquet(out_path, index=False)

    return merged, rows_2026


if __name__ == "__main__":
    root = _project_root()
    settings = _load_settings()
    processed_dir = root / settings["paths"]["processed"]
    out = processed_dir / "matches_2026.parquet"
    df = build_rows(output_path=out)
    if df.empty:
        print("No completed matches found — nothing to write.")
    else:
        print(f"Built {len(df)} training rows from 2026 World Cup results.")
        print(df[["team_home", "team_away", "goals_home", "goals_away", "home_elo_pre", "away_elo_pre"]].to_string(index=False))
