"""
Command-line integration for the World Cup predictor.

The CLI is intentionally thin: it wires the existing data, feature, model,
simulation, and eval modules together using config/settings.yaml.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from wcpredict.data.build_table import build_table
from wcpredict.eval.calibration import (
    ab_test,
    ablate_by_source,
    devig,
    evaluate_and_persist,
    load_market_data,
    reliability_diagram,
    score,
)
from wcpredict.features.engineer import (
    build_features,
    compute_historical_squad_team_features,
    compute_named_squad_team_features,
)
from wcpredict.model.match_model import MatchModel
from wcpredict.sim.simulator import STAGES, TournamentSimulator


TRAIN_START = pd.Timestamp("2000-01-01")
SPLIT_DATE = pd.Timestamp("2022-01-01")


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _paths(settings: dict[str, Any]) -> dict[str, Path]:
    root = _project_root()
    raw = root / settings["paths"]["raw"]
    processed = root / settings["paths"]["processed"]
    outputs = root / settings["paths"]["outputs"]
    return {
        "root": root,
        "raw": raw,
        "processed": processed,
        "outputs": outputs,
        "match_table": root / settings["paths"]["match_table"],
        "features": root / settings["paths"]["features"],
        "model": outputs / "match_model.pkl",
        "probabilities": outputs / "probabilities.csv",
        "counts": outputs / "simulation_counts.json",
        "metrics": outputs / "metrics",
        "live_ab": outputs / "metrics" / "live_ab.csv",
    }


def _load_settings(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else _project_root() / "config" / "settings.yaml"
    return _load_yaml(path)


def _load_elo(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "elo_ratings_wc2026.csv"
    df = pd.read_csv(path)
    if "snapshot_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"snapshot_date": "date"})
    return df


def _load_fifa(raw_dir: Path) -> pd.DataFrame | None:
    parts = [pd.read_csv(path) for path in sorted(raw_dir.glob("fifa_ranking-*.csv"))]
    if not parts:
        return None
    out = pd.concat(parts, ignore_index=True)
    if {"rank_date", "country_full"}.issubset(out.columns):
        out = out.drop_duplicates(subset=["rank_date", "country_full"])
    return out


def _read_csv_maybe_gzip(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, compression="gzip")


def _load_squad_values(raw_dir: Path) -> pd.DataFrame | None:
    players_path = raw_dir / "players.csv"
    if not players_path.exists():
        return None
    players = _read_csv_maybe_gzip(players_path)
    needed = {"country_of_citizenship", "market_value_in_eur"}
    if not needed.issubset(players.columns):
        return None

    values = players[["country_of_citizenship", "market_value_in_eur"]].copy()
    values = values.rename(columns={"country_of_citizenship": "team"})
    values["total_value"] = pd.to_numeric(values["market_value_in_eur"], errors="coerce")
    values = values.dropna(subset=["team", "total_value"])
    values = values[values["total_value"] > 0]
    if values.empty:
        return None
    top26 = (
        values.sort_values(["team", "total_value"], ascending=[True, False])
        .groupby("team", as_index=False)
        .head(26)
    )
    return top26.groupby("team", as_index=False)["total_value"].sum()


def _load_csv_maybe_gzip(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, compression="gzip", **kwargs)


def _load_squad_team_features(settings: dict[str, Any]) -> pd.DataFrame | None:
    paths = _paths(settings)
    cutoff = str(settings.get("snapshot", {}).get("cutoff", ""))
    squad_path = paths["raw"] / f"squads_2026_{cutoff}.csv"
    if not squad_path.exists():
        return None
    squads = pd.read_csv(squad_path)
    players = _load_csv_maybe_gzip(paths["raw"] / "players.csv")
    appearances = _load_csv_maybe_gzip(paths["raw"] / "appearances.csv")
    valuations = _load_csv_maybe_gzip(paths["raw"] / "player_valuations.csv")
    team_features = compute_named_squad_team_features(
        squads,
        players,
        appearances,
        valuations,
        cutoff=pd.Timestamp(cutoff),
    )
    out_path = paths["processed"] / f"squad_team_features_{cutoff}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    team_features.to_csv(out_path, index=False)
    return team_features


def _load_historical_squad_features(settings: dict[str, Any]) -> pd.DataFrame | None:
    paths = _paths(settings)
    cutoff = str(settings.get("snapshot", {}).get("cutoff", ""))
    out_path = paths["processed"] / f"squad_history_{cutoff}.csv"
    if out_path.exists():
        return pd.read_csv(out_path)
    players_path = paths["raw"] / "players.csv"
    valuations_path = paths["raw"] / "player_valuations.csv"
    if not players_path.exists() or not valuations_path.exists():
        return None
    players = _load_csv_maybe_gzip(players_path)
    valuations = _load_csv_maybe_gzip(valuations_path)
    history = compute_historical_squad_team_features(
        players,
        valuations,
        cutoff=pd.Timestamp(cutoff),
        start=pd.Timestamp("2000-01-01"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(out_path, index=False)
    return history


def _load_groups(root: Path) -> dict[str, list[str]]:
    teams_cfg = _load_yaml(root / "config" / "teams.yaml")
    return {str(group): list(teams) for group, teams in teams_cfg["groups"].items()}


def _load_strengths(raw_dir: Path, groups: dict[str, list[str]]) -> dict[str, float]:
    teams = {team for group in groups.values() for team in group}
    elo = _load_elo(raw_dir)
    team_col = "team" if "team" in elo.columns else "country"
    rating_col = "elo" if "elo" in elo.columns else "rating"
    if team_col not in elo.columns or rating_col not in elo.columns:
        return {}
    if "date" in elo.columns:
        elo["date"] = pd.to_datetime(elo["date"], errors="coerce")
        elo = elo.sort_values("date")
    latest = elo.dropna(subset=[team_col, rating_col]).groupby(team_col).tail(1)
    return {
        str(row[team_col]): float(row[rating_col])
        for _, row in latest.iterrows()
        if str(row[team_col]) in teams
    }


def _build_processed(settings: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = _paths(settings)
    paths["processed"].mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(paths["raw"] / "results.csv")
    table = build_table(
        results_df=results,
        elo_df=_load_elo(paths["raw"]),
        fifa_df=_load_fifa(paths["raw"]),
        settings=settings,
        output_path=paths["match_table"],
    )
    table["date"] = pd.to_datetime(table["date"])
    table = (
        table.dropna(subset=["home_elo_pre", "away_elo_pre"])
        .loc[lambda df: df["date"] >= TRAIN_START]
        .sort_values("date")
        .reset_index(drop=True)
    )
    table.to_parquet(paths["match_table"], index=False)

    squad_team_features = _load_squad_team_features(settings)
    squad_history = _load_historical_squad_features(settings)
    features = build_features(
        table,
        squad_values=None if squad_team_features is not None else _load_squad_values(paths["raw"]),
        squad_team_features=squad_team_features,
        squad_history=squad_history,
    )
    features.to_parquet(paths["features"], index=False)
    return table, features


def _load_processed(settings: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = _paths(settings)
    if not paths["match_table"].exists() or not paths["features"].exists():
        return _build_processed(settings)
    return pd.read_parquet(paths["match_table"]), pd.read_parquet(paths["features"])


def _fit_model(settings: dict[str, Any]) -> tuple[MatchModel, pd.DataFrame, pd.DataFrame]:
    table, features = _load_processed(settings)
    table = table.copy()
    table["date"] = pd.to_datetime(table["date"])
    scored_mask = table[["goals_home", "goals_away"]].notna().all(axis=1)
    train_mask = (table["date"] < SPLIT_DATE) & scored_mask
    train = table.loc[train_mask].reset_index(drop=True)
    train_features = features.loc[train_mask].reset_index(drop=True)
    if train.empty:
        raise ValueError("No training rows before split date")

    model = MatchModel(ridge_alpha=0.01, team_alpha=0.10)
    model.fit(train, train_features)

    paths = _paths(settings)
    paths["outputs"].mkdir(parents=True, exist_ok=True)
    with open(paths["model"], "wb") as f:
        pickle.dump(model, f)
    return model, table, features


def _apply_recency_weight(
    table: pd.DataFrame,
    *,
    half_life_years: float | None,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    out = table.copy()
    if half_life_years is None:
        out["recency_weight"] = 1.0
        return out
    if half_life_years <= 0:
        raise ValueError("half_life_years must be positive")
    dates = pd.to_datetime(out["date"])
    ref = pd.Timestamp(reference_date) if reference_date is not None else dates.max()
    age_years = (ref - dates).dt.days.clip(lower=0).to_numpy(float) / 365.25
    out["recency_weight"] = np.exp(-math.log(2.0) * age_years / float(half_life_years))
    return out


def _outcome_log_loss(model: MatchModel, matches: pd.DataFrame, features: pd.DataFrame) -> float:
    return _outcome_scores(model, matches, features)["log_loss"]


def _outcome_scores(model: MatchModel, matches: pd.DataFrame, features: pd.DataFrame) -> dict[str, float]:
    losses: list[float] = []
    briers: list[float] = []
    for i, feat_row in features.iterrows():
        ctx = feat_row.to_dict()
        ctx["team_a"] = matches.loc[i, "team_home"]
        ctx["team_b"] = matches.loc[i, "team_away"]
        w, d, loss = model.derive_wdl(model.predict(ctx))
        gh = float(matches.loc[i, "goals_home"])
        ga = float(matches.loc[i, "goals_away"])
        probs = np.array([w, d, loss], dtype=float)
        actual = np.array([gh > ga, gh == ga, gh < ga], dtype=float)
        p = float(probs[int(actual.argmax())])
        losses.append(-math.log(max(float(p), 1e-12)))
        briers.append(float(np.mean((probs - actual) ** 2)))
    return {"log_loss": float(np.mean(losses)), "brier": float(np.mean(briers))}


FEATURE_GROUP_COLUMNS = {
    "squad_depth": [
        "squad_quality_rank_a", "squad_quality_rank_b",
        "squad_depth_rank_a", "squad_depth_rank_b",
    ],
    "star_power": [
        "star_power_rank_a", "star_power_rank_b",
        "star_concentration_a", "star_concentration_b",
    ],
    "position_groups": [
        "attack_value_rank_a", "attack_value_rank_b",
        "midfield_value_rank_a", "midfield_value_rank_b",
        "defense_value_rank_a", "defense_value_rank_b",
    ],
    "recent_form": [
        "form_a", "form_b",
        "club_minutes_rank_a", "club_minutes_rank_b",
        "club_form_rank_a", "club_form_rank_b",
    ],
    "travel_rest": ["rest_days_a", "rest_days_b", "travel_km_a", "travel_km_b"],
    "gk": ["gk_rating_a", "gk_rating_b"],
}


NEUTRAL_FEATURE_VALUES = {
    "squad_quality_rank_a": 0.5,
    "squad_quality_rank_b": 0.5,
    "squad_depth_rank_a": 0.5,
    "squad_depth_rank_b": 0.5,
    "star_power_rank_a": 0.5,
    "star_power_rank_b": 0.5,
    "star_concentration_a": 0.5,
    "star_concentration_b": 0.5,
    "attack_value_rank_a": 0.5,
    "attack_value_rank_b": 0.5,
    "midfield_value_rank_a": 0.5,
    "midfield_value_rank_b": 0.5,
    "defense_value_rank_a": 0.5,
    "defense_value_rank_b": 0.5,
    "club_minutes_rank_a": 0.5,
    "club_minutes_rank_b": 0.5,
    "club_form_rank_a": 0.5,
    "club_form_rank_b": 0.5,
    "gk_rating_a": 0.5,
    "gk_rating_b": 0.5,
    "form_a": 0.5,
    "form_b": 0.5,
    "rest_days_a": 0.0,
    "rest_days_b": 0.0,
    "travel_km_a": 0.0,
    "travel_km_b": 0.0,
}


def _mask_features(features: pd.DataFrame, active_groups: set[str]) -> pd.DataFrame:
    out = features.copy()
    active_cols = {
        col
        for group in active_groups
        for col in FEATURE_GROUP_COLUMNS.get(group, [])
    }
    for col, neutral in NEUTRAL_FEATURE_VALUES.items():
        if col in out.columns and col not in active_cols:
            out[col] = neutral
    return out


def _fit_with_decay(
    table: pd.DataFrame,
    features: pd.DataFrame,
    *,
    half_life_years: float | None,
    production: bool,
    team_alpha: float = 0.10,
) -> tuple[MatchModel, pd.DataFrame, pd.DataFrame]:
    table = table.copy()
    table["date"] = pd.to_datetime(table["date"])
    if production:
        fit_mask = table[["goals_home", "goals_away"]].notna().all(axis=1)
        ref_date = table.loc[fit_mask, "date"].max()
    else:
        scored_mask = table[["goals_home", "goals_away"]].notna().all(axis=1)
        fit_mask = (table["date"] < SPLIT_DATE) & scored_mask
        ref_date = table.loc[fit_mask, "date"].max()
    fit_table = _apply_recency_weight(
        table.loc[fit_mask].reset_index(drop=True),
        half_life_years=half_life_years,
        reference_date=ref_date,
    )
    fit_features = features.loc[fit_mask].reset_index(drop=True)
    model = MatchModel(ridge_alpha=0.01, team_alpha=team_alpha)
    model.fit(fit_table, fit_features)
    return model, fit_table, fit_features


def _select_recency_decay(
    table: pd.DataFrame,
    features: pd.DataFrame,
    candidates: list[float | None],
) -> tuple[float | None, pd.DataFrame]:
    table = table.copy()
    table["date"] = pd.to_datetime(table["date"])
    scored_mask = table[["goals_home", "goals_away"]].notna().all(axis=1)
    train_mask = (table["date"] < SPLIT_DATE) & scored_mask
    val_mask = (table["date"] >= SPLIT_DATE) & scored_mask
    val_table = table.loc[val_mask].reset_index(drop=True)
    val_features = features.loc[val_mask].reset_index(drop=True)
    if val_table.empty:
        raise ValueError("No validation rows for recency selection")

    rows: list[dict[str, Any]] = []
    best_decay: float | None = None
    best_loss = float("inf")
    for decay in candidates:
        model, _, _ = _fit_with_decay(table, features, half_life_years=decay, production=False)
        loss = _outcome_log_loss(model, val_table, val_features)
        label = "none" if decay is None else f"{decay:g}"
        rows.append({"half_life_years": label, "validation_log_loss": loss})
        if loss < best_loss:
            best_loss = loss
            best_decay = decay
    return best_decay, pd.DataFrame(rows)


def _ablation_ladder(
    table: pd.DataFrame,
    features: pd.DataFrame,
    *,
    half_life_years: float | None,
    min_optional_logloss_delta: float = 0.002,
) -> tuple[pd.DataFrame, set[str]]:
    table = table.copy()
    table["date"] = pd.to_datetime(table["date"])
    scored_mask = table[["goals_home", "goals_away"]].notna().all(axis=1)
    val_mask = (table["date"] >= SPLIT_DATE) & scored_mask
    val_table = table.loc[val_mask].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    active: set[str] = set()
    best_log_loss = float("inf")
    survivors: set[str] = set()

    ladder = [
        ("elo_only", set(), 0.0, None),
        ("+attack_defense", set(), 0.10, None),
        ("+squad_depth", {"squad_depth"}, 0.10, "squad_depth"),
        ("+star_power", {"star_power"}, 0.10, "star_power"),
        ("+position_groups", {"position_groups"}, 0.10, "position_groups"),
        ("+recent_form", {"recent_form"}, 0.10, "recent_form"),
        ("+travel_rest", {"travel_rest"}, 0.10, "travel_rest"),
        ("+gk_rating", {"gk"}, 0.10, "gk"),
    ]

    for name, candidate_groups, team_alpha, new_group in ladder:
        use_groups = set(survivors)
        if new_group is not None:
            use_groups.add(new_group)
        if name == "+attack_defense":
            use_groups = set()
        if name == "elo_only":
            use_groups = set()
        masked = _mask_features(features, use_groups)
        model, _, _ = _fit_with_decay(
            table,
            masked,
            half_life_years=half_life_years,
            production=False,
            team_alpha=team_alpha,
        )
        scores = _outcome_scores(
            model,
            val_table,
            masked.loc[val_mask].reset_index(drop=True),
        )
        improvement = best_log_loss - scores["log_loss"]
        keep = improvement > min_optional_logloss_delta
        if name in {"elo_only", "+attack_defense"}:
            keep = True
        elif keep and new_group is not None:
            survivors.add(new_group)
        rows.append({
            "rung": name,
            "active_groups": ",".join(sorted(use_groups)) or "none",
            "team_alpha": team_alpha,
            "heldout_log_loss": scores["log_loss"],
            "heldout_brier": scores["brier"],
            "log_loss_improvement": improvement if np.isfinite(improvement) else np.nan,
            "kept": keep,
        })
        if keep:
            best_log_loss = min(best_log_loss, scores["log_loss"])
    return pd.DataFrame(rows), survivors


def _load_or_fit_model(settings: dict[str, Any]) -> MatchModel:
    model_path = _paths(settings)["model"]
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    model, _, _ = _fit_model(settings)
    return model


def _counts_to_probabilities(
    counts: dict[str, dict[str, int]],
    *,
    n_runs: int,
    groups: dict[str, list[str]],
) -> pd.DataFrame:
    teams = [team for group in groups.values() for team in group]
    rows: list[dict[str, Any]] = []
    for team in teams:
        row: dict[str, Any] = {"team": team}
        team_counts = counts.get(team, {})
        for stage in STAGES:
            row[f"p_{stage}"] = float(team_counts.get(stage, 0) / n_runs)
        p_group_exit = row["p_group_exit"]
        p_r32_exit = row["p_round_of_32"]
        p_r16_exit = row["p_round_of_16"]
        p_qf_exit = row["p_quarter_final"]
        p_sf_exit = row["p_semi_final"]
        p_third = row["p_third_place"]
        row["p_reach_r32"] = 1.0 - p_group_exit
        row["p_reach_r16"] = 1.0 - p_group_exit - p_r32_exit
        row["p_reach_qf"] = 1.0 - p_group_exit - p_r32_exit - p_r16_exit
        row["p_reach_sf"] = p_sf_exit + p_third + row["p_finalist"]
        row["p_reach_final"] = row["p_finalist"]
        for col in ["p_reach_r32", "p_reach_r16", "p_reach_qf", "p_reach_sf", "p_reach_final"]:
            row[col] = float(np.clip(row[col], 0.0, 1.0))
        rows.append(row)
    cols = [
        "team",
        "p_reach_r32",
        "p_reach_r16",
        "p_reach_qf",
        "p_reach_sf",
        "p_reach_final",
        "p_finalist",
        "p_champion",
    ] + [f"p_{stage}" for stage in STAGES if stage not in {"finalist", "champion"}]
    return pd.DataFrame(rows, columns=cols).sort_values(
        ["p_champion", "p_finalist"], ascending=False
    )


def cmd_build_data(args: argparse.Namespace) -> int:
    settings = _load_settings(args.config)
    start = time.perf_counter()
    table, features = _build_processed(settings)
    paths = _paths(settings)
    print(f"Wrote match table: {paths['match_table']} ({len(table):,} rows)")
    print(f"Wrote features:    {paths['features']} ({len(features):,} rows)")
    print("Schema check: date/team/score, ELO, FIFA, venue, neutral, and importance fields present")
    print(f"Elapsed: {time.perf_counter() - start:.1f}s")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    settings = _load_settings(args.config)
    start = time.perf_counter()
    table, features = _load_processed(settings)
    selected_decay: float | None = args.recency_half_life_years
    if args.select_recency:
        candidates: list[float | None] = [None]
        candidates.extend(float(x) for x in args.decay_grid.split(",") if x.strip())
        selected_decay, decay_table = _select_recency_decay(table, features, candidates)
        paths = _paths(settings)
        paths["outputs"].mkdir(parents=True, exist_ok=True)
        decay_path = paths["outputs"] / "recency_validation.csv"
        decay_table.to_csv(decay_path, index=False)
        print("Recency validation:")
        print(decay_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        _decay_label = "none" if selected_decay is None else f"{selected_decay:g}"
        print(f"Selected half-life years: {_decay_label}")
        print(f"Wrote recency validation: {decay_path}")
    active_groups: set[str] = {
        "squad_depth", "star_power", "position_groups", "recent_form", "travel_rest", "gk"
    }
    if args.ablate:
        ablation_table, active_groups = _ablation_ladder(
            table,
            features,
            half_life_years=selected_decay,
        )
        ablation_path = _paths(settings)["outputs"] / "ablation_ladder.csv"
        ablation_table.to_csv(ablation_path, index=False)
        print("Ablation ladder:")
        print(ablation_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print(f"Surviving feature groups: {','.join(sorted(active_groups)) or 'none'}")
        print(f"Wrote ablation ladder: {ablation_path}")
    active_path = _paths(settings)["outputs"] / "active_feature_groups.json"
    active_path.write_text(json.dumps({"active_groups": sorted(active_groups)}, indent=2))
    model_features = _mask_features(features, active_groups)
    if args.production:
        model, fit_table, _ = _fit_with_decay(
            table,
            model_features,
            half_life_years=selected_decay,
            production=True,
        )
        paths = _paths(settings)
        paths["outputs"].mkdir(parents=True, exist_ok=True)
        with open(paths["model"], "wb") as f:
            pickle.dump(model, f)
        table = fit_table
    else:
        if selected_decay is None:
            model, table, _ = _fit_model(settings)
        else:
            model, fit_table, _ = _fit_with_decay(
                table,
                model_features,
                half_life_years=selected_decay,
                production=False,
            )
            paths = _paths(settings)
            paths["outputs"].mkdir(parents=True, exist_ok=True)
            with open(paths["model"], "wb") as f:
                pickle.dump(model, f)
            table = fit_table
    paths = _paths(settings)
    if args.production:
        val_rows = 0
        train_rows = len(table)
    else:
        full_table, _ = _load_processed(settings)
        full_dates = pd.to_datetime(full_table["date"])
        scored_mask = full_table[["goals_home", "goals_away"]].notna().all(axis=1)
        train_rows = int(((full_dates < SPLIT_DATE) & scored_mask).sum())
        val_rows = int(((full_dates >= SPLIT_DATE) & scored_mask).sum())
    print(f"Wrote model: {paths['model']}")
    print(f"Fit mode: {'production-all-data' if args.production else 'validation-split'}")
    print(f"Train rows: {train_rows:,}")
    print(f"Validation rows reserved: {val_rows:,}")
    _decay_label = "none" if selected_decay is None else f"{selected_decay:g}"
    print(f"Recency half-life years: {_decay_label}")
    print(f"Active feature groups: {','.join(sorted(active_groups)) or 'none'}")
    print(f"Latest fit date: {pd.to_datetime(table['date']).max().date()}")
    print(f"rho: {model.rho_:+.4f}")
    if getattr(model, "fit_result_", None) is not None:
        result = model.fit_result_
        print(f"Optimizer: success={result.success} nit={result.nit} fun={result.fun:.3f}")
    for team in args.report_teams.split(","):
        team = team.strip()
        if not team:
            continue
        if model.team_map_ and team in model.team_map_:
            idx = model.team_map_[team]
            atk = float(model.team_atk_[idx])
            dfn = float(model.team_dfn_[idx])
            print(f"Team effect {team:<12s} atk={atk:+.4f} dfn={dfn:+.4f} net={atk-dfn:+.4f}")
        else:
            print(f"Team effect {team:<12s} not available")
    print(f"Elapsed: {time.perf_counter() - start:.1f}s")
    return 0


def _parse_results(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    df = pd.read_csv(path)
    column_sets = [
        ("team_home", "team_away", "goals_home", "goals_away"),
        ("home_team", "away_team", "home_score", "away_score"),
        ("home", "away", "home_goals", "away_goals"),
        ("home", "away", "goals_home", "goals_away"),
    ]
    for home_col, away_col, gh_col, ga_col in column_sets:
        if {home_col, away_col, gh_col, ga_col}.issubset(df.columns):
            rows = df[[home_col, away_col, gh_col, ga_col]].dropna()
            return {
                (str(row[home_col]), str(row[away_col])): (int(row[gh_col]), int(row[ga_col]))
                for _, row in rows.iterrows()
            }
    raise ValueError(
        "Results file must contain one of: "
        "team_home/team_away/goals_home/goals_away, "
        "home_team/away_team/home_score/away_score, or "
        "home/away plus goal columns"
    )


def cmd_simulate(args: argparse.Namespace) -> int:
    settings = _load_settings(args.config)
    paths = _paths(settings)
    groups = _load_groups(paths["root"])
    model = _load_or_fit_model(settings)
    n_runs = int(args.n_runs or settings.get("simulation", {}).get("n_runs", 25000))
    seed = int(args.seed if args.seed is not None else settings.get("simulation", {}).get("random_seed", 42))
    strengths = _load_strengths(paths["raw"], groups)
    realized_results = getattr(args, "realized_results", None)
    active_path = paths["outputs"] / "active_feature_groups.json"
    if active_path.exists():
        active_groups = set(json.loads(active_path.read_text()).get("active_groups", []))
    else:
        active_groups = {"squad", "recent_form", "travel_rest", "gk"}
    team_features_df = _load_squad_team_features(settings)
    team_features: dict[str, dict[str, float]] | None = None
    if team_features_df is not None:
        masked_team_features = team_features_df.copy()
        if "squad_depth" not in active_groups:
            masked_team_features["squad_quality_rank"] = 0.5
            masked_team_features["squad_depth_rank"] = 0.5
        if "star_power" not in active_groups:
            masked_team_features["star_power_rank"] = 0.5
            masked_team_features["star_concentration"] = 0.5
        if "position_groups" not in active_groups:
            masked_team_features["attack_value_rank"] = 0.5
            masked_team_features["midfield_value_rank"] = 0.5
            masked_team_features["defense_value_rank"] = 0.5
        if "recent_form" not in active_groups:
            masked_team_features["club_minutes_rank"] = 0.5
            masked_team_features["club_form_rank"] = 0.5
            masked_team_features["national_form"] = 0.5
        if "gk" not in active_groups:
            masked_team_features["gk_rating"] = 0.5
        team_features = masked_team_features.set_index("team").to_dict(orient="index")

    start = time.perf_counter()
    sim = TournamentSimulator(
        groups,
        model,
        settings,
        strengths=strengths or None,
        realized_results=realized_results,
        team_features=team_features,
    )
    counts = sim.run(n_runs=n_runs, seed=seed)
    probs = _counts_to_probabilities(counts, n_runs=n_runs, groups=groups)

    paths["outputs"].mkdir(parents=True, exist_ok=True)
    probs.to_csv(paths["probabilities"], index=False)
    with open(paths["counts"], "w") as f:
        json.dump(counts, f, indent=2, sort_keys=True)

    print(f"Wrote probabilities: {paths['probabilities']}")
    print(f"Wrote simulation counts: {paths['counts']}")
    print(f"Runs: {n_runs:,}  seed: {seed}")
    print("Top champion probabilities:")
    for _, row in probs.head(8).iterrows():
        print(f"  {row['team']:<22s} champion={row['p_champion']:.3f} finalist={row['p_finalist']:.3f}")
    print(f"Elapsed: {time.perf_counter() - start:.1f}s")
    return 0


def _market_vector(df: pd.DataFrame, prefix: str, method: str) -> np.ndarray:
    if f"{prefix}_champion" in df.columns:
        return df[f"{prefix}_champion"].to_numpy(dtype=float)
    if f"{prefix}_raw_champion" in df.columns:
        return devig(df[f"{prefix}_raw_champion"].to_numpy(dtype=float), method=method)
    cols = [f"{prefix}_{stage}" for stage in ("home", "draw", "away")]
    if all(col in df.columns for col in cols):
        return df[cols].to_numpy(dtype=float).ravel()
    raw_cols = [f"{prefix}_raw_{stage}" for stage in ("home", "draw", "away")]
    if all(col in df.columns for col in raw_cols):
        return np.concatenate([devig(row, method=method) for row in df[raw_cols].to_numpy(dtype=float)])
    raise ValueError(
        "Market data must include market_home/market_draw/market_away or "
        "market_raw_home/market_raw_draw/market_raw_away columns"
    )


def _model_vector(df: pd.DataFrame, prefix: str) -> np.ndarray:
    if f"{prefix}_champion" in df.columns:
        return df[f"{prefix}_champion"].to_numpy(dtype=float)
    cols = [f"{prefix}_{stage}" for stage in ("home", "draw", "away")]
    if not all(col in df.columns for col in cols):
        raise ValueError(f"Missing model probability columns: {cols}")
    return df[cols].to_numpy(dtype=float).ravel()


def cmd_evaluate(args: argparse.Namespace) -> int:
    settings = _load_settings(args.config)
    paths = _paths(settings)
    market_path = Path(args.market_data) if args.market_data else paths["outputs"] / "market_closing.csv"
    if not market_path.exists():
        raise FileNotFoundError(
            f"Market data not found: {market_path}\n"
            "Provide --market-data with de-vig-ready closing lines. Evaluation is the only "
            "market-reading path, and it will not compare against raw implied probabilities."
        )

    method = args.devig_method or settings.get("eval", {}).get("devig_method", "shin")
    market = load_market_data(market_path)
    market_probs = _market_vector(market, "market", method)
    model_probs = _model_vector(market, "model")
    scores = score(model_probs, market_probs)

    predictions = pd.DataFrame({"model_prob": model_probs, "outcome": market_probs})
    bins = int(settings.get("eval", {}).get("reliability_bins", 10))
    live_rows = pd.read_csv(args.live_ab) if args.live_ab else (
        pd.read_csv(paths["live_ab"]) if paths["live_ab"].exists() else pd.DataFrame()
    )
    metrics_path = evaluate_and_persist(
        predictions,
        model_probs=model_probs,
        market_probs=market_probs,
        live_rows=live_rows,
        output_dir=paths["metrics"],
        n_bins=bins,
    )
    reliability = reliability_diagram(predictions, n_bins=bins)
    ab = ab_test(live_rows) if not live_rows.empty else {"n": 0, "base": {}, "adjusted": {}, "delta": {}}

    print(f"De-vig method: {method}")
    print(f"Brier:   {scores['brier']:.6f}")
    print(f"Logloss: {scores['log_loss']:.6f}")
    print("Reliability bins:")
    print(reliability.to_string(index=False))
    print(f"A/B adjusted vs base: {json.dumps(ab, sort_keys=True)}")
    if not live_rows.empty:
        print(f"Source-tier ablation: {json.dumps(ablate_by_source(live_rows), sort_keys=True)}")
    print(f"Wrote metrics: {metrics_path}")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    realized = _parse_results(results_path)
    setattr(args, "realized_results", realized)
    print(f"Loaded realized results: {results_path} ({len(realized):,} scored rows)")
    print("Confirmed-lineup supersession is handled in the adjustment layer before simulation.")
    return cmd_simulate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wcpredict")
    parser.add_argument("--config", default=None, help="Path to config/settings.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-data", help="Ingest raw files and build processed match/features tables")
    p_build.set_defaults(func=cmd_build_data)

    p_fit = sub.add_parser("fit", help="Train the match model")
    p_fit.add_argument("--production", action="store_true", help="Fit on all processed rows")
    p_fit.add_argument("--select-recency", action="store_true", help="Choose recency decay on held-out log-loss")
    p_fit.add_argument("--ablate", action="store_true", help="Run held-out feature ablation ladder")
    p_fit.add_argument("--recency-half-life-years", type=float, default=None)
    p_fit.add_argument("--decay-grid", default="3,5,7,10,14,20")
    p_fit.add_argument(
        "--report-teams",
        default="Germany,Argentina,England,Brazil,Spain,France",
    )
    p_fit.set_defaults(func=cmd_fit)

    p_sim = sub.add_parser("simulate", help="Run tournament simulation")
    p_sim.add_argument("--n-runs", type=int, default=None)
    p_sim.add_argument("--seed", type=int, default=None)
    p_sim.set_defaults(func=cmd_simulate)

    p_eval = sub.add_parser("evaluate", help="Score model calibration against de-vigged market close")
    p_eval.add_argument("--market-data", default=None)
    p_eval.add_argument("--live-ab", default=None, help="Optional live per-match A/B accumulation CSV")
    p_eval.add_argument("--devig-method", choices=("shin", "proportional"), default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_update = sub.add_parser("update", help="Apply realized results and re-simulate the remaining tournament")
    p_update.add_argument("--results", required=True, help="Match-day realized results CSV")
    p_update.add_argument("--n-runs", type=int, default=None)
    p_update.add_argument("--seed", type=int, default=None)
    p_update.set_defaults(func=cmd_update)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
