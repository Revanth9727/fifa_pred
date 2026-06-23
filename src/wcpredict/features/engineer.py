"""
features/engineer.py — pure transform, no network access, no llm/ or market imports.

Produces the feature vector defined in docs/data_exp.md §2 from the per-match
training table (data_exp.md §1) plus optional squad-value data.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

HOST_NATIONS: frozenset[str] = frozenset({"United States", "Mexico", "Canada"})
_EARTH_RADIUS_KM = 6371.0

FEATURE_COLS = [
    "elo_diff",
    "fifa_rank_diff",
    "spi_diff",
    "squad_quality_rank_a",
    "squad_quality_rank_b",
    "squad_depth_rank_a",
    "squad_depth_rank_b",
    "star_power_rank_a",
    "star_power_rank_b",
    "star_concentration_a",
    "star_concentration_b",
    "attack_value_rank_a",
    "attack_value_rank_b",
    "midfield_value_rank_a",
    "midfield_value_rank_b",
    "defense_value_rank_a",
    "defense_value_rank_b",
    "club_minutes_rank_a",
    "club_minutes_rank_b",
    "club_form_rank_a",
    "club_form_rank_b",
    "gk_rating_a",
    "gk_rating_b",
    "form_a",
    "form_b",
    "host_flag",
    "neutral_flag",
    "rest_days_a",
    "rest_days_b",
    "travel_km_a",
    "travel_km_b",
    "importance_weight",
]


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


def _load_settings(settings_path: Path | str | None = None) -> dict[str, Any]:
    root = _project_root()
    path = Path(settings_path) if settings_path else root / "config" / "settings.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _compute_form(
    team: str,
    before_date: pd.Timestamp,
    matches: pd.DataFrame,
    window_months: int,
) -> float:
    """
    Recency-weighted form for *team* in the *window_months* months before
    *before_date*. Result scored W=1, D=0.5, L=0; weighted by exp(-k*days_ago).
    Returns NaN when no matches exist in the window.
    """
    cutoff = before_date - pd.DateOffset(months=window_months)
    home_mask = matches["team_home"] == team
    away_mask = matches["team_away"] == team
    date_mask = (matches["date"] < before_date) & (matches["date"] >= cutoff)
    sub = matches.loc[(home_mask | away_mask) & date_mask]
    if sub.empty:
        return float("nan")

    k = 0.01  # decay ≈ 70-day half-life
    total_w = 0.0
    weighted_pts = 0.0
    for _, row in sub.iterrows():
        days_ago = max((before_date - row["date"]).days, 0)
        w = math.exp(-k * days_ago)
        gh = int(row["goals_home"]) if pd.notna(row["goals_home"]) else 0
        ga = int(row["goals_away"]) if pd.notna(row["goals_away"]) else 0
        if row["team_home"] == team:
            pts = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
        else:
            pts = 1.0 if ga > gh else (0.5 if gh == ga else 0.0)
        weighted_pts += pts * w
        total_w += w

    return weighted_pts / total_w if total_w > 0 else float("nan")


def compute_squad_quality_ranks(
    squad_values: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
) -> pd.Series:
    """
    Convert a DataFrame with (team, [date,] total_value) into a rank-signal
    Series team -> normalized_rank in [0, 1], where 1.0 = best squad.

    Raw euro levels are NOT exposed — only the relative rank position, so that
    a single-star team cannot inflate its own lambda via an extreme raw number.
    """
    sv = squad_values.copy()
    if "date" in sv.columns and as_of is not None:
        sv["date"] = pd.to_datetime(sv["date"])
        sv = sv[sv["date"] <= as_of].sort_values("date")
        sv = sv.groupby("team").last().reset_index()
    if sv.empty or "total_value" not in sv.columns:
        return pd.Series(dtype=float)
    n = len(sv)
    sv = sv.copy()
    sv["_rank"] = sv["total_value"].rank(ascending=False, method="min")
    sv["quality_rank"] = 1.0 - (sv["_rank"] - 1.0) / max(n - 1, 1)
    return sv.set_index("team")["quality_rank"]


def _rank_signal(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index)
    ranks = valid.rank(ascending=False, method="min")
    out = 1.0 - (ranks - 1.0) / max(len(valid) - 1, 1)
    return out.reindex(values.index)


def _squad_value_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for team, sub in df.dropna(subset=["team", "value"]).groupby("team"):
        vals = sub["value"].sort_values(ascending=False).to_numpy(float)
        if len(vals) == 0:
            continue
        top18 = float(vals[:18].sum())
        top5 = float(vals[:5].sum())
        top3 = float(vals[:3].sum())
        top1 = float(vals[0])

        def _pos_sum(label: str) -> float:
            psub = sub[sub["position_group"] == label]["value"].sort_values(ascending=False)
            return float(psub.head(8).sum())

        rows.append({
            "team": team,
            "squad_depth_value": top18,
            "squad_quality_value": top5,
            "star_power_value": top3,
            "star_top1_value": top1,
            "star_concentration": top3 / top18 if top18 > 0 and len(vals) >= 15 else np.nan,
            "attack_value": _pos_sum("attack"),
            "midfield_value": _pos_sum("midfield"),
            "defense_value": _pos_sum("defense"),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["squad_depth_rank"] = _rank_signal(out.set_index("team")["squad_depth_value"]).to_numpy()
    out["squad_quality_rank"] = _rank_signal(out.set_index("team")["squad_quality_value"]).to_numpy()
    out["star_power_rank"] = _rank_signal(out.set_index("team")["star_power_value"]).to_numpy()
    out["attack_value_rank"] = _rank_signal(out.set_index("team")["attack_value"]).to_numpy()
    out["midfield_value_rank"] = _rank_signal(out.set_index("team")["midfield_value"]).to_numpy()
    out["defense_value_rank"] = _rank_signal(out.set_index("team")["defense_value"]).to_numpy()
    return out


def _position_group(position: Any, sub_position: Any = "") -> str:
    text = f"{position or ''} {sub_position or ''}".lower()
    if "attack" in text or "forward" in text or "winger" in text or "striker" in text:
        return "attack"
    if "midfield" in text:
        return "midfield"
    if "defender" in text or "back" in text:
        return "defense"
    if "goalkeeper" in text:
        return "gk"
    return "other"


def compute_historical_squad_team_features(
    players: pd.DataFrame,
    valuations: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    start: pd.Timestamp | None = None,
    freq: str = "MS",
) -> pd.DataFrame:
    """
    Monthly historical squad/star proxies by citizenship.

    This is for validation only: it uses player citizenship and valuations as
    known by each snapshot date, not the 2026 named squad. It lets squad/star
    features earn their place on historical held-out log-loss.
    """
    cutoff = pd.Timestamp(cutoff)
    val = valuations.copy()
    val["player_id"] = pd.to_numeric(val["player_id"], errors="coerce")
    val["date"] = pd.to_datetime(val["date"], errors="coerce")
    val["market_value_in_eur"] = pd.to_numeric(val["market_value_in_eur"], errors="coerce")
    val = val.dropna(subset=["player_id", "date", "market_value_in_eur"])
    val = val[val["date"] <= cutoff].sort_values(["date", "player_id"])
    if start is None:
        start = max(pd.Timestamp("2000-01-01"), val["date"].min())
    snapshots = pd.date_range(pd.Timestamp(start), cutoff, freq=freq)
    if cutoff not in snapshots:
        snapshots = snapshots.append(pd.DatetimeIndex([cutoff]))

    meta = players[[
        "player_id", "country_of_citizenship", "position", "sub_position"
    ]].copy()
    meta["player_id"] = pd.to_numeric(meta["player_id"], errors="coerce")
    meta = meta.dropna(subset=["player_id", "country_of_citizenship"])
    meta["player_id"] = meta["player_id"].astype(int)
    meta["team"] = meta["country_of_citizenship"].astype(str)
    meta["position_group"] = [
        _position_group(pos, sub)
        for pos, sub in zip(meta["position"], meta["sub_position"])
    ]

    current = pd.Series(dtype=float)
    rows: list[pd.DataFrame] = []
    cursor = 0
    val_dates = val["date"].to_numpy()
    for snap in snapshots:
        end = int(np.searchsorted(val_dates, np.datetime64(snap), side="right"))
        if end > cursor:
            upd = val.iloc[cursor:end]
            latest = upd.groupby("player_id")["market_value_in_eur"].last()
            for pid, value in latest.items():
                current.at[int(pid)] = float(value)
            cursor = end
        if current.empty:
            continue
        snap_df = current.rename("value").reset_index().rename(columns={"index": "player_id"})
        snap_df["player_id"] = snap_df["player_id"].astype(int)
        snap_df = snap_df.merge(meta[["player_id", "team", "position_group"]], on="player_id", how="inner")
        agg = _squad_value_aggregates(snap_df)
        if agg.empty:
            continue
        agg.insert(0, "date", snap)
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_named_squad_team_features(
    squads: pd.DataFrame,
    players: pd.DataFrame,
    appearances: pd.DataFrame | None,
    valuations: pd.DataFrame | None,
    *,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """
    Aggregate current named-squad features by team.

    This is a pure transform over frozen raw inputs. It does not fetch FBref or
    fabricate historical GK columns. ``gk_rating`` is current-snapshot only;
    with the local data available here it is a Transfermarkt/current-minutes
    keeper proxy and should be ablated before use.
    """
    sq = squads.copy()
    if "player_id" not in sq.columns:
        raise ValueError("squads must include player_id")
    sq["player_id"] = pd.to_numeric(sq["player_id"], errors="coerce")
    sq = sq.dropna(subset=["player_id"]).copy()
    sq["player_id"] = sq["player_id"].astype(int)

    cutoff = pd.Timestamp(cutoff)
    player_meta = players.copy()
    player_meta["player_id"] = pd.to_numeric(player_meta["player_id"], errors="coerce")
    player_meta = player_meta.dropna(subset=["player_id"]).copy()
    player_meta["player_id"] = player_meta["player_id"].astype(int)

    if valuations is not None and not valuations.empty:
        val = valuations.copy()
        val["player_id"] = pd.to_numeric(val["player_id"], errors="coerce")
        val["date"] = pd.to_datetime(val["date"], errors="coerce")
        val["market_value_in_eur"] = pd.to_numeric(val["market_value_in_eur"], errors="coerce")
        val = (
            val.dropna(subset=["player_id", "date"])
            .loc[lambda d: d["date"] <= cutoff]
            .sort_values("date")
            .groupby("player_id", as_index=False)
            .tail(1)[["player_id", "market_value_in_eur"]]
        )
    else:
        val = player_meta[["player_id", "market_value_in_eur"]].copy()
        val["market_value_in_eur"] = pd.to_numeric(val["market_value_in_eur"], errors="coerce")

    val = val.rename(columns={"market_value_in_eur": "current_market_value_in_eur"})
    enriched = sq.drop(columns=["market_value_in_eur"], errors="ignore").merge(
        val, on="player_id", how="left"
    )
    team = enriched.groupby("team", as_index=True).agg(
        total_value=("current_market_value_in_eur", "sum"),
        n_squad=("player_id", "count"),
        n_value_missing=("current_market_value_in_eur", lambda s: int(s.isna().sum())),
    )

    if appearances is not None and not appearances.empty:
        app = appearances.copy()
        app["player_id"] = pd.to_numeric(app["player_id"], errors="coerce")
        app["date"] = pd.to_datetime(app["date"], errors="coerce")
        app["minutes_played"] = pd.to_numeric(app["minutes_played"], errors="coerce").fillna(0.0)
        app["goals"] = pd.to_numeric(app.get("goals", 0.0), errors="coerce").fillna(0.0)
        app["assists"] = pd.to_numeric(app.get("assists", 0.0), errors="coerce").fillna(0.0)
        app = app.dropna(subset=["player_id", "date"])
        app = app[app["date"] <= cutoff]
        squad_ids = set(enriched["player_id"])
        app = app[app["player_id"].isin(squad_ids)]

        for days, col in [(90, "club_minutes_90d"), (180, "club_minutes_180d")]:
            sub = app[app["date"] >= cutoff - pd.Timedelta(days=days)]
            agg = (
                sub.groupby("player_id", as_index=False)["minutes_played"].sum()
                .rename(columns={"minutes_played": col})
            )
            enriched = enriched.merge(agg, on="player_id", how="left")
            enriched[col] = enriched[col].fillna(0.0)

        form_sub = app[app["date"] >= cutoff - pd.Timedelta(days=180)].copy()
        form_sub["goal_assist"] = form_sub["goals"] + form_sub["assists"]
        player_form = (
            form_sub.groupby("player_id", as_index=False)
            .agg(club_form_180d=("goal_assist", "sum"))
        )
        enriched = enriched.merge(player_form, on="player_id", how="left")
        enriched["club_form_180d"] = enriched["club_form_180d"].fillna(0.0)
    else:
        for col in ["club_minutes_90d", "club_minutes_180d", "club_form_180d"]:
            enriched[col] = np.nan

    minutes = enriched.groupby("team", as_index=True).agg(
        club_minutes_90d=("club_minutes_90d", "sum"),
        club_minutes_180d=("club_minutes_180d", "sum"),
        club_form_180d=("club_form_180d", "sum"),
    )
    team = team.join(minutes, how="left")

    agg_source = enriched.rename(columns={"current_market_value_in_eur": "value"}).copy()
    agg_source["position_group"] = [
        _position_group(pos, "")
        for pos in agg_source["position"]
    ]
    named_agg = _squad_value_aggregates(agg_source[["team", "value", "position_group"]])
    if not named_agg.empty:
        team = team.drop(columns=["total_value"], errors="ignore").join(
            named_agg.set_index("team"), how="left"
        )

    keepers = enriched[enriched["position"].astype(str).str.upper().eq("GK")].copy()
    if not keepers.empty:
        keepers["_keeper_minutes"] = pd.to_numeric(
            keepers.get("club_minutes_180d", np.nan), errors="coerce"
        ).fillna(0.0)
        keepers["_keeper_value"] = pd.to_numeric(
            keepers.get("current_market_value_in_eur", np.nan), errors="coerce"
        )
        likely = (
            keepers.sort_values(["team", "_keeper_minutes", "_keeper_value"], ascending=[True, False, False])
            .groupby("team", as_index=False)
            .head(1)
            .set_index("team")
        )
        gk_value_rank = _rank_signal(likely["_keeper_value"])
        gk_minutes_rank = _rank_signal(likely["_keeper_minutes"])
        gk_rating = (0.75 * gk_value_rank + 0.25 * gk_minutes_rank).clip(0.0, 1.0)
        team = team.join(gk_rating.rename("gk_rating"), how="left")
        team = team.join(likely["name"].rename("likely_gk"), how="left")
    else:
        team["gk_rating"] = np.nan
        team["likely_gk"] = np.nan

    if "squad_quality_rank" not in team.columns:
        team["squad_quality_rank"] = _rank_signal(team["total_value"])
    if "squad_depth_rank" not in team.columns:
        team["squad_depth_rank"] = team["squad_quality_rank"]
    team["club_minutes_rank"] = _rank_signal(team["club_minutes_180d"])
    team["club_form_rank"] = _rank_signal(team["club_form_180d"])
    return team.reset_index()


def _rest_travel(
    team: str,
    match_date: pd.Timestamp,
    match_lat: float | None,
    match_lon: float | None,
    matches: pd.DataFrame,
) -> tuple[float, float]:
    """
    Returns (rest_days, travel_km) for *team* arriving at the current match.
    Both NaN when no prior match exists or coords are unavailable.
    """
    home_mask = matches["team_home"] == team
    away_mask = matches["team_away"] == team
    prev = matches.loc[(home_mask | away_mask) & (matches["date"] < match_date)]
    if prev.empty:
        return float("nan"), float("nan")

    last = prev.sort_values("date").iloc[-1]
    rest_days = float((match_date - last["date"]).days)

    travel_km = float("nan")
    prev_lat = last.get("venue_lat")
    prev_lon = last.get("venue_lon")
    if (
        match_lat is not None
        and match_lon is not None
        and not math.isnan(match_lat)
        and not math.isnan(match_lon)
        and pd.notna(prev_lat)
        and pd.notna(prev_lon)
    ):
        try:
            travel_km = haversine_km(float(prev_lat), float(prev_lon), match_lat, match_lon)
        except (ValueError, TypeError):
            pass

    return rest_days, travel_km


def _team_appearance_features(
    matches: pd.DataFrame,
    window_months: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute form/rest/travel for both teams in one grouped chronological pass.

    The old implementation scanned the full match table once per row and per
    team, which is O(n^2). This builds a two-row-per-match appearance table and
    scans only within each team's own history.
    """
    n = len(matches)
    if n == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty, empty, empty

    home_win = matches["goals_home"].to_numpy(float) > matches["goals_away"].to_numpy(float)
    away_win = matches["goals_away"].to_numpy(float) > matches["goals_home"].to_numpy(float)
    draw = matches["goals_home"].to_numpy(float) == matches["goals_away"].to_numpy(float)
    home_pts = np.where(home_win, 1.0, np.where(draw, 0.5, 0.0))
    away_pts = np.where(away_win, 1.0, np.where(draw, 0.5, 0.0))

    home_rows = pd.DataFrame({
        "match_idx": np.arange(n, dtype=np.int64),
        "side": "a",
        "team": matches["team_home"].to_numpy(),
        "date": matches["date"].to_numpy(),
        "points": home_pts,
        "venue_lat": matches.get("venue_lat", pd.Series(np.nan, index=matches.index)).to_numpy(float),
        "venue_lon": matches.get("venue_lon", pd.Series(np.nan, index=matches.index)).to_numpy(float),
    })
    away_rows = pd.DataFrame({
        "match_idx": np.arange(n, dtype=np.int64),
        "side": "b",
        "team": matches["team_away"].to_numpy(),
        "date": matches["date"].to_numpy(),
        "points": away_pts,
        "venue_lat": matches.get("venue_lat", pd.Series(np.nan, index=matches.index)).to_numpy(float),
        "venue_lon": matches.get("venue_lon", pd.Series(np.nan, index=matches.index)).to_numpy(float),
    })
    long = pd.concat([home_rows, away_rows], ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["team", "date", "match_idx", "side"]).reset_index(drop=True)

    form = np.full(len(long), np.nan)
    rest = np.full(len(long), np.nan)
    travel = np.full(len(long), np.nan)
    k = 0.01

    for _, idx in long.groupby("team", sort=False).indices.items():
        positions = np.asarray(idx, dtype=np.int64)
        order = np.lexsort((
            long.iloc[positions]["side"].to_numpy(),
            long.iloc[positions]["match_idx"].to_numpy(),
            pd.to_datetime(long.iloc[positions]["date"]).astype("int64").to_numpy(),
        ))
        positions = positions[order]
        sub = long.iloc[positions]
        dates = pd.to_datetime(sub["date"]).reset_index(drop=True)
        date_ns = dates.to_numpy(dtype="datetime64[ns]").astype("int64")
        day_num = (date_ns // 86_400_000_000_000).astype(float)
        pts = sub["points"].to_numpy(float)
        lat = sub["venue_lat"].to_numpy(float)
        lon = sub["venue_lon"].to_numpy(float)

        for local_i, global_i in enumerate(positions):
            current = dates.iloc[local_i]
            cutoff = current - pd.DateOffset(months=window_months)
            current_ns = pd.Timestamp(current).value
            start = int(np.searchsorted(date_ns, cutoff.value, side="left"))
            end = int(np.searchsorted(date_ns, current_ns, side="left"))

            if end > start:
                days_ago = day_num[local_i] - day_num[start:end]
                weights = np.exp(-k * days_ago)
                total_w = float(weights.sum())
                if total_w > 0:
                    form[global_i] = float(np.dot(pts[start:end], weights) / total_w)

            prev_i = end - 1
            if prev_i >= 0:
                rest[global_i] = float((current - dates.iloc[prev_i]).days)
                if (
                    not np.isnan(lat[local_i])
                    and not np.isnan(lon[local_i])
                    and not np.isnan(lat[prev_i])
                    and not np.isnan(lon[prev_i])
                ):
                    travel[global_i] = haversine_km(
                        float(lat[prev_i]), float(lon[prev_i]),
                        float(lat[local_i]), float(lon[local_i]),
                    )

    long["form"] = form
    long["rest"] = rest
    long["travel"] = travel

    home = long[long["side"] == "a"].sort_values("match_idx")
    away = long[long["side"] == "b"].sort_values("match_idx")
    return (
        home["form"].to_numpy(float),
        away["form"].to_numpy(float),
        home["rest"].to_numpy(float),
        away["rest"].to_numpy(float),
        home["travel"].to_numpy(float),
        away["travel"].to_numpy(float),
    )


def build_features(
    matches: pd.DataFrame,
    squad_values: pd.DataFrame | None = None,
    squad_team_features: pd.DataFrame | None = None,
    squad_history: pd.DataFrame | None = None,
    settings: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Produce the feature vector (docs/data_exp.md §2) for every row in *matches*.

    Parameters
    ----------
    matches:
        Per-match table from build_table(). Required columns: date, team_home,
        team_away, goals_home, goals_away, neutral, importance_weight,
        home_elo_pre, away_elo_pre, home_fifa_rank, away_fifa_rank,
        venue_lat, venue_lon.
    squad_values:
        Optional DataFrame with (team, [date,] total_value). If None,
        squad_quality_rank_a/b are NaN.
    settings:
        Parsed config/settings.yaml. Loaded from disk if None.

    Returns
    -------
    pd.DataFrame with columns matching FEATURE_COLS, index aligned with *matches*.

    Notes
    -----
    - NO llm/ or market imports (ai_rules.md #2, #3).
    - Pure transform — no network, no file I/O.
    """
    if settings is None:
        settings = _load_settings()

    form_window = int(settings.get("match_model", {}).get("form_window_months", 9))

    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # elo_diff: team_a (home) minus team_b (away) — the workhorse strength signal
    df["elo_diff"] = (
        pd.to_numeric(df.get("home_elo_pre"), errors="coerce")
        - pd.to_numeric(df.get("away_elo_pre"), errors="coerce")
    )

    # fifa_rank_diff: lower rank number = better team, so away - home → positive = home stronger
    df["fifa_rank_diff"] = (
        pd.to_numeric(df.get("away_fifa_rank"), errors="coerce")
        - pd.to_numeric(df.get("home_fifa_rank"), errors="coerce")
    )

    # spi_diff: not in the standard raw sources; reserved for future data
    df["spi_diff"] = float("nan")

    # host_flag: 1 if either team is a 2026 host nation
    df["host_flag"] = (
        df["team_home"].isin(HOST_NATIONS) | df["team_away"].isin(HOST_NATIONS)
    ).astype(int)

    # neutral_flag: 1 if neutral venue
    df["neutral_flag"] = df["neutral"].astype(int) if "neutral" in df.columns else 0

    # Squad quality ranks — rank-signal, not raw euro level (ai_rules, arch.md)
    squad_cols = [
        "squad_quality_rank", "squad_depth_rank", "star_power_rank",
        "star_concentration", "attack_value_rank", "midfield_value_rank",
        "defense_value_rank",
    ]
    if squad_history is not None and not squad_history.empty:
        hist = squad_history.copy()
        hist["date"] = pd.to_datetime(hist["date"])
        left = pd.concat([
            df[["date", "team_home"]].rename(columns={"team_home": "team"}).assign(_side="a", _row=np.arange(len(df))),
            df[["date", "team_away"]].rename(columns={"team_away": "team"}).assign(_side="b", _row=np.arange(len(df))),
        ], ignore_index=True).sort_values(["team", "date"])
        right = hist[["date", "team"] + [c for c in squad_cols if c in hist.columns]].sort_values(["team", "date"])
        merged_parts = []
        for team, sub in left.groupby("team", sort=False):
            rsub = right[right["team"] == team]
            if rsub.empty:
                continue
            merged_parts.append(pd.merge_asof(sub.sort_values("date"), rsub.sort_values("date"), on="date", by="team", direction="backward"))
        if merged_parts:
            merged = pd.concat(merged_parts, ignore_index=True)
            for col in squad_cols:
                if col in merged.columns:
                    a_map = merged[merged["_side"] == "a"].set_index("_row")[col]
                    b_map = merged[merged["_side"] == "b"].set_index("_row")[col]
                    df[f"{col}_a"] = df.index.to_series().map(a_map)
                    df[f"{col}_b"] = df.index.to_series().map(b_map)

    if squad_team_features is not None and not squad_team_features.empty:
        stf = squad_team_features.set_index("team")
        snapshot_cutoff = pd.Timestamp(settings.get("snapshot", {}).get("cutoff", df["date"].max()))
        current_mask = df["date"] >= snapshot_cutoff
        for col in squad_cols:
            if col in stf:
                if f"{col}_a" not in df:
                    df[f"{col}_a"] = np.nan
                    df[f"{col}_b"] = np.nan
                df.loc[current_mask, f"{col}_a"] = df.loc[current_mask, f"{col}_a"].fillna(
                    df.loc[current_mask, "team_home"].map(stf[col])
                )
                df.loc[current_mask, f"{col}_b"] = df.loc[current_mask, f"{col}_b"].fillna(
                    df.loc[current_mask, "team_away"].map(stf[col])
                )
        ranks = df.get("squad_quality_rank_a", pd.Series(np.nan, index=df.index))
        club_minutes = stf["club_minutes_rank"] if "club_minutes_rank" in stf else pd.Series(dtype=float)
        club_form = stf["club_form_rank"] if "club_form_rank" in stf else pd.Series(dtype=float)
        gk = stf["gk_rating"] if "gk_rating" in stf else pd.Series(dtype=float)
        if "squad_quality_rank_a" not in df:
            df["squad_quality_rank_a"] = df["team_home"].map(stf["squad_quality_rank"]) if "squad_quality_rank" in stf else np.nan
            df["squad_quality_rank_b"] = df["team_away"].map(stf["squad_quality_rank"]) if "squad_quality_rank" in stf else np.nan
        for col in ["club_minutes_rank", "club_form_rank", "gk_rating"]:
            df[f"{col}_a"] = np.nan
            df[f"{col}_b"] = np.nan
        df.loc[current_mask, "club_minutes_rank_a"] = df.loc[current_mask, "team_home"].map(club_minutes)
        df.loc[current_mask, "club_minutes_rank_b"] = df.loc[current_mask, "team_away"].map(club_minutes)
        df.loc[current_mask, "club_form_rank_a"] = df.loc[current_mask, "team_home"].map(club_form)
        df.loc[current_mask, "club_form_rank_b"] = df.loc[current_mask, "team_away"].map(club_form)
        df.loc[current_mask, "gk_rating_a"] = df.loc[current_mask, "team_home"].map(gk)
        df.loc[current_mask, "gk_rating_b"] = df.loc[current_mask, "team_away"].map(gk)
    elif squad_values is not None:
        ranks = compute_squad_quality_ranks(squad_values)
        df["squad_quality_rank_a"] = df["team_home"].map(ranks)
        df["squad_quality_rank_b"] = df["team_away"].map(ranks)
        df["squad_depth_rank_a"] = df["squad_quality_rank_a"]
        df["squad_depth_rank_b"] = df["squad_quality_rank_b"]
        for col in ["star_power_rank", "star_concentration", "attack_value_rank", "midfield_value_rank", "defense_value_rank"]:
            df[f"{col}_a"] = float("nan")
            df[f"{col}_b"] = float("nan")
        df["club_minutes_rank_a"] = float("nan")
        df["club_minutes_rank_b"] = float("nan")
        df["club_form_rank_a"] = float("nan")
        df["club_form_rank_b"] = float("nan")
        df["gk_rating_a"] = float("nan")
        df["gk_rating_b"] = float("nan")
    else:
        for col in squad_cols:
            if f"{col}_a" not in df:
                df[f"{col}_a"] = float("nan")
                df[f"{col}_b"] = float("nan")
        df["club_minutes_rank_a"] = float("nan")
        df["club_minutes_rank_b"] = float("nan")
        df["club_form_rank_a"] = float("nan")
        df["club_form_rank_b"] = float("nan")
        df["gk_rating_a"] = float("nan")
        df["gk_rating_b"] = float("nan")

    (
        df["form_a"],
        df["form_b"],
        df["rest_days_a"],
        df["rest_days_b"],
        df["travel_km_a"],
        df["travel_km_b"],
    ) = _team_appearance_features(df, form_window)

    return df[FEATURE_COLS].copy()
