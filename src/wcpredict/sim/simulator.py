"""
simulator.py
Monte Carlo tournament simulator for the 2026 FIFA World Cup.

Structure (arch.md):
  12 groups of 4 → 32 advance (24 auto + 8 best-third) → single-elimination
  R32 → R16 → QF → SF → Final; 3rd-place playoff.

Loop per run (50k default):
  1. Play group matches via sampler.sample_match()
  2. Apply group tiebreakers → standings
  3. Rank third-placed teams → select best 8
  4. Build R32 via bracket.build_r32()
  5. Resolve knockout rounds → record stage reached per team

Outputs: dict mapping team_id -> {stage: count} for each stage.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import functools

import numpy as np
import yaml

from wcpredict.model.match_model import GoalsDist, MatchModel
from wcpredict.sim.bracket import build_r32, Matchup
from wcpredict.sim.sampler import (
    sample_match, MatchAdjustment, MatchResult,
    _flat_grid_cached, _build_grid, _vectorized_draw,
)


# ---------------------------------------------------------------------------
# Default team strengths (ELO proxy — overridden by real table in Phase 7)
# ---------------------------------------------------------------------------

_DEFAULT_STRENGTHS: dict[str, float] = {
    # Group A
    "Mexico": 1700, "South Africa": 1450, "South Korea": 1640, "Czechia": 1550,
    # Group B
    "Canada": 1620, "Bosnia and Herzegovina": 1500, "Qatar": 1440, "Switzerland": 1720,
    # Group C
    "Brazil": 1920, "Morocco": 1730, "Haiti": 1360, "Scotland": 1610,
    # Group D
    "United States": 1650, "Paraguay": 1540, "Australia": 1600, "Türkiye": 1600,
    # Group E
    "Germany": 1850, "Curaçao": 1310, "Côte d'Ivoire": 1600, "Ecuador": 1650,
    # Group F
    "Netherlands": 1830, "Japan": 1680, "Sweden": 1650, "Tunisia": 1570,
    # Group G
    "Belgium": 1800, "Egypt": 1560, "Iran": 1530, "New Zealand": 1360,
    # Group H
    "Spain": 1910, "Cabo Verde": 1400, "Saudi Arabia": 1510, "Uruguay": 1770,
    # Group I
    "France": 1920, "Senegal": 1680, "Iraq": 1420, "Norway": 1660,
    # Group J
    "Argentina": 1960, "Algeria": 1550, "Austria": 1660, "Jordan": 1380,
    # Group K
    "Portugal": 1840, "DR Congo": 1490, "Uzbekistan": 1410, "Colombia": 1770,
    # Group L
    "England": 1880, "Croatia": 1710, "Ghana": 1540, "Panama": 1490,
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[4]


def _load_yaml(name: str) -> dict:
    with open(_project_root() / "config" / name) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tiebreaker helpers
# ---------------------------------------------------------------------------

_TB_KEYS = ["points", "goal_difference", "goals_scored", "team_conduct_score", "fifa_ranking"]


def _team_stats(
    team: str,
    matches: list[tuple[str, str, int, int]],
) -> dict[str, float]:
    """
    Compute group-stage stats for tiebreaking.
    matches: list of (home, away, g_home, g_away) tuples.
    """
    pts = gd = gs = gc = 0
    for h, a, gh, ga in matches:
        if h == team:
            gs += gh; gc += ga; gd += gh - ga
            pts += 3 if gh > ga else (1 if gh == ga else 0)
        elif a == team:
            gs += ga; gc += gh; gd += ga - gh
            pts += 3 if ga > gh else (1 if ga == gh else 0)
    return {"points": pts, "goal_difference": gd, "goals_scored": gs}


def _rank_group(
    teams: list[str],
    matches: list[tuple[str, str, int, int]],
    strengths: dict[str, float],
) -> list[str]:
    """
    Rank 4 teams in a group by tiebreaker order.
    team_conduct_score and fifa_ranking use strength as a proxy
    (lower strength rank = higher numeric value → worse rank).
    """
    stats = {t: _team_stats(t, matches) for t in teams}

    def sort_key(team: str):
        s = stats[team]
        return (
            -s["points"],
            -s["goal_difference"],
            -s["goals_scored"],
            strengths.get(team, 0.5),   # proxy for conduct/fifa (higher elo = lower rank value)
        )

    return sorted(teams, key=sort_key)


def _rank_third_placed(
    third_teams: list[tuple[str, str, dict]],  # (team, group, stats)
    strengths: dict[str, float],
) -> list[tuple[str, str]]:
    """
    Rank the 12 third-placed teams; return the top 8 as list of (team, group).
    """
    def sort_key(entry):
        team, group, s = entry
        return (
            -s["points"],
            -s["goal_difference"],
            -s["goals_scored"],
            strengths.get(team, 0.5),
        )

    ranked = sorted(third_teams, key=sort_key)
    return [(t, g) for t, g, _ in ranked[:8]]


# ---------------------------------------------------------------------------
# Match-playing helpers
# ---------------------------------------------------------------------------

def _play_group_stage(
    group_letter: str,
    teams: list[str],
    model: MatchModel,
    rng: np.random.Generator,
    settings: dict,
    adjustments: dict[tuple[str, str], MatchAdjustment],
    predict_cache: dict,
    strengths: dict,
    realized_results: dict[tuple[str, str], tuple[int, int]] | None = None,
    team_features: dict[str, dict[str, float]] | None = None,
) -> tuple[list[str], list[tuple[str, str, int, int]]]:
    """
    Play a single round-robin group using vectorized batch sampling.
    All 6 fixtures are sampled in one numpy operation.
    """
    max_goals = int(settings.get("match_model", {}).get("max_goals", 10))
    fixtures: list[tuple[str, str]] = [
        (teams[0], teams[1]), (teams[0], teams[2]), (teams[0], teams[3]),
        (teams[1], teams[2]), (teams[1], teams[3]), (teams[2], teams[3]),
    ]
    records: list[tuple[str, str, int, int] | None] = [None] * len(fixtures)
    open_fixtures: list[tuple[str, str]] = []
    open_indices: list[int] = []
    fixed = realized_results or {}
    for idx, (home, away) in enumerate(fixtures):
        if (home, away) in fixed:
            gh, ga = fixed[(home, away)]
            records[idx] = (home, away, int(gh), int(ga))
        elif (away, home) in fixed:
            ga, gh = fixed[(away, home)]
            records[idx] = (home, away, int(gh), int(ga))
        else:
            open_fixtures.append((home, away))
            open_indices.append(idx)

    if open_fixtures:
        grids, _ = _build_grids_for_matchups(
            open_fixtures, model, predict_cache, settings, strengths,
            neutral=True, adjustments=adjustments, rng=rng,
            team_features=team_features,
        )
        goals = _vectorized_draw(grids, rng, max_goals)
        for draw_idx, fixture_idx in enumerate(open_indices):
            home, away = fixtures[fixture_idx]
            records[fixture_idx] = (
                home,
                away,
                int(goals[draw_idx, 0]),
                int(goals[draw_idx, 1]),
            )

    played_records = [record for record in records if record is not None]
    ranked = _rank_group(teams, played_records, strengths)
    return ranked, played_records


def _ko_round_vectorized(
    matchups: list[tuple[str, str]],
    model: MatchModel,
    rng: np.random.Generator,
    settings: dict,
    predict_cache: dict,
    strengths: dict,
    adjustments: dict[tuple[str, str], MatchAdjustment],
    team_features: dict[str, dict[str, float]] | None = None,
    locked_ko_winners: dict[tuple[str, str], str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Play all matches in one knockout round via vectorized sampling.
    90' draws → ET (vectorized) → shootout (scalar, rare).
    Returns (winners, losers) in matchup order.
    locked_ko_winners maps (home, away) → actual winner name; those matches
    are used as-is (handles pens correctly) instead of being re-simulated.
    """
    if not matchups:
        return [], []

    fixed = locked_ko_winners or {}

    # Partition into locked (real result) vs open (to simulate)
    open_matchups: list[tuple[str, str]] = []
    open_indices: list[int] = []
    pre_winners: dict[int, str] = {}
    pre_losers: dict[int, str] = {}
    pre_records: dict[int, tuple[str, str, int, int]] = {}
    for idx, (home, away) in enumerate(matchups):
        w: str | None = fixed.get((home, away)) or fixed.get((away, home))
        if w is not None:
            l = away if w == home else home
            pre_winners[idx] = w; pre_losers[idx] = l
            pre_records[idx] = (home, away, 1, 0) if w == home else (home, away, 0, 1)
        else:
            open_matchups.append((home, away))
            open_indices.append(idx)

    if not open_matchups:
        winners = [pre_winners[i] for i in range(len(matchups))]
        losers  = [pre_losers[i]  for i in range(len(matchups))]
        records = [pre_records[i] for i in range(len(matchups))]
        return winners, losers, records

    max_goals = int(settings.get("match_model", {}).get("max_goals", 10))
    et_scale  = float(settings.get("simulation", {}).get("et_lambda_scale", 0.33))
    so_coef   = float(settings.get("simulation", {}).get("shootout_strength_coef", 0.10))
    n = len(open_matchups)

    # --- 90 minutes (open matchups only) ---
    grids_90, dists = _build_grids_for_matchups(
        open_matchups, model, predict_cache, settings, strengths,
        neutral=True, adjustments=adjustments, rng=rng,
        team_features=team_features,
    )
    goals_90 = _vectorized_draw(grids_90, rng, max_goals)      # (n, 2)
    total    = goals_90.astype(np.int32).copy()

    # --- Extra time for drawn matches ---
    draw_idx = np.where(goals_90[:, 0] == goals_90[:, 1])[0]
    if len(draw_idx):
        et_grids = np.array([
            _flat_grid_cached(
                float(dists[i].lambda_a * et_scale),
                float(dists[i].lambda_b * et_scale),
                float(dists[i].dependence),
                max_goals,
            )
            for i in draw_idx
        ], dtype=np.float64)
        et_goals = _vectorized_draw(et_grids, rng, max_goals)  # (draws, 2)
        total[draw_idx] += et_goals

        # --- Penalty shootout for still-drawn after ET ---
        for j, i in enumerate(draw_idx):
            if total[i, 0] == total[i, 1]:
                home, away = open_matchups[i]
                sd = (strengths.get(home, 1600.0) - strengths.get(away, 1600.0)) / 400.0
                p_home = float(np.clip(0.5 + so_coef * sd, 0.05, 0.95))
                if rng.random() < p_home:
                    total[i, 0] += 1
                else:
                    total[i, 1] += 1

    sim_winners: list[str] = []
    sim_losers:  list[str] = []
    sim_records: list[tuple[str, str, int, int]] = []
    for i, (home, away) in enumerate(open_matchups):
        if total[i, 0] >= total[i, 1]:
            sim_winners.append(home); sim_losers.append(away)
        else:
            sim_winners.append(away); sim_losers.append(home)
        sim_records.append((home, away, int(total[i, 0]), int(total[i, 1])))

    # Merge locked and simulated results back into original matchup order
    winners: list[str] = []
    losers:  list[str] = []
    records: list[tuple[str, str, int, int]] = []
    sim_ptr = 0
    for idx in range(len(matchups)):
        if idx in pre_winners:
            winners.append(pre_winners[idx])
            losers.append(pre_losers[idx])
            records.append(pre_records[idx])
        else:
            winners.append(sim_winners[sim_ptr])
            losers.append(sim_losers[sim_ptr])
            records.append(sim_records[sim_ptr])
            sim_ptr += 1
    return winners, losers, records


def _build_context(
    home: str,
    away: str,
    neutral: bool,
    settings: dict,
    model: MatchModel,
    *,
    elo_diff: float = 0.0,
    rest_days_home: float = 0.0,
    rest_days_away: float = 0.0,
    travel_km_home: float = 0.0,
    travel_km_away: float = 0.0,
    team_features: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """
    Build a context dict for model.predict().
    elo_diff is team-strength-aware (strengths[home] - strengths[away]).
    Path-dependent fields (rest_days, travel_km) are recomputed per knockout
    path rather than precomputed once outside the loop.
    """
    tf = team_features or {}
    home_f = tf.get(home, {})
    away_f = tf.get(away, {})
    return {
        "team_a": home,
        "team_b": away,
        "neutral_flag": 1.0 if neutral else 0.0,
        "host_flag": 0.0,
        "elo_diff": elo_diff,
        "form_a": home_f.get("national_form", 0.5),
        "form_b": away_f.get("national_form", 0.5),
        "squad_quality_rank_a": home_f.get("squad_quality_rank", 0.5),
        "squad_quality_rank_b": away_f.get("squad_quality_rank", 0.5),
        "squad_depth_rank_a": home_f.get("squad_depth_rank", 0.5),
        "squad_depth_rank_b": away_f.get("squad_depth_rank", 0.5),
        "star_power_rank_a": home_f.get("star_power_rank", 0.5),
        "star_power_rank_b": away_f.get("star_power_rank", 0.5),
        "star_concentration_a": home_f.get("star_concentration", 0.5),
        "star_concentration_b": away_f.get("star_concentration", 0.5),
        "attack_value_rank_a": home_f.get("attack_value_rank", 0.5),
        "attack_value_rank_b": away_f.get("attack_value_rank", 0.5),
        "midfield_value_rank_a": home_f.get("midfield_value_rank", 0.5),
        "midfield_value_rank_b": away_f.get("midfield_value_rank", 0.5),
        "defense_value_rank_a": home_f.get("defense_value_rank", 0.5),
        "defense_value_rank_b": away_f.get("defense_value_rank", 0.5),
        "club_minutes_rank_a": home_f.get("club_minutes_rank", 0.5),
        "club_minutes_rank_b": away_f.get("club_minutes_rank", 0.5),
        "club_form_rank_a": home_f.get("club_form_rank", 0.5),
        "club_form_rank_b": away_f.get("club_form_rank", 0.5),
        "gk_rating_a": home_f.get("gk_rating", 0.5),
        "gk_rating_b": away_f.get("gk_rating", 0.5),
        "rest_days_a": rest_days_home,
        "rest_days_b": rest_days_away,
        "travel_km_a": travel_km_home,
        "travel_km_b": travel_km_away,
    }


def _build_grids_for_matchups(
    matchups: list[tuple[str, str]],
    model: MatchModel,
    predict_cache: dict,
    settings: dict,
    strengths: dict,
    neutral: bool,
    adjustments: dict,
    rng: np.random.Generator,
    team_features: dict[str, dict[str, float]] | None = None,
) -> tuple[np.ndarray, list]:
    """
    Build an (N, 121) float64 grid matrix for vectorized batch sampling.
    - Unadjusted / channel-1 matches: use _flat_grid_cached (single build per unique λ).
    - Channel-2 (dispersion>0) matches: draw perturbed λ fresh each run — bypass cache.

    Returns (grids, dists) where dists[i] is the base GoalsDist before any adjustment.
    """
    max_goals = int(settings.get("match_model", {}).get("max_goals", 10))
    rows: list[np.ndarray] = []
    dists_out: list[GoalsDist] = []

    for home, away in matchups:
        ediff = strengths.get(home, 1600.0) - strengths.get(away, 1600.0)
        ctx = _build_context(
            home, away, neutral=neutral, settings=settings, model=model,
            elo_diff=ediff,
            team_features=team_features,
        )
        dist = _cached_predict(model, ctx, predict_cache)
        dists_out.append(dist)

        la, lb = dist.lambda_a, dist.lambda_b
        adj = adjustments.get((home, away))
        if adj is not None:
            la = max(la + adj.lambda_delta_home, 1e-4)
            lb = max(lb + adj.lambda_delta_away, 1e-4)

        disp = adj.dispersion if adj is not None else 0.0
        if disp > 0.0:
            sigma = disp * 0.5
            # Channel-2: bias-corrected log-normal draw — not cached (per-run)
            la = max(float(la) * float(np.exp(rng.normal(-sigma**2 / 2.0, sigma))), 1e-4)
            lb = max(float(lb) * float(np.exp(rng.normal(-sigma**2 / 2.0, sigma))), 1e-4)
            flat = _build_grid(
                GoalsDist(lambda_a=la, lambda_b=lb, dependence=dist.dependence), max_goals
            ).ravel()
        else:
            flat = _flat_grid_cached(float(la), float(lb), float(dist.dependence), max_goals)

        rows.append(flat)

    return np.array(rows, dtype=np.float64), dists_out


# ---------------------------------------------------------------------------
# In-tournament Elo update helpers
# ---------------------------------------------------------------------------

def _elo_delta(
    elo_a: float,
    elo_b: float,
    score_a: int,
    score_b: int,
    K: float = 40.0,
) -> float:
    """Standard Elo delta for team A given a match result."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    actual_a = 1.0 if score_a > score_b else (0.5 if score_a == score_b else 0.0)
    return K * (actual_a - expected_a)


def _update_strengths(
    strengths: dict[str, float],
    records: list[tuple[str, str, int, int]],
    K: float,
) -> None:
    """Update Elo strengths in-place from match records (home, away, g_home, g_away)."""
    for home, away, gh, ga in records:
        delta = _elo_delta(
            strengths.get(home, 1600.0),
            strengths.get(away, 1600.0),
            gh, ga, K,
        )
        strengths[home] = strengths.get(home, 1600.0) + delta
        strengths[away] = strengths.get(away, 1600.0) - delta


# ---------------------------------------------------------------------------
# GoalsDist predict cache
# Keyed on (context tuple, model id) so predictions are not rebuilt per match
# in the inner loop when the context is identical.
# ---------------------------------------------------------------------------

def _ctx_key(ctx: dict[str, float]) -> tuple:
    """Convert a context dict to a hashable tuple for caching."""
    return tuple(sorted(ctx.items()))


def _cached_predict(
    model: MatchModel,
    ctx: dict[str, float],
    cache: dict,
) -> "GoalsDist":
    key = (id(model), _ctx_key(ctx))
    if key not in cache:
        cache[key] = model.predict(ctx)
    return cache[key]


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

STAGES = [
    "group_exit",
    "round_of_32",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "third_place",
    "finalist",
    "champion",
]


class TournamentSimulator:
    """
    Monte Carlo simulator for the 2026 FIFA World Cup.

    Parameters
    ----------
    groups  : dict mapping group letter -> list of 4 team IDs
    model   : fitted MatchModel (used in predict() calls inside the loop)
    settings: parsed config/settings.yaml
    adjustments: optional dict mapping (home, away) -> MatchAdjustment
    """

    def __init__(
        self,
        groups: dict[str, list[str]],
        model: MatchModel,
        settings: dict[str, Any] | None = None,
        adjustments: dict[tuple[str, str], MatchAdjustment] | None = None,
        strengths: dict[str, float] | None = None,
        realized_results: dict[tuple[str, str], tuple[int, int]] | None = None,
        team_features: dict[str, dict[str, float]] | None = None,
        elo_k: float = 40.0,
        locked_ko_winners: dict[tuple[str, str], str] | None = None,
    ) -> None:
        if settings is None:
            settings = _load_yaml("settings.yaml")
        self.groups = groups
        self.model = model
        self.settings = settings
        self.adjustments: dict[tuple[str, str], MatchAdjustment] = adjustments or {}
        self.strengths: dict[str, float] = strengths if strengths is not None else _DEFAULT_STRENGTHS
        self.realized_results: dict[tuple[str, str], tuple[int, int]] = realized_results or {}
        self.team_features: dict[str, dict[str, float]] = team_features or {}
        self.elo_k: float = float(elo_k)
        self.locked_ko_winners: dict[tuple[str, str], str] = locked_ko_winners or {}

    def run(self, n_runs: int | None = None, seed: int | None = None) -> dict[str, dict[str, int]]:
        """
        Execute Monte Carlo simulation.

        Returns
        -------
        results : dict mapping team_id -> {stage: count}
            Where stage is one of STAGES and count is how many runs the team
            reached AT LEAST that stage.
        """
        sim_cfg = self.settings.get("simulation", {})
        n = int(n_runs if n_runs is not None else sim_cfg.get("n_runs", 25000))
        s = int(seed if seed is not None else sim_cfg.get("random_seed", 42))

        rng = np.random.default_rng(s)
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        use_param_uncertainty = getattr(self.model, "param_cov_", None) is not None
        # Group-stage predictions use pre-tournament Elo (identical across runs) → safe to share.
        shared_cache: dict = {}

        for _ in range(n):
            run_strengths = dict(self.strengths)  # fresh mutable copy — updated per run
            if use_param_uncertainty:
                # Each run draws fresh coefficients → predictions differ → no cross-run cache.
                self.model._perturbed_coef = self.model.sample_coef(rng)
                per_run_cache: dict = {}
            else:
                per_run_cache = shared_cache
            self._run_once(rng, counts, per_run_cache, run_strengths)

        if use_param_uncertainty:
            self.model._perturbed_coef = None  # restore clean state

        return {team: dict(stage_counts) for team, stage_counts in counts.items()}

    def _run_once(
        self,
        rng: np.random.Generator,
        counts: dict[str, dict[str, int]],
        predict_cache: dict,
        run_strengths: dict,
    ) -> None:
        # ---- Group stage (vectorized: 6 matches per group in one numpy op) ----
        standings: dict[str, list[str]] = {}
        third_records: list[tuple[str, str, dict]] = []
        all_group_records: list[tuple[str, str, int, int]] = []

        for grp, teams in self.groups.items():
            ranked, records = _play_group_stage(
                grp, teams, self.model, rng, self.settings,
                self.adjustments, predict_cache, run_strengths,
                self.realized_results,
                self.team_features,
            )
            standings[grp] = ranked
            all_group_records.extend(records)
            third = ranked[2]
            third_records.append((third, grp, _team_stats(third, records)))

        # ---- Update Elo from group stage before knockouts ----
        if self.elo_k > 0.0:
            _update_strengths(run_strengths, all_group_records, self.elo_k)
            # Knockout predictions use updated Elo → can't reuse the group-stage cache.
            ko_cache: dict = {}
        else:
            ko_cache = predict_cache

        # ---- Best-8 third-placed teams ----
        top8 = _rank_third_placed(third_records, run_strengths)
        qualifying_groups = {g for _, g in top8}

        # ---- Group exits ----
        for grp, ranked in standings.items():
            counts[ranked[3]]["group_exit"] += 1
            if grp not in qualifying_groups:
                counts[ranked[2]]["group_exit"] += 1

        # ---- R32 bracket ----
        r32 = build_r32(standings, qualifying_groups)
        current: list[tuple[str, str]] = [(m.home, m.away) for m in r32]

        # ---- Knockout rounds (vectorized per round, Elo updated between rounds) ----
        semi_losers: list[str] = []
        for round_name in ["round_of_32", "round_of_16", "quarter_final", "semi_final"]:
            winners, losers, ko_recs = _ko_round_vectorized(
                current, self.model, rng, self.settings,
                ko_cache, run_strengths, self.adjustments,
                self.team_features,
                self.locked_ko_winners,
            )
            if self.elo_k > 0.0:
                _update_strengths(run_strengths, ko_recs, self.elo_k)
            if round_name == "semi_final":
                semi_losers = losers
            else:
                for loser in losers:
                    counts[loser][round_name] += 1
            current = [(winners[i], winners[i + 1]) for i in range(0, len(winners), 2)]

        # ---- 3rd-place playoff ----
        if len(semi_losers) == 2:
            w3, l3, _ = _ko_round_vectorized(
                [(semi_losers[0], semi_losers[1])],
                self.model, rng, self.settings,
                ko_cache, run_strengths, self.adjustments,
                self.team_features,
            )
            counts[w3[0]]["third_place"] += 1
            counts[l3[0]]["semi_final"] += 1

        # ---- Final ----
        if current:
            finalists = list(current[0])
            for f in finalists:
                counts[f]["finalist"] += 1
            w_final, _, _ = _ko_round_vectorized(
                [tuple(finalists)],
                self.model, rng, self.settings,
                ko_cache, run_strengths, self.adjustments,
                self.team_features,
            )
            counts[w_final[0]]["champion"] += 1

    def _get_semi_losers(
        self,
        final_four: list[str],
        standings: dict[str, list[str]],
        qualifying_third_groups: set[str],
    ) -> list[str]:
        """
        Placeholder: returns the two teams that lost their semi-finals.
        In a full implementation, these would be tracked during the SF round.
        For now, returns empty list (3rd-place playoff is skipped in that case).
        """
        return []
