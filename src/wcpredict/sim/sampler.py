"""
sampler.py
The seam between the match model and the simulator (see docs/arch.md §Seam).

Contract (ai_rules.md #6-9):
  1. Sample, never average — draw a real scoreline every call.
  2. Preserve goal dependence — sample from the joint DC-corrected grid.
  3. ET / shootout handoff — 90' draw → ET at scaled λ → shootout.
  4. Channel-2 dispersion — sampler draws λ from a mixture when lineup
     uncertainty is non-zero (plugs in here cleanly).

Public API
----------
sample_match(dist, rng, settings, *, et_allowed, adj) -> MatchResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import functools

import numpy as np
from scipy.stats import poisson

from wcpredict.model.match_model import GoalsDist


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    goals_home: int
    goals_away: int
    et_goals_home: int = 0
    et_goals_away: int = 0
    penalties_home: int | None = None
    penalties_away: int | None = None
    went_to_et: bool = False
    went_to_shootout: bool = False
    winner: str = ""   # "home" | "away" | "" (draw, group-stage only)


# ---------------------------------------------------------------------------
# Core scoreline sampler (honours joint DC-corrected distribution)
# ---------------------------------------------------------------------------

def _build_grid(
    dist: GoalsDist,
    max_goals: int = 10,
) -> np.ndarray:
    """
    Build the (max_goals+1) x (max_goals+1) joint probability matrix.
    DC tau correction is applied to the low-score cells (x,y) in {0,1}^2.
    """
    la, lb, rho = dist.lambda_a, dist.lambda_b, dist.dependence

    # Independent Poisson probabilities (vectorised — no Python-level factorial loop)
    xs = np.arange(max_goals + 1)
    pa = poisson.pmf(xs, la)
    pb = poisson.pmf(xs, lb)

    # Joint grid
    grid = np.outer(pa, pb)

    # Apply Dixon-Coles correction to {0,1}x{0,1}
    _dc = {
        (0, 0): 1.0 - la * lb * rho,
        (1, 0): 1.0 + lb * rho,
        (0, 1): 1.0 + la * rho,
        (1, 1): 1.0 - rho,
    }
    for (i, j), tau in _dc.items():
        if i <= max_goals and j <= max_goals:
            grid[i, j] *= max(tau, 1e-10)

    grid = np.maximum(grid, 0.0)
    total = grid.sum()
    if total > 0:
        grid /= total
    else:
        grid[:] = 1.0 / grid.size

    return grid


@functools.lru_cache(maxsize=512)
def _flat_grid_cached(
    la: float, lb: float, rho: float, max_goals: int
) -> np.ndarray:
    """
    LRU-cached flat probability array for (la, lb, rho, max_goals).
    Called ~5M times per 50k simulation run but produces at most a handful
    of distinct grids (one per unique context bucket).  The cache ensures
    _build_grid is called only once per unique GoalsDist, not per match.
    Returns a read-only C-contiguous array — never mutate it.
    """
    dist = GoalsDist(lambda_a=la, lambda_b=lb, dependence=rho)
    flat = _build_grid(dist, max_goals).ravel()
    flat.flags.writeable = False
    return flat


def _vectorized_draw(
    grids: np.ndarray,
    rng: np.random.Generator,
    n_goals: int = 10,
) -> np.ndarray:
    """
    Vectorized inverse-CDF categorical draw across N matches simultaneously.
    Each row of `grids` is sampled from its OWN distribution — not a shared
    probability vector.  Replaces N sequential rng.choice(121) calls with one
    numpy operation.

    Parameters
    ----------
    grids   : (N, (n_goals+1)**2) — each row is a flat normalised joint PMF
    rng     : seeded Generator (caller owns seed)
    n_goals : max goals per team (must match grid width dimension)

    Returns
    -------
    (N, 2) int32 array — column 0 = home_goals, column 1 = away_goals
    """
    n = grids.shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=np.int32)
    u = rng.random(n)                               # (N,) uniform [0,1)
    cumsum = np.cumsum(grids, axis=1)               # (N, cells) cumulative probs
    # count of positions where cumsum < u  →  first position where cumsum ≥ u
    indices = (u[:, None] > cumsum).sum(axis=1)     # (N,) vectorised inverse CDF
    indices = np.minimum(indices, grids.shape[1] - 1)   # guard against fp rounding
    home_goals = (indices // (n_goals + 1)).astype(np.int32)
    away_goals = (indices  % (n_goals + 1)).astype(np.int32)
    return np.stack([home_goals, away_goals], axis=1)


def _sample_scoreline(
    dist: GoalsDist,
    rng: np.random.Generator,
    max_goals: int = 10,
) -> tuple[int, int]:
    """
    Draw one scoreline from the joint DC-corrected distribution.
    Never samples marginals independently.
    Uses the LRU-cached flat grid to avoid rebuilding on every call.
    """
    flat = _flat_grid_cached(dist.lambda_a, dist.lambda_b, dist.dependence, max_goals)
    idx = rng.choice(len(flat), p=flat)
    i, j = divmod(idx, max_goals + 1)
    return int(i), int(j)


# ---------------------------------------------------------------------------
# Channel-2 dispersion (lineup uncertainty)
# ---------------------------------------------------------------------------

def _draw_lambdas(
    dist: GoalsDist,
    dispersion: float,
    rng: np.random.Generator,
) -> GoalsDist:
    """
    If dispersion > 0, draw λ from a small mixture centred on the base
    estimate.  High certainty (dispersion→0) collapses to a point.
    Implements arch.md channel-2 logic inside the sampler loop.
    """
    if dispersion <= 0.0:
        return dist
    sigma = dispersion * 0.5   # scale dispersion to a log-normal std
    # Bias-corrected draws: E[exp(N(-σ²/2, σ))] = 1 exactly → mean-preserving
    la = dist.lambda_a * float(np.exp(rng.normal(-sigma**2 / 2.0, sigma)))
    lb = dist.lambda_b * float(np.exp(rng.normal(-sigma**2 / 2.0, sigma)))
    la = max(la, 1e-4)
    lb = max(lb, 1e-4)
    return GoalsDist(lambda_a=la, lambda_b=lb, dependence=dist.dependence)


# ---------------------------------------------------------------------------
# Shootout resolver
# ---------------------------------------------------------------------------

def _resolve_shootout(
    strength_diff: float,
    rng: np.random.Generator,
    coef: float = 0.10,
) -> str:
    """
    Resolve a penalty shootout.  Near-coinflip with a small shrunk lean.
    Returns "home" or "away".
    """
    p_home = float(np.clip(0.5 + coef * strength_diff, 0.05, 0.95))
    return "home" if rng.random() < p_home else "away"


def _sample_shootout_score(
    winner: str,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """
    Sample a plausible displayed penalty score.

    This is presentation/state detail, not a model feature: the shootout winner
    is decided by _resolve_shootout(), then this generates a scoreline consistent
    with that winner.
    """
    losing_score = int(rng.choice([3, 4, 5], p=[0.18, 0.58, 0.24]))
    margin = int(rng.choice([1, 2], p=[0.88, 0.12]))
    winning_score = losing_score + margin
    if winner == "home":
        return winning_score, losing_score
    return losing_score, winning_score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class MatchAdjustment:
    """Pre-match adjustment from the LLM/adjustment layer."""
    lambda_delta_home: float = 0.0   # channel 1: signed delta on home λ
    lambda_delta_away: float = 0.0   # channel 1: signed delta on away λ
    dispersion: float = 0.0          # channel 2: lineup-uncertainty term


def sample_match(
    dist: GoalsDist,
    rng: np.random.Generator,
    settings: dict[str, Any],
    *,
    et_allowed: bool = True,
    adj: MatchAdjustment | None = None,
    strength_diff: float = 0.0,
) -> MatchResult:
    """
    Sample a full match result, including ET and shootout if applicable.

    Parameters
    ----------
    dist          : GoalsDist from match_model.predict()
    rng           : seeded numpy Generator (caller owns the seed)
    settings      : parsed config/settings.yaml dict
    et_allowed    : False in group stage (draws stand); True in knockouts
    adj           : optional pre-match adjustment (channels 1 & 2)
    strength_diff : normalised strength gap, used by shootout sub-model
    """
    sim_cfg = settings.get("simulation", {})
    et_scale: float = float(sim_cfg.get("et_lambda_scale", 0.33))
    so_coef: float  = float(sim_cfg.get("shootout_strength_coef", 0.10))
    max_goals: int  = int(settings.get("match_model", {}).get("max_goals", 10))

    # Apply channel-1 (availability) deltas
    la = dist.lambda_a
    lb = dist.lambda_b
    if adj is not None:
        la = max(la + adj.lambda_delta_home, 1e-4)
        lb = max(lb + adj.lambda_delta_away, 1e-4)

    working_dist = GoalsDist(lambda_a=la, lambda_b=lb, dependence=dist.dependence)

    # Channel-2 (lineup uncertainty): draw λ from mixture if dispersion > 0
    dispersion = adj.dispersion if adj is not None else 0.0
    working_dist = _draw_lambdas(working_dist, dispersion, rng)

    # 90-minute scoreline
    g_home, g_away = _sample_scoreline(working_dist, rng, max_goals)

    result = MatchResult(goals_home=g_home, goals_away=g_away)

    if g_home != g_away or not et_allowed:
        result.winner = "home" if g_home > g_away else (
            "away" if g_away > g_home else ""
        )
        return result

    # Extra time (draws in knockout stage)
    result.went_to_et = True
    et_dist = GoalsDist(
        lambda_a=working_dist.lambda_a * et_scale,
        lambda_b=working_dist.lambda_b * et_scale,
        dependence=working_dist.dependence,
    )
    et_home, et_away = _sample_scoreline(et_dist, rng, max_goals)
    result.et_goals_home = et_home
    result.et_goals_away = et_away

    total_home = g_home + et_home
    total_away = g_away + et_away

    if total_home != total_away:
        result.winner = "home" if total_home > total_away else "away"
        return result

    # Penalty shootout
    result.went_to_shootout = True
    result.winner = _resolve_shootout(strength_diff, rng, coef=so_coef)
    result.penalties_home, result.penalties_away = _sample_shootout_score(result.winner, rng)
    return result
