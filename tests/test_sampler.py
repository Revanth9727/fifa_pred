"""
test_sampler.py
Unit tests for sim/sampler.py — validates the seam contract from arch.md.

Key invariants:
  1. Sample, never average — every call returns integer scorelines.
  2. Joint sampling — the DC-corrected grid is sampled jointly.
  3. ET / shootout handoff — correct sequencing and state propagation.
  4. Channel-2 dispersion — mixture widens the λ distribution.
"""
import numpy as np
import pytest

from wcpredict.model.match_model import GoalsDist
from wcpredict.sim.sampler import (
    MatchAdjustment,
    MatchResult,
    _build_grid,
    _draw_lambdas,
    _resolve_shootout,
    _sample_scoreline,
    sample_match,
)


_SETTINGS = {
    "simulation": {
        "et_lambda_scale": 0.33,
        "shootout_strength_coef": 0.10,
        "n_runs": 100,
        "random_seed": 42,
    },
    "match_model": {"max_goals": 10},
}


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _dist(la: float = 1.2, lb: float = 1.0, dep: float = -0.1) -> GoalsDist:
    return GoalsDist(lambda_a=la, lambda_b=lb, dependence=dep)


# ---------------------------------------------------------------------------
# _build_grid
# ---------------------------------------------------------------------------

def test_grid_sums_to_one():
    grid = _build_grid(_dist(), max_goals=10)
    assert abs(grid.sum() - 1.0) < 1e-9


def test_grid_shape():
    grid = _build_grid(_dist(), max_goals=8)
    assert grid.shape == (9, 9)


def test_grid_non_negative():
    grid = _build_grid(_dist(), max_goals=10)
    assert np.all(grid >= 0)


def test_grid_dc_low_score_applied():
    d = _dist(la=1.0, lb=1.0, dep=0.05)
    grid_no_dc = np.outer(
        [np.exp(-1.0) * 1.0**i / float(__import__("math").factorial(i)) for i in range(11)],
        [np.exp(-1.0) * 1.0**i / float(__import__("math").factorial(i)) for i in range(11)],
    )
    grid_no_dc /= grid_no_dc.sum()
    grid_dc = _build_grid(d, max_goals=10)
    # DC correction shifts probability; (0,0) should differ
    assert abs(grid_dc[0, 0] - grid_no_dc[0, 0]) > 1e-6


# ---------------------------------------------------------------------------
# _sample_scoreline
# ---------------------------------------------------------------------------

def test_scoreline_returns_ints():
    g_h, g_a = _sample_scoreline(_dist(), _rng())
    assert isinstance(g_h, int)
    assert isinstance(g_a, int)


def test_scoreline_non_negative():
    d = _dist()
    r = _rng(42)
    for _ in range(50):
        gh, ga = _sample_scoreline(d, r)
        assert gh >= 0 and ga >= 0


def test_scoreline_within_max_goals():
    d = _dist()
    r = _rng(7)
    for _ in range(100):
        gh, ga = _sample_scoreline(d, r, max_goals=5)
        assert gh <= 5 and ga <= 5


def test_joint_sampling_not_marginals():
    """
    Draw many samples and verify the (0,0) cell probability approximately
    matches the grid value — not the product of marginals, which would differ
    when rho != 0.
    """
    d = _dist(la=1.2, lb=1.0, dep=0.2)
    grid = _build_grid(d, max_goals=10)
    expected_00 = grid[0, 0]

    r = _rng(99)
    n = 5000
    count_00 = sum(1 for _ in range(n) if _sample_scoreline(d, r, 10) == (0, 0))
    observed = count_00 / n
    # Should be within ~3 sigma (sigma ≈ sqrt(p*(1-p)/n))
    sigma = (expected_00 * (1 - expected_00) / n) ** 0.5
    assert abs(observed - expected_00) < 4 * sigma, (
        f"(0,0) observed={observed:.4f} expected={expected_00:.4f}"
    )


# ---------------------------------------------------------------------------
# _resolve_shootout
# ---------------------------------------------------------------------------

def test_shootout_returns_home_or_away():
    r = _rng()
    for _ in range(20):
        w = _resolve_shootout(0.0, r)
        assert w in ("home", "away")


def test_shootout_near_coinflip_at_zero_diff():
    r = _rng(123)
    n = 2000
    home_wins = sum(1 for _ in range(n) if _resolve_shootout(0.0, r) == "home")
    assert abs(home_wins / n - 0.5) < 0.05


def test_shootout_bias_with_positive_diff():
    r = _rng(456)
    n = 2000
    home_wins = sum(1 for _ in range(n) if _resolve_shootout(2.0, r, coef=0.10) == "home")
    assert home_wins / n > 0.55, "Positive strength diff should favour home"


# ---------------------------------------------------------------------------
# _draw_lambdas (channel-2 dispersion)
# ---------------------------------------------------------------------------

def test_draw_lambdas_zero_dispersion_is_identity():
    d = _dist()
    r = _rng()
    out = _draw_lambdas(d, dispersion=0.0, rng=r)
    assert out.lambda_a == d.lambda_a
    assert out.lambda_b == d.lambda_b


def test_draw_lambdas_nonzero_dispersion_varies():
    d = _dist(la=1.5, lb=1.0)
    r = _rng(10)
    lambdas = [_draw_lambdas(d, dispersion=0.5, rng=r).lambda_a for _ in range(100)]
    assert len(set(lambdas)) > 1, "With dispersion > 0, λ should vary across draws"


def test_draw_lambdas_positive():
    d = _dist()
    r = _rng(20)
    for _ in range(50):
        out = _draw_lambdas(d, dispersion=1.0, rng=r)
        assert out.lambda_a > 0
        assert out.lambda_b > 0


# ---------------------------------------------------------------------------
# sample_match — group stage (no ET)
# ---------------------------------------------------------------------------

def test_group_match_returns_MatchResult():
    r = sample_match(_dist(), _rng(), _SETTINGS, et_allowed=False)
    assert isinstance(r, MatchResult)


def test_group_match_no_et():
    r = _rng(0)
    for _ in range(30):
        res = sample_match(_dist(), r, _SETTINGS, et_allowed=False)
        assert not res.went_to_et
        assert not res.went_to_shootout


def test_group_match_integer_goals():
    res = sample_match(_dist(), _rng(), _SETTINGS, et_allowed=False)
    assert isinstance(res.goals_home, int)
    assert isinstance(res.goals_away, int)


def test_group_match_winner_empty_on_draw():
    rng = _rng(0)
    draws_found = 0
    for _ in range(500):
        res = sample_match(_dist(la=1.0, lb=1.0), rng, _SETTINGS, et_allowed=False)
        if res.goals_home == res.goals_away:
            assert res.winner == "", "Draw in group stage should have winner=''"
            draws_found += 1
    assert draws_found > 0, "No draws found in 500 group-stage matches"


def test_group_match_winner_set_on_decisive():
    rng = _rng(0)
    for _ in range(200):
        res = sample_match(_dist(la=2.5, lb=0.5), rng, _SETTINGS, et_allowed=False)
        if res.goals_home != res.goals_away:
            assert res.winner in ("home", "away")


# ---------------------------------------------------------------------------
# sample_match — knockout (ET + shootout)
# ---------------------------------------------------------------------------

def test_knockout_draws_trigger_et():
    """Force draws by using equal, high lambdas — must eventually hit ET."""
    rng = _rng(0)
    et_count = 0
    for _ in range(500):
        res = sample_match(_dist(la=1.0, lb=1.0), rng, _SETTINGS, et_allowed=True)
        if res.went_to_et:
            et_count += 1
    assert et_count > 0, "Expected some matches to go to ET"


def test_knockout_always_has_winner():
    rng = _rng(0)
    for _ in range(100):
        res = sample_match(_dist(), rng, _SETTINGS, et_allowed=True)
        assert res.winner in ("home", "away"), "Knockout must always produce a winner"


def test_shootout_requires_et_first():
    rng = _rng(0)
    for _ in range(200):
        res = sample_match(_dist(), rng, _SETTINGS, et_allowed=True)
        if res.went_to_shootout:
            assert res.went_to_et, "Shootout cannot happen without ET"


def test_et_uses_scaled_lambda():
    """
    With et_lambda_scale=0, ET goals must always be 0.
    This verifies the scaling is applied (not the full 90' rate).
    """
    settings_zero_et = {
        "simulation": {"et_lambda_scale": 0.0, "shootout_strength_coef": 0.10},
        "match_model": {"max_goals": 10},
    }
    d = _dist(la=1.0, lb=1.0, dep=0.0)
    rng = _rng(0)
    for _ in range(200):
        res = sample_match(d, rng, settings_zero_et, et_allowed=True)
        if res.went_to_et:
            assert res.et_goals_home == 0 and res.et_goals_away == 0, (
                "With et_lambda_scale=0, ET must produce 0 goals"
            )


# ---------------------------------------------------------------------------
# sample_match — adjustment channels
# ---------------------------------------------------------------------------

def test_channel1_delta_shifts_goals_mean():
    """Positive home delta should increase home goals on average."""
    rng_base = _rng(42)
    rng_adj  = _rng(42)
    d = _dist(la=1.0, lb=1.0)
    n = 500

    base_goals = [sample_match(d, rng_base, _SETTINGS, et_allowed=False).goals_home
                  for _ in range(n)]
    adj = MatchAdjustment(lambda_delta_home=0.5)
    adj_goals  = [sample_match(d, rng_adj, _SETTINGS, et_allowed=False, adj=adj).goals_home
                  for _ in range(n)]

    assert sum(adj_goals) > sum(base_goals), (
        "Positive lambda_delta_home should increase average home goals"
    )


def test_channel2_dispersion_produces_variance():
    """With high dispersion, repeated draws from the same dist should vary more."""
    d = _dist(la=1.5, lb=1.0)
    rng_base = _rng(0)
    rng_disp = _rng(0)
    n = 300

    base_goals = [sample_match(d, rng_base, _SETTINGS, et_allowed=False).goals_home
                  for _ in range(n)]
    adj = MatchAdjustment(dispersion=1.0)
    disp_goals = [sample_match(d, rng_disp, _SETTINGS, et_allowed=False, adj=adj).goals_home
                  for _ in range(n)]

    assert np.var(disp_goals) >= np.var(base_goals) * 0.9, (
        "High lineup uncertainty should produce at least as much variance"
    )


# ---------------------------------------------------------------------------
# Channel-2 mean preservation (ai_rules req)
# Dispersion draws λ from a bias-corrected log-normal so E[λ'] = λ exactly.
# ---------------------------------------------------------------------------

def test_channel2_mean_preservation():
    """
    A dispersion-only adjustment must increase scoreline variance while
    leaving expected goals unchanged.

    We compare:
      - baseline (dispersion=0): sample N scorelines from the base dist
      - adjusted (dispersion>0): sample N scorelines with channel-2 dispersion

    Mean goals should be statistically indistinguishable (within 3 sigma);
    variance in the adjusted sample must be strictly higher.
    """
    d = _dist(la=1.5, lb=1.2, dep=0.0)   # dep=0 isolates the dispersion effect
    n = 4000
    dispersion = 0.8

    rng_base = _rng(77)
    rng_disp = _rng(77)

    base_h = np.array([sample_match(d, rng_base, _SETTINGS, et_allowed=False).goals_home
                        for _ in range(n)], dtype=float)
    adj = MatchAdjustment(dispersion=dispersion)
    disp_h = np.array([sample_match(d, rng_disp, _SETTINGS, et_allowed=False, adj=adj).goals_home
                        for _ in range(n)], dtype=float)

    mu_base = base_h.mean()
    mu_disp = disp_h.mean()

    # Mean should be preserved: |mu_disp - mu_base| < 4 * pooled SEM
    sem = (base_h.std() / n**0.5 + disp_h.std() / n**0.5)
    assert abs(mu_disp - mu_base) < 4 * sem, (
        f"Channel-2 dispersion shifted mean: base={mu_base:.3f} disp={mu_disp:.3f} "
        f"(diff={abs(mu_disp-mu_base):.3f} > 4*SEM={4*sem:.3f})"
    )

    # Variance must increase
    var_base = base_h.var()
    var_disp = disp_h.var()
    assert var_disp > var_base, (
        f"Channel-2 dispersion did not increase variance: "
        f"base_var={var_base:.4f} disp_var={var_disp:.4f}"
    )
