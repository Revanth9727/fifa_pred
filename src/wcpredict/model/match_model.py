"""
model/match_model.py — bivariate Poisson with Dixon-Coles low-score correction.

Trains exclusively on structured historical features (ai_rules.md #2).
Must NOT import llm/, adjust/, or any market module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson


@dataclass
class GoalsDist:
    """Goals-distribution parameters sufficient to sample a full scoreline."""
    lambda_a: float
    lambda_b: float
    dependence: float


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


def _load_settings(path: Path | str | None = None) -> dict[str, Any]:
    root = _project_root()
    p = Path(path) if path else root / "config" / "settings.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def _dc_tau(x: int, y: int, la: float, lb: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1.0 - la * lb * rho
    if x == 1 and y == 0:
        return 1.0 + lb * rho
    if x == 0 and y == 1:
        return 1.0 + la * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _dc_tau_vec(
    ya: np.ndarray, yb: np.ndarray,
    la: np.ndarray, lb: np.ndarray,
    rho: float,
) -> np.ndarray:
    tau = np.ones(len(ya))
    tau[(ya == 0) & (yb == 0)] = (1.0 - la * lb * rho)[(ya == 0) & (yb == 0)]
    tau[(ya == 1) & (yb == 0)] = (1.0 + lb * rho)[(ya == 1) & (yb == 0)]
    tau[(ya == 0) & (yb == 1)] = (1.0 + la * rho)[(ya == 0) & (yb == 1)]
    tau[(ya == 1) & (yb == 1)] = 1.0 - rho
    return np.maximum(tau, 1e-10)


def _build_team_map(
    home_teams: pd.Series | list,
    away_teams: pd.Series | list,
) -> dict[str, int]:
    """Assign a sorted integer index to every distinct team across home and away."""
    teams = sorted(set(home_teams) | set(away_teams))
    return {t: i for i, t in enumerate(teams)}


def _extract_beta_cov(result: Any, n_beta: int) -> np.ndarray | None:
    """
    Extract an approximate n_beta x n_beta covariance matrix for the feature
    coefficients from an L-BFGS-B OptimizeResult's hess_inv.

    Uses column-matvec to materialise only the top-left n_beta x n_beta block,
    symmetrises, and projects onto the PSD cone.  Returns None on any failure.
    """
    if result is None or not hasattr(result, "hess_inv") or result.hess_inv is None:
        return None
    try:
        H = result.hess_inv
        n_total = H.shape[0]
        cols: list[np.ndarray] = []
        e = np.zeros(n_total)
        for i in range(n_beta):
            e[:] = 0.0
            e[i] = 1.0
            cols.append(H.matvec(e)[:n_beta])
        cov = np.column_stack(cols)
        cov = (cov + cov.T) / 2.0  # symmetrise
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-12)  # project onto PSD cone
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    except Exception:
        return None


def _build_xy(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X_a, X_b) design matrices from a features DataFrame.

    Parameterisation (symmetric):
      log(lambda_a) = X_a @ beta
      log(lambda_b) = X_b @ beta
    where directional features are negated for team_b.

    Columns include Elo, form, squad depth, star power/concentration,
    position-group strength, rest/travel, GK, and host terms.
    """
    n = len(features)

    def _get(col: str, default: float = 0.0) -> np.ndarray:
        if col in features.columns:
            return pd.to_numeric(features[col], errors="coerce").fillna(default).to_numpy(float)
        return np.full(n, default)

    neutral = _get("neutral_flag", 0.0)
    home_adv = 1.0 - neutral

    elo_s = _get("elo_diff", 0.0) / 400.0
    attack_value_diff = _get("attack_value_rank_a", 0.5) - _get("attack_value_rank_b", 0.5)
    midfield_value_diff = _get("midfield_value_rank_a", 0.5) - _get("midfield_value_rank_b", 0.5)
    defense_value_diff = _get("defense_value_rank_a", 0.5) - _get("defense_value_rank_b", 0.5)
    host = _get("host_flag", 0.0)

    X_a = np.column_stack([
        np.ones(n), home_adv, elo_s,
        attack_value_diff, midfield_value_diff, defense_value_diff, host,
    ])
    X_b = np.column_stack([
        np.ones(n), -home_adv, -elo_s,
        -attack_value_diff, -midfield_value_diff, -defense_value_diff, host,
    ])
    return X_a, X_b


def _neg_log_lik(
    params: np.ndarray,
    X_a: np.ndarray,
    X_b: np.ndarray,
    ya: np.ndarray,
    yb: np.ndarray,
    weights: np.ndarray,
) -> float:
    beta, rho = params[:-1], float(params[-1])
    la = np.exp(X_a @ beta)
    lb = np.exp(X_b @ beta)
    ll_a = ya * np.log(np.maximum(la, 1e-12)) - la - gammaln(ya + 1)
    ll_b = yb * np.log(np.maximum(lb, 1e-12)) - lb - gammaln(yb + 1)
    tau = _dc_tau_vec(ya, yb, la, lb, rho)
    ll = ll_a + ll_b + np.log(tau)
    return -float(np.sum(weights * ll))


def _neg_log_lik_regularized(
    params: np.ndarray,
    X_a: np.ndarray,
    X_b: np.ndarray,
    ya: np.ndarray,
    yb: np.ndarray,
    weights: np.ndarray,
    ridge_alpha: float,
) -> float:
    """
    DC bivariate-Poisson log-likelihood + L2 (ridge) on non-intercept
    feature coefficients.  Tempers extrapolation when ELO differences are
    extreme without distorting the intercept or rho.
    """
    beta, rho = params[:-1], float(params[-1])
    la = np.exp(X_a @ beta)
    lb = np.exp(X_b @ beta)
    ll_a = ya * np.log(np.maximum(la, 1e-12)) - la - gammaln(ya + 1)
    ll_b = yb * np.log(np.maximum(lb, 1e-12)) - lb - gammaln(yb + 1)
    tau = _dc_tau_vec(ya, yb, la, lb, rho)
    ll = ll_a + ll_b + np.log(tau)
    neg_ll = -float(np.sum(weights * ll))
    reg = ridge_alpha * float(np.sum(beta[1:] ** 2))   # skip intercept at index 0
    return neg_ll + reg


def _neg_log_lik_with_teams(
    params: np.ndarray,
    X_a: np.ndarray,
    X_b: np.ndarray,
    ya: np.ndarray,
    yb: np.ndarray,
    weights: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    n_teams: int,
    n_beta: int,
    ridge_alpha: float,
    team_alpha: float,
) -> float:
    """
    DC bivariate-Poisson loss with:

    - Per-team attack / defence intercepts (partial pooling via L2 penalty
      with strength *team_alpha*).  The model is:
        log(lambda_a) = beta @ x_a + atk[home] - dfn[away]
        log(lambda_b) = beta @ x_b + atk[away] - dfn[home]

    - Ridge (L2) on non-intercept feature coefficients (*ridge_alpha*) to
      temper extrapolation at extreme ELO differences.
    """
    beta = params[:n_beta]
    atk  = params[n_beta:n_beta + n_teams]
    dfn  = params[n_beta + n_teams:n_beta + 2 * n_teams]
    rho  = float(params[-1])

    eta_a = X_a @ beta + atk[home_idx] - dfn[away_idx]
    eta_b = X_b @ beta + atk[away_idx] - dfn[home_idx]
    if (
        not np.all(np.isfinite(eta_a))
        or not np.all(np.isfinite(eta_b))
        or eta_a.max(initial=0.0) > 50.0
        or eta_b.max(initial=0.0) > 50.0
        or eta_a.min(initial=0.0) < -50.0
        or eta_b.min(initial=0.0) < -50.0
    ):
        return 1e100

    la = np.exp(eta_a)
    lb = np.exp(eta_b)

    ll_a = ya * np.log(np.maximum(la, 1e-12)) - la - gammaln(ya + 1)
    ll_b = yb * np.log(np.maximum(lb, 1e-12)) - lb - gammaln(yb + 1)
    tau  = _dc_tau_vec(ya, yb, la, lb, rho)
    ll   = ll_a + ll_b + np.log(tau)
    neg_ll = -float(np.sum(weights * ll))
    if not np.isfinite(neg_ll):
        return 1e100

    reg_feat = ridge_alpha * float(np.sum(beta[1:] ** 2))
    reg_team = team_alpha  * float(np.sum(atk ** 2) + np.sum(dfn ** 2))
    return neg_ll + reg_feat + reg_team


class MatchModel:
    """
    Bivariate Poisson match model with Dixon-Coles low-score correction.

    fit(matches, features)  — MLE on importance-weighted history.
    predict(context)        — returns GoalsDist for a matchup.
    derive_wdl(dist)        — W/D/L probabilities from the scoreline grid.

    Regularisation (Phase 2 enhancement)
    -------------------------------------
    ridge_alpha : float
        L2 penalty on non-intercept feature coefficients.  Tempers
        extrapolation at extreme ELO differences.
    team_alpha : float
        L2 partial-pooling penalty on per-team attack / defence intercepts.
        Only active when ``matches`` contains ``team_home`` / ``team_away``
        columns in ``fit()``.
    """

    N_PARAMS = 7   # intercept + home_adv + elo + attack_diff + midfield_diff + defense_diff + host

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        *,
        ridge_alpha: float = 0.01,
        team_alpha: float = 0.10,
        fixed_rho: float | None = -0.10,
    ) -> None:
        if settings is None:
            settings = _load_settings()
        mm = settings.get("match_model", {})
        self._max_goals: int = int(mm.get("max_goals", 10))
        self.ridge_alpha: float = float(ridge_alpha)
        self.team_alpha: float  = float(team_alpha)
        self.fixed_rho: float | None = fixed_rho
        self.coef_: np.ndarray | None = None
        self.rho_: float = 0.0
        self.team_map_: dict[str, int] | None = None
        self.team_atk_: np.ndarray | None = None
        self.team_dfn_: np.ndarray | None = None
        self.fit_result_: Any | None = None
        # Approximate covariance of beta (N_PARAMS x N_PARAMS) from L-BFGS-B hess_inv.
        # Set after fit(); used by sample_coef() for parameter-uncertainty simulation.
        self.param_cov_: np.ndarray | None = None
        # Scratch slot for the simulator to set a perturbed coef per run.
        # None means use self.coef_ (default, no perturbation).
        self._perturbed_coef: np.ndarray | None = None
        # Optional post-hoc WDL calibrators (callables: float -> float).
        self.cal_win_: Any | None = None
        self.cal_draw_: Any | None = None
        self.cal_loss_: Any | None = None
        self._fitted = False

    def fit(self, matches: pd.DataFrame, features: pd.DataFrame) -> "MatchModel":
        """
        Train by MLE on importance-weighted match history.

        Parameters
        ----------
        matches:
            Must include goals_home, goals_away, importance_weight.
            If ``team_home`` and ``team_away`` columns are present, per-team
            attack / defence intercepts are fitted with L2 partial pooling.
        features:
            Feature DataFrame aligned row-for-row with matches (from engineer.py).
        """
        ya = pd.to_numeric(matches["goals_home"], errors="coerce").fillna(0).to_numpy(float)
        yb = pd.to_numeric(matches["goals_away"], errors="coerce").fillna(0).to_numpy(float)
        weights = (
            pd.to_numeric(matches.get("importance_weight", 1.0), errors="coerce")
            .fillna(1.0)
            .to_numpy(float)
        )
        if "recency_weight" in matches.columns:
            recency_weights = (
                pd.to_numeric(matches["recency_weight"], errors="coerce")
                .fillna(1.0)
                .to_numpy(float)
            )
            weights = weights * recency_weights

        X_a, X_b = _build_xy(features)

        x0 = np.zeros(self.N_PARAMS + 1)
        x0[0] = np.log(1.35)   # intercept: average ~1.35 goals per side
        x0[1] = 0.20            # home advantage
        x0[2] = 0.50            # elo coefficient (starting guess)

        bounds = [(-5, 5)] * self.N_PARAMS + [(-0.5, 0.5)]  # rho bounded

        has_teams = (
            "team_home" in matches.columns
            and "team_away" in matches.columns
            and self.team_alpha > 0.0
        )

        if has_teams:
            self.team_map_ = _build_team_map(matches["team_home"], matches["team_away"])
            n_teams = len(self.team_map_)
            home_idx = matches["team_home"].map(self.team_map_).to_numpy(int)
            away_idx = matches["team_away"].map(self.team_map_).to_numpy(int)

            if self.fixed_rho is None:
                x0_ext = np.concatenate([
                    x0[:-1],
                    np.zeros(2 * n_teams),
                    [x0[-1]],
                ])
                bounds_ext = (
                    bounds[:-1]
                    + [(-2.0, 2.0)] * (2 * n_teams)
                    + [bounds[-1]]
                )
                objective = _neg_log_lik_with_teams
                args = (
                    X_a, X_b, ya, yb, weights,
                    home_idx, away_idx, n_teams, self.N_PARAMS,
                    self.ridge_alpha, self.team_alpha,
                )
            else:
                fixed_rho = float(self.fixed_rho)
                x0_ext = np.concatenate([
                    x0[:-1],
                    np.zeros(2 * n_teams),
                ])
                bounds_ext = bounds[:-1] + [(-2.0, 2.0)] * (2 * n_teams)

                def objective(params: np.ndarray, *args) -> float:
                    full_params = np.concatenate([params, [fixed_rho]])
                    return _neg_log_lik_with_teams(full_params, *args)

                args = (
                    X_a, X_b, ya, yb, weights,
                    home_idx, away_idx, n_teams, self.N_PARAMS,
                    self.ridge_alpha, self.team_alpha,
                )

            result = minimize(
                objective,
                x0_ext,
                args=args,
                method="L-BFGS-B",
                bounds=bounds_ext,
                options={
                    "maxiter": 1000,
                    "maxfun": 60000,
                    "ftol": 1e-10,
                    "gtol": 1e-5,
                    "maxls": 100,
                },
            )
            self.fit_result_ = result
            self.coef_     = result.x[:self.N_PARAMS]
            self.team_atk_ = result.x[self.N_PARAMS:self.N_PARAMS + n_teams].copy()
            self.team_dfn_ = result.x[self.N_PARAMS + n_teams:self.N_PARAMS + 2 * n_teams].copy()
            self.rho_      = float(result.x[-1] if self.fixed_rho is None else self.fixed_rho)
        elif self.ridge_alpha > 0.0:
            result = minimize(
                _neg_log_lik_regularized,
                x0,
                args=(X_a, X_b, ya, yb, weights, self.ridge_alpha),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-10},
            )
            self.fit_result_ = result
            self.coef_     = result.x[:-1]
            self.rho_      = float(result.x[-1])
            self.team_map_ = None
            self.team_atk_ = None
            self.team_dfn_ = None
        else:
            result = minimize(
                _neg_log_lik,
                x0,
                args=(X_a, X_b, ya, yb, weights),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-10},
            )
            self.fit_result_ = result
            self.coef_     = result.x[:-1]
            self.rho_      = float(result.x[-1])
            self.team_map_ = None
            self.team_atk_ = None
            self.team_dfn_ = None

        self.param_cov_ = _extract_beta_cov(self.fit_result_, self.N_PARAMS)
        self._fitted = True
        return self

    def predict(self, context: dict[str, float] | pd.Series) -> GoalsDist:
        """
        Return goals-distribution parameters for a single matchup.

        Parameters
        ----------
        context:
            Dict or Series with feature values (output of engineer.build_features
            for one row). Missing keys are imputed with neutral defaults.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        def _v(key: str, default: float = 0.0) -> float:
            val = context.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val)

        neutral = _v("neutral_flag", 0.0)
        home_adv = 1.0 - neutral
        elo_s = _v("elo_diff", 0.0) / 400.0
        attack_value_diff = _v("attack_value_rank_a", 0.5) - _v("attack_value_rank_b", 0.5)
        midfield_value_diff = _v("midfield_value_rank_a", 0.5) - _v("midfield_value_rank_b", 0.5)
        defense_value_diff = _v("defense_value_rank_a", 0.5) - _v("defense_value_rank_b", 0.5)
        host = _v("host_flag", 0.0)

        x_a = np.array([
            1.0, home_adv, elo_s,
            attack_value_diff, midfield_value_diff, defense_value_diff, host,
        ])
        x_b = np.array([
            1.0, -home_adv, -elo_s,
            -attack_value_diff, -midfield_value_diff, -defense_value_diff, host,
        ])

        _coef = getattr(self, "_perturbed_coef", None)
        _coef = _coef if _coef is not None else self.coef_
        la_log = float(x_a @ _coef)
        lb_log = float(x_b @ _coef)

        # Apply per-team attack / defence effects when the model was fitted
        # with team names and the caller passes "team_a" / "team_b" in context.
        if self.team_atk_ is not None:
            team_a_name = context.get("team_a") if hasattr(context, "get") else None
            team_b_name = context.get("team_b") if hasattr(context, "get") else None
            if team_a_name is not None and team_a_name in self.team_map_:
                idx_a = self.team_map_[team_a_name]
                la_log += float(self.team_atk_[idx_a])   # home attack
                lb_log -= float(self.team_dfn_[idx_a])   # home defence → fewer away goals
            if team_b_name is not None and team_b_name in self.team_map_:
                idx_b = self.team_map_[team_b_name]
                la_log -= float(self.team_dfn_[idx_b])   # away defence → fewer home goals
                lb_log += float(self.team_atk_[idx_b])   # away attack

        la = float(np.exp(la_log))
        lb = float(np.exp(lb_log))
        return GoalsDist(lambda_a=la, lambda_b=lb, dependence=self.rho_)

    def sample_coef(self, rng: np.random.Generator) -> np.ndarray:
        """
        Draw a perturbed feature coefficient vector from the approximate
        Gaussian posterior N(coef_, param_cov_).  If param_cov_ is None
        (e.g. fit failed to produce a Hessian), returns a copy of coef_.
        """
        param_cov = getattr(self, "param_cov_", None)
        if param_cov is None or self.coef_ is None:
            return self.coef_.copy() if self.coef_ is not None else np.array([])
        try:
            return rng.multivariate_normal(self.coef_, param_cov)
        except np.linalg.LinAlgError:
            return self.coef_.copy()

    def derive_wdl(
        self,
        dist: GoalsDist,
        max_goals: int | None = None,
    ) -> tuple[float, float, float]:
        """
        Integrate the scoreline grid to derive P(win_a), P(draw), P(loss_a).

        Returns a normalised tuple that sums to 1.0.
        """
        mg = max_goals if max_goals is not None else self._max_goals
        la, lb, rho = dist.lambda_a, dist.lambda_b, dist.dependence

        p_win = p_draw = p_loss = 0.0
        for i in range(mg + 1):
            for j in range(mg + 1):
                p = max(0.0, poisson.pmf(i, la) * poisson.pmf(j, lb) * _dc_tau(i, j, la, lb, rho))
                if i > j:
                    p_win += p
                elif i == j:
                    p_draw += p
                else:
                    p_loss += p

        total = p_win + p_draw + p_loss
        if total <= 0:
            return 1 / 3, 1 / 3, 1 / 3
        pw = p_win / total
        pd_ = p_draw / total
        pl = p_loss / total

        if getattr(self, "cal_win_", None) is not None:
            pw = float(np.clip(self.cal_win_(np.array([pw]))[0], 0.0, 1.0))
            pd_ = float(np.clip(self.cal_draw_(np.array([pd_]))[0], 0.0, 1.0))
            pl = float(np.clip(self.cal_loss_(np.array([pl]))[0], 0.0, 1.0))
            total2 = pw + pd_ + pl
            if total2 > 0:
                pw, pd_, pl = pw / total2, pd_ / total2, pl / total2

        return pw, pd_, pl


def derive_wdl(
    dist: GoalsDist,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    """Module-level convenience wrapper around MatchModel.derive_wdl."""
    _m = MatchModel.__new__(MatchModel)
    _m._max_goals = max_goals
    _m.coef_ = None
    _m.rho_ = dist.dependence
    _m._fitted = True
    return _m.derive_wdl(dist, max_goals=max_goals)
