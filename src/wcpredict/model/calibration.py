"""
model/calibration.py — goals/WDL calibration diagnostic and post-hoc calibrators.

calibration_table()   — Elo-bucketed predicted vs actual goals/WDL bias analysis.
fit_wdl_calibration() — fit isotonic (PAVA) calibrators on val split; stores on model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wcpredict.model.match_model import MatchModel


# ---------------------------------------------------------------------------
# PAVA isotonic regression (no sklearn dependency)
# ---------------------------------------------------------------------------

def _pava(y: np.ndarray) -> np.ndarray:
    """Pool Adjacent Violators Algorithm — returns isotonically non-decreasing y."""
    out = y.astype(float).copy()
    n = len(out)
    i = 0
    while i < n - 1:
        if out[i] > out[i + 1]:
            # merge block: replace both with their mean, then re-check leftward
            j = i + 1
            block_sum = out[i] + out[j]
            block_len = 2
            out[i] = out[j] = block_sum / block_len
            while i > 0 and out[i - 1] > out[i]:
                i -= 1
                block_sum += out[i]
                block_len += 1
                out[i: i + block_len] = block_sum / block_len
        else:
            i += 1
    return out


class IsotonicCalibrator:
    """
    Piecewise-linear monotone calibrator trained via PAVA on binned data.
    Extrapolates as flat (clips to [min_x, max_x] range).
    """

    def __init__(self, n_bins: int = 20) -> None:
        self.n_bins = n_bins
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, p_pred: np.ndarray, p_actual: np.ndarray) -> "IsotonicCalibrator":
        p_pred = np.asarray(p_pred, float)
        p_actual = np.asarray(p_actual, float)
        order = np.argsort(p_pred)
        p_pred = p_pred[order]
        p_actual = p_actual[order]
        # Bin into n_bins quantile buckets, take mean of pred and actual per bucket
        idx = np.array_split(np.arange(len(p_pred)), self.n_bins)
        x_pts = np.array([p_pred[i].mean() for i in idx if len(i)])
        y_pts = np.array([p_actual[i].mean() for i in idx if len(i)])
        # Enforce isotonicity via PAVA
        y_pts = _pava(y_pts)
        self._x = x_pts
        self._y = y_pts
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        if self._x is None:
            return np.asarray(p, float)
        return np.interp(np.asarray(p, float), self._x, self._y)


def fit_wdl_calibration(
    matches: pd.DataFrame,
    features: pd.DataFrame,
    model: MatchModel,
    *,
    n_bins: int = 20,
) -> None:
    """
    Fit three isotonic calibrators (win, draw, loss) on the provided matches
    and store them as model.cal_win_, model.cal_draw_, model.cal_loss_.

    Parameters
    ----------
    matches  : table with goals_home, goals_away (rows aligned with features)
    features : feature DataFrame (same row order as matches)
    model    : fitted MatchModel — calibrators are attached in-place
    n_bins   : number of quantile bins for PAVA fitting
    """
    pred_w, pred_d, pred_l = [], [], []
    act_w, act_d, act_l = [], [], []

    for i in range(len(matches)):
        row_m = matches.iloc[i]
        row_f = features.iloc[i]
        gh = row_m.get("goals_home")
        ga = row_m.get("goals_away")
        if pd.isna(gh) or pd.isna(ga):
            continue
        gh, ga = float(gh), float(ga)
        ctx = row_f.to_dict()
        ctx["team_a"] = matches.iloc[i].get("team_home", "")
        ctx["team_b"] = matches.iloc[i].get("team_away", "")
        dist = model.predict(ctx)
        w, d, l = _raw_wdl(model, dist)
        pred_w.append(w); pred_d.append(d); pred_l.append(l)
        act_w.append(1.0 if gh > ga else 0.0)
        act_d.append(1.0 if gh == ga else 0.0)
        act_l.append(1.0 if gh < ga else 0.0)

    if not pred_w:
        return

    model.cal_win_  = IsotonicCalibrator(n_bins).fit(np.array(pred_w), np.array(act_w))
    model.cal_draw_ = IsotonicCalibrator(n_bins).fit(np.array(pred_d), np.array(act_d))
    model.cal_loss_ = IsotonicCalibrator(n_bins).fit(np.array(pred_l), np.array(act_l))


def _raw_wdl(model: MatchModel, dist) -> tuple[float, float, float]:
    """derive_wdl without applying calibration (avoids infinite recursion)."""
    from scipy.stats import poisson
    from wcpredict.model.match_model import _dc_tau
    mg = model._max_goals
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
    return p_win / total, p_draw / total, p_loss / total


_ELO_BINS = [-float("inf"), -500, -200, 200, 500, float("inf")]
_ELO_LABELS = ["≤-500", "-500..-200", "-200..+200", "+200..+500", "≥+500"]


def calibration_table(
    matches: pd.DataFrame,
    features: pd.DataFrame,
    model: MatchModel,
    elo_bins: list[float] | None = None,
    elo_labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Bucket *matches* by elo_diff and report predicted vs actual goals/WDL.

    Parameters
    ----------
    matches:
        Per-match table aligned row-for-row with *features*.
        Must include goals_home, goals_away.
    features:
        Feature DataFrame (output of engineer.build_features), same row order.
    model:
        Fitted MatchModel.
    elo_bins, elo_labels:
        Bin edges and labels for pd.cut.  Defaults cover elo_diff extremes.

    Returns
    -------
    DataFrame with columns: bucket, n, pred_goals_a, actual_goals_a,
    pred_goals_b, actual_goals_b, pred_W, actual_W, pred_D, actual_D,
    pred_L, actual_L, goals_a_bias, goals_b_bias, W_bias, D_bias, L_bias.
    Bias columns are (pred - actual); large_bias flag is appended.
    """
    bins = elo_bins if elo_bins is not None else _ELO_BINS
    labels = elo_labels if elo_labels is not None else _ELO_LABELS

    records: list[dict] = []
    for i in range(len(matches)):
        row_m = matches.iloc[i]
        row_f = features.iloc[i]
        ctx = row_f.to_dict()
        dist = model.predict(ctx)
        w, d, l = model.derive_wdl(dist)

        ga = float(row_m["goals_home"]) if pd.notna(row_m.get("goals_home")) else float("nan")
        gb = float(row_m["goals_away"]) if pd.notna(row_m.get("goals_away")) else float("nan")
        if np.isnan(ga) or np.isnan(gb):
            actual_w = actual_d = actual_l = float("nan")
        else:
            actual_w = 1.0 if ga > gb else 0.0
            actual_d = 1.0 if ga == gb else 0.0
            actual_l = 1.0 if ga < gb else 0.0

        ed = float(ctx.get("elo_diff", 0.0) or 0.0)
        records.append({
            "elo_diff": ed,
            "pred_lambda_a": dist.lambda_a,
            "pred_lambda_b": dist.lambda_b,
            "actual_goals_a": ga,
            "actual_goals_b": gb,
            "pred_W": w, "pred_D": d, "pred_L": l,
            "actual_W": actual_w, "actual_D": actual_d, "actual_L": actual_l,
        })

    df = pd.DataFrame(records)
    df["bucket"] = pd.cut(df["elo_diff"], bins=bins, labels=labels)

    agg = (
        df.groupby("bucket", observed=True)
        .agg(
            n=("elo_diff", "count"),
            pred_goals_a=("pred_lambda_a", "mean"),
            actual_goals_a=("actual_goals_a", "mean"),
            pred_goals_b=("pred_lambda_b", "mean"),
            actual_goals_b=("actual_goals_b", "mean"),
            pred_W=("pred_W", "mean"),
            actual_W=("actual_W", "mean"),
            pred_D=("pred_D", "mean"),
            actual_D=("actual_D", "mean"),
            pred_L=("pred_L", "mean"),
            actual_L=("actual_L", "mean"),
        )
        .reset_index()
    )

    agg["goals_a_bias"] = agg["pred_goals_a"] - agg["actual_goals_a"]
    agg["goals_b_bias"] = agg["pred_goals_b"] - agg["actual_goals_b"]
    agg["W_bias"] = agg["pred_W"] - agg["actual_W"]
    agg["D_bias"] = agg["pred_D"] - agg["actual_D"]
    agg["L_bias"] = agg["pred_L"] - agg["actual_L"]

    _GOALS_THRESH = 0.50
    _WDL_THRESH = 0.12
    agg["large_bias"] = (
        (agg["goals_a_bias"].abs() > _GOALS_THRESH)
        | (agg["goals_b_bias"].abs() > _GOALS_THRESH)
        | (agg["W_bias"].abs() > _WDL_THRESH)
        | (agg["D_bias"].abs() > _WDL_THRESH)
        | (agg["L_bias"].abs() > _WDL_THRESH)
    )

    return agg


def print_calibration_table(table: pd.DataFrame) -> None:
    """Pretty-print the calibration table to stdout."""
    cols = [
        "bucket", "n",
        "pred_goals_a", "actual_goals_a",
        "pred_goals_b", "actual_goals_b",
        "pred_W", "actual_W",
        "pred_D", "actual_D",
        "pred_L", "actual_L",
        "large_bias",
    ]
    display = table[cols].copy()
    float_cols = [c for c in cols if c not in ("bucket", "n", "large_bias")]
    for c in float_cols:
        display[c] = display[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NaN")
    display["large_bias"] = display["large_bias"].map(lambda x: ">>> BIAS <<<" if x else "ok")
    print(display.to_string(index=False))
