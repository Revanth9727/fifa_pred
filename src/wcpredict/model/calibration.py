"""
model/calibration.py — goals/WDL calibration diagnostic.

Buckets historical matches by elo_diff and compares predicted vs actual
goals and W/D/L rates.  Pure analysis; writes nothing to disk.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from wcpredict.model.match_model import MatchModel


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
