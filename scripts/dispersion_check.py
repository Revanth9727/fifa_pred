"""
scripts/dispersion_check.py — dispersion diagnostic for the goals model.

Diagnostic only; no model changes.

Loads the fitted model and processed training data, bins matches by predicted
total lambda (lambda_a + lambda_b), and computes

    var(actual_total_goals) / mean(actual_total_goals)

per bin.  A ratio > 1.5 suggests overdispersion in that region and may warrant
revisiting a Negative Binomial parameterisation.

Usage:
    python scripts/dispersion_check.py [--threshold FLOAT] [--n_bins INT]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


OVERDISPERSION_THRESHOLD = 1.5


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[1]


def load_settings(root: Path) -> dict:
    with open(root / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dispersion check for the goals model.")
    parser.add_argument(
        "--threshold", type=float, default=OVERDISPERSION_THRESHOLD,
        help="Variance/mean ratio above which a bin is flagged (default: %(default)s)",
    )
    parser.add_argument(
        "--n_bins", type=int, default=8,
        help="Number of lambda bins (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    root = project_root()
    settings = load_settings(root)
    paths = settings.get("paths", {})

    processed_dir = root / paths.get("processed", "data/processed")
    outputs_dir = root / paths.get("outputs", "outputs")

    match_table_path = processed_dir / "match_table.parquet"
    features_path = processed_dir / "features.parquet"
    model_path = outputs_dir / "match_model.pkl"

    for p in (match_table_path, features_path, model_path):
        if not p.exists():
            print(f"ERROR: required file not found: {p}", file=sys.stderr)
            print("Run `wcpredict build-data` and `wcpredict fit` first.", file=sys.stderr)
            return 1

    print("Loading data and model…")
    table = pd.read_parquet(match_table_path)
    features = pd.read_parquet(features_path)
    table["date"] = pd.to_datetime(table["date"])

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    scored = table[["goals_home", "goals_away"]].notna().all(axis=1)
    table = table.loc[scored].reset_index(drop=True)
    features = features.loc[scored].reset_index(drop=True)

    n = len(table)
    print(f"Scored matches available: {n}")
    if n == 0:
        print("No scored matches — nothing to check.")
        return 0

    print("Computing predicted lambdas for each match…")
    lambda_totals: list[float] = []
    actual_totals: list[float] = []

    for i in range(n):
        row_m = table.iloc[i]
        row_f = features.iloc[i]
        ctx = row_f.to_dict()
        ctx["team_a"] = row_m.get("team_home", "")
        ctx["team_b"] = row_m.get("team_away", "")
        dist = model.predict(ctx)
        lambda_totals.append(float(dist.lambda_a) + float(dist.lambda_b))
        actual_totals.append(float(row_m["goals_home"]) + float(row_m["goals_away"]))

    lam = np.array(lambda_totals)
    actual = np.array(actual_totals)

    edges = np.percentile(lam, np.linspace(0, 100, args.n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    header = (
        f"{'Bin':>3}  {'λ range':>18}  {'n':>6}  {'mean λ':>8}  "
        f"{'mean act':>9}  {'var(act)':>9}  {'var/mean':>9}  {'flag':>6}"
    )
    print()
    print(header)
    print("-" * len(header))

    any_flagged = False
    for b in range(args.n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (lam > lo) & (lam <= hi)
        sub_lam = lam[mask]
        sub_act = actual[mask]
        if len(sub_act) < 2:
            continue
        mean_lam = float(np.mean(sub_lam))
        mean_act = float(np.mean(sub_act))
        var_act = float(np.var(sub_act, ddof=1))
        ratio = var_act / mean_act if mean_act > 0 else float("nan")
        flagged = ratio > args.threshold
        if flagged:
            any_flagged = True
        flag_str = "⚠ HIGH" if flagged else "ok"
        print(
            f"{b + 1:>3}  [{lo:6.3f}, {hi:6.3f}]  {len(sub_act):>6}  "
            f"{mean_lam:>8.3f}  {mean_act:>9.3f}  {var_act:>9.3f}  "
            f"{ratio:>9.3f}  {flag_str:>6}"
        )

    print()
    if any_flagged:
        print(
            f"At least one bin exceeds var/mean > {args.threshold}. "
            "Consider a Negative Binomial re-parameterisation before concluding."
        )
    else:
        print(
            f"All bins within var/mean ≤ {args.threshold}. "
            "No evidence of systematic overdispersion."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
