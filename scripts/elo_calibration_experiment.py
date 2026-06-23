#!/usr/bin/env python3
"""
Experiment: Linear ELO vs Quadratic ELO  ×  Calibration Off vs On

Trains four model variants on the train split (pre-SPLIT_DATE) and evaluates
each on the validation split, reporting:
  - Mean log loss
  - Mean Brier score
  - Expected Calibration Error (ECE, 10-bin on win probability)
  - Reliability diagram data

Usage:
  uv run python scripts/elo_calibration_experiment.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import wcpredict.model.match_model as _mm_module
from wcpredict.model.match_model import MatchModel, _build_xy as _build_xy_quad
from wcpredict.model.calibration import fit_wdl_calibration

SPLIT_DATE = pd.Timestamp("2022-01-01")
N_CAL_BINS = 10


# ---------------------------------------------------------------------------
# Linear-only variant of _build_xy (drops the signed-square Elo term)
# ---------------------------------------------------------------------------

def _build_xy_linear(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(features)

    def _get(col: str, default: float = 0.0) -> np.ndarray:
        if col in features.columns:
            return pd.to_numeric(features[col], errors="coerce").fillna(default).to_numpy(float)
        return np.full(n, default)

    neutral = _get("neutral_flag", 0.0)
    home_adv = 1.0 - neutral
    elo_s = _get("elo_diff", 0.0) / 400.0
    sq_a = _get("squad_quality_rank_a", 0.5)
    sq_b = _get("squad_quality_rank_b", 0.5)
    squad_diff = sq_a - sq_b
    squad_depth_diff = _get("squad_depth_rank_a", 0.5) - _get("squad_depth_rank_b", 0.5)
    star_power_diff = _get("star_power_rank_a", 0.5) - _get("star_power_rank_b", 0.5)
    star_conc_diff = _get("star_concentration_a", 0.5) - _get("star_concentration_b", 0.5)
    attack_diff = _get("attack_value_rank_a", 0.5) - _get("attack_value_rank_b", 0.5)
    mid_diff = _get("midfield_value_rank_a", 0.5) - _get("midfield_value_rank_b", 0.5)
    def_diff = _get("defense_value_rank_a", 0.5) - _get("defense_value_rank_b", 0.5)
    club_min_diff = _get("club_minutes_rank_a", 0.5) - _get("club_minutes_rank_b", 0.5)
    club_form_diff = _get("club_form_rank_a", 0.5) - _get("club_form_rank_b", 0.5)
    gk_diff = _get("gk_rating_a", 0.5) - _get("gk_rating_b", 0.5)
    rest_diff = np.clip((_get("rest_days_a") - _get("rest_days_b")) / 14.0, -2.0, 2.0)
    trav_diff = np.clip((_get("travel_km_a") - _get("travel_km_b")) / 5000.0, -2.0, 2.0)
    host = _get("host_flag", 0.0)

    X_a = np.column_stack([
        np.ones(n), home_adv, elo_s,
        squad_diff, squad_depth_diff, star_power_diff, star_conc_diff,
        attack_diff, mid_diff, def_diff,
        club_min_diff, club_form_diff, rest_diff, trav_diff, gk_diff, host,
    ])
    X_b = np.column_stack([
        np.ones(n), -home_adv, -elo_s,
        -squad_diff, -squad_depth_diff, -star_power_diff, -star_conc_diff,
        -attack_diff, -mid_diff, -def_diff,
        -club_min_diff, -club_form_diff, -rest_diff, -trav_diff, -gk_diff, host,
    ])
    return X_a, X_b


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    model: MatchModel,
    matches: pd.DataFrame,
    features: pd.DataFrame,
    n_bins: int = N_CAL_BINS,
) -> dict[str, Any]:
    pred_w_list, act_w_list = [], []
    log_losses, briers = [], []

    for i in range(len(matches)):
        row_m = matches.iloc[i]
        gh = row_m.get("goals_home")
        ga = row_m.get("goals_away")
        if pd.isna(gh) or pd.isna(ga):
            continue
        gh, ga = float(gh), float(ga)
        ctx = features.iloc[i].to_dict()
        ctx["team_a"] = row_m.get("team_home", "")
        ctx["team_b"] = row_m.get("team_away", "")
        dist = model.predict(ctx)
        w, d, l = model.derive_wdl(dist)
        act_w = 1.0 if gh > ga else 0.0
        act_d = 1.0 if gh == ga else 0.0
        act_l = 1.0 if gh < ga else 0.0
        p_outcome = w if gh > ga else (d if gh == ga else l)
        ll = -math.log(max(float(p_outcome), 1e-12))
        brier = float(np.mean((np.array([w, d, l]) - np.array([act_w, act_d, act_l])) ** 2))
        log_losses.append(ll)
        briers.append(brier)
        pred_w_list.append(w)
        act_w_list.append(act_w)

    pred_arr = np.array(pred_w_list)
    act_arr = np.array(act_w_list)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    reliability = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (pred_arr >= lo) & (pred_arr < hi)
        if mask.sum() == 0:
            continue
        n_bin = int(mask.sum())
        mean_pred = float(pred_arr[mask].mean())
        obs_rate = float(act_arr[mask].mean())
        ece += (n_bin / len(pred_arr)) * abs(mean_pred - obs_rate)
        reliability.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": n_bin,
            "mean_pred": round(mean_pred, 4),
            "obs_rate": round(obs_rate, 4),
            "gap": round(mean_pred - obs_rate, 4),
        })

    return {
        "log_loss": round(float(np.mean(log_losses)), 5),
        "brier": round(float(np.mean(briers)), 5),
        "ece": round(float(ece), 5),
        "n": len(log_losses),
        "reliability": reliability,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    proc = ROOT / "data" / "processed"
    matches = pd.read_parquet(proc / "matches.parquet")
    features = pd.read_parquet(proc / "features.parquet")
    matches["date"] = pd.to_datetime(matches["date"])

    scored = matches[["goals_home", "goals_away"]].notna().all(axis=1)
    train_mask = (matches["date"] < SPLIT_DATE) & scored
    val_mask = (matches["date"] >= SPLIT_DATE) & scored

    train_m = matches.loc[train_mask].reset_index(drop=True)
    train_f = features.loc[train_mask].reset_index(drop=True)
    val_m = matches.loc[val_mask].reset_index(drop=True)
    val_f = features.loc[val_mask].reset_index(drop=True)

    print(f"Train rows: {len(train_m):,}   Val rows: {len(val_m):,}")
    print()

    conditions = [
        ("Linear ELO",   _build_xy_linear, 16, False),
        ("Linear ELO",   _build_xy_linear, 16, True),
        ("Quad ELO",     _build_xy_quad,   17, False),
        ("Quad ELO",     _build_xy_quad,   17, True),
    ]

    results = []
    for label, build_fn, n_params, use_cal in conditions:
        cal_label = "Cal On" if use_cal else "Cal Off"
        print(f"  Fitting: {label} + {cal_label} ...", end=" ", flush=True)

        _mm_module._build_xy = build_fn
        model = MatchModel(ridge_alpha=0.01, team_alpha=0.10)
        model.N_PARAMS = n_params
        model.fit(train_m, train_f)
        if n_params == 16:
            # predict() always builds a 17-element vector; insert 0 for elo_signed_sq
            model.coef_ = np.insert(model.coef_, 3, 0.0)
            model.N_PARAMS = 17

        if use_cal:
            fit_wdl_calibration(val_m, val_f, model)

        metrics = _evaluate(model, val_m, val_f)
        print(f"log_loss={metrics['log_loss']:.5f}  brier={metrics['brier']:.5f}  ece={metrics['ece']:.5f}")
        metrics["model"] = label
        metrics["calibration"] = cal_label
        results.append(metrics)

    _mm_module._build_xy = _build_xy_quad  # restore

    # Summary table
    print()
    print("=" * 68)
    print(f"{'Model':<16} {'Calibration':<12} {'Log Loss':>10} {'Brier':>10} {'ECE':>10} {'N':>6}")
    print("-" * 68)
    for r in results:
        print(f"{r['model']:<16} {r['calibration']:<12} {r['log_loss']:>10.5f} {r['brier']:>10.5f} {r['ece']:>10.5f} {r['n']:>6}")
    print("=" * 68)

    # Save summary
    out_path = ROOT / "outputs" / "elo_calibration_experiment.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = [{k: v for k, v in r.items() if k != "reliability"} for r in results]
    pd.DataFrame(summary).to_csv(out_path, index=False)
    print(f"\nSaved summary: {out_path}")

    # Reliability diagrams
    print()
    for r in results:
        print(f"Reliability — {r['model']} + {r['calibration']}:")
        rel = pd.DataFrame(r["reliability"])
        if not rel.empty:
            print(rel.to_string(index=False))
        print()


if __name__ == "__main__":
    main()
