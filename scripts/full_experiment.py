#!/usr/bin/env python3
"""
Full experiment suite: steps 2-8.
Outputs one summary row per variant + slice table + feature audit + dispersion.
Usage: uv run python scripts/full_experiment.py
"""
from __future__ import annotations
import math, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wcpredict.model.match_model import MatchModel
from wcpredict.model.calibration import fit_wdl_calibration

SPLIT_DATE = pd.Timestamp("2022-01-01")
FEAT_NAMES = [
    "intercept","home_adv","elo",
    "squad_diff","squad_depth_diff","star_power_diff","star_conc_diff",
    "attack_diff","midfield_diff","defense_diff",
    "club_min_diff","club_form_diff","rest_diff","trav_diff","gk_diff","host",
]
SQUAD_COLS = [
    "squad_quality_rank_a","squad_quality_rank_b",
    "squad_depth_rank_a","squad_depth_rank_b",
    "star_power_rank_a","star_power_rank_b",
    "star_concentration_a","star_concentration_b",
    "attack_value_rank_a","attack_value_rank_b",
    "midfield_value_rank_a","midfield_value_rank_b",
    "defense_value_rank_a","defense_value_rank_b",
    "club_minutes_rank_a","club_minutes_rank_b",
    "club_form_rank_a","club_form_rank_b",
    "gk_rating_a","gk_rating_b",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    proc = ROOT / "data" / "processed"
    matches = pd.read_parquet(proc / "matches.parquet")
    features = pd.read_parquet(proc / "features.parquet")
    matches["date"] = pd.to_datetime(matches["date"])
    scored = matches[["goals_home","goals_away"]].notna().all(axis=1)
    train_mask = (matches["date"] < SPLIT_DATE) & scored
    val_mask   = (matches["date"] >= SPLIT_DATE) & scored
    return (
        matches.loc[train_mask].reset_index(drop=True),
        features.loc[train_mask].reset_index(drop=True),
        matches.loc[val_mask].reset_index(drop=True),
        features.loc[val_mask].reset_index(drop=True),
    )

# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------
def apply_recency_weight(matches: pd.DataFrame, half_life_days: float) -> pd.DataFrame:
    out = matches.copy()
    ref = pd.to_datetime(out["date"]).max()
    age = (ref - pd.to_datetime(out["date"])).dt.days.clip(lower=0).to_numpy(float)
    out["recency_weight"] = np.exp(-math.log(2.0) * age / half_life_days)
    return out

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def eval_model(model, val_m, val_f):
    pred_w, act_w, elo_d, neutr, lls, brs = [], [], [], [], [], []
    for i in range(len(val_m)):
        row = val_m.iloc[i]
        gh, ga = row.get("goals_home"), row.get("goals_away")
        if pd.isna(gh) or pd.isna(ga): continue
        gh, ga = float(gh), float(ga)
        ctx = val_f.iloc[i].to_dict()
        ctx["team_a"] = row.get("team_home","")
        ctx["team_b"] = row.get("team_away","")
        dist = model.predict(ctx)
        w, d, l = model.derive_wdl(dist)
        aw = 1.0 if gh > ga else 0.0
        ad = 1.0 if gh == ga else 0.0
        al = 1.0 if gh < ga else 0.0
        p_out = w if gh > ga else (d if gh == ga else l)
        lls.append(-math.log(max(p_out, 1e-12)))
        brs.append(float(np.mean((np.array([w,d,l]) - np.array([aw,ad,al]))**2)))
        pred_w.append(w); act_w.append(aw)
        elo_d.append(float(ctx.get("elo_diff", 0.0) or 0.0))
        neutr.append(float(ctx.get("neutral_flag", 0.0) or 0.0))
    return (np.array(pred_w), np.array(act_w),
            np.array(elo_d), np.array(neutr),
            np.array(lls), np.array(brs))

def ece(pred_w, act_w, n_bins=10):
    edges = np.linspace(0, 1, n_bins+1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred_w >= lo) & (pred_w < hi)
        if m.sum() == 0: continue
        total += m.sum()/len(pred_w) * abs(pred_w[m].mean() - act_w[m].mean())
    return total

def slice_row(label, mask, pred_w, act_w, lls, brs):
    if mask.sum() == 0: return None
    pw = pred_w[mask]; aw = act_w[mask]
    e = ece(pw, aw)
    return {"slice": label, "n": int(mask.sum()),
            "ll": round(lls[mask].mean(), 5),
            "brier": round(brs[mask].mean(), 6),
            "ece": round(e, 5)}

def slices(pred_w, act_w, elo_d, neutr, lls, brs):
    rows = []
    for label, mask in [
        ("Elo≤-200",    elo_d <= -200),
        ("-200<Elo<+200",(elo_d > -200) & (elo_d < 200)),
        ("Elo≥+200",    elo_d >= 200),
        ("Neutral",     neutr == 1.0),
        ("Home/Away",   neutr == 0.0),
    ]:
        r = slice_row(label, mask, pred_w, act_w, lls, brs)
        if r: rows.append(r)
    return rows

def summary_row(name, data_state, pred_w, act_w, lls, brs,
                pred_w_cal=None, act_w_cal=None, lls_cal=None, brs_cal=None,
                elo_d=None, coefs=None):
    e_raw = ece(pred_w, act_w)
    ll_cal = brier_cal = ece_cal = None
    if lls_cal is not None:
        e_cal = ece(pred_w_cal, act_w_cal)
        ll_cal = round(lls_cal.mean(), 5)
        brier_cal = round(brs_cal.mean(), 6)
        ece_cal = round(e_cal, 5)
    big_fav_mask = (elo_d >= 200) if elo_d is not None else np.zeros(len(lls), bool)
    sl = slice_row("Elo≥+200", big_fav_mask, pred_w, act_w, lls, brs)
    return {
        "model": name, "data": data_state,
        "ll_raw": round(lls.mean(), 5), "ll_cal": ll_cal,
        "brier_raw": round(brs.mean(), 6), "brier_cal": brier_cal,
        "ece_raw": round(e_raw, 5), "ece_cal": ece_cal,
        "elo200_ll": sl["ll"] if sl else None,
        "elo200_brier": sl["brier"] if sl else None,
        "elo200_ece": sl["ece"] if sl else None,
        "coefs": coefs,
    }

def fit_model(train_m, train_f):
    m = MatchModel(ridge_alpha=0.01, team_alpha=0.10)
    m.fit(train_m, train_f)
    return m

def calibrate(model, val_m, val_f):
    """Fit PAVA calibration on val; return calibrated wdl arrays."""
    fit_wdl_calibration(val_m, val_f, model)

def eval_calibrated(model, val_m, val_f):
    pred_w, act_w, lls, brs = [], [], [], []
    for i in range(len(val_m)):
        row = val_m.iloc[i]
        gh, ga = row.get("goals_home"), row.get("goals_away")
        if pd.isna(gh) or pd.isna(ga): continue
        gh, ga = float(gh), float(ga)
        ctx = val_f.iloc[i].to_dict()
        ctx["team_a"] = row.get("team_home","")
        ctx["team_b"] = row.get("team_away","")
        dist = model.predict(ctx)
        w, d, l = model.derive_wdl(dist)
        aw = 1.0 if gh > ga else 0.0
        ad = 1.0 if gh == ga else 0.0
        al = 1.0 if gh < ga else 0.0
        p_out = w if gh > ga else (d if gh == ga else l)
        lls.append(-math.log(max(p_out, 1e-12)))
        brs.append(float(np.mean((np.array([w,d,l]) - np.array([aw,ad,al]))**2)))
        pred_w.append(w); act_w.append(aw)
    return np.array(pred_w), np.array(act_w), np.array(lls), np.array(brs)

def zero_squad(f: pd.DataFrame) -> pd.DataFrame:
    out = f.copy()
    for c in SQUAD_COLS:
        if c in out.columns:
            out[c] = 0.5
    return out

def zero_elo(f: pd.DataFrame) -> pd.DataFrame:
    out = f.copy()
    if "elo_diff" in out.columns:
        out["elo_diff"] = 0.0
    return out

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...", flush=True)
    train_m, train_f, val_m, val_f = load_data()
    print(f"  train={len(train_m)}  val={len(val_m)}\n")

    summary_rows = []
    all_slices   = {}

    # -----------------------------------------------------------------------
    # STEP 1 — Clean baseline (already fit; reload)
    # -----------------------------------------------------------------------
    print("Step 1: Clean baseline...", flush=True)
    with open(ROOT / "outputs" / "match_model.pkl", "rb") as f:
        baseline = pickle.load(f)
    pw, aw, ed, nt, lls, brs = eval_model(baseline, val_m, val_f)
    coefs_baseline = dict(zip(FEAT_NAMES, baseline.coef_))
    r = summary_row("1_baseline", "clean+linear_elo", pw, aw, lls, brs, elo_d=ed, coefs=coefs_baseline)
    summary_rows.append(r)
    all_slices["1_baseline"] = slices(pw, aw, ed, nt, lls, brs)
    print(f"  ll={r['ll_raw']}  brier={r['brier_raw']}  ece={r['ece_raw']}")

    # -----------------------------------------------------------------------
    # STEP 2 — Calibration layer (PAVA, fit on val, eval on val)
    # -----------------------------------------------------------------------
    print("Step 2: Calibration...", flush=True)
    with open(ROOT / "outputs" / "match_model.pkl", "rb") as f:
        cal_model = pickle.load(f)
    calibrate(cal_model, val_m, val_f)
    pw_c, aw_c, lls_c, brs_c = eval_calibrated(cal_model, val_m, val_f)
    r2 = summary_row("2_calibrated", "clean+linear_elo+PAVA",
                     pw, aw, lls, brs,
                     pw_c, aw_c, lls_c, brs_c,
                     elo_d=ed, coefs=coefs_baseline)
    summary_rows.append(r2)
    all_slices["2_calibrated"] = slices(pw_c, aw_c, ed, nt, lls_c, brs_c)
    print(f"  raw ll={r2['ll_raw']}  cal ll={r2['ll_cal']}  cal ece={r2['ece_cal']}")

    # -----------------------------------------------------------------------
    # STEP 4 — Squad ablation (3 variants; step 3 is printed separately)
    # -----------------------------------------------------------------------
    print("Step 4a: Elo-only model...", flush=True)
    elo_only_f_tr = zero_squad(train_f)
    elo_only_f_va = zero_squad(val_f)
    m_elo = fit_model(train_m, elo_only_f_tr)
    pw4a, aw4a, ed4a, nt4a, lls4a, brs4a = eval_model(m_elo, val_m, elo_only_f_va)
    r4a = summary_row("4a_elo_only", "clean, no squad",
                      pw4a, aw4a, lls4a, brs4a, elo_d=ed4a,
                      coefs=dict(zip(FEAT_NAMES, m_elo.coef_)))
    summary_rows.append(r4a)
    all_slices["4a_elo_only"] = slices(pw4a, aw4a, ed4a, nt4a, lls4a, brs4a)
    print(f"  ll={r4a['ll_raw']}  brier={r4a['brier_raw']}  ece={r4a['ece_raw']}")

    print("Step 4b: Elo+squad (=baseline, already done)", flush=True)

    print("Step 4c: Squad-only model...", flush=True)
    squad_only_f_tr = zero_elo(train_f)
    squad_only_f_va = zero_elo(val_f)
    m_sq = fit_model(train_m, squad_only_f_tr)
    pw4c, aw4c, ed4c, nt4c, lls4c, brs4c = eval_model(m_sq, val_m, squad_only_f_va)
    r4c = summary_row("4c_squad_only", "clean, no elo",
                      pw4c, aw4c, lls4c, brs4c, elo_d=ed4c,
                      coefs=dict(zip(FEAT_NAMES, m_sq.coef_)))
    summary_rows.append(r4c)
    all_slices["4c_squad_only"] = slices(pw4c, aw4c, ed4c, nt4c, lls4c, brs4c)
    print(f"  ll={r4c['ll_raw']}  brier={r4c['brier_raw']}  ece={r4c['ece_raw']}")

    # -----------------------------------------------------------------------
    # STEP 5 — Temporal weighting (800 / 1200 / 1600 days)
    # -----------------------------------------------------------------------
    for hl in [800, 1200, 1600]:
        print(f"Step 5: Temporal weight half-life={hl}d...", flush=True)
        train_m_tw = apply_recency_weight(train_m, hl)
        m_tw = fit_model(train_m_tw, train_f)
        pw5, aw5, ed5, nt5, lls5, brs5 = eval_model(m_tw, val_m, val_f)
        r5 = summary_row(f"5_tw_{hl}d", f"clean+tw{hl}d",
                         pw5, aw5, lls5, brs5, elo_d=ed5,
                         coefs=dict(zip(FEAT_NAMES, m_tw.coef_)))
        summary_rows.append(r5)
        all_slices[f"5_tw_{hl}d"] = slices(pw5, aw5, ed5, nt5, lls5, brs5)
        print(f"  ll={r5['ll_raw']}  brier={r5['brier_raw']}  ece={r5['ece_raw']}")

    # -----------------------------------------------------------------------
    # STEP 7 — Dispersion diagnostic (on baseline model)
    # -----------------------------------------------------------------------
    print("\n=== STEP 7: DISPERSION DIAGNOSTIC ===")
    rows_disp = []
    for i in range(len(val_m)):
        row = val_m.iloc[i]
        gh, ga = row.get("goals_home"), row.get("goals_away")
        if pd.isna(gh) or pd.isna(ga): continue
        ctx = val_f.iloc[i].to_dict()
        ctx["team_a"] = row.get("team_home","")
        ctx["team_b"] = row.get("team_away","")
        dist = baseline.predict(ctx)
        for la, actual in [(dist.lambda_a, float(gh)), (dist.lambda_b, float(ga))]:
            rows_disp.append({"pred_rate": la, "actual": actual})
    disp_df = pd.DataFrame(rows_disp)
    disp_df["bucket"] = pd.cut(disp_df["pred_rate"],
                                bins=[0, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 99],
                                labels=["<0.75","0.75-1.0","1.0-1.25","1.25-1.5",
                                        "1.5-1.75","1.75-2.0","2.0-3.0",">3.0"])
    disp_tbl = (disp_df.groupby("bucket", observed=True)["actual"]
                .agg(n="count", mean="mean", var="var")
                .assign(dispersion=lambda x: x["var"] / x["mean"])
                .reset_index())
    print(disp_tbl.to_string(index=False))

    # -----------------------------------------------------------------------
    # STEP 3 — Feature audit (correlation matrix on val features)
    # -----------------------------------------------------------------------
    print("\n=== STEP 3: FEATURE AUDIT ===")
    audit_cols = ["elo_diff",
                  "squad_quality_rank_a","squad_depth_rank_a",
                  "star_power_rank_a","star_concentration_a",
                  "attack_value_rank_a","midfield_value_rank_a","defense_value_rank_a",
                  "travel_km_a"]
    audit_cols = [c for c in audit_cols if c in train_f.columns]
    af = train_f[audit_cols].dropna()
    print("\nMeans / Stds:")
    for c in audit_cols:
        print(f"  {c:<35} mean={af[c].mean():+.4f}  std={af[c].std():.4f}")
    print("\nCorrelation matrix (|r|>0.5 flagged with *):")
    corr = af.corr()
    for i, ci in enumerate(audit_cols):
        row_str = f"  {ci[:20]:<20}"
        for j, cj in enumerate(audit_cols):
            v = corr.iloc[i, j]
            flag = "*" if abs(v) > 0.5 and i != j else " "
            row_str += f" {v:+.2f}{flag}"
        print(row_str)

    # -----------------------------------------------------------------------
    # OUTPUT — Summary table
    # -----------------------------------------------------------------------
    print("\n" + "="*110)
    print(f"{'Model':<20} {'Data':<22} {'ll_raw':>8} {'ll_cal':>8} {'br_raw':>8} {'br_cal':>8} "
          f"{'ece_raw':>8} {'ece_cal':>8} {'e200_ll':>8} {'e200_br':>8} {'e200_ece':>9}")
    print("-"*110)
    for r in summary_rows:
        print(
            f"{r['model']:<20} {r['data']:<22} "
            f"{r['ll_raw']:>8.5f} {str(r['ll_cal'] or '-'):>8} "
            f"{r['brier_raw']:>8.6f} {str(r['brier_cal'] or '-'):>8} "
            f"{r['ece_raw']:>8.5f} {str(r['ece_cal'] or '-'):>8} "
            f"{str(r['elo200_ll'] or '-'):>8} "
            f"{str(r['elo200_brier'] or '-'):>8} "
            f"{str(r['elo200_ece'] or '-'):>9}"
        )
    print("="*110)

    print("\n=== SLICE DETAILS BY MODEL ===")
    for name, sl_rows in all_slices.items():
        print(f"\n  {name}:")
        print(f"    {'Slice':<20} {'n':>5} {'ll':>9} {'brier':>9} {'ece':>9}")
        for s in sl_rows:
            print(f"    {s['slice']:<20} {s['n']:>5} {s['ll']:>9.5f} {s['brier']:>9.6f} {s['ece']:>9.5f}")

    print("\n=== COEFFICIENTS BY MODEL ===")
    for r in summary_rows:
        if r["coefs"]:
            print(f"\n  {r['model']}:")
            for k, v in r["coefs"].items():
                print(f"    {k:<22} {v:+.4f}")

    print("\n=== CALIBRATION BINS — baseline vs calibrated ===")
    print(f"  {'Bin':<10} {'n':>5} {'pred_raw':>10} {'pred_cal':>10} {'actual':>8} {'gap_raw':>9} {'gap_cal':>9}")
    edges = np.linspace(0,1,11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m_r = (pw >= lo) & (pw < hi)
        m_c = (pw_c >= lo) & (pw_c < hi)
        if m_r.sum() == 0: continue
        pr = pw[m_r].mean(); pc = pw_c[m_c].mean() if m_c.sum() else float("nan")
        ob = aw[m_r].mean()
        print(f"  {lo:.1f}-{hi:.1f}     {m_r.sum():>5} {pr:>10.4f} {pc:>10.4f} {ob:>8.4f} {pr-ob:>+9.4f} {pc-ob:>+9.4f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
