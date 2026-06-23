#!/usr/bin/env python3
import math, pickle, sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPLIT_DATE = pd.Timestamp("2022-01-01")
FEATURE_NAMES = [
    "intercept","home_adv","elo",
    "attack_diff","midfield_diff","defense_diff","host",
]

with open(ROOT / "outputs" / "match_model.pkl", "rb") as f:
    model = pickle.load(f)

proc = ROOT / "data" / "processed"
matches = pd.read_parquet(proc / "matches.parquet")
features = pd.read_parquet(proc / "features.parquet")
matches["date"] = pd.to_datetime(matches["date"])
scored = matches[["goals_home","goals_away"]].notna().all(axis=1)
val_mask = (matches["date"] >= SPLIT_DATE) & scored
val_m = matches.loc[val_mask].reset_index(drop=True)
val_f = features.loc[val_mask].reset_index(drop=True)

# --- Coefficients ---
print("=== COEFFICIENTS ===")
for name, coef in zip(FEATURE_NAMES, model.coef_):
    print(f"  {name:<22} {coef:+.4f}")

# --- Evaluate all at once ---
pred_w, act_w, elo_diffs, neutrals, log_losses, briers = [], [], [], [], [], []
for i in range(len(val_m)):
    row_m = val_m.iloc[i]
    gh, ga = row_m.get("goals_home"), row_m.get("goals_away")
    if pd.isna(gh) or pd.isna(ga): continue
    gh, ga = float(gh), float(ga)
    ctx = val_f.iloc[i].to_dict()
    ctx["team_a"] = row_m.get("team_home","")
    ctx["team_b"] = row_m.get("team_away","")
    dist = model.predict(ctx)
    w, d, l = model.derive_wdl(dist)
    aw = 1.0 if gh > ga else 0.0
    ad = 1.0 if gh == ga else 0.0
    al = 1.0 if gh < ga else 0.0
    p_out = w if gh > ga else (d if gh == ga else l)
    pred_w.append(w); act_w.append(aw)
    elo_diffs.append(float(ctx.get("elo_diff", 0.0) or 0.0))
    neutrals.append(float(ctx.get("neutral_flag", 0.0) or 0.0))
    log_losses.append(-math.log(max(p_out, 1e-12)))
    briers.append(float(np.mean((np.array([w,d,l]) - np.array([aw,ad,al]))**2)))

pred_w = np.array(pred_w); act_w = np.array(act_w)
elo_diffs = np.array(elo_diffs); neutrals = np.array(neutrals)
log_losses = np.array(log_losses); briers = np.array(briers)

# ECE
bin_edges = np.linspace(0, 1, 11)
ece = sum(
    (((pred_w >= lo) & (pred_w < hi)).sum() / len(pred_w))
    * abs(pred_w[(pred_w >= lo) & (pred_w < hi)].mean() - act_w[(pred_w >= lo) & (pred_w < hi)].mean())
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:])
    if ((pred_w >= lo) & (pred_w < hi)).sum() > 0
)

print(f"\n=== OVERALL METRICS (val n={len(log_losses)}) ===")
print(f"  log_loss = {log_losses.mean():.5f}")
print(f"  brier    = {briers.mean():.5f}")
print(f"  ece      = {ece:.5f}")

print("\n=== SLICE METRICS ===")
print(f"  {'Slice':<25} {'n':>5} {'LogLoss':>9} {'Brier':>8} {'ECE':>8}")

def slice_stats(mask, label):
    if mask.sum() == 0: return
    pw = pred_w[mask]; aw2 = act_w[mask]
    ll = log_losses[mask].mean(); br = briers[mask].mean()
    be = np.linspace(0,1,11)
    e = sum(
        ((pw>=lo)&(pw<hi)).sum()/mask.sum()*abs(pw[(pw>=lo)&(pw<hi)].mean()-aw2[(pw>=lo)&(pw<hi)].mean())
        for lo,hi in zip(be[:-1],be[1:]) if ((pw>=lo)&(pw<hi)).sum()>0
    )
    print(f"  {label:<25} {mask.sum():>5} {ll:>9.4f} {br:>8.6f} {e:>8.5f}")

slice_stats(elo_diffs <= -200,      "Elo ≤ -200")
slice_stats((elo_diffs > -200) & (elo_diffs < 200), "-200 < Elo < +200")
slice_stats(elo_diffs >= 200,       "Elo ≥ +200")
slice_stats(neutrals == 1.0,        "Neutral venue")
slice_stats(neutrals == 0.0,        "Home/Away venue")

print("\n=== CORRUPTION REPORT ===")
sf = pd.read_csv(ROOT / "data" / "processed" / "squad_team_features_2026-06-06.csv")
low = sf[sf["n_squad"] < 5]
print(f"  n_squad < 5: {len(low)} teams")
for _, r in low.iterrows():
    print(f"    {r['team']}  n_squad={r['n_squad']}  star_conc={r.get('star_concentration','?')}")
ones = sf[sf["star_concentration"].round(3) == 1.0]
print(f"  star_concentration=1.0: {len(ones)} teams")
for _, r in ones.iterrows():
    print(f"    {r['team']}  n_squad={r['n_squad']}  n_missing={r['n_value_missing']}")

print("\n=== CALIBRATION BINS (win probability) ===")
print(f"  {'Bin':<10} {'n':>5} {'pred':>8} {'actual':>8} {'gap':>8}")
for lo, hi in zip(np.arange(0,1,0.1), np.arange(0.1,1.1,0.1)):
    m = (pred_w >= lo) & (pred_w < hi)
    if m.sum() == 0: continue
    mp = pred_w[m].mean(); ob = act_w[m].mean()
    print(f"  {lo:.1f}-{hi:.1f}     {m.sum():>5} {mp:>8.4f} {ob:>8.4f} {mp-ob:>+8.4f}")
