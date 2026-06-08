"""
scripts/fit_real_model.py
=========================
Phase 2 enhancement: temporal train/val split + regularised match-model fit
on real historical data, followed by λ-calibration diagnostics on the
held-out validation set.

Usage
-----
    python3.11 scripts/fit_real_model.py

Output
------
    outputs/lambda_calibration.csv   — λ and W/D/L by elo_diff bucket
    outputs/fit_summary.txt          — fitted coefficients + metadata

Design notes
------------
- Training  : all matches 2000-01-01 .. SPLIT_DATE-1 with valid ELO on both sides.
- Validation: all matches from SPLIT_DATE onwards with valid ELO.
- Assertion : val.date.min() > train.date.max()  (no temporal leakage).
- Features  : full Phase-2 feature matrix from features/engineer.py:
  Elo/FIFA rank, form, squad-quality rank signal, neutral/host, rest, travel,
  and importance weight.  No LLM or market features.
- Regularisation: ridge_alpha=0.01 on feature coefficients;
  team_alpha=0.10 partial-pooling on per-team attack/defence intercepts.
- ELO source : data/raw/elo_ratings_wc2026.csv (annual snapshots, renamed
  snapshot_date → date before passing to build_table).
- FIFA source: all fifa_ranking-*.csv files in data/raw/, concatenated.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Project-root bootstrap
# --------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from wcpredict.data.build_table import build_table
from wcpredict.features.engineer import build_features
from wcpredict.model.match_model import MatchModel, GoalsDist

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SPLIT_DATE = pd.Timestamp("2022-01-01")   # training < this date, validation ≥
TRAIN_START = pd.Timestamp("2000-01-01")  # ignore pre-modern football

RAW_DIR = _ROOT / "data" / "raw"
OUT_DIR = _ROOT / "outputs"

ELO_BINS = [-np.inf, -400, -200, 0, 200, 400, np.inf]
BUCKET_LABELS = [
    "< -400", "-400 to -200", "-200 to 0",
    "0 to 200", "200 to 400", "> 400",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _load_elo() -> pd.DataFrame:
    """Load the WC-2026 ELO file and rename snapshot_date → date."""
    df = pd.read_csv(RAW_DIR / "elo_ratings_wc2026.csv")
    if "snapshot_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"snapshot_date": "date"})
    return df


def _load_fifa() -> pd.DataFrame:
    """Concatenate all fifa_ranking-*.csv files in data/raw/."""
    parts = []
    for p in sorted(RAW_DIR.glob("fifa_ranking-*.csv")):
        parts.append(pd.read_csv(p))
    if not parts:
        print("WARNING: no fifa_ranking-*.csv files found; FIFA rank will be NaN.")
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["rank_date", "country_full"])
    return df


def _read_csv_maybe_gzip(path: Path) -> pd.DataFrame:
    """Read CSV files even when a .csv path contains gzip-compressed bytes."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, compression="gzip")


def _load_squad_values() -> pd.DataFrame | None:
    """
    Build a national-team squad-quality proxy from Transfermarkt player data.

    This stays rank-style in features/engineer.py. We aggregate by player
    citizenship and keep the top 26 current player market values per country.
    It is imperfect, but it wires a real structured squad signal without
    exposing raw euro levels to the model.
    """
    players_path = RAW_DIR / "players.csv"
    if not players_path.exists():
        print("WARNING: players.csv not found; squad quality will be NaN.")
        return None

    players = _read_csv_maybe_gzip(players_path)
    needed = {"country_of_citizenship", "market_value_in_eur"}
    if not needed.issubset(players.columns):
        print(
            "WARNING: players.csv lacks country/value columns; "
            "squad quality will be NaN."
        )
        return None

    sv = players[["country_of_citizenship", "market_value_in_eur"]].copy()
    sv = sv.rename(columns={"country_of_citizenship": "team"})
    sv["total_value"] = pd.to_numeric(sv["market_value_in_eur"], errors="coerce")
    sv = sv.dropna(subset=["team", "total_value"])
    sv = sv[sv["total_value"] > 0]
    if sv.empty:
        print("WARNING: no positive player market values found; squad quality will be NaN.")
        return None

    top26 = (
        sv.sort_values(["team", "total_value"], ascending=[True, False])
        .groupby("team", as_index=False)
        .head(26)
    )
    return (
        top26.groupby("team", as_index=False)["total_value"]
        .sum()
        .sort_values("total_value", ascending=False)
    )


def _wdl_actual(ya: pd.Series, yb: pd.Series) -> tuple[float, float, float]:
    """W/D/L rates from actual scorelines (home team's perspective)."""
    w = (ya > yb).mean()
    d = (ya == yb).mean()
    loss = (ya < yb).mean()
    return float(w), float(d), float(loss)


def _wdl_predicted(model: MatchModel, rows: list[dict]) -> tuple[float, float, float]:
    """Mean predicted W/D/L probabilities across a list of context dicts."""
    if not rows:
        return 0.0, 0.0, 0.0
    ws, ds, ls = [], [], []
    for ctx in rows:
        dist = model.predict(ctx)
        w, d, l = model.derive_wdl(dist)
        ws.append(w); ds.append(d); ls.append(l)
    return float(np.mean(ws)), float(np.mean(ds)), float(np.mean(ls))


def _predict_frame(
    model: MatchModel,
    matches: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach lambda and W/D/L predictions to a match frame."""
    out = matches.copy()
    pred_la, pred_lb = [], []
    pred_w, pred_d, pred_l = [], [], []
    for i, row in features.iterrows():
        ctx = row.to_dict()
        ctx["team_a"] = matches.loc[i, "team_home"]
        ctx["team_b"] = matches.loc[i, "team_away"]
        dist = model.predict(ctx)
        pred_la.append(dist.lambda_a)
        pred_lb.append(dist.lambda_b)
        w, d, lv = model.derive_wdl(dist)
        pred_w.append(w); pred_d.append(d); pred_l.append(lv)

    out["pred_la"] = pred_la
    out["pred_lb"] = pred_lb
    out["pred_w"] = pred_w
    out["pred_d"] = pred_d
    out["pred_l"] = pred_l
    out["elo_diff"] = features["elo_diff"].values
    out["neutral_flag"] = features["neutral_flag"].values
    return out


def _diagnostic_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the exact lambda/WDL bucket diagnostic table."""
    df = df.copy()
    df["bucket"] = pd.cut(df["elo_diff"], bins=ELO_BINS, labels=BUCKET_LABELS)
    rows_out = []
    for bucket in BUCKET_LABELS:
        sub = df[df["bucket"] == bucket]
        if len(sub) == 0:
            continue
        scored = sub.dropna(subset=["goals_home", "goals_away"])
        ya = scored["goals_home"].astype(float)
        yb = scored["goals_away"].astype(float)
        act_w, act_d, act_l = _wdl_actual(ya, yb)
        pred_w = float(sub["pred_w"].mean() * 100)
        row = {
            "elo_diff_bucket": bucket,
            "n_matches": len(sub),
            "n_scored": len(scored),
            "neutral_share": sub["neutral_flag"].mean(),
            "mean_elo_diff": sub["elo_diff"].mean(),
            "actual_goals_home": ya.mean(),
            "actual_goals_away": yb.mean(),
            "pred_lambda_a": sub["pred_la"].mean(),
            "pred_lambda_b": sub["pred_lb"].mean(),
            "bias_lambda_a": sub["pred_la"].mean() - ya.mean(),
            "bias_lambda_b": sub["pred_lb"].mean() - yb.mean(),
            "actual_W%": act_w * 100,
            "actual_D%": act_d * 100,
            "actual_L%": act_l * 100,
            "pred_W%": pred_w,
            "pred_D%": sub["pred_d"].mean() * 100,
            "pred_L%": sub["pred_l"].mean() * 100,
            "win_bias_pct": pred_w - act_w * 100,
        }
        rows_out.append(row)
    return pd.DataFrame(rows_out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # -----------------------------------------------------------------------
    # 1. Load raw data
    # -----------------------------------------------------------------------
    print("Loading raw data …")
    results_df = pd.read_csv(RAW_DIR / "results.csv")
    elo_df     = _load_elo()
    fifa_df    = _load_fifa()

    # -----------------------------------------------------------------------
    # 2. Build training table (ELO + FIFA as-of-date joins)
    # -----------------------------------------------------------------------
    print("Building match table …")
    tbl = build_table(
        results_df=results_df,
        elo_df=elo_df,
        fifa_df=fifa_df if not fifa_df.empty else None,
    )
    tbl["date"] = pd.to_datetime(tbl["date"])

    # Keep only rows with valid ELO on both sides
    tbl = tbl.dropna(subset=["home_elo_pre", "away_elo_pre"]).copy()
    print(f"  Total rows with valid ELO: {len(tbl):,}")

    # Filter to modern football
    tbl = (
        tbl[tbl["date"] >= TRAIN_START]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )
    print(f"  Rows from {TRAIN_START.date()} onwards: {len(tbl):,}")

    # -----------------------------------------------------------------------
    # 3. Build feature matrix once over the full chronological history.
    #    The feature builder uses strictly prior matches for form/rest/travel,
    #    so validation rows may use training-era history without leakage.
    # -----------------------------------------------------------------------
    print("Building full Phase-2 features …")
    squad_values = _load_squad_values()
    t_feat = time.perf_counter()
    feats_all = build_features(tbl, squad_values=squad_values)
    feat_secs = time.perf_counter() - t_feat
    print(f"  Feature build complete in {feat_secs:.1f}s")
    missing_rates = feats_all.isna().mean().sort_values(ascending=False)
    print("  Highest feature missing rates:")
    for name, rate in missing_rates.head(5).items():
        print(f"    {name:<24s} {rate:.3f}")

    # -----------------------------------------------------------------------
    # 4. Temporal split — strict, no leakage
    # -----------------------------------------------------------------------
    train_mask = tbl["date"] < SPLIT_DATE
    val_mask = ~train_mask
    train_df = tbl.loc[train_mask].copy().reset_index(drop=True)
    val_df   = tbl.loc[val_mask].copy().reset_index(drop=True)
    train_feats = feats_all.loc[train_mask].copy().reset_index(drop=True)
    val_feats   = feats_all.loc[val_mask].copy().reset_index(drop=True)

    train_max = train_df["date"].max()
    val_min   = val_df["date"].min()

    assert val_min > train_max, (
        f"Temporal leakage: earliest val match ({val_min.date()}) "
        f"does not postdate latest train match ({train_max.date()})"
    )
    print(
        f"\nSplit: train {train_df['date'].min().date()} – {train_max.date()} "
        f"({len(train_df):,} rows)"
    )
    print(
        f"       val   {val_min.date()} – {val_df['date'].max().date()} "
        f"({len(val_df):,} rows)"
    )
    print("  Temporal assertion: PASS  (val_min > train_max)")

    # -----------------------------------------------------------------------
    # 5. Fit regularised model on training data
    # -----------------------------------------------------------------------
    n_teams_train = len(set(train_df["team_home"]) | set(train_df["team_away"]))
    n_params = MatchModel.N_PARAMS + 2 * n_teams_train + 1
    print(
        f"\nFitting model: {len(train_df):,} rows, {n_teams_train} teams, "
        f"{n_params} parameters  (ridge_alpha=0.01, team_alpha=0.10) …"
    )
    t_fit = time.perf_counter()
    model = MatchModel(ridge_alpha=0.01, team_alpha=0.10)
    model.fit(train_df, train_feats)
    fit_secs = time.perf_counter() - t_fit
    print(f"  Fit complete in {fit_secs:.1f}s")
    fit_result = getattr(model, "fit_result_", None)
    if fit_result is not None:
        print(
            f"  Optimizer: success={fit_result.success} "
            f"nit={fit_result.nit} nfev={fit_result.nfev} fun={fit_result.fun:.3f}"
        )

    # -----------------------------------------------------------------------
    # 6. Predict on validation set
    # -----------------------------------------------------------------------
    print("\nPredicting on validation set …")
    val_df = _predict_frame(model, val_df, val_feats)

    # -----------------------------------------------------------------------
    # 7. Bucket diagnostic table
    # -----------------------------------------------------------------------
    print("\n=== λ-Calibration Diagnostic: Validation Set ===\n")
    val_df["bucket"] = pd.cut(val_df["elo_diff"], bins=ELO_BINS, labels=BUCKET_LABELS)
    diag_df = _diagnostic_table(val_df)

    # Print table
    float_cols = [c for c in diag_df.columns if c not in ("elo_diff_bucket", "n_matches")]
    print(diag_df.to_string(
        index=False,
        float_format=lambda x: f"{x:7.3f}",
    ))

    equal_neutral = {
        "neutral_flag": 1.0, "elo_diff": 0.0,
        "form_a": 0.5, "form_b": 0.5,
        "squad_quality_rank_a": 0.5, "squad_quality_rank_b": 0.5,
        "rest_days_a": 7.0, "rest_days_b": 7.0,
        "travel_km_a": 500.0, "travel_km_b": 500.0,
        "host_flag": 0.0,
    }
    equal_home = {**equal_neutral, "neutral_flag": 0.0}
    d_neutral = model.predict(equal_neutral)
    d_home = model.predict(equal_home)
    print("\n--- Home/neutral sanity ---")
    print(f"  home_adv coefficient: {model.coef_[1]:+.4f}")
    print(f"  validation neutral share: {val_df['neutral_flag'].mean():.3f}")
    print(
        "  equal teams, neutral non-host: "
        f"λa={d_neutral.lambda_a:.4f}, λb={d_neutral.lambda_b:.4f}, "
        f"ratio={d_neutral.lambda_a / d_neutral.lambda_b:.4f}"
    )
    print(
        "  equal teams, non-neutral:      "
        f"λa={d_home.lambda_a:.4f}, λb={d_home.lambda_b:.4f}, "
        f"ratio={d_home.lambda_a / d_home.lambda_b:.4f}"
    )

    # -----------------------------------------------------------------------
    # 8. Extreme-bucket detail: raw λ distribution
    # -----------------------------------------------------------------------
    print("\n--- λ distribution at extremes ---")
    for bucket in [BUCKET_LABELS[0], BUCKET_LABELS[-1]]:
        sub = val_df[val_df["bucket"] == bucket]
        if len(sub) == 0:
            continue
        print(f"\n{bucket}  (n={len(sub)})")
        print(f"  pred_λa : min={sub['pred_la'].min():.3f}  "
              f"p5={sub['pred_la'].quantile(0.05):.3f}  "
              f"median={sub['pred_la'].median():.3f}  "
              f"p95={sub['pred_la'].quantile(0.95):.3f}  "
              f"max={sub['pred_la'].max():.3f}")
        print(f"  pred_λb : min={sub['pred_lb'].min():.3f}  "
              f"p5={sub['pred_lb'].quantile(0.05):.3f}  "
              f"median={sub['pred_lb'].median():.3f}  "
              f"p95={sub['pred_lb'].quantile(0.95):.3f}  "
              f"max={sub['pred_lb'].max():.3f}")
        print(f"  elo_diff: min={sub['elo_diff'].min():.0f}  "
              f"max={sub['elo_diff'].max():.0f}")
        # P(0 home goals) ≈ Poisson(λa) PMF at 0 = exp(-λa)
        p_zero_home = np.exp(-sub["pred_la"]).mean()
        p_zero_away = np.exp(-sub["pred_lb"]).mean()
        print(f"  P(0 home goals): {p_zero_home:.3f}  "
              f"P(0 away goals): {p_zero_away:.3f}")

    # -----------------------------------------------------------------------
    # 9. Save outputs
    # -----------------------------------------------------------------------
    csv_path = OUT_DIR / "lambda_calibration.csv"
    diag_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    summary_path = OUT_DIR / "fit_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== fit_real_model.py — Regularised Match Model Summary ===\n\n")
        f.write(f"Training : {train_df['date'].min().date()} – {train_max.date()}"
                f"  ({len(train_df):,} rows)\n")
        f.write(f"Validation: {val_min.date()} – {val_df['date'].max().date()}"
                f"  ({len(val_df):,} rows)\n")
        f.write(f"Teams in train: {n_teams_train}\n")
        f.write(f"Parameters: {n_params}\n")
        f.write(f"ridge_alpha: 0.01   team_alpha: 0.10\n")
        f.write(f"Feature build time: {feat_secs:.1f}s\n")
        f.write(f"Fit time: {fit_secs:.1f}s\n\n")
        fit_result = getattr(model, "fit_result_", None)
        if fit_result is not None:
            f.write(
                "Optimizer:\n"
                f"  success: {fit_result.success}\n"
                f"  status:  {fit_result.status}\n"
                f"  message: {fit_result.message}\n"
                f"  nit:     {fit_result.nit}\n"
                f"  nfev:    {fit_result.nfev}\n"
                f"  fun:     {fit_result.fun:.6f}\n\n"
            )
        feat_names = [
            "intercept", "home_adv", "elo_s (÷400)",
            "form_diff", "squad_diff", "rest_diff (÷14, clipped)",
            "travel_diff (÷5000)", "host_flag",
        ]
        f.write("Feature coefficients:\n")
        for name, val in zip(feat_names, model.coef_):
            f.write(f"  {name:<25s}: {val:+.4f}\n")
        f.write(f"\n  rho (DC dependence): {model.rho_:+.4f}\n\n")
        if model.team_atk_ is not None:
            train_counts = pd.concat([train_df["team_home"], train_df["team_away"]]).value_counts()
            team_rows = []
            for team, idx in model.team_map_.items():
                atk = float(model.team_atk_[idx])
                dfn = float(model.team_dfn_[idx])
                team_rows.append({
                    "team": team,
                    "appearances": int(train_counts.get(team, 0)),
                    "atk": atk,
                    "dfn": dfn,
                    "net": atk - dfn,
                    "abs_net": abs(atk - dfn),
                })
            team_effects_df = pd.DataFrame(team_rows).sort_values("abs_net", ascending=False)

            f.write("Team-effect shrinkage sanity:\n")
            f.write(
                f"  median |net|: {team_effects_df['abs_net'].median():.4f}\n"
                f"  p90 |net|:    {team_effects_df['abs_net'].quantile(0.90):.4f}\n"
                f"  max |net|:    {team_effects_df['abs_net'].max():.4f}\n\n"
            )

            wc_teams = [
                "Argentina", "Brazil", "France", "England", "Spain",
                "Portugal", "Germany", "Netherlands", "Colombia", "Uruguay",
                "Qatar", "Jordan", "Haiti", "Curaçao", "Saudi Arabia",
            ]
            f.write("Per-team effects (attack, defence) for selected teams:\n")
            f.write(f"  {'Team':<25s} {'apps':>6s} {'atk':>8s} {'dfn':>8s} {'net':>8s}\n")
            for team in wc_teams:
                if team in model.team_map_:
                    idx = model.team_map_[team]
                    atk = float(model.team_atk_[idx])
                    dfn = float(model.team_dfn_[idx])
                    apps = int(train_counts.get(team, 0))
                    f.write(f"  {team:<25s} {apps:6d} {atk:+8.4f} {dfn:+8.4f} {atk-dfn:+8.4f}\n")
                else:
                    f.write(f"  {team:<25s} {'(not in training data)':>33s}\n")
        f.write(f"\n{'='*60}\n")
        f.write(diag_df.to_string(index=False) + "\n")

    print(f"Saved: {summary_path}")
    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
