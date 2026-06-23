"""
Market-only calibration validator.

This module is the only place odds/market data may be read. Prediction modules
must not import it. It de-vigs closing lines, scores calibration, persists
timestamped metrics records, and accumulates live adjusted-vs-unadjusted A/B
results as tournament matches are played.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import json
import math

import numpy as np
import pandas as pd


def load_market_data(path: Path | str) -> pd.DataFrame:
    """Load market/odds data. This market read is intentionally eval-only."""
    return pd.read_csv(path)


def odds_to_implied(decimal_odds: Iterable[float]) -> np.ndarray:
    odds = np.asarray(list(decimal_odds), dtype=float)
    if np.any(odds <= 1.0):
        raise ValueError("Decimal odds must be > 1")
    return 1.0 / odds


def devig_proportional(raw_implied: Iterable[float]) -> np.ndarray:
    probs = np.asarray(list(raw_implied), dtype=float)
    total = probs.sum()
    if total <= 0:
        raise ValueError("Raw implied probabilities must sum positive")
    return probs / total


def _shin_sum(z: float, probs: np.ndarray) -> float:
    # Jullien-Salanie/Shin transform. z=0 reduces to proportional after
    # normalization; z approaches 1 as insider overround increases.
    return float(np.sum((np.sqrt(z * z + 4.0 * (1.0 - z) * probs * probs / probs.sum()) - z) / (2.0 * (1.0 - z))))


def devig_shin(raw_implied: Iterable[float], *, tol: float = 1e-12, max_iter: int = 100) -> np.ndarray:
    probs = np.asarray(list(raw_implied), dtype=float)
    if np.any(probs <= 0):
        raise ValueError("Raw implied probabilities must be positive")
    if len(probs) == 2:
        # Shin is weakly identified for two outcomes; proportional is stable.
        return devig_proportional(probs)
    if probs.sum() <= 1.0:
        return devig_proportional(probs)

    lo, hi = 0.0, 0.999999
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        s = _shin_sum(mid, probs)
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2.0
    out = (np.sqrt(z * z + 4.0 * (1.0 - z) * probs * probs / probs.sum()) - z) / (2.0 * (1.0 - z))
    return out / out.sum()


def devig(raw_implied: Iterable[float], method: str = "shin") -> np.ndarray:
    if method == "proportional":
        return devig_proportional(raw_implied)
    if method == "shin":
        return devig_shin(raw_implied)
    raise ValueError(f"Unknown de-vig method: {method}")


def score(model_probs: Iterable[float], market_probs: Iterable[float]) -> dict[str, float]:
    """
    Score model probabilities against de-vigged market probabilities.

    This is distribution-vs-distribution scoring: Brier is mean squared
    probability error and log-loss is cross entropy of market under model.
    """
    p = np.asarray(list(model_probs), dtype=float)
    q = np.asarray(list(market_probs), dtype=float)
    if p.shape != q.shape:
        raise ValueError("model_probs and market_probs must have same shape")
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    q = np.clip(q, 0.0, 1.0)
    q = q / q.sum()
    return {
        "brier": float(np.mean((p - q) ** 2)),
        "log_loss": float(-np.sum(q * np.log(p))),
    }


def reliability_diagram(
    predictions: pd.DataFrame,
    *,
    prob_col: str = "model_prob",
    target_col: str = "outcome",
    n_bins: int = 10,
) -> pd.DataFrame:
    df = predictions[[prob_col, target_col]].dropna().copy()
    df["bin"] = pd.cut(df[prob_col], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    return (
        df.groupby("bin", observed=True)
        .agg(
            n=(prob_col, "count"),
            mean_pred=(prob_col, "mean"),
            observed_rate=(target_col, "mean"),
        )
        .reset_index()
        .assign(bin=lambda x: x["bin"].astype(str))
    )


@dataclass(frozen=True)
class MetricsRecord:
    captured_at: str
    brier: float
    log_loss: float
    reliability_bins: list[dict[str, Any]]
    ab_test: dict[str, Any]
    source_tier_ablation: dict[str, Any]


def persist_metrics(
    metrics: MetricsRecord | dict[str, Any],
    output_dir: Path | str = "outputs/metrics",
    *,
    timestamp: datetime | None = None,
) -> Path:
    ts = timestamp or datetime.now(timezone.utc)
    payload = asdict(metrics) if isinstance(metrics, MetricsRecord) else dict(metrics)
    payload.setdefault("captured_at", ts.isoformat())
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"metrics_{ts.strftime('%Y%m%dT%H%M%SZ')}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def ab_test(
    rows: pd.DataFrame,
    *,
    market_cols: tuple[str, ...] = ("market_home", "market_draw", "market_away"),
    base_cols: tuple[str, ...] = ("base_home", "base_draw", "base_away"),
    adjusted_cols: tuple[str, ...] = ("adjusted_home", "adjusted_draw", "adjusted_away"),
) -> dict[str, Any]:
    """Score adjusted vs unadjusted predictions against de-vigged close rows."""
    base_scores = []
    adj_scores = []
    for _, row in rows.dropna(subset=list(market_cols + base_cols + adjusted_cols)).iterrows():
        market = [row[c] for c in market_cols]
        base = [row[c] for c in base_cols]
        adj = [row[c] for c in adjusted_cols]
        base_scores.append(score(base, market))
        adj_scores.append(score(adj, market))
    return _summarize_ab(base_scores, adj_scores)


def _summarize_ab(base_scores: list[dict[str, float]], adj_scores: list[dict[str, float]]) -> dict[str, Any]:
    if not base_scores:
        return {"n": 0, "base": {}, "adjusted": {}, "delta": {}}
    base_df = pd.DataFrame(base_scores)
    adj_df = pd.DataFrame(adj_scores)
    base_mean = base_df.mean().to_dict()
    adj_mean = adj_df.mean().to_dict()
    return {
        "n": len(base_scores),
        "base": {k: float(v) for k, v in base_mean.items()},
        "adjusted": {k: float(v) for k, v in adj_mean.items()},
        "delta": {k: float(adj_mean[k] - base_mean[k]) for k in base_mean},
    }


def append_live_match(
    path: Path | str,
    *,
    match_id: str,
    market_probs: Iterable[float],
    base_probs: Iterable[float],
    adjusted_probs: Iterable[float],
    source_type: str | None = None,
    captured_at: datetime | None = None,
) -> pd.DataFrame:
    """Append one live match A/B observation scored against its closing line."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = captured_at or datetime.now(timezone.utc)
    market = list(market_probs)
    base = list(base_probs)
    adj = list(adjusted_probs)
    row = {
        "match_id": match_id,
        "captured_at": ts.isoformat(),
        "source_type": source_type or "none",
        "market_home": market[0], "market_draw": market[1], "market_away": market[2],
        "base_home": base[0], "base_draw": base[1], "base_away": base[2],
        "adjusted_home": adj[0], "adjusted_draw": adj[1], "adjusted_away": adj[2],
    }
    if p.exists():
        df = pd.read_csv(p)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(p, index=False)
    return df


def evaluate_by_slice(
    matches: pd.DataFrame,
    features: pd.DataFrame,
    model: Any,
) -> list[dict[str, Any]]:
    """
    Score the model's WDL predictions by Elo-gap bucket and by neutral/home venue.

    Returns a list of dicts, one per slice, with fields:
      bucket, n, pred_W, actual_W, pred_D, actual_D, pred_L, actual_L,
      W_bias, goals_bias, brier, log_loss, large_bias.

    goals_bias = mean(predicted_total_goals) - mean(actual_total_goals); positive
    means the model over-predicts goals in that slice.
    """
    import math

    ELO_BINS = [
        (-float("inf"), -200, "Elo ≤ −200"),
        (-200, 200, "−200 < Elo < +200"),
        (200, float("inf"), "Elo ≥ +200"),
    ]

    rows: list[dict] = []
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
        elo_diff = float(ctx.get("elo_diff", 0.0) or 0.0)
        neutral = float(ctx.get("neutral_flag", 0.0) or 0.0)
        dist = model.predict(ctx)
        w, d, l = model.derive_wdl(dist)
        act_w = 1.0 if gh > ga else 0.0
        act_d = 1.0 if gh == ga else 0.0
        act_l = 1.0 if gh < ga else 0.0
        p_outcome = w if gh > ga else (d if gh == ga else l)
        brier = float(np.mean((np.array([w, d, l]) - np.array([act_w, act_d, act_l])) ** 2))
        ll = -math.log(max(float(p_outcome), 1e-12))
        rows.append({
            "elo_diff": elo_diff,
            "neutral": neutral,
            "lambda_a": float(dist.lambda_a),
            "lambda_b": float(dist.lambda_b),
            "goals_home": gh,
            "goals_away": ga,
            "pred_W": w, "pred_D": d, "pred_L": l,
            "act_W": act_w, "act_D": act_d, "act_L": act_l,
            "brier": brier, "log_loss": ll,
        })

    if not rows:
        return []

    df = pd.DataFrame(rows)

    def _make_slice(label: str, sub: pd.DataFrame) -> dict[str, Any]:
        pw = float(sub["pred_W"].mean())
        pd_ = float(sub["pred_D"].mean())
        pl = float(sub["pred_L"].mean())
        aw = float(sub["act_W"].mean())
        goals_bias = float(
            (sub["lambda_a"] + sub["lambda_b"]).mean()
            - (sub["goals_home"] + sub["goals_away"]).mean()
        )
        return {
            "bucket": label,
            "n": int(len(sub)),
            "pred_W": round(pw, 4),
            "actual_W": round(aw, 4),
            "pred_D": round(pd_, 4),
            "actual_D": round(float(sub["act_D"].mean()), 4),
            "pred_L": round(pl, 4),
            "actual_L": round(float(sub["act_L"].mean()), 4),
            "W_bias": round(pw - aw, 4),
            "goals_bias": round(goals_bias, 4),
            "brier": round(float(sub["brier"].mean()), 6),
            "log_loss": round(float(sub["log_loss"].mean()), 4),
            "large_bias": abs(pw - aw) > 0.08,
        }

    slices: list[dict[str, Any]] = []

    for lo, hi, label in ELO_BINS:
        sub = df[(df["elo_diff"] > lo) & (df["elo_diff"] <= hi)]
        if not sub.empty:
            slices.append(_make_slice(label, sub))

    for flag, label in [(1.0, "Neutral venue"), (0.0, "Home/Away venue")]:
        sub = df[df["neutral"] == flag]
        if not sub.empty:
            slices.append(_make_slice(label, sub))

    return slices


def ablate_by_source(live_rows: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "source_type" not in live_rows.columns:
        return out
    for source_type, sub in live_rows.groupby("source_type"):
        out[str(source_type)] = ab_test(sub)
    return out


def evaluate_and_persist(
    predictions: pd.DataFrame,
    *,
    model_probs: Iterable[float],
    market_probs: Iterable[float],
    live_rows: pd.DataFrame | None = None,
    output_dir: Path | str = "outputs/metrics",
    n_bins: int = 10,
) -> Path:
    scores = score(model_probs, market_probs)
    reliability = reliability_diagram(predictions, n_bins=n_bins).to_dict(orient="records")
    live = live_rows if live_rows is not None else pd.DataFrame()
    record = MetricsRecord(
        captured_at=datetime.now(timezone.utc).isoformat(),
        brier=scores["brier"],
        log_loss=scores["log_loss"],
        reliability_bins=reliability,
        ab_test=ab_test(live) if not live.empty else {"n": 0, "base": {}, "adjusted": {}, "delta": {}},
        source_tier_ablation=ablate_by_source(live) if not live.empty else {},
    )
    return persist_metrics(record, output_dir=output_dir)
