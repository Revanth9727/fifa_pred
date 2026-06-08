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
