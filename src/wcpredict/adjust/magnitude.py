"""Magnitude mapping for Phase-4 availability adjustments."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

from wcpredict.llm.schema import Status


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


def _load_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    if settings is not None:
        return settings
    with open(_project_root() / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


_BASE_STATUS_EFFECT = {
    Status.RULED_OUT: (-0.24, 0.90),
    Status.DOUBTFUL: (-0.14, 0.65),
    Status.FIT: (0.0, 0.85),
    Status.UNCONFIRMED: (0.0, 0.35),
}


def magnitude(
    status: str | Status,
    importance_tier: str,
    recency: float = 1.0,
    settings: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """
    Return (signed_delta, confidence).

    The signed delta is an attacking mean-lambda adjustment for the affected
    team. It comes from a small club-football prior, config shrinkage, tier
    scaling, and the availability clip. LLM output never supplies this number.
    """
    cfg = _load_settings(settings)
    st = Status(str(status))
    base_delta, base_conf = _BASE_STATUS_EFFECT[st]
    shrinkage = float(cfg.get("magnitude", {}).get("shrinkage", 0.5))
    tier_mult = float(cfg.get("magnitude", {}).get("importance_tiers", {}).get(importance_tier, 0.2))
    recency_clamped = max(0.0, min(float(recency), 1.0))
    raw_delta = base_delta * shrinkage * tier_mult * recency_clamped
    cap = float(cfg.get("adjustment", {}).get("clip", {}).get("availability_max_abs_delta", 0.40))
    delta = max(-cap, min(cap, raw_delta))
    confidence = max(0.0, min(1.0, base_conf * recency_clamped))
    return float(delta), float(confidence)
