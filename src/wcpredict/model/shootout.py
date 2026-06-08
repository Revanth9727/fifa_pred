"""
model/shootout.py — deliberately simple penalty-shootout sub-model.

Design principle (arch.md): near-coinflip with a small, heavily-shrunk
strength lean. A "deep" shootout model manufactures fake signal; simulator
sensitivity to it is low.  Coefficient is config-driven, not hardcoded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


def _load_settings(path: Path | str | None = None) -> dict[str, Any]:
    root = _project_root()
    p = Path(path) if path else root / "config" / "settings.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


def resolve_shootout(
    team_a: str,
    team_b: str,
    strength_diff: float,
    rng: np.random.Generator,
    coef: float | None = None,
    settings: dict[str, Any] | None = None,
) -> str:
    """
    Resolve a penalty shootout.  Returns the winner's name.

    Parameters
    ----------
    team_a, team_b:
        Names of the two teams.
    strength_diff:
        Signed strength advantage of team_a over team_b.  Typically
        elo_diff / 400 (already small).  The coefficient shrinks it further.
    rng:
        numpy Generator — deterministic given a seed (ai_rules.md: every
        stochastic component takes an explicit seed).
    coef:
        Override the config's ``simulation.shootout_strength_coef``.
        If None, the value is read from settings.yaml.
    settings:
        Pre-loaded settings dict; loaded from disk if None and coef is None.

    Notes
    -----
    P(team_a wins) = clip(0.5 + coef * strength_diff, 0.05, 0.95)
    The clip prevents pathological inputs from making the result deterministic.
    """
    if coef is None:
        if settings is None:
            settings = _load_settings()
        coef = float(settings.get("simulation", {}).get("shootout_strength_coef", 0.10))

    p_a = float(np.clip(0.5 + coef * strength_diff, 0.05, 0.95))
    return team_a if rng.random() < p_a else team_b
