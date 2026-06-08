"""
Output-time narration grounded in produced probabilities and attributions.

This module never runs in training or eval. Claude may only turn provided
numbers into prose; generated text is rejected if it contains numbers that were
not present in the supplied outputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import re

import yaml


_NUMBER_RE = re.compile(r"(?<![A-Za-z])-?\d+(?:\.\d+)?%?")


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


def _collect_numbers(obj: Any) -> set[str]:
    text = json.dumps(obj, sort_keys=True, default=str)
    nums = set(_NUMBER_RE.findall(text))
    more = set()
    for n in nums:
        if n.endswith("%"):
            more.add(n[:-1])
        else:
            try:
                more.add(f"{float(n) * 100:.1f}%")
                more.add(f"{float(n):.1f}")
                more.add(f"{float(n):.2f}")
            except ValueError:
                pass
    return nums | more


def validate_grounding(text: str, allowed_data: dict[str, Any]) -> None:
    allowed = _collect_numbers(allowed_data)
    invented = [n for n in _NUMBER_RE.findall(text) if n not in allowed]
    if invented:
        raise ValueError(f"Narration invented unsupported numbers: {invented}")


def build_prompt(probabilities: dict[str, Any], attributions: dict[str, Any]) -> str:
    return (
        "Write a concise rationale grounded strictly in the supplied outputs.\n"
        "Do not add facts, injuries, rankings, odds, or numbers not present here.\n"
        "Use only these probabilities and attributions.\n\n"
        f"Probabilities JSON:\n{json.dumps(probabilities, sort_keys=True)}\n\n"
        f"Attributions JSON:\n{json.dumps(attributions, sort_keys=True)}"
    )


def narrate(
    probabilities: dict[str, Any],
    attributions: dict[str, Any],
    settings: dict[str, Any] | None = None,
) -> str:
    cfg = _load_settings(settings)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for narration")
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package is required for narration") from exc

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=cfg.get("llm", {}).get("model", "claude-sonnet-4-6"),
        max_tokens=int(cfg.get("llm", {}).get("max_tokens", 1024)),
        temperature=float(cfg.get("llm", {}).get("temperature", 0.0)),
        messages=[{"role": "user", "content": build_prompt(probabilities, attributions)}],
    )
    text = "\n".join(getattr(block, "text", "") for block in getattr(message, "content", []))
    validate_grounding(text, {"probabilities": probabilities, "attributions": attributions})
    return text
