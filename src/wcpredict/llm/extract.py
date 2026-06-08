"""
Claude extraction wrapper.

This module is the only Phase-4 LLM prediction touchpoint. It sends raw news
text to Claude and accepts only JSON that validates against llm/schema.py.
The response is structure only; no probabilities, coefficients, or deltas.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import re

import yaml

from wcpredict.llm.schema import Extraction, validate_extractions


_MARKET_TEXT = re.compile(r"\b(?:odds|bookmaker|sportsbook|market|line|spread|price|vig|favorite)\b", re.I)


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


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        elif isinstance(block, dict) and "text" in block:
            parts.append(str(block["text"]))
    return "\n".join(parts)


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start < 0 or end < start:
            raise
        data = json.loads(text[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("Claude extraction output must be a JSON array")
    if not all(isinstance(x, dict) for x in data):
        raise ValueError("Claude extraction output must contain JSON objects")
    return data


def build_prompt(raw_news: str, match_context: dict[str, Any]) -> str:
    return (
        "Extract only structured injury/team-news/lineup information.\n"
        "Return ONLY a JSON array. Each object must have exactly these fields:\n"
        "match_id, team, channel, player_id, status, role, source_type, evidence_span, as_of.\n"
        "Allowed channel: availability, lineup_certainty.\n"
        "Allowed status: ruled_out, doubtful, fit, unconfirmed.\n"
        "Allowed role: starter, rotation.\n"
        "Allowed source_type: official, beat_report, rumor.\n"
        "Do not output probabilities, percentages, coefficients, lambda deltas, ratings, odds, or magnitudes.\n"
        "If no valid extraction exists, return [].\n\n"
        f"Match context JSON:\n{json.dumps(match_context, default=str)}\n\n"
        f"Raw news text:\n{raw_news}"
    )


def extract(raw_news: str, match_context: dict[str, Any], settings: dict[str, Any] | None = None) -> list[Extraction]:
    if _MARKET_TEXT.search(raw_news):
        raise ValueError("Extraction input appears to contain market/odds text")
    cfg = _load_settings(settings)
    llm_cfg = cfg.get("llm", {})
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for Claude extraction")

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package is required for Claude extraction") from exc

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=llm_cfg.get("model", "claude-sonnet-4-6"),
        max_tokens=int(llm_cfg.get("max_tokens", 1024)),
        temperature=float(llm_cfg.get("temperature", 0.0)),
        messages=[{"role": "user", "content": build_prompt(raw_news, match_context)}],
    )
    return validate_extractions(_parse_json_array(_message_text(message)))
