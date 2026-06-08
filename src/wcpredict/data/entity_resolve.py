"""
Build-time entity resolution.

This module maps source-specific names to canonical config/teams.yaml names.
It is intentionally build-time only: model/, sim/, and eval/ must not import it.
Claude is optional and only used when an API key is present and the caller asks
for it; the default resolver is deterministic/offline for CI and data builds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import difflib
import json
import os

import yaml


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


def _key(value: str) -> str:
    return (
        value.lower()
        .replace(".", "")
        .replace("&", "and")
        .replace("'", "")
        .strip()
    )


@dataclass(frozen=True)
class ResolutionReport:
    resolved: dict[str, str]
    unmatched: list[str]


def load_canonical_teams(path: Path | str | None = None) -> list[str]:
    p = Path(path) if path else _project_root() / "config" / "teams.yaml"
    with open(p) as f:
        data = yaml.safe_load(f)
    return [team["name"] for team in data.get("teams", [])]


def make_resolver(
    canonical_names: Iterable[str],
    aliases: dict[str, str] | None = None,
) -> tuple[callable, ResolutionReport]:
    canonical = list(canonical_names)
    alias_map = {_key(name): name for name in canonical}
    for raw, target in (aliases or {}).items():
        alias_map[_key(raw)] = target

    resolved: dict[str, str] = {}
    unmatched: list[str] = []

    def resolve(name: str) -> str:
        if not isinstance(name, str):
            return str(name)
        raw = name.strip()
        k = _key(raw)
        if k in alias_map:
            out = alias_map[k]
            resolved[raw] = out
            return out
        for suffix in (" football association", " fa", " national team"):
            stripped = k.removesuffix(suffix)
            if stripped != k and stripped in alias_map:
                out = alias_map[stripped]
                resolved[raw] = out
                return out
        match = difflib.get_close_matches(k, alias_map.keys(), n=1, cutoff=0.92)
        if match:
            out = alias_map[match[0]]
            resolved[raw] = out
            return out
        if raw not in unmatched:
            unmatched.append(raw)
        return raw

    return resolve, ResolutionReport(resolved=resolved, unmatched=unmatched)


def resolve_batch(
    names: Iterable[str],
    canonical_names: Iterable[str],
    aliases: dict[str, str] | None = None,
) -> ResolutionReport:
    resolver, report = make_resolver(canonical_names, aliases)
    for name in names:
        resolver(name)
    return report


def resolve_with_claude(
    names: Iterable[str],
    canonical_names: Iterable[str],
    *,
    settings: dict[str, Any],
) -> ResolutionReport:
    """
    Ask Claude to map unresolved names to canonical names.

    Returns only mappings whose targets are in canonical_names. Invalid or
    unmatched targets are reported as unmatched rather than silently accepted.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for Claude entity resolution")
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package is required for Claude entity resolution") from exc

    canonical = list(canonical_names)
    prompt = (
        "Map each source name to exactly one canonical team name, or null if uncertain.\n"
        "Return ONLY a JSON object of {source_name: canonical_name_or_null}.\n"
        f"Canonical names: {json.dumps(canonical)}\n"
        f"Source names: {json.dumps(list(names))}"
    )
    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=settings.get("llm", {}).get("model_cheap", "claude-haiku-4-5-20251001"),
        max_tokens=int(settings.get("llm", {}).get("max_tokens", 1024)),
        temperature=float(settings.get("llm", {}).get("temperature", 0.0)),
        messages=[{"role": "user", "content": prompt}],
    )
    text = "\n".join(getattr(block, "text", "") for block in getattr(message, "content", []))
    data = json.loads(text[text.find("{"):text.rfind("}") + 1])
    valid = set(canonical)
    resolved: dict[str, str] = {}
    unmatched: list[str] = []
    for raw in names:
        target = data.get(raw)
        if target in valid:
            resolved[str(raw)] = str(target)
        else:
            unmatched.append(str(raw))
    return ResolutionReport(resolved=resolved, unmatched=unmatched)
