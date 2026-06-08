from __future__ import annotations

from types import SimpleNamespace
import ast
import sys
import types

import pytest

from wcpredict.data.entity_resolve import load_canonical_teams, make_resolver, resolve_with_claude
from wcpredict.explain.narrate import narrate, validate_grounding


def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch, response_text: str):
    class _Messages:
        def create(self, **kwargs):
            return SimpleNamespace(content=[SimpleNamespace(text=response_text)])

    class _Anthropic:
        def __init__(self, api_key: str):
            self.messages = _Messages()

    module = types.ModuleType("anthropic")
    module.Anthropic = _Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def test_entity_resolver_offline_aliases_and_unmatched():
    resolver, report = make_resolver(["United States", "Côte d'Ivoire"], {
        "usa": "United States",
        "ivory coast": "Côte d'Ivoire",
    })
    assert resolver("USA") == "United States"
    assert resolver("Ivory Coast") == "Côte d'Ivoire"
    assert resolver("Atlantis") == "Atlantis"
    assert "Atlantis" in report.unmatched


def test_entity_resolve_with_mocked_claude(monkeypatch):
    _install_fake_anthropic(monkeypatch, '{"USA": "United States", "Atlantis": null}')
    report = resolve_with_claude(
        ["USA", "Atlantis"],
        ["United States", "Mexico"],
        settings={"llm": {"model_cheap": "claude-test", "max_tokens": 128, "temperature": 0}},
    )
    assert report.resolved == {"USA": "United States"}
    assert report.unmatched == ["Atlantis"]


def test_narration_rejects_invented_numbers():
    with pytest.raises(ValueError):
        validate_grounding(
            "Team A has a 33% chance and a made-up 99% scenario.",
            {"probabilities": {"Team A": {"champion": 0.33}}, "attributions": {}},
        )


def test_narrate_with_mocked_claude_uses_only_given_numbers(monkeypatch):
    _install_fake_anthropic(
        monkeypatch,
        "Team A's 33.0% title chance is supported by the +0.12 Elo contribution.",
    )
    text = narrate(
        {"Team A": {"champion": 0.33}},
        {"Team A": {"elo": 0.12}},
        settings={"llm": {"model": "claude-test", "max_tokens": 128, "temperature": 0}},
    )
    assert "33.0%" in text
    assert "+0.12" in text


def test_training_sim_eval_do_not_import_side_modules():
    root = __import__("pathlib").Path(__file__).resolve().parents[1] / "src" / "wcpredict"
    checked = []
    violations = []
    for folder in ("model", "sim", "eval"):
        for path in (root / folder).rglob("*.py"):
            checked.append(path)
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if "data.entity_resolve" in name or "explain.narrate" in name:
                        violations.append(f"{path}: imports {name}")
    assert checked
    assert not violations, "\n".join(violations)


def test_canonical_teams_loads_48_names():
    assert len(load_canonical_teams()) == 48
