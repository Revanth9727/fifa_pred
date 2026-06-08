from __future__ import annotations

import pytest

from wcpredict.cli import _counts_to_probabilities, _parse_results, build_parser, main


def test_cli_exposes_phase7_subcommands():
    parser = build_parser()
    help_text = parser.format_help()
    for command in ["build-data", "fit", "simulate", "evaluate", "update"]:
        assert command in help_text


def test_update_requires_results_argument():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["update"])


def test_counts_to_probabilities_includes_all_stage_columns():
    groups = {"A": ["A1", "A2", "A3", "A4"]}
    counts = {
        "A1": {"champion": 3, "finalist": 5},
        "A2": {"group_exit": 10},
    }
    out = _counts_to_probabilities(counts, n_runs=10, groups=groups)
    assert list(out["team"]) == ["A1", "A2", "A3", "A4"]
    assert out.loc[out["team"] == "A1", "p_champion"].item() == pytest.approx(0.3)
    assert out.loc[out["team"] == "A1", "p_finalist"].item() == pytest.approx(0.5)
    assert out.loc[out["team"] == "A2", "p_group_exit"].item() == pytest.approx(1.0)


def test_parse_results_accepts_match_table_schema(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("team_home,team_away,goals_home,goals_away\nA1,A2,2,1\n")
    out = _parse_results(path)
    assert out == {("A1", "A2"): (2, 1)}


def test_main_returns_error_for_missing_market_file(tmp_path, capsys):
    code = main(["evaluate", "--market-data", str(tmp_path / "missing.csv")])
    assert code == 1
    assert "Market data not found" in capsys.readouterr().err
