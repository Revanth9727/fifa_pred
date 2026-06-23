from __future__ import annotations

import pandas as pd

from wcpredict.model.match_model import GoalsDist
from wcpredict.sim.sampler import MatchResult
from wcpredict.web.service import PredictionService


def test_group_fixtures_cover_72_matches():
    service = PredictionService()
    assert len(service.group_fixtures()) == 72


def test_group_fixtures_have_stable_match_ids():
    service = PredictionService()
    fixtures = service.group_fixtures()
    assert fixtures[0]["match_id"] == "GA1"
    assert fixtures[0]["stage"] == "Group"
    assert fixtures[-1]["match_id"] == "GL6"


def test_result_lookup_accepts_reversed_fixture_order(tmp_path):
    service = PredictionService()
    service.results_path = tmp_path / "results.csv"
    service.results_path.write_text(
        "team_home,team_away,goals_home,goals_away\n"
        "South Africa,Mexico,1,2\n"
    )
    assert service._result_for("Mexico", "South Africa") == (2, 1)


def test_live_state_reset_and_simulate_one_match(tmp_path):
    service = PredictionService()
    service.live_state_path = tmp_path / "live_state.json"
    state = service.reset_live_state()
    assert len(state["matches"]) == 72
    assert state["matches"][0]["available"] is True
    out = service.simulate_live_match("GA1")
    match = next(item for item in out["matches"] if item["match_id"] == "GA1")
    assert match["locked"] is True
    assert match["home_score"] is not None
    assert out["standings"]["A"][0]["played"] >= 0


def _lock_stage_home_wins(state: dict, stage: str) -> None:
    for match in state["matches"]:
        if match["stage"] != stage or match.get("locked"):
            continue
        match.update({
            "home_score": 1,
            "away_score": 0,
            "winner": match["home"],
            "went_to_et": False,
            "went_to_shootout": False,
            "locked": True,
            "status": "final",
        })


def test_live_state_advances_from_groups_to_champion(tmp_path):
    service = PredictionService()
    service.live_state_path = tmp_path / "live_state.json"
    state = service.reset_live_state()

    _lock_stage_home_wins(state, "Group")
    state = service._advance_live_state(state)
    assert len([m for m in state["matches"] if m["stage"] == "Round of 32"]) == 16

    for stage, next_stage, expected_count in [
        ("Round of 32", "Round of 16", 8),
        ("Round of 16", "Quarter-final", 4),
        ("Quarter-final", "Semi-final", 2),
    ]:
        _lock_stage_home_wins(state, stage)
        state = service._advance_live_state(state)
        assert len([m for m in state["matches"] if m["stage"] == next_stage]) == expected_count

    _lock_stage_home_wins(state, "Semi-final")
    state = service._advance_live_state(state)
    assert len([m for m in state["matches"] if m["stage"] == "Third-place playoff"]) == 1
    assert len([m for m in state["matches"] if m["stage"] == "Final"]) == 1

    _lock_stage_home_wins(state, "Third-place playoff")
    _lock_stage_home_wins(state, "Final")
    state = service._advance_live_state(state)
    assert state["status"] == "complete"
    assert state["champion"]
    assert len(state["matches"]) == 104


def test_dashboard_uses_live_tournament_state_not_projected_slots(tmp_path):
    service = PredictionService()
    service.live_state_path = tmp_path / "live_state.json"
    service.reset_live_state()

    dashboard = service.dashboard()

    assert dashboard["summary"]["matches_total"] == 72
    assert all(match["stage"] == "Group" for match in dashboard["matches"])


def test_real_result_sync_locks_live_state_match(tmp_path):
    service = PredictionService()
    service.live_state_path = tmp_path / "live_state.json"
    service.processed_dir = tmp_path
    service.outputs_dir = tmp_path / "outputs"
    service.reset_live_state()
    fetched = pd.DataFrame([{
        "team_home": "Mexico",
        "team_away": "South Africa",
        "goals_home": 2,
        "goals_away": 1,
        "source": "espn",
    }])

    synced = service._sync_wikipedia_results_to_live_state(fetched)
    state = service.load_live_state()
    match = next(m for m in state["matches"] if m["match_id"] == "GA1")

    assert synced == 1
    assert match["locked"] is True
    assert match["home_score"] == 2
    assert match["away_score"] == 1


def test_knockout_shootout_details_are_saved(monkeypatch, tmp_path):
    service = PredictionService()
    service.live_state_path = tmp_path / "live_state.json"
    state = service.reset_live_state()
    _lock_stage_home_wins(state, "Group")
    state = service._advance_live_state(state)
    service.save_live_state(state)

    monkeypatch.setattr(
        service,
        "_match_probability_row",
        lambda **_: {
            "home_win": 0.4,
            "draw": 0.2,
            "away_win": 0.4,
            "lambda_home": 1.0,
            "lambda_away": 1.0,
        },
    )

    class FakeModel:
        def predict(self, _ctx):
            return GoalsDist(lambda_a=1.0, lambda_b=1.0, dependence=0.0)

    monkeypatch.setattr(service, "_load_model", lambda: FakeModel())
    monkeypatch.setattr(service, "_load_strengths", lambda: {})
    monkeypatch.setattr(service, "_team_features", lambda: {})

    import wcpredict.web.service as web_service

    monkeypatch.setattr(
        web_service,
        "sample_match",
        lambda *_, **__: MatchResult(
            goals_home=1,
            goals_away=1,
            et_goals_home=0,
            et_goals_away=0,
            penalties_home=5,
            penalties_away=4,
            went_to_et=True,
            went_to_shootout=True,
            winner="home",
        ),
    )

    out = service.simulate_live_match("R321")
    match = next(item for item in out["matches"] if item["match_id"] == "R321")

    assert match["home_score"] == 1
    assert match["away_score"] == 1
    assert match["winner"] == match["home"]
    assert match["went_to_shootout"] is True
    assert match["penalties_home"] == 5
    assert match["penalties_away"] == 4
    assert "on penalties" in match["result_note"]
