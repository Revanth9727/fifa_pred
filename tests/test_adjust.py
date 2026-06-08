from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from wcpredict.adjust.layer import apply_adjustments
from wcpredict.adjust.provenance import ProvenanceLog
from wcpredict.llm.schema import Extraction
from wcpredict.model.match_model import GoalsDist


SETTINGS = {
    "adjustment": {
        "confidence_threshold": 0.5,
        "clip": {
            "availability_max_abs_delta": 0.20,
            "lineup_dispersion_max": 0.50,
        },
        "decay": {"news_half_life_days": 2.0, "kickoff_proximity_boost": True},
    },
    "magnitude": {
        "shrinkage": 0.5,
        "importance_tiers": {"talisman": 1.0, "key_starter": 0.6, "rotation": 0.2},
        "rule_version": "v-test",
    },
}


NOW = datetime(2026, 6, 1, 12, tzinfo=timezone.utc)
KICKOFF = datetime(2026, 6, 1, 20, tzinfo=timezone.utc)


def _ex(**kwargs) -> Extraction:
    base = {
        "match_id": "M1",
        "team": "A",
        "channel": "availability",
        "player_id": "p-alpha",
        "status": "ruled_out",
        "role": "starter",
        "source_type": "beat_report",
        "evidence_span": "Player is unavailable",
        "as_of": NOW.isoformat(),
    }
    base.update(kwargs)
    return Extraction.from_dict(base)


def test_schema_rejects_numeric_fields():
    with pytest.raises(ValueError):
        Extraction.from_dict({
            "match_id": "M1", "team": "A", "channel": "availability",
            "player_id": "p", "status": "ruled_out", "role": "starter",
            "source_type": "official", "evidence_span": "out",
            "as_of": NOW.isoformat(), "delta": 0.2,
        })


def test_schema_rejects_magnitude_language_in_evidence():
    with pytest.raises(ValueError):
        _ex(evidence_span="This is a 20 percent boost")


def test_channel2_never_moves_mean_lambda():
    dist = GoalsDist(lambda_a=1.4, lambda_b=1.1, dependence=-0.1)
    res = apply_adjustments(
        dist, "A", "B",
        [_ex(channel="lineup_certainty", status="unconfirmed", source_type="beat_report")],
        kickoff=KICKOFF, now=NOW, settings=SETTINGS,
    )
    assert res.lambda_a == pytest.approx(dist.lambda_a)
    assert res.lambda_b == pytest.approx(dist.lambda_b)
    assert res.lambda_delta_a == 0.0
    assert res.dispersion_a > 0.0


def test_availability_delta_is_clipped():
    dist = GoalsDist(lambda_a=1.4, lambda_b=1.1, dependence=-0.1)
    items = [_ex(player_id=f"p{i}", source_type="official") for i in range(5)]
    res = apply_adjustments(dist, "A", "B", items, kickoff=KICKOFF, now=NOW, settings=SETTINGS)
    assert res.lambda_delta_a == pytest.approx(-0.20)
    assert res.lambda_a == pytest.approx(1.20)


def test_low_confidence_rumor_is_noop():
    dist = GoalsDist(lambda_a=1.4, lambda_b=1.1, dependence=-0.1)
    res = apply_adjustments(
        dist, "A", "B",
        [_ex(channel="lineup_certainty", status="unconfirmed", source_type="rumor")],
        kickoff=KICKOFF, now=NOW, settings=SETTINGS,
    )
    assert res.dispersion_a == 0.0
    assert res.lambda_a == pytest.approx(dist.lambda_a)


def test_decay_reduces_magnitude_with_age_and_distance():
    dist = GoalsDist(lambda_a=1.4, lambda_b=1.1, dependence=-0.1)
    fresh = apply_adjustments(dist, "A", "B", [_ex(as_of=NOW.isoformat())],
                              kickoff=KICKOFF, now=NOW, settings=SETTINGS)
    old_far_now = NOW - timedelta(days=5)
    old = apply_adjustments(
        dist, "A", "B",
        [_ex(as_of=old_far_now.isoformat())],
        kickoff=KICKOFF + timedelta(days=10), now=NOW, settings=SETTINGS,
    )
    assert abs(old.lambda_delta_a) < abs(fresh.lambda_delta_a)


def test_supersession_collapses_channel2_and_restricts_channel1():
    dist = GoalsDist(lambda_a=1.4, lambda_b=1.1, dependence=-0.1)
    items = [
        _ex(channel="lineup_certainty", status="unconfirmed", source_type="beat_report"),
        _ex(status="doubtful", source_type="beat_report"),
        _ex(status="ruled_out", source_type="official", player_id="confirmed-out"),
    ]
    res = apply_adjustments(dist, "A", "B", items, kickoff=KICKOFF, now=NOW, settings=SETTINGS)
    assert res.dispersion_a == 0.0
    assert res.lambda_delta_a < 0.0
    assert len([r for r in res.provenance if r.superseded]) == 2
    assert all(r.applied_value == 0.0 for r in res.provenance if r.superseded)


def test_provenance_records_are_written(tmp_path):
    log = ProvenanceLog()
    dist = GoalsDist(lambda_a=1.4, lambda_b=1.1, dependence=-0.1)
    res = apply_adjustments(
        dist, "A", "B", [_ex(source_type="official")],
        kickoff=KICKOFF, now=NOW, settings=SETTINGS, provenance_log=log,
    )
    assert len(res.provenance) == 1
    assert len(log.records) == 1
    out = tmp_path / "prov.jsonl"
    log.write_jsonl(out)
    row = json.loads(out.read_text().strip())
    assert row["rule_version"] == "v-test"
    assert row["channel"] == "availability"
