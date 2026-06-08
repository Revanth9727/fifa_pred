from __future__ import annotations

import pytest

from wcpredict.adjust.magnitude import magnitude


SETTINGS = {
    "adjustment": {"clip": {"availability_max_abs_delta": 0.08}},
    "magnitude": {
        "shrinkage": 0.5,
        "importance_tiers": {"talisman": 1.0, "key_starter": 0.6, "rotation": 0.2},
    },
}


def test_magnitude_clips_after_shrinkage_and_tier():
    delta, conf = magnitude("ruled_out", "talisman", recency=1.0, settings=SETTINGS)
    assert delta == pytest.approx(-0.08)
    assert 0.0 <= conf <= 1.0


def test_importance_tiers_scale_magnitude():
    talisman, _ = magnitude("ruled_out", "talisman", recency=1.0, settings=SETTINGS)
    starter, _ = magnitude("ruled_out", "key_starter", recency=1.0, settings=SETTINGS)
    rotation, _ = magnitude("ruled_out", "rotation", recency=1.0, settings=SETTINGS)
    assert abs(talisman) > abs(starter) > abs(rotation)


def test_fit_and_unconfirmed_are_noop_deltas():
    fit_delta, fit_conf = magnitude("fit", "talisman", recency=1.0, settings=SETTINGS)
    unc_delta, unc_conf = magnitude("unconfirmed", "talisman", recency=1.0, settings=SETTINGS)
    assert fit_delta == 0.0
    assert unc_delta == 0.0
    assert fit_conf > unc_conf


def test_recency_reduces_delta_and_confidence():
    fresh_delta, fresh_conf = magnitude("doubtful", "key_starter", recency=1.0, settings=SETTINGS)
    stale_delta, stale_conf = magnitude("doubtful", "key_starter", recency=0.25, settings=SETTINGS)
    assert abs(stale_delta) < abs(fresh_delta)
    assert stale_conf < fresh_conf
