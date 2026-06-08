from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import pytest

from wcpredict.eval.calibration import (
    ab_test,
    ablate_by_source,
    append_live_match,
    devig_proportional,
    devig_shin,
    evaluate_and_persist,
    persist_metrics,
    reliability_diagram,
    score,
)


def test_proportional_devig_sums_to_one():
    out = devig_proportional([0.55, 0.30, 0.25])
    assert out.sum() == pytest.approx(1.0)
    assert np.all(out > 0)


def test_shin_devig_sums_to_one():
    out = devig_shin([0.55, 0.30, 0.25])
    assert out.sum() == pytest.approx(1.0)
    assert np.all(out > 0)


def test_score_matches_toy_values():
    model = [0.5, 0.3, 0.2]
    market = [0.4, 0.4, 0.2]
    s = score(model, market)
    assert s["brier"] == pytest.approx(((0.5 - 0.4) ** 2 + (0.3 - 0.4) ** 2) / 3)
    assert s["log_loss"] == pytest.approx(-(0.4 * np.log(0.5) + 0.4 * np.log(0.3) + 0.2 * np.log(0.2)))


def test_reliability_diagram_bins():
    preds = pd.DataFrame({"model_prob": [0.1, 0.2, 0.8, 0.9], "outcome": [0, 0, 1, 1]})
    rel = reliability_diagram(preds, n_bins=2)
    assert rel["n"].sum() == 4
    assert set(rel.columns) == {"bin", "n", "mean_pred", "observed_rate"}


def test_metrics_persist_timestamped_record(tmp_path):
    path = persist_metrics(
        {
            "brier": 0.1,
            "log_loss": 1.2,
            "reliability_bins": [{"n": 2}],
            "ab_test": {"n": 0},
            "source_tier_ablation": {},
        },
        output_dir=tmp_path,
        timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    assert path.name == "metrics_20260601T000000Z.json"
    payload = json.loads(path.read_text())
    assert payload["brier"] == 0.1
    assert payload["reliability_bins"] == [{"n": 2}]


def test_live_ab_accumulates_per_match(tmp_path):
    path = tmp_path / "live.csv"
    df = append_live_match(
        path,
        match_id="M1",
        market_probs=[0.45, 0.30, 0.25],
        base_probs=[0.50, 0.25, 0.25],
        adjusted_probs=[0.46, 0.29, 0.25],
        source_type="official",
        captured_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    df = append_live_match(
        path,
        match_id="M2",
        market_probs=[0.30, 0.30, 0.40],
        base_probs=[0.35, 0.25, 0.40],
        adjusted_probs=[0.31, 0.29, 0.40],
        source_type="beat_report",
        captured_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
    )
    assert len(df) == 2
    result = ab_test(df)
    assert result["n"] == 2
    assert result["adjusted"]["brier"] < result["base"]["brier"]


def test_source_ablation_groups_live_rows(tmp_path):
    path = tmp_path / "live.csv"
    df = append_live_match(
        path,
        match_id="M1",
        market_probs=[0.45, 0.30, 0.25],
        base_probs=[0.50, 0.25, 0.25],
        adjusted_probs=[0.46, 0.29, 0.25],
        source_type="official",
    )
    df = append_live_match(
        path,
        match_id="M2",
        market_probs=[0.30, 0.30, 0.40],
        base_probs=[0.35, 0.25, 0.40],
        adjusted_probs=[0.31, 0.29, 0.40],
        source_type="rumor",
    )
    out = ablate_by_source(df)
    assert set(out) == {"official", "rumor"}
    assert out["official"]["n"] == 1


def test_evaluate_and_persist_writes_full_payload(tmp_path):
    preds = pd.DataFrame({"model_prob": [0.2, 0.8], "outcome": [0, 1]})
    live = pd.DataFrame({
        "market_home": [0.45], "market_draw": [0.30], "market_away": [0.25],
        "base_home": [0.50], "base_draw": [0.25], "base_away": [0.25],
        "adjusted_home": [0.46], "adjusted_draw": [0.29], "adjusted_away": [0.25],
        "source_type": ["official"],
    })
    path = evaluate_and_persist(
        preds,
        model_probs=[0.5, 0.3, 0.2],
        market_probs=[0.45, 0.30, 0.25],
        live_rows=live,
        output_dir=tmp_path,
        n_bins=2,
    )
    payload = json.loads(path.read_text())
    assert "brier" in payload
    assert "log_loss" in payload
    assert payload["ab_test"]["n"] == 1
    assert "official" in payload["source_tier_ablation"]
