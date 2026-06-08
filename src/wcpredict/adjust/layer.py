"""Pre-match adjustment layer applied to match-model output at inference."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable
import math

from wcpredict.adjust.magnitude import magnitude
from wcpredict.adjust.provenance import AdjustmentRecord, ProvenanceLog
from wcpredict.llm.schema import Channel, Extraction, SourceType, Status
from wcpredict.model.match_model import GoalsDist
from wcpredict.sim.sampler import MatchAdjustment


@dataclass(frozen=True)
class LayerResult:
    lambda_a: float
    lambda_b: float
    lambda_delta_a: float
    lambda_delta_b: float
    dispersion_a: float
    dispersion_b: float
    provenance: list[AdjustmentRecord]

    def sampler_adjustment(self) -> MatchAdjustment:
        return MatchAdjustment(
            lambda_delta_home=self.lambda_delta_a,
            lambda_delta_away=self.lambda_delta_b,
            dispersion=max(self.dispersion_a, self.dispersion_b),
        )


def _decay(as_of: datetime, kickoff: datetime, now: datetime, settings: dict[str, Any]) -> float:
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    half_life = float(settings.get("adjustment", {}).get("decay", {}).get("news_half_life_days", 2.0))
    age_days = max((now - as_of).total_seconds() / 86400.0, 0.0)
    age_factor = 0.5 ** (age_days / max(half_life, 1e-6))
    days_to_kickoff = max((kickoff - now).total_seconds() / 86400.0, 0.0)
    proximity = 1.0 / (1.0 + days_to_kickoff / max(half_life, 1e-6))
    return float(max(0.0, min(1.0, age_factor * proximity)))


def _importance_tier(ex: Extraction) -> str:
    if ex.role.value == "rotation":
        return "rotation"
    return "key_starter"


def _has_official_lineup(extractions: Iterable[Extraction]) -> bool:
    return any(ex.source_type == SourceType.OFFICIAL for ex in extractions)


def apply_adjustments(
    dist: GoalsDist,
    team_a: str,
    team_b: str,
    extractions: list[Extraction],
    *,
    kickoff: datetime,
    now: datetime,
    settings: dict[str, Any],
    provenance_log: ProvenanceLog | None = None,
) -> LayerResult:
    adj_cfg = settings.get("adjustment", {})
    clip_cfg = adj_cfg.get("clip", {})
    conf_threshold = float(adj_cfg.get("confidence_threshold", 0.5))
    availability_cap = float(clip_cfg.get("availability_max_abs_delta", 0.40))
    dispersion_cap = float(clip_cfg.get("lineup_dispersion_max", 0.50))
    rule_version = str(settings.get("magnitude", {}).get("rule_version", "v1"))

    official_lineup = _has_official_lineup(extractions)
    delta = {team_a: 0.0, team_b: 0.0}
    dispersion = {team_a: 0.0, team_b: 0.0}
    records: list[AdjustmentRecord] = []

    for ex in extractions:
        if ex.team not in delta:
            continue
        decay_factor = _decay(ex.as_of, kickoff, now, settings)
        superseded = False
        raw_mag = 0.0
        applied = 0.0
        confidence = 0.0

        if official_lineup and ex.channel == Channel.LINEUP_CERTAINTY:
            superseded = True
        elif official_lineup and ex.channel == Channel.AVAILABILITY and ex.source_type != SourceType.OFFICIAL:
            superseded = True
        elif ex.channel == Channel.AVAILABILITY:
            raw_mag, confidence = magnitude(ex.status, _importance_tier(ex), 1.0, settings)
            applied = raw_mag * decay_factor
            if confidence >= conf_threshold:
                delta[ex.team] += applied
        elif ex.channel == Channel.LINEUP_CERTAINTY:
            source_conf = {
                SourceType.OFFICIAL: 0.95,
                SourceType.BEAT_REPORT: 0.70,
                SourceType.RUMOR: 0.45,
            }[ex.source_type]
            confidence = source_conf * decay_factor
            raw_mag = dispersion_cap * (1.0 if ex.status == Status.UNCONFIRMED else 0.5)
            applied = raw_mag * decay_factor
            if confidence >= conf_threshold:
                dispersion[ex.team] = max(dispersion[ex.team], applied)

        delta[ex.team] = max(-availability_cap, min(availability_cap, delta[ex.team]))
        dispersion[ex.team] = max(0.0, min(dispersion_cap, dispersion[ex.team]))
        rec = AdjustmentRecord(
            match_id=ex.match_id,
            team=ex.team,
            channel=ex.channel.value,
            status=ex.status.value,
            source_type=ex.source_type.value,
            raw_magnitude=float(raw_mag),
            applied_value=float(applied if not superseded else 0.0),
            confidence=float(confidence),
            decay_factor=float(decay_factor),
            superseded=bool(superseded),
            rule_version=rule_version,
        )
        records.append(rec)
        if provenance_log is not None:
            provenance_log.append(rec)

    da = max(-availability_cap, min(availability_cap, delta[team_a]))
    db = max(-availability_cap, min(availability_cap, delta[team_b]))
    return LayerResult(
        lambda_a=max(dist.lambda_a + da, 1e-4),
        lambda_b=max(dist.lambda_b + db, 1e-4),
        lambda_delta_a=da,
        lambda_delta_b=db,
        dispersion_a=dispersion[team_a],
        dispersion_b=dispersion[team_b],
        provenance=records,
    )
