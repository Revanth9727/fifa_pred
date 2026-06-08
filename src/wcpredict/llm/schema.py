"""
Typed LLM extraction schema.

Claude is allowed to emit structure only: categorical fields, identifiers,
timestamps, and evidence text. Magnitudes, probabilities, coefficients, and
lambda deltas belong in adjust/magnitude.py, never in this schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
import re


class Channel(StrEnum):
    AVAILABILITY = "availability"
    LINEUP_CERTAINTY = "lineup_certainty"


class Status(StrEnum):
    RULED_OUT = "ruled_out"
    DOUBTFUL = "doubtful"
    FIT = "fit"
    UNCONFIRMED = "unconfirmed"


class Role(StrEnum):
    STARTER = "starter"
    ROTATION = "rotation"


class SourceType(StrEnum):
    OFFICIAL = "official"
    BEAT_REPORT = "beat_report"
    RUMOR = "rumor"


_MAGNITUDE_WORDS = re.compile(
    r"\b(?:probability|prob|chance|odds|price|line|market|percent|percentage|"
    r"coefficient|coef|lambda|xg|delta|boost|rating|edge)\b",
    re.IGNORECASE,
)


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError("as_of must be an ISO datetime string or datetime")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _reject_numeric_fields(data: dict[str, Any]) -> None:
    for key, value in data.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            raise ValueError(f"{key} must not be numeric")
        if key == "evidence_span" and isinstance(value, str) and _MAGNITUDE_WORDS.search(value):
            raise ValueError(f"{key} contains magnitude/probability language")


@dataclass(frozen=True)
class Extraction:
    match_id: str
    team: str
    channel: Channel
    player_id: str
    status: Status
    role: Role
    source_type: SourceType
    evidence_span: str
    as_of: datetime

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Extraction":
        required = {
            "match_id", "team", "channel", "player_id", "status", "role",
            "source_type", "evidence_span", "as_of",
        }
        extra = set(data) - required
        missing = required - set(data)
        if missing or extra:
            raise ValueError(f"Extraction fields mismatch: missing={sorted(missing)} extra={sorted(extra)}")
        _reject_numeric_fields(data)
        return cls(
            match_id=str(data["match_id"]),
            team=str(data["team"]),
            channel=Channel(str(data["channel"])),
            player_id=str(data["player_id"]),
            status=Status(str(data["status"])),
            role=Role(str(data["role"])),
            source_type=SourceType(str(data["source_type"])),
            evidence_span=str(data["evidence_span"]),
            as_of=_parse_datetime(data["as_of"]),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "match_id": self.match_id,
            "team": self.team,
            "channel": self.channel.value,
            "player_id": self.player_id,
            "status": self.status.value,
            "role": self.role.value,
            "source_type": self.source_type.value,
            "evidence_span": self.evidence_span,
            "as_of": self.as_of.isoformat(),
        }


def validate_extractions(items: list[dict[str, Any]] | list[Extraction]) -> list[Extraction]:
    out: list[Extraction] = []
    for item in items:
        out.append(item if isinstance(item, Extraction) else Extraction.from_dict(item))
    return out
