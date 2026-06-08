"""Adjustment provenance records used by Phase-5 ablations."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class AdjustmentRecord:
    match_id: str
    team: str
    channel: str
    status: str
    source_type: str
    raw_magnitude: float
    applied_value: float
    confidence: float
    decay_factor: float
    superseded: bool
    rule_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProvenanceLog:
    def __init__(self) -> None:
        self.records: list[AdjustmentRecord] = []

    def append(self, record: AdjustmentRecord) -> None:
        self.records.append(record)

    def write_jsonl(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            for rec in self.records:
                f.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")
