"""Piano motorio — intenzione prima della produzione vocale."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MotorPlan:
    text: str
    syllables: list[str]
    neuron_ids: list[int] = field(default_factory=list)
    from_motor: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "syllables": self.syllables,
            "from_motor": self.from_motor,
        }
