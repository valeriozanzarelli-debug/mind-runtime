"""Sensation circuits — shared feeling, resonance between stimuli."""

from __future__ import annotations

from mind.types import Circuit, Cue, Fragment


class SensationRegistry:
    def __init__(self, circuits: list[Circuit] | None = None) -> None:
        self._circuits: dict[str, Circuit] = {c.id: c for c in (circuits or [])}

    def add(self, circuit: Circuit) -> None:
        self._circuits[circuit.id] = circuit

    def get(self, circuit_id: str) -> Circuit | None:
        return self._circuits.get(circuit_id)

    def circuits_for_fragments(self, fragments: list[Fragment]) -> list[str]:
        seen: list[str] = []
        for f in fragments:
            if f.sensation_id and f.sensation_id not in seen:
                seen.append(f.sensation_id)
        return seen

    def resonate(self, cue_a: str, cue_b: str, *, jaccard_threshold: float = 0.2) -> bool:
        """K reminds X — token overlap as lightweight resonance proxy."""
        ta = set(_tokens(cue_a))
        tb = set(_tokens(cue_b))
        if not ta or not tb:
            return False
        inter = len(ta & tb)
        union = len(ta | tb)
        return (inter / union) >= jaccard_threshold

    def same_circuit_from_cues(
        self, cue_new: str, cue_past: str, fragments_past: list[Fragment]
    ) -> str | None:
        """If new cue resonates with past cue, return shared sensation circuit."""
        if not self.resonate(cue_new, cue_past):
            return None
        circuits = self.circuits_for_fragments(fragments_past)
        return circuits[0] if circuits else None


def _tokens(text: str) -> list[str]:
    return [t for t in text.lower().split() if len(t) > 2]
