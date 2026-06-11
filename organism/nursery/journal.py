"""Thought journal — ogni ciclo come 'pensiero' osservabile."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThoughtEntry:
    cycle: int
    phase: str
    timestamp: float
    input_summary: str
    thoughts: list[str]  # symbols stream — il "flusso di coscienza"
    mind_action: str | None
    mind_fragments: list[str]
    expression: str
    learning: dict[str, Any] | None
    brain: dict[str, Any]
    new_fragment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "phase": self.phase,
            "timestamp": self.timestamp,
            "input": self.input_summary,
            "thoughts": self.thoughts,
            "action": self.mind_action,
            "fragments": self.mind_fragments,
            "expression": self.expression,
            "learning": self.learning,
            "brain": self.brain,
            "new_fragment": self.new_fragment,
        }


class ThoughtJournal:
    def __init__(self) -> None:
        self.entries: list[ThoughtEntry] = []
        self.birth_record: dict[str, Any] | None = None

    def log_birth(self, snapshot: dict[str, Any]) -> None:
        self.birth_record = {
            "timestamp": time.time(),
            "message": "DNA dispiegato — struttura neurale generata",
            **snapshot,
        }
        self.entries.clear()

    def record(
        self,
        *,
        cycle: int,
        phase: str,
        input_data: dict[str, Any],
        thought_symbols: list[str],
        mind_action: str | None,
        mind_fragments: list[str],
        expression_text: str,
        learning: dict[str, Any] | None,
        brain_snapshot: dict[str, Any],
        new_fragment: str | None = None,
    ) -> ThoughtEntry:
        summary = _summarize_input(input_data)
        entry = ThoughtEntry(
            cycle=cycle,
            phase=phase,
            timestamp=time.time(),
            input_summary=summary,
            thoughts=list(thought_symbols),
            mind_action=mind_action,
            mind_fragments=mind_fragments,
            expression=expression_text,
            learning=learning,
            brain=brain_snapshot,
            new_fragment=new_fragment,
        )
        self.entries.append(entry)
        return entry

    def recent(self, n: int = 30) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.entries[-n:]]

    def stream_text(self, n: int = 15) -> list[str]:
        """Human-readable thought lines for the UI."""
        lines: list[str] = []
        for e in self.entries[-n:]:
            lines.append(f"[#{e.cycle}|{e.phase}] stimolo: {e.input_summary}")
            for t in e.thoughts:
                lines.append(f"  → {t}")
            if e.mind_action:
                lines.append(f"  ⚡ azione: {e.mind_action}")
            if e.expression:
                lines.append(f"  💬 {e.expression[:120]}")
            if e.new_fragment:
                lines.append(f"  🧬 nuovo frammento: {e.new_fragment}")
        return lines


def _summarize_input(data: dict[str, Any]) -> str:
    if "text" in data:
        return str(data["text"])[:80]
    if "shapes" in data:
        return f"visione: {data['shapes'][:60]}"
    if "tone_hz" in data:
        return f"udito: {data['tone_hz']}Hz"
    if "image" in data:
        return "visione: immagine"
    return str(data)[:60]
