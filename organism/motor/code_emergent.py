"""Codice emergente — token appresi da coppie insegnate, non snippet fissi."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodeTokenLearner:
    _weights: dict[str, float] = field(default_factory=dict)

    def absorb(self, code: str, *, boost: float = 1.0) -> list[str]:
        tokens = tokenize_code(code)
        for tok in tokens:
            self._weights[tok] = self._weights.get(tok, 0.0) + boost
        return tokens

    @property
    def count(self) -> int:
        return len(self._weights)

    def to_dict(self) -> dict[str, Any]:
        return {"weights": dict(self._weights)}

    def load_dict(self, data: dict[str, Any]) -> None:
        self._weights = {str(k): float(v) for k, v in data.get("weights", {}).items()}


def tokenize_code(code: str) -> list[str]:
    return [t for t in re.findall(r"\w+|[^\w\s]", code) if t.strip()]


class EmergentCodeMotor:
    def __init__(self) -> None:
        self.tokens = CodeTokenLearner()
        self._learned_snippets: dict[str, str] = {}

    def learn_snippet(self, trigger: str, code: str) -> None:
        from organism.teaching.dialogue import normalize_text

        self._learned_snippets[normalize_text(trigger)] = code.strip()
        self.tokens.absorb(code, boost=1.2)

    def produce(self, trigger: str) -> str:
        from organism.teaching.dialogue import normalize_text

        key = normalize_text(trigger)
        if key in self._learned_snippets:
            return self._learned_snippets[key]
        return self.assemble_program(trigger)

    def assemble_program(self, request: str) -> str:
        """Assembla programma da token appresi — fallback emergente."""
        from organism.teaching.dialogue import normalize_text

        key = normalize_text(request)
        for trig, code in self._learned_snippets.items():
            if trig in key or key in trig:
                return code
        weights = sorted(self.tokens._weights.items(), key=lambda x: -x[1])
        top = [t for t, _ in weights[:40]]
        if len(top) < 4:
            return ""
        if "print" in top or "def" in top:
            lines: list[str] = []
            if "def" in top:
                fn = next((t for t in top if t.isalpha() and t not in ("def", "return", "if")), "azione")
                lines.append(f"def {fn}():")
                lines.append("    pass")
                lines.append(f"print({fn}())")
            else:
                msg = next((t for t in top if t.isalpha() and t not in ("print",)), "ciao")
                lines.append(f"print('{msg}')")
            return "\n".join(lines)
        return ""
