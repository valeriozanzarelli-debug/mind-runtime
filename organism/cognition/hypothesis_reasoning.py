"""Multi-hypothesis reasoning — credenze parallele con active inference."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hypothesis:
    claim: str
    confidence: float = 0.5
    evidence_for: int = 0
    evidence_against: int = 0
    updated_t: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "confidence": round(self.confidence, 3),
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
        }


class HypothesisEngine:
    """Mantiene ipotesi parallele incerte; aggiorna da osservazioni."""

    def __init__(self, *, max_hypotheses: int = 8) -> None:
        self.max_hypotheses = max_hypotheses
        self._hypotheses: dict[str, Hypothesis] = {}

    def propose(self, claim: str, *, prior: float = 0.5) -> Hypothesis:
        key = claim.lower().strip()[:80]
        if key in self._hypotheses:
            h = self._hypotheses[key]
            h.confidence = min(1.0, h.confidence + 0.05)
            h.updated_t = time.time()
            return h
        h = Hypothesis(claim=key, confidence=prior)
        self._hypotheses[key] = h
        if len(self._hypotheses) > self.max_hypotheses:
            weakest = min(self._hypotheses.values(), key=lambda x: x.confidence)
            del self._hypotheses[weakest.claim]
        return h

    def observe(self, *, supports: str = "", refutes: str = "", delta: float = 0.12) -> None:
        if supports:
            h = self.propose(supports)
            h.evidence_for += 1
            h.confidence = min(1.0, h.confidence + delta)
            h.updated_t = time.time()
        if refutes:
            h = self.propose(refutes)
            h.evidence_against += 1
            h.confidence = max(0.05, h.confidence - delta)
            h.updated_t = time.time()

    def parallel_beliefs(self, *, min_confidence: float = 0.3) -> list[Hypothesis]:
        return sorted(
            [h for h in self._hypotheses.values() if h.confidence >= min_confidence],
            key=lambda x: -x.confidence,
        )[:4]

    def test_action_hint(self) -> str:
        """Suggerisce azione per testare ipotesi più incerta tra le forti."""
        beliefs = self.parallel_beliefs(min_confidence=0.4)
        if len(beliefs) < 2:
            return ""
        # Ipotesi con confidence media — massima incertezza utile
        target = min(beliefs, key=lambda h: abs(h.confidence - 0.55))
        return f"verificare:{target.claim[:40]}"

    def replan_after_failure(self, action: str, cause: str = "") -> str:
        """Dopo fallimento azione — ipotesi alternativa."""
        self.observe(refutes=action, delta=0.15)
        alt = f"alternativa:{cause or action}"[:60]
        h = self.propose(alt, prior=0.55)
        return h.claim

    def theme_hints(self) -> list[str]:
        out: list[str] = []
        for h in self.parallel_beliefs()[:3]:
            for w in h.claim.replace(":", " ").split():
                if len(w) > 2 and w not in out:
                    out.append(w)
        return out[:6]

    def stats(self) -> dict[str, Any]:
        return {"count": len(self._hypotheses), "top": [h.to_dict() for h in self.parallel_beliefs()[:3]]}

    def to_dict(self) -> dict[str, Any]:
        return {"hypotheses": [h.to_dict() for h in self._hypotheses.values()]}

    def load_dict(self, data: dict[str, Any]) -> None:
        self._hypotheses = {}
        for row in data.get("hypotheses", []):
            claim = str(row.get("claim", ""))
            if claim:
                self._hypotheses[claim] = Hypothesis(
                    claim=claim,
                    confidence=float(row.get("confidence", 0.5)),
                    evidence_for=int(row.get("evidence_for", 0)),
                    evidence_against=int(row.get("evidence_against", 0)),
                )
