"""Super-io digitale — norme interiorizzate, veto e difese prima del parlato.

Metafora freudiana come architettura (non clinica):
  Id    → impulso motorio / lessico grezzo
  Ego   → PsycheEngine (comprensione)
  Super-io → questo modulo: filtra ciò che l'ego vuole dire
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from organism.cognition.discourse import is_babble, semantic_overlap
from organism.cognition.neural_lexicon import FILLER_WORDS

Action = Literal["allow", "substitute", "block"]


@dataclass
class InternalizedNorm:
    """Regola da correzione caregiver — «non dire X, di' Y»."""

    trigger: str
    preferred: str
    strength: float = 1.0


@dataclass
class SuperegoVerdict:
    action: Action
    text: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "reason": self.reason, "text_len": len(self.text)}


@dataclass
class SuperegoEngine:
    """Vincoli morali-cognitivi: appropriatezza, grounding, onestà."""

    _norms: list[InternalizedNorm] = field(default_factory=list)

    def internalize(self, wrong: str, right: str, *, boost: float = 0.25) -> None:
        w = wrong.strip().lower()[:80]
        r = right.strip()
        if not w or not r:
            return
        for norm in self._norms:
            if norm.trigger == w:
                norm.preferred = r
                norm.strength = min(3.0, norm.strength + boost)
                return
        self._norms.append(InternalizedNorm(trigger=w, preferred=r, strength=1.0))
        self._norms = self._norms[-40:]

    def review(
        self,
        planned: str,
        *,
        heard: str = "",
        frame: Any | None = None,
        semantic: Any | None = None,
        articulable: Any | None = None,
    ) -> SuperegoVerdict:
        text = planned.strip()
        if not text:
            return SuperegoVerdict("block", "", "vuoto")

        tl = text.lower()
        for norm in sorted(self._norms, key=lambda n: -n.strength):
            if norm.trigger in tl or _overlap_ratio(norm.trigger, tl) > 0.55:
                return SuperegoVerdict("substitute", norm.preferred, "norma_correzione")

        depth = float(getattr(frame, "depth", 0.0) or 0.0)
        intent = str(getattr(frame, "intent", "") or "")
        inhibit = bool(getattr(frame, "inhibit_lexicon_dump", False))

        if inhibit and is_babble(text, min_words=4):
            fixed = self._sublimate(frame, semantic, articulable)
            if fixed:
                return SuperegoVerdict("substitute", fixed, "sublimazione_babble")
            return SuperegoVerdict("block", self._honest(heard), "repressione_balbettio")

        if intent == "social" and _is_lexicon_dump(tl):
            return SuperegoVerdict(
                "substitute",
                "ciao, ti ascolto. dimmi pure.",
                "norma_sociale",
            )

        if depth >= 0.7 and semantic and articulable:
            ungrounded = _ungrounded_ratio(text, semantic, articulable)
            if ungrounded > 0.45:
                fixed = self._sublimate(frame, semantic, articulable)
                if fixed:
                    return SuperegoVerdict("substitute", fixed, "razionalizzazione_semantica")

        if heard and _echoes_prompt(tl, heard) and intent.startswith("narrative"):
            fixed = self._sublimate(frame, semantic, articulable)
            if fixed:
                return SuperegoVerdict("substitute", fixed, "difesa_prompt_echo")

        if intent == "vision" and _story_leakage(tl, heard or ""):
            return SuperegoVerdict("block", "Non so cosa vedo. Cos'è questo?", "repressione_visiva")

        if heard and intent in ("explore", "social", "word_meaning", "taught"):
            if _story_leakage(tl, heard) and semantic_overlap(heard, text) < 0.22:
                fixed = self._sublimate(frame, semantic, articulable)
                if fixed and semantic_overlap(heard, fixed) >= semantic_overlap(heard, text):
                    return SuperegoVerdict("substitute", fixed, "difesa_contaminazione")
                return SuperegoVerdict("block", self._honest(heard), "repressione_off_topic")

        return SuperegoVerdict("allow", text, "ok")

    def _sublimate(
        self,
        frame: Any | None,
        semantic: Any | None,
        articulable: Any | None,
    ) -> str:
        """Difesa: trasforma impulso inaccettabile in espressione ammessa."""
        if not frame or not semantic:
            return ""
        from organism.cognition.narrative import compose_narrative

        beats = list(getattr(frame, "beats", []) or [])
        if beats and articulable:
            body = compose_narrative(
                beats[: int(getattr(frame, "narrative_max_beats", 3) or 3)],
                definition_fn=semantic.definition,
                grounded_fn=semantic.is_grounded,
                articulable_fn=articulable,
                max_beats=int(getattr(frame, "narrative_max_beats", 3) or 3),
                focus_entity=str(getattr(frame, "focus_entity", "") or ""),
            )
            if body and not is_babble(body, min_words=3):
                return body
        taught = str(getattr(frame, "taught_say", "") or "").strip()
        if taught:
            return taught
        return ""

    def _honest(self, heard: str) -> str:
        hw = [w for w in re.findall(r"[a-zàèéìòù']+", heard.lower()) if len(w) > 3]
        focus = hw[-1] if hw else "questo"
        return f"Non sono sicuro su {focus}, sto ancora imparando."

    def to_dict(self) -> dict[str, Any]:
        return {
            "norms": [
                {"trigger": n.trigger, "preferred": n.preferred[:100], "strength": n.strength}
                for n in self._norms[-30:]
            ]
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self._norms = [
            InternalizedNorm(
                trigger=str(n.get("trigger", "")),
                preferred=str(n.get("preferred", "")),
                strength=float(n.get("strength", 1.0)),
            )
            for n in data.get("norms", [])
        ]


def _overlap_ratio(a: str, b: str) -> float:
    aw = {w for w in a.split() if len(w) > 2}
    bw = {w for w in b.split() if len(w) > 2}
    if not aw:
        return 0.0
    return len(aw & bw) / len(aw)


def _is_lexicon_dump(tl: str) -> bool:
    words = [w for w in tl.split() if w.isalpha()]
    if len(words) < 6:
        return False
    if "ciao" in tl and len(words) > 8:
        return True
    unique = len(set(words)) / len(words)
    return unique > 0.85 and not tl.endswith("?")


def _ungrounded_ratio(text: str, semantic: Any, articulable: Any) -> float:
    words = [w for w in re.findall(r"[a-zàèéìòù']+", text.lower()) if len(w) >= 4]
    if not words:
        return 0.0
    bad = 0
    for w in words:
        if w in FILLER_WORDS:
            bad += 1
        elif not semantic.is_grounded(w) and not articulable(w):
            bad += 1
    return bad / len(words)


_STORY_MARKERS = frozenset(
    {
        "pinocchio",
        "legno",
        "corvo",
        "geppetto",
        "fata",
        "balena",
        "volpe",
        "gatto",
        "burattino",
        "catasta",
        "bottega",
    }
)


def _story_leakage(text: str, heard: str) -> bool:
    """Parole da favole passate che non c'entrano con la domanda attuale."""
    hw = {w for w in re.findall(r"[a-zàèéìòù']+", heard.lower()) if len(w) > 2}
    tw = {w for w in re.findall(r"[a-zàèéìòù']+", text.lower()) if len(w) > 2}
    leaks = (tw & _STORY_MARKERS) - hw
    if not heard.strip():
        return len(leaks) >= 1
    return len(leaks) >= 2 or (len(leaks) == 1 and len(tw) >= 6)


def _echoes_prompt(tl: str, heard: str) -> bool:
    hw = {w for w in heard.lower().split() if len(w) > 3}
    tw = set(tl.split())
    return len(hw & tw) >= 2
