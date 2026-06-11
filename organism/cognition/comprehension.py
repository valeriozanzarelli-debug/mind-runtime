"""Comprensione profonda — ego esecutivo, schemi, profondità di elaborazione (Craik & Lockhart).

Non è un LLM: è routing cognitivo tra memoria semantica, episodica, causale e sociale
prima che l'id motorio (lessico grezzo) parli a caso.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from organism.cognition.causal_graph import CausalGraph
from organism.cognition.discourse import semantic_overlap
from organism.teaching.dialogue import normalize_text
from organism.teaching.semantic_knowledge import SemanticKnowledge, is_narrative_request

Intent = Literal[
    "social",
    "narrative_full",
    "narrative_identity",
    "causal",
    "vision",
    "taught",
    "explore",
    "word_meaning",
    "silent",
]

_GREETINGS = frozenset(
    {
        "ciao",
        "salve",
        "buongiorno",
        "buonasera",
        "hey",
        "ehi",
        "hello",
        "hi",
    }
)
_THANKS = frozenset({"grazie", "ringrazio", "thanks"})
_SMALLTALK = (
    "come stai",
    "come va",
    "come stai tu",
    "tutto bene",
    "va tutto bene",
    "che fai",
    "cosa fai",
)
_IDENTITY = ("chi è", "chi e", "cos è", "cos e", "cos'è", "cosa è", "cosa e")
_VISION = ("cosa vedi", "cosa guardi", "che vedi", "vedi qualcosa", "descrivi cosa vedi")
_CAUSAL = ("perché", "perche", "perchè")
_CONDITIONAL = ("se ", "cosa succede", "cosa accade", "e se ")


@dataclass
class ComprehensionFrame:
    """Quadro compreso — l'ego sa *cosa* sta succedendo prima di parlare."""

    intent: Intent
    depth: float
    confidence: float
    heard: str = ""
    schema_id: str = ""
    focus_entity: str = ""
    themes: list[str] = field(default_factory=list)
    beats: list[Any] = field(default_factory=list)
    taught_say: str = ""
    taught_verbatim: bool = True
    causal_effect: str = ""
    episodic_hint: str = ""
    narrative_max_beats: int = 5
    inhibit_shallow: bool = False
    inhibit_lexicon_dump: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "depth": round(self.depth, 2),
            "confidence": round(self.confidence, 2),
            "schema_id": self.schema_id,
            "focus_entity": self.focus_entity,
            "themes": self.themes[:8],
            "beats": len(self.beats),
            "inhibit_shallow": self.inhibit_shallow,
        }


class PsycheEngine:
    """Corteccia prefrontale digitale — integra id (drive), schemi (semantica), superego (vincoli)."""

    def __init__(self) -> None:
        self.causal = CausalGraph()
        self._seeded = False

    def ensure_seeded(self) -> None:
        if self._seeded:
            return
        from organism.teaching.corpus import REASONING

        self.causal.seed_from_pairs(list(REASONING))
        self._seeded = True

    def comprehend(
        self,
        heard: str,
        *,
        semantic: SemanticKnowledge,
        dialogue_respond: Any,
        episodic_recall: Any | None = None,
        wm_context: list[str] | None = None,
        visual_themes: list[str] | None = None,
    ) -> ComprehensionFrame:
        self.ensure_seeded()
        tl = normalize_text(heard)
        if not tl:
            return ComprehensionFrame(intent="silent", depth=0.0, confidence=1.0)

        tokens = _content_tokens(tl)

        # --- Sociale (sistema limbico + norme: superego) ---
        social = self._social_frame(heard, tl, tokens, dialogue_respond)
        if social:
            return social

        smalltalk = self._smalltalk_frame(heard, tl, dialogue_respond)
        if smalltalk:
            return smalltalk

        # --- Narrazione identità (schema focalizzato, non monologo) ---
        if is_narrative_request(heard) and _is_identity(tl):
            entity = _story_entity(tl, semantic)
            beats = semantic.narrate_beats(heard)
            if entity:
                beats = _beats_for_entity(beats, entity)
            themes = semantic.recall_story_themes(heard, limit=10)
            sid = semantic.story_ids_for_trigger(heard)
            if entity and beats:
                themes = [entity] + [t for t in themes if t != entity]
            return ComprehensionFrame(
                intent="narrative_identity",
                depth=0.82,
                confidence=0.85 if beats else 0.4,
                heard=heard,
                schema_id=sid[0] if sid else "",
                focus_entity=entity,
                themes=themes[:6],
                beats=beats,
                narrative_max_beats=1,
                inhibit_shallow=True,
                inhibit_lexicon_dump=True,
            )

        # --- Narrazione completa (filo episodico ippocampale) ---
        if is_narrative_request(heard):
            beats = semantic.narrate_beats(heard)
            themes = semantic.recall_story_themes(heard)
            sid = semantic.story_ids_for_trigger(heard)
            return ComprehensionFrame(
                intent="narrative_full",
                depth=0.78,
                confidence=0.88 if beats else 0.35,
                heard=heard,
                schema_id=sid[0] if sid else "",
                themes=themes,
                beats=beats,
                narrative_max_beats=5,
                inhibit_shallow=True,
                inhibit_lexicon_dump=True,
            )

        # --- Parola singola con significato appreso (es. «formaggio») ---
        if len(tokens) <= 2 and len(tl) < 40 and not _is_identity(tl):
            word = next(iter(tokens), "")
            if word and semantic.is_grounded(word):
                return ComprehensionFrame(
                    intent="word_meaning",
                    depth=0.8,
                    confidence=0.88,
                    heard=heard,
                    focus_entity=word,
                    themes=[word],
                    inhibit_shallow=True,
                    inhibit_lexicon_dump=True,
                )

        # --- Causale (corteccia prefrontale + grafo procedurale) ---
        causal = self._causal_frame(tl, heard, dialogue_respond)
        if causal:
            return causal

        # --- Visione (corteccia visiva → binding) ---
        if any(v in tl for v in _VISION):
            vthemes = list(visual_themes or wm_context or [])[:8]
            return ComprehensionFrame(
                intent="vision",
                depth=0.65,
                confidence=0.75 if vthemes else 0.45,
                heard=heard,
                themes=vthemes,
                inhibit_lexicon_dump=True,
            )

        # --- Percorso insegnato verbatim (Q&A brevi) ---
        say, _kind, verbatim = dialogue_respond(heard)
        if say and verbatim and len(say.split()) >= 4:
            return ComprehensionFrame(
                intent="taught",
                depth=0.72,
                confidence=0.8,
                heard=heard,
                taught_say=say,
                taught_verbatim=verbatim,
                inhibit_lexicon_dump=True,
            )

        # --- Episodico (solo se pertinente alla domanda) ---
        hint = ""
        if episodic_recall and tokens:
            for ep in episodic_recall(heard, limit=3):
                spoke = str(ep.get("spoke", "")).strip()
                if len(spoke) < 12:
                    continue
                if semantic_overlap(heard, spoke) >= 0.25:
                    hint = spoke[:120]
                    break

        # --- Esplorazione: temi filtrati, mai dump Pinocchio/Corvo a caso ---
        depth = 0.45
        relevant = _themes_for_query(tokens, wm_context, semantic)
        if semantic and any(semantic.is_grounded(t) for t in tokens):
            depth = 0.62

        return ComprehensionFrame(
            intent="explore",
            depth=depth,
            confidence=0.5,
            heard=heard,
            themes=relevant[:6],
            episodic_hint=hint,
            inhibit_shallow=True,
            inhibit_lexicon_dump=True,
        )

    def _smalltalk_frame(
        self,
        heard: str,
        tl: str,
        dialogue_respond: Any,
    ) -> ComprehensionFrame | None:
        if not any(p in tl for p in _SMALLTALK):
            return None
        say, _kind, verbatim = dialogue_respond(heard)
        if not say:
            for p in _SMALLTALK:
                if p in tl:
                    say, _kind, verbatim = dialogue_respond(p)
                    if say:
                        break
        if not say:
            say = "sto bene, grazie. dimmi cosa vuoi sapere."
        return ComprehensionFrame(
            intent="social",
            depth=0.4,
            confidence=0.85,
            heard=heard,
            taught_say=say,
            taught_verbatim=verbatim,
            inhibit_shallow=True,
            inhibit_lexicon_dump=True,
        )

    def _social_frame(
        self,
        heard: str,
        tl: str,
        tokens: set[str],
        dialogue_respond: Any,
    ) -> ComprehensionFrame | None:
        is_greet = tl in _GREETINGS or (len(tokens) <= 2 and tokens & _GREETINGS)
        is_thanks = bool(tokens & _THANKS)
        if not is_greet and not is_thanks:
            return None
        probe = heard.strip() if len(heard.strip()) < 40 else tl
        if is_greet and tokens & _GREETINGS:
            probe = next(iter(tokens & _GREETINGS))
        say, _kind, verbatim = dialogue_respond(probe)
        if not say and is_greet:
            say = "ciao, ti ascolto. dimmi pure."
        elif not say and is_thanks:
            say = "prego, è un piacere."
        return ComprehensionFrame(
            intent="social",
            depth=0.35,
            confidence=0.92,
            heard=tl,
            taught_say=say or "",
            taught_verbatim=verbatim,
            inhibit_shallow=True,
            inhibit_lexicon_dump=True,
        )

    def _causal_frame(
        self,
        tl: str,
        heard: str,
        dialogue_respond: Any,
    ) -> ComprehensionFrame | None:
        is_why = any(c in tl for c in _CAUSAL)
        is_cond = any(c in tl for c in _CONDITIONAL)
        if not is_why and not is_cond:
            return None
        link = self.causal.lookup(heard)
        say, _kind, verbatim = dialogue_respond(heard)
        effect = ""
        if link:
            effect = link.effect
            self.causal.reinforce(link.when)
        elif say:
            effect = say
        if not effect:
            return None
        return ComprehensionFrame(
            intent="causal",
            depth=0.88,
            confidence=0.9 if link else 0.75,
            heard=heard,
            taught_say=effect,
            taught_verbatim=verbatim,
            causal_effect=effect,
            inhibit_shallow=True,
            inhibit_lexicon_dump=True,
        )


def _content_tokens(text: str) -> set[str]:
    stop = {
        "che", "chi", "come", "cosa", "dove", "quando", "perché", "perche",
        "il", "la", "un", "una", "di", "da", "in", "con", "per", "non", "del", "della",
        "raccontami", "racconta", "storia", "favola", "dimmi",
    }
    return {w for w in re.findall(r"[a-zàèéìòù']+", text.lower()) if len(w) > 2 and w not in stop}


def _is_identity(tl: str) -> bool:
    return any(i in tl for i in _IDENTITY)


def _story_entity(tl: str, semantic: SemanticKnowledge) -> str:
    for sid in semantic.story_ids_for_trigger(tl):
        if sid in tl:
            return sid
    for w in _content_tokens(tl):
        if semantic.is_grounded(w):
            return w
    return ""


def _beats_for_entity(beats: list[Any], entity: str) -> list[Any]:
    el = entity.lower()
    focused = [b for b in beats if el in [e.lower() for e in getattr(b, "entities", [])]]
    if focused:
        return focused[-1:]
    for b in reversed(beats):
        if el in [h.lower() for h in getattr(b, "hooks", [])]:
            return [b]
    return beats[-1:] if beats else []


def heard_fix(tl: str) -> str:
    for g in _GREETINGS:
        if g in tl:
            return g
    return tl


def _themes_for_query(
    tokens: set[str],
    wm_context: list[str] | None,
    semantic: SemanticKnowledge,
) -> list[str]:
    """Solo temi legati alla domanda — evita contaminazione da storie passate."""
    out: list[str] = []
    for t in tokens:
        if len(t) > 2:
            out.append(t)
    for w in wm_context or []:
        wl = w.lower().strip()
        if not wl or wl in out:
            continue
        if wl in tokens:
            out.append(wl)
            continue
        if any(wl in tok or tok in wl for tok in tokens if len(tok) > 3):
            out.append(wl)
    if not out and len(tokens) == 1:
        w = next(iter(tokens))
        if semantic.is_grounded(w):
            out.append(w)
    return out
