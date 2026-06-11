"""Conoscenza semantica — parole spiegate, beat di storia, narrazione emergente."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from organism.teaching.dialogue import _content_words, normalize_text

_TOKEN_RE = re.compile(r"[a-zàèéìòù']+")


@dataclass
class WordGrounding:
    word: str
    definition: str
    related: list[str] = field(default_factory=list)
    story_id: str = ""


@dataclass
class StoryBeat:
    story_id: str
    order: int
    summary: str
    entities: list[str]
    hooks: list[str]


def is_narrative_request(text: str) -> bool:
    tl = normalize_text(text)
    cues = (
        "raccontami",
        "racconta",
        "continua",
        "storia",
        "favola",
        "narra",
        "dimmi la storia",
    )
    if any(c in tl for c in cues):
        return True
    identity = ("chi è", "chi e", "cos è", "cos e", "cos'è", "come inizia", "cosa insegna")
    stories = ("pinocchio", "corvo")
    if any(i in tl for i in identity) and any(s in tl for s in stories):
        return True
    return False


class SemanticKnowledge:
    """Lessico spiegato + trama a beat — non monologhi da recitare."""

    def __init__(self) -> None:
        self._words: dict[str, WordGrounding] = {}
        self._beats: dict[str, list[StoryBeat]] = {}

    def teach_word(
        self,
        word: str,
        definition: str,
        *,
        related: list[str] | None = None,
        story_id: str = "",
    ) -> dict[str, Any]:
        w = word.lower().strip()
        if len(w) < 2 or not definition.strip():
            return {"ok": False}
        self._words[w] = WordGrounding(
            word=w,
            definition=definition.strip(),
            related=[r.lower() for r in (related or []) if r],
            story_id=story_id,
        )
        return {"ok": True, "word": w, "story_id": story_id}

    def teach_beat(
        self,
        story_id: str,
        order: int,
        summary: str,
        *,
        entities: list[str] | None = None,
        hooks: list[str] | None = None,
    ) -> dict[str, Any]:
        sid = story_id.lower().strip()
        ents = [e.lower().strip() for e in (entities or []) if e]
        hks = [h.lower().strip() for h in (hooks or ents)]
        beat = StoryBeat(
            story_id=sid,
            order=int(order),
            summary=summary.strip(),
            entities=ents,
            hooks=hks,
        )
        beats = self._beats.setdefault(sid, [])
        beats = [b for b in beats if b.order != beat.order]
        beats.append(beat)
        beats.sort(key=lambda b: b.order)
        self._beats[sid] = beats
        return {"ok": True, "story_id": sid, "order": order, "entities": ents}

    def definition(self, word: str) -> str | None:
        g = self._words.get(word.lower().strip())
        return g.definition if g else None

    def is_grounded(self, word: str) -> bool:
        return word.lower().strip() in self._words

    def grounded_words(self, story_id: str | None = None) -> list[str]:
        if story_id:
            sid = story_id.lower()
            return [w for w, g in self._words.items() if g.story_id == sid]
        return list(self._words.keys())

    def story_ids_for_trigger(self, text: str) -> list[str]:
        tl = normalize_text(text)
        found: list[str] = []
        for sid in self._beats:
            if sid in tl or sid.replace("_", " ") in tl:
                found.append(sid)
        if "pinocchio" in tl and "pinocchio" not in found:
            found.append("pinocchio")
        if "corvo" in tl and "corvo" not in found:
            found.append("corvo")
        return found

    def recall_story_themes(self, trigger: str, *, limit: int = 14) -> list[str]:
        """Temi ordinati per beat — solo entità con parola spiegata."""
        themes: list[str] = []
        cue = _content_words(normalize_text(trigger))
        for sid in self.story_ids_for_trigger(trigger):
            for beat in self._beats.get(sid, []):
                if cue and not (cue & set(beat.hooks) | cue & set(beat.entities)):
                    if sid not in cue and not any(h in cue for h in beat.hooks):
                        continue
                for ent in beat.entities:
                    if self.is_grounded(ent) and ent not in themes:
                        themes.append(ent)
                for w in _tokens(beat.summary):
                    if self.is_grounded(w) and w not in themes:
                        themes.append(w)
                if len(themes) >= limit:
                    return themes[:limit]
        return themes[:limit]

    def recall_beats(self, story_id: str, trigger: str = "") -> list[StoryBeat]:
        beats = list(self._beats.get(story_id.lower(), []))
        if not trigger:
            return beats
        cue = _content_words(normalize_text(trigger))
        if not cue:
            return beats
        scored: list[tuple[int, StoryBeat]] = []
        for b in beats:
            overlap = len(cue & set(b.entities) | cue & set(b.hooks))
            scored.append((overlap, b))
        scored.sort(key=lambda x: (-x[0], x[1].order))
        return [b for s, b in scored if s > 0] or beats

    def narrate_beats(self, trigger: str) -> list[StoryBeat]:
        """Beat episodici ordinati — filo completo della storia riconosciuta dal trigger."""
        out: list[StoryBeat] = []
        for sid in self.story_ids_for_trigger(trigger):
            for beat in self._beats.get(sid, []):
                if any(self.is_grounded(e) for e in beat.entities):
                    out.append(beat)
        if out:
            out.sort(key=lambda b: (b.story_id, b.order))
            return out
        cue = _content_words(normalize_text(trigger))
        if not cue:
            return []
        scored: list[tuple[int, StoryBeat]] = []
        for beats in self._beats.values():
            for beat in beats:
                overlap = len(cue & set(beat.entities) | cue & set(beat.hooks))
                if overlap:
                    scored.append((overlap, beat))
        scored.sort(key=lambda x: (-x[0], x[1].story_id, x[1].order))
        return [b for _, b in scored[:6]]

    def narrate_plan(self, trigger: str) -> list[str]:
        """Frasi-beat corte — l'organismo le compone, non le recita."""
        return [b.summary for b in self.narrate_beats(trigger)]

    def coverage(self, story_id: str) -> dict[str, Any]:
        beats = self._beats.get(story_id.lower(), [])
        if not beats:
            return {"story_id": story_id, "beats": 0, "grounded_entities": 0, "ratio": 0.0}
        all_ents = {e for b in beats for e in b.entities}
        grounded = {e for e in all_ents if self.is_grounded(e)}
        return {
            "story_id": story_id,
            "beats": len(beats),
            "entities_total": len(all_ents),
            "grounded_entities": len(grounded),
            "ratio": round(len(grounded) / max(1, len(all_ents)), 2),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "words": {
                w: {
                    "definition": g.definition,
                    "related": g.related,
                    "story_id": g.story_id,
                }
                for w, g in self._words.items()
            },
            "beats": {
                sid: [
                    {
                        "order": b.order,
                        "summary": b.summary,
                        "entities": b.entities,
                        "hooks": b.hooks,
                    }
                    for b in beats
                ]
                for sid, beats in self._beats.items()
            },
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self._words = {}
        for w, g in data.get("words", {}).items():
            self._words[str(w)] = WordGrounding(
                word=str(w),
                definition=str(g.get("definition", "")),
                related=[str(r) for r in g.get("related", [])],
                story_id=str(g.get("story_id", "")),
            )
        self._beats = {}
        for sid, beats in data.get("beats", {}).items():
            self._beats[str(sid)] = [
                StoryBeat(
                    story_id=str(sid),
                    order=int(b.get("order", 0)),
                    summary=str(b.get("summary", "")),
                    entities=[str(e) for e in b.get("entities", [])],
                    hooks=[str(h) for h in b.get("hooks", [])],
                )
                for b in beats
            ]


def _tokens(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if len(w) >= 4]
