"""Narrazione emergente — filo episodico + significati appresi, non generazione LLM."""

from __future__ import annotations

import re
from typing import Any, Callable

from organism.cognition.discourse import is_babble
from organism.cognition.neural_lexicon import FILLER_WORDS

_TOKEN_RE = re.compile(r"[a-zàèéìòù']+")

_DEF_STOP = FILLER_WORDS | {
    "che", "della", "del", "dei", "delle", "degli", "una", "uno", "un",
    "il", "la", "lo", "gli", "le", "di", "da", "nel", "nella", "nella",
    "con", "per", "non", "più", "molto", "viene", "vieni", "vengono",
    "essere", "sono", "era", "cera", "questo", "questa", "quello", "quella",
    "semplice", "pezzo", "dove", "quando", "come", "dalla", "dalle", "dagli",
    "degli", "alle", "agli", "sul", "sulla", "sui", "sulle",
}


def simplify_definition(defn: str, *, max_words: int = 7) -> str:
    """Estrae nucleo semantico da una definizione insegnata — parole proprie, non copia storia."""
    words: list[str] = []
    for w in _TOKEN_RE.findall(defn.lower()):
        if len(w) < 3 or w in _DEF_STOP:
            continue
        if w not in words:
            words.append(w)
        if len(words) >= max_words:
            break
    return " ".join(words)


def _pick_entities(beat: Any, *, grounded: Callable[[str], bool]) -> list[str]:
    ordered: list[str] = []
    for h in getattr(beat, "hooks", []) or []:
        hl = str(h).lower().strip()
        if hl and grounded(hl) and hl not in ordered:
            ordered.append(hl)
    for ent in getattr(beat, "entities", []) or []:
        el = str(ent).lower().strip()
        if el and grounded(el) and el not in ordered:
            ordered.append(el)
    return ordered


def compose_beat_utterance(
    beat: Any,
    *,
    definition_fn: Callable[[str], str | None],
    grounded_fn: Callable[[str], bool],
    articulable_fn: Callable[[str], bool],
    max_entities: int = 1,
) -> str:
    """Un episodio = una clausola dal significato appreso — slot di memoria di lavoro."""
    entities = [
        e
        for e in _pick_entities(beat, grounded=grounded_fn)
        if articulable_fn(e) and definition_fn(e)
    ][:max_entities]
    if not entities:
        return ""

    clauses: list[str] = []
    for ent in entities:
        core = simplify_definition(definition_fn(ent) or "")
        if not core:
            continue
        el = ent.lower()
        if el in core.split():
            clause = f"{ent.capitalize()} {core}"
        else:
            clause = f"{ent.capitalize()} {core}"
        clauses.append(clause.rstrip("."))

    if not clauses:
        return ""
    return ". ".join(clauses) + "."


def relate_clauses(clauses: list[str], *, focus_entity: str = "") -> str:
    """Collegamento causale leggero tra episodi — non sintassi LLM."""
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0] if clauses[0].endswith(".") else clauses[0] + "."
    out: list[str] = []
    prev_ent = ""
    for clause in clauses:
        c = clause.rstrip(".")
        words = c.lower().split()
        ent = words[0] if words else (focus_entity.lower() if focus_entity else "")
        if prev_ent and ent and prev_ent != ent and not c.lower().startswith(("poi", "dopo")):
            c = f"poi {c[0].lower()}{c[1:]}" if c else c
        out.append(c)
        prev_ent = ent
    return ". ".join(out) + "."


def compose_narrative(
    beats: list[Any],
    *,
    definition_fn: Callable[[str], str | None],
    grounded_fn: Callable[[str], bool],
    articulable_fn: Callable[[str], bool],
    max_beats: int = 5,
    focus_entity: str = "",
) -> str:
    """Fil narrativo — beat in ordine episodico, ognuno espresso con parole spiegate."""
    sentences: list[str] = []
    seen: set[str] = set()
    for beat in beats[:max_beats]:
        line = compose_beat_utterance(
            beat,
            definition_fn=definition_fn,
            grounded_fn=grounded_fn,
            articulable_fn=articulable_fn,
        )
        if not line or is_babble(line, min_words=4):
            continue
        norm = line.rstrip(".").lower()
        if norm in seen:
            continue
        seen.add(norm)
        sentences.append(line.rstrip("."))
    if not sentences:
        return ""
    return relate_clauses(sentences, focus_entity=focus_entity)


def narrative_quality(text: str, *, expected_entities: list[str]) -> float:
    """Quanto la narrazione ancorata alle entità attese (0–1) — per evoluzione/test."""
    if not text.strip():
        return 0.0
    tl = text.lower()
    hits = sum(1 for e in expected_entities if e.lower() in tl)
    if not expected_entities:
        return 0.0
    entity_score = hits / len(expected_entities)
    words = tl.split()
    if len(words) < 6:
        return entity_score * 0.5
    if is_babble(text, min_words=5):
        return entity_score * 0.4
    return min(1.0, entity_score * 0.75 + 0.25)
