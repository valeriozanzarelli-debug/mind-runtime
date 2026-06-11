"""Coerenza temi — il pensiero resta sul discorso, non sul rumore di memoria."""

from __future__ import annotations

import re

_STOP = frozenset(
    {
        "che", "chi", "come", "cosa", "dove", "quando", "perché", "perche", "perch",
        "il", "la", "lo", "un", "una", "di", "da", "in", "con", "per", "non", "mi", "ti",
        "sei", "sono", "hai", "ho", "è", "era", "sto", "sta", "del", "della", "questo", "questa",
    }
)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zàèéìòù']+", text.lower()) if len(w) > 2 and w not in _STOP}


def score_theme(theme: str, *, heard: set[str], pathway: set[str], context: set[str]) -> float:
    t = theme.lower().strip()
    if not t or t.startswith(("VIS:", "OBJ:", "COL:", "MEM:", "BRAIN:")):
        return -1.0
    score = 0.0
    if t in heard:
        score += 3.0
    if t in pathway:
        score += 2.5
    if t in context:
        score += 1.5
    if heard and any(h in t or t in h for h in heard if len(h) > 3):
        score += 1.2
    if pathway and any(p in t or t in p for p in pathway if len(p) > 3):
        score += 1.0
    if t in _STOP:
        score -= 2.0
    return score


def coherent_themes(
    themes: list[str],
    *,
    heard: str = "",
    pathway_words: list[str] | None = None,
    context_words: list[str] | None = None,
    max_themes: int = 6,
    min_score: float = 0.5,
) -> list[str]:
    """Filtra temi — solo ciò che c'entra con domanda, percorso appreso, contesto."""
    heard_s = _tokens(heard)
    pathway_s = _tokens(" ".join(pathway_words or []))
    context_s = _tokens(" ".join(context_words or []))

    if not themes:
        return []

    ranked: list[tuple[float, int, str]] = []
    for i, t in enumerate(themes):
        s = score_theme(t, heard=heard_s, pathway=pathway_s, context=context_s)
        if s >= min_score:
            ranked.append((s, -i, t))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    out: list[str] = []
    for _, _, t in ranked:
        if t not in out:
            out.append(t)
        if len(out) >= max_themes:
            break

    # Percorso appreso: mantieni ordine naturale della risposta insegnata
    if pathway_words and pathway_s:
        ordered = [w for w in pathway_words if w.lower() in {x.lower() for x in out} or w.lower() in pathway_s]
        if ordered:
            merged = []
            for w in ordered:
                wl = w.lower()
                for t in out:
                    if t.lower() == wl and t not in merged:
                        merged.append(t)
            for t in out:
                if t not in merged:
                    merged.append(t)
            return merged[:max_themes]

    return out
