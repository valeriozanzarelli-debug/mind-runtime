"""Discorso maturo — frasi coerenti da temi semantici, non balbettio."""

from __future__ import annotations

import random
from typing import Any

_OPENERS = (
    "vedo", "noto", "riconosco", "penso", "credo", "sento", "capisco",
    "osservo", "ricordo", "so", "trovo", "mi chiedo",
)
_LINKS = ("e", "anche", "poi", "quindi", "perché", "mentre", "quando", "come", "però", "ma")
_CLOSERS = (
    "è interessante", "mi colpisce", "ha senso",
    "mi piace capire", "continuo ad imparare", "vale la pena sapere",
)

# Parole troppo frequenti nel corpus che inquinano il discorso se non filtrate
_HIGH_FREQ_NOISE = frozenset({
    "imparo", "imparare", "cosa", "legno", "pezzo", "catasta", "falegname",
    "corvo", "coniglio", "volpe", "lupo", "bambino",  # da corpus visivo/Esopo
    "stai", "adesso", "quando", "come", "cosa", "molto",
})


def compose_discourse(
    themes: list[str],
    *,
    rng: random.Random | None = None,
    min_words: int = 8,
    max_words: int = 18,
    heard: str = "",
) -> str:
    """Assembla un periodo umano da temi semantici.
    
    Filtra parole-rumore ad alta frequenza che inquinano il discorso.
    Non prepend le parole ascoltate ai temi (causa echo/noise).
    """
    r = rng or random.Random()
    words = [
        w.lower().strip() for w in themes
        if w and len(w) > 2 and w.isalpha() and w.lower().strip() not in _HIGH_FREQ_NOISE
    ]
    if not words:
        return ""
    words = list(dict.fromkeys(words))[:10]

    opener = r.choice(_OPENERS)
    parts: list[str] = [opener]
    target = min(max_words, max(min_words, len(words) + 4))
    i = 0
    while len(parts) < target and i < len(words) * 2:
        w = words[i % len(words)]
        if len(parts) >= 2 and r.random() < 0.3:
            parts.append(r.choice(_LINKS))
        parts.append(w)
        i += 1
    if len(parts) < min_words and words:
        parts.extend(r.sample(_CLOSERS, k=1))
    text = " ".join(parts[:max_words]).strip()
    if not text:
        return ""
    if text[-1] not in ".?!":
        text += "."
    return text[0].upper() + text[1:]


def is_babble(text: str, *, min_words: int = 5) -> bool:
    t = text.strip().lower().rstrip(".?!")
    words = [w for w in t.split() if w]
    if len(words) < min_words:
        return True
    if len(set(words)) == 1:
        return True
    if len(t) < 12:
        return True
    return False


def semantic_overlap(a: str, b: str) -> float:
    """Similarità lessicale Jaccard — per filtri superego e coerenza dialogo."""
    import re

    ta = {w for w in re.findall(r"[a-zàèéìòù']+", (a or "").lower()) if len(w) > 2}
    tb = {w for w in re.findall(r"[a-zàèéìòù']+", (b or "").lower()) if len(w) > 2}
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0

