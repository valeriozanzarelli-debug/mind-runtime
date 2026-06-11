"""Pensieri interiori assimilabili a monologo umano."""

from __future__ import annotations

import random
from typing import Any


_REFLECT = (
    "mi chiedo",
    "mi viene in mente",
    "noto dentro di me",
    "sento che",
    "mi accorgo che",
    "sto collegando",
)
_BRIDGE = ("perché", "mentre", "quando", "se", "anche se", "così")
_CLOSER = (
    "ha senso per me",
    "voglio capire meglio",
    "continuo a riflettere",
    "resto attento",
    "mi affido a quello che sento",
)


def thought_coherence(themes: list[str], *, heard: str = "") -> float:
    """0–1 — temi distinti e collegati allo stimolo."""
    if not themes:
        return 0.0
    unique = len({t.lower() for t in themes if t})
    span = unique / max(1, len(themes))
    link = 0.0
    if heard:
        hw = {w for w in heard.lower().split() if len(w) > 2}
        overlap = sum(1 for t in themes if t.lower() in hw or any(w in t.lower() for w in hw))
        link = min(1.0, overlap / max(1, len(hw)))
    return min(1.0, 0.55 * span + 0.45 * link)


def compose_inner_voice(
    *,
    themes: list[str],
    heard: str = "",
    emotion: str = "",
    objects: list[str] | None = None,
    memory_line: str = "",
    rng: random.Random | None = None,
) -> str:
    """Un pensiero interno in prima persona — non è output parlato."""
    r = rng or random.Random()
    pool = [t.lower().strip() for t in themes if t and len(t) > 2 and t.isalpha()]
    pool = list(dict.fromkeys(pool))[:8]
    if objects:
        for o in objects:
            if o and o not in pool:
                pool.insert(0, o.lower())
    if heard:
        hw = [w for w in heard.lower().split() if len(w) > 2][:2]
        pool = list(dict.fromkeys(hw + pool))[:8]
    if not pool:
        pool = ["qualcosa", "attimo", "silenzio"]

    opener = r.choice(_REFLECT)
    parts = [opener]
    if emotion and emotion not in ("", "neutro"):
        parts.append(f"sono {emotion}")
        parts.append(r.choice(_BRIDGE))

    target = min(14, max(7, len(pool) + 3))
    i = 0
    while len(parts) < target and i < len(pool) * 2:
        if len(parts) > 2 and r.random() < 0.3:
            parts.append(r.choice(_BRIDGE))
        parts.append(pool[i % len(pool)])
        i += 1

    if memory_line:
        parts.append("e")
        parts.append(memory_line[:40])

    parts.append(r.choice(_CLOSER))
    text = " ".join(parts[:target])
    return text[0].upper() + text[1:] + "."


def enrich_consciousness_lines(
    lines: list[str],
    *,
    inner_voice: str,
    coherence: float,
) -> list[str]:
    out = list(lines)
    if inner_voice:
        out.append(f"pensiero: {inner_voice}")
    if coherence > 0.1:
        out.append(f"coerenza: {coherence:.2f}")
    return out
