"""Parsing frasi di insegnamento visivo — «questa è una cassa rosa»."""

from __future__ import annotations

import re
from dataclasses import dataclass

_COLORS = frozenset(
    {
        "rosso", "rossa", "verde", "blu", "giallo", "gialla", "nero", "nera",
        "bianco", "bianca", "grigio", "grigia", "rosa", "arancione", "marrone",
        "azzurro", "azzurra", "viola",
    }
)
_ARTICLES = frozenset({"un", "una", "il", "la", "lo", "l", "questo", "questa", "quello", "quella"})
_STOP = frozenset(
    {
        "ciao", "come", "stai", "sono", "grazie", "prego", "bene", "male", "chi", "sei",
        "cosa", "dove", "quando", "perché", "perche", "sì", "si", "no", "addio",
    }
)


@dataclass
class VisionTeachPhrase:
    object_name: str
    color: str = ""
    raw: str = ""
    phrase: str = ""

    @property
    def has_object(self) -> bool:
        return bool(self.object_name)


def _norm_color(word: str) -> str:
    w = word.lower().strip()
    if w.endswith("a") and w[:-1] in _COLORS:
        return w[:-1] if w[:-1] != "azzurr" else "azzurro"
    if w.endswith("o") and w[:-1] + "a" in _COLORS:
        return w
    return w


def parse_vision_teaching(phrase: str) -> VisionTeachPhrase:
    """Estrae oggetto e colore da frasi naturali del caregiver."""
    raw = phrase.strip()
    tl = raw.lower()
    if not tl:
        return VisionTeachPhrase(object_name="", raw=raw)

    patterns = [
        r"quest[aoèe]?\s+(?:è|e)\s+(?:un|una|il|la|lo)?\s*(?P<obj>[a-zàèéìòù']+)(?:\s+(?P<col>[a-zàèéìòù']+))?",
        r"quell[ao]\s+(?:è|e)\s+(?:un|una|il|la|lo)?\s*(?P<obj>[a-zàèéìòù']+)(?:\s+(?P<col>[a-zàèéìòù']+))?",
        r"(?:vedo|guarda|guarda\s+quest[ao])\s+(?:un|una|il|la)?\s*(?P<obj>[a-zàèéìòù']+)(?:\s+(?P<col>[a-zàèéìòù']+))?",
        r"(?:è|e)\s+(?:un|una|il|la|lo)\s*(?P<obj>[a-zàèéìòù']+)(?:\s+(?P<col>[a-zàèéìòù']+))?",
        r"(?:questo|questa)\s+(?:è|e)\s+(?:il|la|lo)\s*(?P<obj>[a-zàèéìòù']+)",
        r"(?P<obj>[a-zàèéìòù']+)\s+(?P<col>rosso|rossa|verde|blu|giallo|gialla|nero|nera|bianco|bianca|rosa)",
        r"si\s+chiama\s+(?P<obj>[a-zàèéìòù']+)",
    ]
    for pat in patterns:
        m = re.search(pat, tl)
        if not m:
            continue
        obj = m.group("obj").strip()
        if obj in _ARTICLES or obj in _STOP or len(obj) < 2:
            continue
        col = ""
        if m.lastgroup and m.groupdict().get("col"):
            cw = m.group("col").strip()
            if cw in _COLORS or cw.rstrip("a") in {c.rstrip("a") for c in _COLORS}:
                col = _norm_color(cw)
        return VisionTeachPhrase(object_name=obj, color=col, raw=raw, phrase=raw)

    tokens = [w for w in re.findall(r"[a-zàèéìòù']+", tl) if w not in _ARTICLES and w not in _STOP and len(w) > 2]
    if len(tokens) == 1 and len(tokens[0]) >= 4:
        return VisionTeachPhrase(object_name=tokens[0], raw=raw, phrase=raw)
    if not tokens:
        return VisionTeachPhrase(object_name="", raw=raw, phrase=raw)
    if not any(k in tl for k in ("quest", "vedo", "guarda", "è ", " e ")):
        return VisionTeachPhrase(object_name="", raw=raw, phrase=raw)
    obj = tokens[0]
    col = ""
    for w in tokens[1:]:
        if w in _COLORS or w.rstrip("a") in {c.rstrip("a") for c in _COLORS}:
            col = _norm_color(w)
            break
    return VisionTeachPhrase(object_name=obj, color=col, raw=raw, phrase=raw)


def is_vision_naming_phrase(phrase: str) -> bool:
    """True se la frase nomina un oggetto visivo (es. «valigia», «questa è una cassa»)."""
    p = parse_vision_teaching(phrase)
    if p.has_object:
        return True
    tl = phrase.strip().lower()
    tokens = [w for w in re.findall(r"[a-zàèéìòù']+", tl) if w not in _ARTICLES and w not in _STOP]
    return len(tokens) == 1 and len(tokens[0]) >= 4

