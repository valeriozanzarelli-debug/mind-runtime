"""Tono sociale del caregiver — emerge da segnali testuali (STT), non hardcoded risposte."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SocialTone:
    valence: float  # -1 arrabbiato … +1 affettuoso
    arousal: float  # 0 calmo … 1 intenso
    is_angry: bool
    is_correction: bool
    is_praise: bool
    is_warm: bool
    markers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "is_angry": self.is_angry,
            "is_correction": self.is_correction,
            "is_praise": self.is_praise,
            "is_warm": self.is_warm,
            "markers": self.markers[:8],
        }


_ANGER = (
    "arrabbiato", "arrabbiata", "furioso", "furiosa", "basta", "smettila",
    "idiota", "stupido", "stupida", "zitto", "cazzo", "merda", "porco",
    "incazzato", "incazzata", "che schifo", "odio",
)
_CORRECTION = (
    "sbagliato", "scorretto", "non è", "non e ", "si dice", "sì dice",
    "correggi", "correzione", "intendo", "intendevo", "volevo dire",
    "non hai capito", "non capisci", "ripeti bene", "dovevi dire",
    "non così", "è così", "giusto è",
)
_PRAISE = ("bravo", "brava", "bene", "esatto", "perfetto", "ottimo", "giusto", "così si fa")
_WARM = ("ti voglio bene", "amore", "caro", "cara", "grazie", "mi piaci", "forte", "bello")


def analyze_social_tone(text: str, *, last_spoke: str = "") -> SocialTone:
    t = text.strip()
    tl = t.lower()
    markers: list[str] = []
    valence = 0.0
    arousal = 0.15

    # --- Segnali energetici dal testo (tono, non solo contenuto) ---

    # Lettere ripetute: "CIAOOO" → alta eccitazione; "ohhh" → sorpresa/emozione
    import re as _re
    repeated = _re.findall(r"([a-zA-Z])\1{2,}", t)
    if repeated:
        markers.append("repeated_letters")
        arousal += min(0.5, len(repeated) * 0.15)
        valence += 0.1  # di solito positivo (entusiasmo)

    # Puntini di sospensione: "ciao..." → riflessivo, incerto, lento
    if "..." in t or "…" in t:
        markers.append("ellipsis")
        arousal -= 0.1
        valence -= 0.05  # incertezza/esitazione

    # "eh" o "no?" finale: "ciao eh" → aspetta risposta, tono piatto
    if re.search(r"\beh\b", tl) or tl.endswith(" no?") or tl.endswith(" eh?"):
        markers.append("trailing_check")
        arousal -= 0.08
        # tono interrogativo/aspettativo

    # Maiuscole in tutto: "CIAO" → enfasi/volume alto
    if t.isupper() and len(t) > 2:
        markers.append("all_caps")
        arousal += 0.3

    angry_hits = sum(1 for w in _ANGER if w in tl)
    if angry_hits:
        markers.append("anger_lex")
        valence -= 0.35 * min(3, angry_hits)
        arousal += 0.25 * angry_hits
    if "!" in t:
        ex = t.count("!")
        markers.append("exclamation")
        arousal += min(0.4, ex * 0.12)
        if ex >= 2:
            valence -= 0.15
    caps = sum(1 for w in t.split() if len(w) > 2 and w.isupper())
    if caps:
        markers.append("caps")
        arousal += min(0.35, caps * 0.1)
        valence -= 0.1

    is_correction = any(m in tl for m in _CORRECTION)
    if is_correction:
        markers.append("correction")
        valence -= 0.08
        arousal += 0.2
    if last_spoke and is_correction:
        markers.append("corrects_baby")

    is_praise = any(m in tl for m in _PRAISE)
    if is_praise:
        markers.append("praise")
        valence += 0.35
        arousal += 0.1

    is_warm = any(m in tl for m in _WARM)
    if is_warm:
        markers.append("warm")
        valence += 0.25

    is_angry = angry_hits > 0 or (valence < -0.35 and arousal > 0.45)

    return SocialTone(
        valence=max(-1.0, min(1.0, valence)),
        arousal=min(1.0, arousal),
        is_angry=is_angry,
        is_correction=is_correction,
        is_praise=is_praise,
        is_warm=is_warm or is_praise,
        markers=markers,
    )


def extract_correction_payload(text: str) -> str | None:
    """Estrae la forma corretta da una frase di correzione."""
    t = text.strip()
    patterns = [
        r"(?:si dice|sì dice|dovevi dire|volevo dire|intendo|corretto[,:]?)\s+(.+)",
        r"sbagliato[,! ]+(.+)",
        r"non (?:è|e) (.+)[, ]+(?:ma |piuttosto )?(.+)",
        r"^no[,! ]+(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, t, re.I)
        if m:
            groups = [g.strip(" .!?") for g in m.groups() if g]
            if groups:
                return groups[-1]
    if len(t.split()) <= 8 and not t.lower().startswith("no"):
        return t
    return None
