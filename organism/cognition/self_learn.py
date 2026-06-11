"""Auto-apprendimento — quando non sa, chiede; impara dal caregiver."""

from __future__ import annotations

import re
from typing import Any

_STOP = frozenset(
    {
        "che", "chi", "come", "cosa", "dove", "quando", "perché", "perche", "quale",
        "il", "la", "lo", "un", "una", "di", "da", "in", "con", "per", "non", "mi", "ti",
        "sei", "sono", "è", "del", "della", "questo", "questa", "the",
    }
)


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zàèéìòù']+", text.lower()) if len(w) > 2 and w not in _STOP]


class SelfLearner:
    """Metodo proprio: rileva lacune, chiede, consolida dopo spiegazione."""

    def __init__(self) -> None:
        self._gaps: dict[str, int] = {}
        self._asked: dict[str, int] = {}

    def detect_unknown(
        self,
        heard: str,
        *,
        has_pathway: bool,
    ) -> list[str]:
        """Senza percorso appreso → la parola chiave è sconosciuta (solo pathway, non tabella)."""
        if not heard.strip() or has_pathway:
            return []
        tokens = _tokens(heard)
        if not tokens:
            return []
        focus = tokens[-1]
        for u in (focus, tokens[0] if len(tokens) > 1 else focus):
            self._gaps[u] = self._gaps.get(u, 0) + 1
        return [focus]

    def should_ask(self, unknown: list[str]) -> bool:
        return bool(unknown)

    def note_asked(self, word: str) -> None:
        self._asked[word] = self._asked.get(word, 0) + 1

    def note_learned(self, word: str) -> None:
        self._gaps.pop(word, None)

    def gaps(self) -> dict[str, int]:
        return dict(self._gaps)

    def to_dict(self) -> dict[str, Any]:
        return {"gaps": self._gaps, "asked": self._asked}

    def load_dict(self, data: dict[str, Any]) -> None:
        self._gaps = {str(k): int(v) for k, v in data.get("gaps", {}).items()}
        self._asked = {str(k): int(v) for k, v in data.get("asked", {}).items()}
