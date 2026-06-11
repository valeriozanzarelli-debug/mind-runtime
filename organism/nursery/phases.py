"""Curriculum — come insegnare: nascita → sensi → linguaggio → mondo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Lesson:
    id: str
    label: str
    input_data: dict[str, Any]
    kwargs: dict[str, Any]
    modality: str = "speech"


@dataclass
class Phase:
    id: str
    label: str
    description: str
    lessons: list[Lesson]


def _grid_bulb() -> list[list[int]]:
    """16x16 grayscale — bright blob center (lamp-like)."""
    g = [[20] * 16 for _ in range(16)]
    for y in range(5, 11):
        for x in range(5, 11):
            g[y][x] = 200
    return g


CURRICULUM: list[Phase] = [
    Phase(
        id="birth",
        label="🌱 Nascita",
        description="Il DNA dispiega la topologia neurale — come nasce il cervello.",
        lessons=[],
    ),
    Phase(
        id="vision",
        label="👁 Vista",
        description="Primi stimoli visivi — pattern, forme, bordi.",
        lessons=[
            Lesson(
                "see_circle_pattern",
                "Vedi forme con cerchio",
                {"shapes": "quadrato+cerchio,triangolo+cerchio,rettangolo+"},
                {},
                "text",
            ),
            Lesson(
                "see_edges",
                "Immagine sintetica (bordi)",
                {"image": _grid_bulb(), "width": 16, "height": 16},
                {},
                "speech",
            ),
        ],
    ),
    Phase(
        id="hearing",
        label="👂 Udito",
        description="Frequenze e suoni — attivazione bande audio.",
        lessons=[
            Lesson("hear_440", "Tono La 440Hz", {"tone_hz": 440.0}, {}, "song"),
            Lesson("hear_220", "Tono grave 220Hz", {"tone_hz": 220.0}, {}, "song"),
        ],
    ),
    Phase(
        id="language",
        label="💬 Linguaggio",
        description="Prime parole — collegamento suono/significato/azione.",
        lessons=[
            Lesson("word_hello", "Ciao", {"text": "Ciao, vorrei prenotare per giovedì"}, {}, "speech"),
            Lesson(
                "word_quote",
                "Preventivo",
                {"text": "preventivo tattoo braccio realistico"},
                {},
                "speech",
            ),
            Lesson(
                "word_lamp",
                "Problema lampadina",
                {"text": "la lampadina non si accende"},
                {},
                "speech",
            ),
        ],
    ),
    Phase(
        id="social",
        label="🤝 Sociale",
        description="Emozione e risonanza — imparare dagli umani.",
        lessons=[
            Lesson(
                "social_distrust",
                "Cliente diffidente",
                {"text": "cliente whatsapp diffidente chiede preventivo braccio"},
                {"resonate_with": "cliente marzo diffidente whatsapp"},
                "speech",
            ),
        ],
    ),
    Phase(
        id="world",
        label="🌍 Mondo",
        description="Mix multimodale — come esperienza reale.",
        lessons=[
            Lesson(
                "world_mix",
                "Voce + testo",
                {"text": "Ciao preventivo braccio", "tone_hz": 330.0},
                {},
                "full",
            ),
        ],
    ),
]


def get_phase(phase_id: str) -> Phase | None:
    for p in CURRICULUM:
        if p.id == phase_id:
            return p
    return None


def phases_dict() -> list[dict[str, Any]]:
    return [
        {
            "id": p.id,
            "label": p.label,
            "description": p.description,
            "lessons": [{"id": l.id, "label": l.label} for l in p.lessons],
        }
        for p in CURRICULUM
    ]
