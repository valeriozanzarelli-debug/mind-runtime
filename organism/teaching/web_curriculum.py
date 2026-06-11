"""Curriculum visivo da internet — oggetti, volti, emozioni."""

from __future__ import annotations

import time
from typing import Any, Callable

from organism.teaching.web_fetch import fetch_image_rgba, picsum_url

# Immagini stabili da internet (seed per nome — senza API Wikimedia)
CURATED_URLS: dict[str, list[str]] = {
    name: [picsum_url(name)]
    for name in (
        "mela", "banana", "cane", "gatto", "casa", "albero", "sole", "mare",
        "libro", "palla", "macchina", "fiori", "felice", "triste", "arrabbiato",
        "sorpreso", "calmo", "spaventato", "volto", "persona",
    )
}

OBJECT_LESSONS: list[tuple[str, str, str]] = [
    ("mela", "red apple fruit", "questa è una mela"),
    ("banana", "banana fruit yellow", "questa è una banana"),
    ("cane", "dog animal portrait", "questo è un cane"),
    ("gatto", "cat animal face", "questo è un gatto"),
    ("casa", "house building", "questa è una casa"),
    ("albero", "tree nature green", "questo è un albero"),
    ("sole", "sun sky bright", "questo è il sole"),
    ("mare", "sea ocean horizon", "questo è il mare"),
    ("libro", "book reading", "questo è un libro"),
    ("palla", "ball round toy", "questa è una palla"),
    ("macchina", "car automobile", "questa è una macchina"),
    ("fiori", "flowers colorful", "questi sono fiori"),
]

EMOTION_LESSONS: list[tuple[str, str, str]] = [
    ("felice", "happy smiling face person", "questa persona è felice"),
    ("triste", "sad crying face person", "questa persona è triste"),
    ("arrabbiato", "angry face expression", "questa persona è arrabbiata"),
    ("sorpreso", "surprised face expression", "questa persona è sorpresa"),
    ("calmo", "calm peaceful face person", "questa persona è calma"),
    ("spaventato", "fear scared face person", "questa persona ha paura"),
]

FACE_LESSONS: list[tuple[str, str, str]] = [
    ("volto", "human face portrait", "questo è un volto"),
    ("persona", "person portrait face", "questa è una persona"),
]


def run_web_curriculum(
    teach_fn: Callable[..., dict[str, Any]],
    *,
    objects: bool = True,
    emotions: bool = True,
    faces: bool = True,
    images_per_lesson: int = 1,
    pause_s: float = 1.0,
) -> dict[str, Any]:
    """Esegue curriculum — teach_fn = baby.teach_from_web_image."""
    log: list[dict[str, Any]] = []
    errors: list[str] = []
    taught_objects = 0
    taught_emotions = 0
    taught_faces = 0

    def _teach_block(
        lessons: list[tuple[str, str, str]],
        kind: str,
    ) -> int:
        count = 0
        for name, query, phrase in lessons:
            urls = list(CURATED_URLS.get(name, [picsum_url(name)]))
            if not urls:
                errors.append(f"no_image:{name}")
                continue
            deduped: list[str] = []
            seen: set[str] = set()
            for u in urls:
                if u not in seen:
                    seen.add(u)
                    deduped.append(u)
            urls = deduped
            for url in urls[:images_per_lesson]:
                try:
                    r = teach_fn(
                        url,
                        name=name,
                        phrase=phrase,
                        kind=kind,
                    )
                    log.append({"name": name, "kind": kind, "url": url[:80], **r})
                    if r.get("ok"):
                        count += 1
                except Exception as e:
                    errors.append(f"{name}:{e!s}"[:120])
                time.sleep(pause_s)
        return count

    if objects:
        taught_objects = _teach_block(OBJECT_LESSONS, "object")
    if faces:
        taught_faces = _teach_block(FACE_LESSONS, "face")
    if emotions:
        taught_emotions = _teach_block(EMOTION_LESSONS, "emotion")

    return {
        "ok": taught_objects + taught_emotions + taught_faces > 0,
        "taught_objects": taught_objects,
        "taught_faces": taught_faces,
        "taught_emotions": taught_emotions,
        "lessons": log[-40:],
        "errors": errors[:20],
    }
