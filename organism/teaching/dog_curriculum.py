"""Training visivo cane — razze/pose diverse per generalizzazione."""

from __future__ import annotations

import time
from typing import Any, Callable

# Frasi variate — stesso oggetto, contesti diversi (generalizzazione)
_DOG_PHRASES: list[str] = [
    "questo è un cane",
    "vedo un cane marrone",
    "il cane sta correndo",
    "il cane è seduto",
    "un cane con la coda lunga",
    "cane al guinzaglio",
    "cane che abbaia",
    "cane piccolo bianco",
    "cane grande nero",
    "cane nel prato",
    "cane che gioca",
    "cane che dorme",
    "cucciolo di cane",
    "cane da pastore",
    "cane labrador",
    "cane pastore tedesco",
    "cane che salta",
    "cane nella strada",
    "cane al parco",
    "cane con le orecchie lunghe",
]

# Seed diversi → immagini diverse da picsum
_DOG_SEEDS: list[str] = [
    "cane", "dog", "puppy", "labrador", "shepherd", "beagle", "husky", "poodle",
    "dog-run", "dog-sit", "dog-play", "dog-park", "dog-brown", "dog-white",
    "dog-black", "dog-small", "dog-big", "dog-walk", "dog-sleep", "dog-jump",
    "dog1", "dog2", "dog3", "dog4", "dog5", "dog6", "dog7", "dog8", "dog9", "dog10",
    "cane1", "cane2", "cane3", "cane4", "cane5", "cane6", "cane7", "cane8",
    "cane-corre", "cane-dorme", "cane-gioca", "cane-parco", "cane-strada",
    "cane-marrone", "cane-bianco", "cane-nero", "cane-piccolo", "cane-grande",
    "cucciolo", "pastore", "labrador-retriever", "golden-retriever",
]


def run_dog_curriculum(
    teach_fn: Callable[..., dict[str, Any]],
    *,
    limit: int = 50,
    pause_s: float = 0.3,
    hd_size: int = 256,
) -> dict[str, Any]:
    """Insegna cane con immagini e frasi variate."""
    from organism.teaching.web_fetch import fetch_image_rgba, picsum_url

    taught = 0
    errors: list[str] = []
    seeds = (_DOG_SEEDS * ((limit // len(_DOG_SEEDS)) + 1))[:limit]
    for i, seed in enumerate(seeds):
        phrase = _DOG_PHRASES[i % len(_DOG_PHRASES)]
        url = picsum_url(seed, size=hd_size)
        try:
            img = fetch_image_rgba(url, size=hd_size)
            r = teach_fn(
                url=url,
                name="cane",
                phrase=phrase,
                kind="object",
                image_prefetch=img,
            )
            if r.get("ok") or r.get("consolidated") or r.get("learned"):
                taught += 1
        except Exception as e:
            errors.append(f"{seed}:{e!s}"[:80])
        time.sleep(pause_s)
    return {
        "ok": taught > 0,
        "taught": taught,
        "total": len(seeds),
        "target_object": "cane",
        "errors": errors[:15],
    }
