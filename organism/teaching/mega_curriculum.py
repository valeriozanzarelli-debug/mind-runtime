"""Curriculum 1000+ oggetti — visione HD da internet."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from organism.teaching.web_fetch import fetch_image_rgba, picsum_url

_DATA = Path(__file__).resolve().parents[2] / "data" / "objects_it_1000.txt"
HD_SIZE = 256


def load_object_names(*, limit: int = 0) -> list[str]:
    if not _DATA.exists():
        return []
    names: list[str] = []
    for line in _DATA.read_text(encoding="utf-8").splitlines():
        w = line.strip().lower()
        if w and not w.startswith("#"):
            names.append(w)
        if limit and len(names) >= limit:
            break
    return names


def phrase_for(name: str) -> str:
    vowels = "aeiouàèéìòù"
    if name and name[0] in vowels:
        return f"questa è una {name}"
    return f"questo è un {name}"


def run_mega_curriculum(
    teach_fn: Callable[..., dict[str, Any]],
    *,
    limit: int = 1000,
    pause_s: float = 0.35,
    hd_size: int = HD_SIZE,
    batch_log: int = 50,
) -> dict[str, Any]:
    """Insegna oggetti da lista — immagine stabile per seed."""
    names = load_object_names(limit=limit)
    if not names:
        return {"ok": False, "reason": "no_object_list", "path": str(_DATA)}

    log: list[dict[str, Any]] = []
    errors: list[str] = []
    taught = 0
    for i, name in enumerate(names):
        url = picsum_url(name, size=hd_size)
        try:
            img = fetch_image_rgba(url, size=hd_size)
            r = teach_fn(
                url=url,
                name=name,
                phrase=phrase_for(name),
                kind="object",
                image_prefetch=img,
            )
            if r.get("ok"):
                taught += 1
            if i % batch_log == 0 or not r.get("ok"):
                log.append({"i": i, "name": name, "ok": r.get("ok"), "url": url[:60]})
        except Exception as e:
            errors.append(f"{name}:{e!s}"[:100])
        time.sleep(pause_s)

    return {
        "ok": taught > 0,
        "taught": taught,
        "total": len(names),
        "hd_size": hd_size,
        "lessons": log[-30:],
        "errors": errors[:25],
    }
