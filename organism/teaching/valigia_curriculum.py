"""Curriculum visivo HD — solo valigie, immagini reali da Wikimedia."""

from __future__ import annotations

import time
from typing import Any, Callable

from organism.teaching.web_fetch import fetch_image_hd, wikimedia_image_urls

OBJECT_NAME = "valigia"

# Immagini stabili (tag lock) — fallback se Wikimedia rate-limita
LOREMFLICKR_TAGS = ("suitcase", "luggage", "baggage", "travel-bag")
LOCK_COUNT = 24

SEARCH_QUERIES = (
    "suitcase",
    "rolling suitcase",
    "luggage",
    "travel suitcase",
    "hard shell suitcase",
    "valigia",
    "baggage",
    "airport luggage",
    "suitcase photography",
    "leather suitcase",
)


def loremflickr_valigia_urls(*, count: int = LOCK_COUNT) -> list[str]:
    """Foto reali valigie/viaggio — 1920×1080, seed per varietà."""
    urls: list[str] = []
    for i in range(count):
        tag = LOREMFLICKR_TAGS[i % len(LOREMFLICKR_TAGS)]
        urls.append(f"https://loremflickr.com/1920/1080/{tag}?lock={i + 17}")
    return urls


def collect_valigia_urls(*, per_query: int = 5, thumb_width: int = 1280) -> list[str]:
    """Raccoglie URL unici di valigie — Wikimedia + fallback LoremFlickr HD."""
    seen: set[str] = set()
    out: list[str] = []
    for q in SEARCH_QUERIES[:4]:
        try:
            urls = wikimedia_image_urls(q, limit=per_query, thumb_width=thumb_width)
        except Exception:
            urls = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                out.append(u)
        time.sleep(1.2)
    for u in loremflickr_valigia_urls():
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def purge_valigia_bindings(visual_binder: Any) -> int:
    """Rimuove binding valigia vecchi (curriculum rumoroso) prima del retrain HD."""
    removed = 0
    name = OBJECT_NAME
    visual_binder._prototypes.pop(name, None)
    visual_binder._user_taught.discard(name)
    for sig, n in list(visual_binder._object_names.items()):
        if n == name:
            del visual_binder._object_names[sig]
            removed += 1
    for sig in list(visual_binder._object_features.keys()):
        if visual_binder._object_names.get(sig) == name:
            del visual_binder._object_features[sig]
    return removed


def run_valigia_hd_curriculum(
    teach_fn: Callable[..., dict[str, Any]],
    *,
    purge: bool = True,
    per_query: int = 8,
    pause_s: float = 0.8,
) -> dict[str, Any]:
    """Insegna molte valigie HD — riconoscimento rapido su forma+colore reali."""
    urls = collect_valigia_urls(per_query=per_query)
    if not urls:
        return {"ok": False, "reason": "no_images_found", "queries": list(SEARCH_QUERIES)}

    log: list[dict[str, Any]] = []
    errors: list[str] = []
    taught = 0
    for i, url in enumerate(urls):
        try:
            img = fetch_image_hd(url, max_dim=1920)
            w, h = int(img.get("image_w", 0)), int(img.get("image_h", 0))
            if w < 200 or h < 200:
                errors.append(f"small:{url[:50]}")
                continue
            phrase = f"questa è una {OBJECT_NAME}"
            r = teach_fn(
                url=url,
                name=OBJECT_NAME,
                phrase=phrase,
                kind="object",
                image_prefetch=img,
            )
            if r.get("consolidated") or r.get("name") == OBJECT_NAME or r.get("learned"):
                taught += 1
            log.append(
                {
                    "i": i,
                    "ok": bool(r.get("ok") or r.get("consolidated")),
                    "w": w,
                    "h": h,
                    "url": url[:80],
                }
            )
        except Exception as e:
            errors.append(f"{i}:{e!s}"[:100])
        time.sleep(pause_s)

    return {
        "ok": taught >= 5,
        "object": OBJECT_NAME,
        "taught": taught,
        "total_urls": len(urls),
        "hd": True,
        "lessons": log[-20:],
        "errors": errors[:15],
        "purge": purge,
    }
