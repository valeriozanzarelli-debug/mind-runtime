"""Scarica immagini da URL / Wikimedia per curriculum visivo."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "OrganismBaby/1.0 (educational; inkconscius.eu)"
DEFAULT_SIZE = 96


def picsum_url(seed: str, *, size: int = DEFAULT_SIZE) -> str:
    """Immagine stabile per seed — fallback se Wikimedia non disponibile."""
    safe = "".join(c if c.isalnum() else "-" for c in seed.lower())[:40]
    return f"https://picsum.photos/seed/{safe}/{size}/{size}"


def _request(url: str, *, timeout: float = 30.0, retries: int = 4) -> bytes:
    import time

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            last_err = e
            code = getattr(e, "code", None)
            if code in (429, 503) or attempt < retries - 1:
                time.sleep(1.5 * (2**attempt))
                continue
            raise
    raise last_err  # type: ignore[misc]


def fetch_image_rgba(url: str, *, size: int = DEFAULT_SIZE) -> dict[str, Any]:
    """URL → gray flat + rgba flat + rgb medio."""
    raw = _request(url)
    return decode_image_bytes(raw, size=size, source_url=url)


def decode_image_bytes(data: bytes, *, size: int = DEFAULT_SIZE, source_url: str = "") -> dict[str, Any]:
    try:
        from io import BytesIO

        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow richiesto per curriculum web (pip install pillow)") from e

    img = Image.open(BytesIO(data)).convert("RGBA")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    gray: list[int] = []
    rgba: list[int] = []
    r_sum = g_sum = b_sum = 0.0
    for r, g, b, a in pixels:
        lum = int(0.299 * r + 0.587 * g + 0.114 * b)
        gray.append(lum)
        rgba.extend([r, g, b, a])
        r_sum += r
        g_sum += g
        b_sum += b
    n = max(1, len(pixels))
    return {
        "image_gray": gray,
        "image_rgba": rgba,
        "image_w": size,
        "image_h": size,
        "color_rgb": {"r": r_sum / n, "g": g_sum / n, "b": b_sum / n},
        "source_url": source_url,
    }


def wikimedia_image_urls(query: str, *, limit: int = 3, thumb_width: int = 128) -> list[str]:
    """Cerca su Wikimedia Commons — immagini libere."""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",
            "gsrlimit": str(limit),
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": str(thumb_width),
        }
    )
    url = f"https://commons.wikimedia.org/w/api.php?{params}"
    raw = _request(url)
    data = json.loads(raw.decode("utf-8"))
    pages = data.get("query", {}).get("pages", {})
    out: list[str] = []
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        thumb = info.get("thumburl") or info.get("url")
        if thumb and _safe_image_url(thumb):
            out.append(thumb)
    return out


def _safe_image_url(url: str) -> bool:
    if not url.startswith("https://"):
        return False
    if re.search(r"\.(jpe?g|png|webp|gif)(\?|$)", url, re.I):
        return True
    return "thumb" in url or "upload.wikimedia.org" in url


def wikipedia_thumb(title: str, *, width: int = 128) -> str | None:
    """Thumbnail principale di una voce Wikipedia."""
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "pageimages",
            "pithumbsize": str(width),
        }
    )
    url = f"https://it.wikipedia.org/w/api.php?{params}"
    try:
        data = json.loads(_request(url).decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            thumb = page.get("thumbnail", {}).get("source")
            if thumb:
                return thumb
    except Exception:
        return None
    return None
