"""Navigazione web autonoma — legge pagine e estrae conoscenza."""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any

USER_AGENT = "OrganismBaby/1.0 (educational; inkconscius.eu)"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = False
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip = False
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3"):
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip and data.strip():
            self._chunks.append(data.strip())

    def text(self) -> str:
        raw = " ".join(self._chunks)
        return re.sub(r"\s+", " ", raw).strip()


def fetch_page_text(url: str, *, timeout: float = 25.0, max_chars: int = 8000) -> dict[str, Any]:
    """Scarica URL e restituisce testo leggibile."""
    if not url.startswith(("http://", "https://")):
        return {"ok": False, "reason": "invalid_url"}
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read(512_000)
    except Exception as e:
        return {"ok": False, "reason": str(e)[:120], "url": url[:120]}

    if "html" in ctype.lower() or raw[:15].lower().startswith(b"<!doctype") or b"<html" in raw[:500].lower():
        try:
            html = raw.decode("utf-8", errors="replace")
        except Exception:
            html = raw.decode("latin-1", errors="replace")
        parser = _TextExtractor()
        parser.feed(html)
        text = parser.text()
        title_m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.I)
        title = title_m.group(1).strip() if title_m else ""
    else:
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = raw.decode("latin-1", errors="replace")
        title = urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]

    text = text[:max_chars]
    words = [w for w in re.findall(r"[a-zàèéìòùA-ZÀÈÉÌÒÙ']+", text.lower()) if len(w) > 2]
    return {
        "ok": bool(text),
        "url": url[:200],
        "title": title[:200],
        "text": text,
        "word_count": len(words),
        "keywords": list(dict.fromkeys(words))[:40],
    }


def search_duckduckgo_lite(query: str, *, limit: int = 5) -> list[dict[str, str]]:
    """Ricerca leggera — restituisce titoli e URL (no API key)."""
    q = urllib.parse.quote_plus(query.strip())
    url = f"https://lite.duckduckgo.com/lite/?q={q}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    results: list[dict[str, str]] = []
    for m in re.finditer(
        r'class="result-link"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
        html,
        re.I,
    ):
        href, title = m.group(1), m.group(2).strip()
        if href.startswith("//"):
            href = "https:" + href
        if href.startswith("http"):
            results.append({"url": href, "title": title})
        if len(results) >= limit:
            break
    return results
