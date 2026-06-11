"""Ricerca su come sentiamo/vediamo — Wikipedia in italiano."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from mind.memory import MemoryGraph
from mind.types import Fragment


def research_human_senses(
    memory: MemoryGraph,
    *,
    topics: list[str] | None = None,
    lang: str = "it",
) -> list[dict[str, Any]]:
    """Fetch Wikipedia summaries — argomenti dal DNA del neonato, non frasi fisse."""
    titles = topics or []
    results = []
    for title in titles:
        summary = _wiki_summary(title, lang=lang)
        if not summary:
            continue
        fid = f"research_{title.lower().replace(' ', '_')}"
        hooks = [w.lower() for w in title.split() if len(w) > 2]
        hooks += [w.lower() for w in summary.split()[:12] if len(w) > 4][:6]
        memory.add(
            Fragment(
                id=fid,
                title=summary[:200],
                weight=0.4,
                sensation_id="curiosity_knowledge",
                hooks=hooks,
                functions=["learned"],
            )
        )
        results.append({"topic": title, "title": summary[:120], "id": fid})
    return results


def _wiki_summary(title: str, *, lang: str = "it") -> str:
    host = f"https://{lang}.wikipedia.org"
    url = f"{host}/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OrganismNursery/0.5"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
            return str(data.get("extract", ""))
    except Exception:
        return ""
