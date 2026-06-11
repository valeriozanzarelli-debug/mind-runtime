"""Curriculum Wikipedia italiano — testo enciclopedico reale per associazioni profonde.

Usa l'API Wikipedia in italiano per estrarre testi su argomenti rilevanti.
Non hardcoda Q→A: il testo viene assorbito dal lessico neurale e crea
co-attivazioni semantiche naturali.
"""

from __future__ import annotations

import re
import time
from typing import Any

# Argomenti Wikipedia rilevanti per Baby — cognizione, mondo, linguaggio
WIKI_TOPICS = [
    "Neuroscienza",
    "Apprendimento automatico",
    "Linguaggio",
    "Memoria",
    "Coscienza",
    "Cervello umano",
    "Intelligenza",
    "Emozione",
    "Percezione",
    "Linguistica",
    "Filosofia della mente",
    "Psicologia cognitiva",
    "Acqua",
    "Sole",
    "Terra",
    "Gravità",
    "Matematica",
    "Fisica",
    "Algoritmo",
    "Informatica",
]

# Argomenti codice / ingegneria
CODE_TOPICS = [
    "Python (linguaggio di programmazione)",
    "Algoritmo",
    "Struttura dati",
    "Programmazione orientata agli oggetti",
    "Funzione (informatica)",
    "Ricorsione (informatica)",
    "Repository di codice sorgente",
]


def _clean_wiki_text(raw: str) -> str:
    """Pulisce il testo Wikipedia rimuovendo markup e caratteri speciali."""
    text = re.sub(r"==+[^=]+=+=*", "", raw)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _split_into_sentences(text: str, max_len: int = 200) -> list[str]:
    """Divide testo in frasi di lunghezza gestibile."""
    parts = re.split(r"[.!?]+", text)
    result: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p.split()) >= 4:
            result.append(p[:max_len])
    return result


def fetch_wiki_text(topic: str, *, chars: int = 2000) -> dict[str, Any]:
    """Scarica il testo di una voce Wikipedia italiana via API."""
    try:
        import urllib.request
        import urllib.parse
        import json

        params = urllib.parse.urlencode({
            "action": "query",
            "format": "json",
            "titles": topic,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "exsectionformat": "plain",
            "redirects": True,
        })
        url = f"https://it.wikipedia.org/w/api.php?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": "MindRuntime/0.5 (educational)"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return {"ok": False, "topic": topic}
        page = next(iter(pages.values()))
        if "missing" in page:
            return {"ok": False, "topic": topic, "reason": "not_found"}
        extract = page.get("extract", "")[:chars]
        text = _clean_wiki_text(extract)
        sentences = _split_into_sentences(text)
        return {
            "ok": True,
            "topic": topic,
            "title": page.get("title", topic),
            "text": text,
            "sentences": sentences,
            "chars": len(text),
        }
    except Exception as e:
        return {"ok": False, "topic": topic, "error": str(e)[:80]}


def run_wiki_curriculum(
    baby_agent: Any,
    *,
    topics: list[str] | None = None,
    chars_per_topic: int = 1500,
    pause_s: float = 0.2,
    include_code: bool = True,
) -> dict[str, Any]:
    """Scarica e assorbe testi Wikipedia in italiano.

    Ogni testo viene:
    1. Assorbito nel lessico neurale (crea esposizione parole)
    2. Processato frasi per frasi dal cervello (crea sinapsi contestuali)
    3. Alimentato al canale di lettura (stream semantico)
    """
    all_topics = list(topics or WIKI_TOPICS)
    if include_code:
        all_topics.extend(CODE_TOPICS[:3])

    results: list[dict[str, Any]] = []
    absorbed_chars = 0
    failed = 0

    for topic in all_topics:
        page = fetch_wiki_text(topic, chars=chars_per_topic)
        if not page.get("ok"):
            failed += 1
            results.append({"topic": topic, "ok": False})
            if pause_s > 0:
                time.sleep(pause_s * 0.5)
            continue

        text = page["text"]
        sentences = page.get("sentences", [])

        # Assorbi il testo completo nel lessico
        baby_agent.composer.absorb(text, boost=0.35)

        # Assorbi frasi singole (crea associazioni sinaptiche più forti)
        for sent in sentences[:20]:
            baby_agent.composer.absorb(sent, boost=0.2)
            if baby_agent.speech and len(sent.split()) >= 4:
                baby_agent.speech.hear(sent[:100], boost=0.08)

        # Usa il canale di lettura per stream semantico continuo
        if hasattr(baby_agent, "read") and len(text) >= 50:
            baby_agent.read(text[:800])

        absorbed_chars += len(text)
        results.append({
            "topic": topic,
            "ok": True,
            "title": page.get("title", topic),
            "chars": len(text),
            "sentences": len(sentences),
        })

        if pause_s > 0:
            time.sleep(pause_s)

    # Normalizza esposizione finale
    baby_agent.composer.lexicon.squash_overexposed()

    # Hebbian pass
    if baby_agent.org and baby_agent.org.brain.plasticity:
        baby_agent.org.brain.propagate(steps=2)
        baby_agent.org.brain.plasticity.apply_hebbian(
            baby_agent.org.brain, baby_agent.org.brain.tick
        )

    baby_agent._persist()

    return {
        "ok": True,
        "topics_attempted": len(all_topics),
        "topics_ok": len([r for r in results if r.get("ok")]),
        "topics_failed": failed,
        "absorbed_chars": absorbed_chars,
        "vocab_after": baby_agent.composer.lexicon.count,
        "results": results,
    }
