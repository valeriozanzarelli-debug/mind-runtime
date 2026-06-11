"""GitHub integration — Baby legge codice dal repository.

Baby può:
1. Leggere file da qualsiasi repository GitHub pubblico
2. Associare nomi di funzioni/classi a descrizioni italiane
3. Usare browse_web() per accedere all'API raw GitHub

Nessun hardcoding: le associazioni emergono dall'assorbimento del codice.
"""

from __future__ import annotations

import re
from typing import Any


def _extract_code_concepts(code: str) -> list[tuple[str, str]]:
    """Estrae concetti dal codice Python: (nome, descrizione_it)."""
    concepts: list[tuple[str, str]] = []

    # Funzioni
    for m in re.finditer(r"^def (\w+)\s*\(([^)]*)\).*?:", code, re.MULTILINE):
        name = m.group(1)
        params = m.group(2).strip()
        if name.startswith("_"):
            desc = f"{name} è una funzione interna con parametri {params}"
        else:
            desc = f"{name} è una funzione che prende {params} come parametri"
        concepts.append((name, desc))

    # Classi
    for m in re.finditer(r"^class (\w+)(?:\([^)]*\))?:", code, re.MULTILINE):
        name = m.group(1)
        concepts.append((name, f"{name} è una classe che definisce un tipo di oggetto"))

    # Import principali
    for m in re.finditer(r"^(?:import|from)\s+([\w.]+)", code, re.MULTILINE):
        mod = m.group(1)
        concepts.append((mod, f"{mod} è un modulo importato nel codice"))

    return concepts[:30]


def _code_to_italian_summary(code: str, filename: str = "") -> str:
    """Genera un sommario italiano del file codice."""
    lines = [ln for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    funcs = re.findall(r"^def (\w+)", code, re.MULTILINE)
    classes = re.findall(r"^class (\w+)", code, re.MULTILINE)

    parts = []
    if filename:
        parts.append(f"Il file {filename} contiene codice Python.")
    if classes:
        parts.append(f"Definisce le classi: {', '.join(classes[:5])}.")
    if funcs:
        parts.append(f"Ha le funzioni: {', '.join(funcs[:8])}.")
    parts.append(f"Totale: circa {len(lines)} righe di codice.")
    return " ".join(parts)


def fetch_github_file(
    owner: str,
    repo: str,
    path: str,
    *,
    branch: str = "main",
) -> dict[str, Any]:
    """Scarica un file da GitHub via raw content."""
    try:
        import urllib.request

        url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "MindRuntime/0.5 (educational)"}
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        return {
            "ok": True,
            "owner": owner,
            "repo": repo,
            "path": path,
            "branch": branch,
            "url": url,
            "content": content,
            "lines": len(content.splitlines()),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:100], "path": path}


def absorb_github_file(
    baby_agent: Any,
    owner: str,
    repo: str,
    path: str,
    *,
    branch: str = "main",
) -> dict[str, Any]:
    """Scarica e assorbe un file GitHub nel cervello di Baby."""
    result = fetch_github_file(owner, repo, path, branch=branch)
    if not result.get("ok"):
        return result

    content = result["content"]
    filename = path.split("/")[-1]

    # Sommario italiano
    summary = _code_to_italian_summary(content, filename)
    baby_agent.composer.absorb(summary, boost=0.6)
    if baby_agent.speech:
        baby_agent.speech.hear(summary[:120], boost=0.2)

    # Concetti estratti
    concepts = _extract_code_concepts(content)
    for name, desc in concepts:
        baby_agent.composer.absorb(f"{name}: {desc}", boost=0.4)
        # Insegna come dialogo: "cos'è {name}" → descrizione
        q = f"cos'è {name}"
        baby_agent.dialogue.teach(q, desc)
        baby_agent.dialogue.teach(q, desc)
        baby_agent.dialogue.teach(q, desc)

    # Assorbi il codice stesso (crea pattern sintattici)
    # Solo le prime N righe per evitare overhead
    code_snippet = "\n".join(content.splitlines()[:80])
    baby_agent.composer.absorb(code_snippet, boost=0.2)

    # Leggi il sommario come testo continuo
    if hasattr(baby_agent, "read"):
        baby_agent.read(summary + " " + " ".join(f"{n} {d}" for n, d in concepts[:10]))

    baby_agent._persist()
    result.update({
        "summary": summary,
        "concepts": len(concepts),
        "vocab_after": baby_agent.composer.lexicon.count,
    })
    return result


def absorb_own_codebase(baby_agent: Any, owner: str = "valeriozanzarelli-debug", repo: str = "mind-runtime") -> dict[str, Any]:
    """Baby legge i file principali del proprio codebase."""
    key_files = [
        "organism/autonomous/baby_agent.py",
        "organism/cognition/working_memory.py",
        "organism/motor/compose_speech.py",
        "organism/cognition/neural_lexicon.py",
        "mind/runtime.py",
        "mind/memory.py",
    ]
    results = []
    for path in key_files:
        r = absorb_github_file(baby_agent, owner, repo, path)
        results.append({
            "path": path,
            "ok": r.get("ok"),
            "concepts": r.get("concepts", 0),
            "error": r.get("error"),
        })
    return {
        "ok": True,
        "files_processed": len([r for r in results if r.get("ok")]),
        "total_vocab": baby_agent.composer.lexicon.count,
        "results": results,
    }
