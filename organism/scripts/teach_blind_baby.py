#!/usr/bin/env python3
"""Insegna al baby (fase cieca) — parole e frasi via API locale."""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/organism"

# Fase cieca: chiavi solo udito/testo, niente visione
CURRICULUM: list[tuple[str, str, str]] = [
    # (stimulus_key, phrase, categoria)
    ("blind:word:ciao", "ciao", "parola"),
    ("blind:word:si", "sì", "parola"),
    ("blind:word:no", "no", "parola"),
    ("blind:word:grazie", "grazie", "parola"),
    ("blind:word:aiuto", "aiuto", "parola"),
    ("blind:word:mamma", "mamma", "parola"),
    ("blind:word:papa", "papà", "parola"),
    ("blind:word:acqua", "acqua", "parola"),
    ("blind:word:bene", "bene", "parola"),
    ("blind:word:male", "male", "parola"),
    ("blind:word:voce", "voce", "parola"),
    ("blind:word:qui", "qui", "parola"),
    ("blind:phrase:saluto", "ciao, come stai", "frase"),
    ("blind:phrase:sto_bene", "sto bene", "frase"),
    ("blind:phrase:grazie_mille", "grazie mille", "frase"),
    ("blind:phrase:aiuto", "ho bisogno di aiuto", "frase"),
    ("blind:phrase:sono_qui", "sono qui", "frase"),
    ("blind:phrase:ti_sento", "ti sento", "frase"),
    ("blind:phrase:chi_sei", "chi sei tu", "frase"),
    ("blind:phrase:imparare", "voglio imparare", "frase"),
    ("blind:phrase:sono_bambino", "sono un bambino", "frase"),
    ("blind:phrase:mi_chiamo", "mi chiamo organism", "frase"),
    ("blind:phrase:parla", "parla con me", "frase"),
    ("blind:phrase:ascolto", "ti ascolto", "frase"),
    ("blind:phrase:ripeti", "ripeti per favore", "frase"),
]


def post(path: str, body: dict) -> dict:
    url = f"{BASE}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as resp:
        return json.loads(resp.read().decode())


def teach_item(key: str, phrase: str) -> dict:
    last = {}
    for i in range(3):
        last = post("/api/baby/teach", {"phrase": phrase, "stimulus_key": key})
        if last.get("learned"):
            break
    # rinforzo uditivo
    post("/api/baby/sense", {"text": phrase})
    return last


def main() -> None:
    print(f"=== Insegnamento fase cieca → {BASE} ===\n")
    learned = 0
    for key, phrase, cat in CURRICULUM:
        r = teach_item(key, phrase)
        ok = r.get("learned", False)
        if ok:
            learned += 1
        mark = "✓" if ok else f"{r.get('count', '?')}/3"
        print(f"  [{mark}] {cat:6} «{phrase}»  (+{r.get('new_wires', 0)} fili)")
    state = get("/api/baby/state")
    phrases = state.get("learned_phrases", {})
    print(f"\n=== Fatto: {learned}/{len(CURRICULUM)} consolidati ===")
    print(f"sinapsi: {state.get('brain', {}).get('synapses', '?')} (+{state.get('brain', {}).get('synapses_grown', 0)})")
    print(f"sillabe: {state.get('syllables_known')}")
    print(f"frasi in memoria: {len(phrases)}")
    for k, v in sorted(phrases.items(), key=lambda x: x[1]):
        print(f"    · «{v}»")


if __name__ == "__main__":
    main()
