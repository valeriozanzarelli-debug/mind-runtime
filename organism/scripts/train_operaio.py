#!/usr/bin/env python3
"""Allena l'operaio — dialogo vero + prime abilità codice."""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/organism"

# Quando senti X → rispondi Y (non eco cieca)
DIALOGUE: list[tuple[str, str]] = [
    ("ciao", "ciao"),
    ("ciao organism", "ciao"),
    ("ehi", "ciao"),
    ("come stai", "sto bene"),
    ("come va", "sto bene, grazie"),
    ("tutto bene", "sì, sto bene"),
    ("chi sei", "mi chiamo organism"),
    ("chi sei tu", "sono organism, un operaio che impara"),
    ("come ti chiami", "mi chiamo organism"),
    ("cosa sei", "sono un organismo che impara a parlare e scrivere codice"),
    ("cosa sai fare", "parlo, capisco e sto imparando a scrivere codice"),
    ("grazie", "prego"),
    ("grazie mille", "di niente"),
    ("aiuto", "sono qui, dimmi cosa ti serve"),
    ("ho bisogno di aiuto", "sono qui, ti ascolto"),
    ("parla", "ti ascolto"),
    ("parla con me", "ci sono, dimmi"),
    ("ti sento", "anch'io ti sento"),
    ("voglio imparare", "anch'io, insegnami"),
    ("sei vivo", "sì, ci sto provando"),
    ("addio", "ciao, a presto"),
    ("buonanotte", "notte, riposa"),
    ("buongiorno", "buongiorno"),
    ("ok", "ok"),
    ("sì", "sì"),
    ("no", "no"),
]

CODE: list[tuple[str, str]] = [
    ("scrivi hello in python", 'print("hello")'),
    ("stampa ciao", 'print("ciao")'),
    ("somma uno piu uno", "1 + 1"),
    ("scrivi una funzione che somma", "def somma(a, b):\n    return a + b"),
]


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def teach_dialogue(when: str, say: str, *, kind: str = "speech") -> dict:
    last: dict = {}
    for _ in range(3):
        last = post("/api/baby/teach-dialogue", {"when": when, "say": say, "kind": kind})
        if last.get("learned"):
            break
    post("/api/baby/sense", {"text": when})
    return last


def main() -> None:
    print(f"=== Training operaio @ {BASE} ===\n")
    n = 0
    for when, say in DIALOGUE:
        r = teach_dialogue(when, say)
        if r.get("learned"):
            n += 1
        print(f"  [{'✓' if r.get('learned') else '?'}] «{when}» → «{say}»")
    print("\n--- codice ---\n")
    for when, code in CODE:
        r = teach_dialogue(when, code, kind="code")
        if r.get("learned"):
            n += 1
        print(f"  [{'✓' if r.get('learned') else '?'}] «{when}» → codice")
    print("\n--- prova dialogo ---\n")
    for q, expect in [
        ("ciao", "ciao"),
        ("come stai", "sto bene"),
        ("chi sei tu", "sono organism"),
        ("grazie", "prego"),
        ("scrivi hello in python", None),
    ]:
        m = post("/api/baby/sense", {"text": q})["moment"]
        out = m.get("code") or m.get("spoke", "")
        ok = expect is None or expect in out
        print(f"  tu: «{q}»\n  lui: «{out[:80]}» {'✓' if ok else '✗'}")


if __name__ == "__main__":
    main()
