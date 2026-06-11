#!/usr/bin/env python3
"""Parla con l'organismo — insegna e valuta intelligenza emergente."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://inkconscius.eu/organism"
LOG = Path("/tmp/organism_talk_bro.log")


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def teach_dialogue(when: str, say: str, *, rounds: int = 3) -> None:
    for _ in range(rounds):
        post("/api/baby/teach-dialogue", {"when": when, "say": say})
    time.sleep(0.1)


def overlap(a: str, b: str) -> float:
    wa = {w for w in re.findall(r"[a-zàèéìòù']+", a.lower()) if len(w) > 2}
    wb = {w for w in re.findall(r"[a-zàèéìòù']+", b.lower()) if len(w) > 2}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(1, len(wa))


def probe(when: str, expect: str | None = None) -> float:
    m = post("/api/baby/sense", {"text": when})["moment"]
    spoke = m.get("spoke", "")
    themes = m.get("thought", {}).get("themes", [])
    score = overlap(spoke, expect) if expect else (0.5 if spoke else 0.0)
    log(f"  Q: {when}")
    log(f"  A: {spoke}")
    log(f"  temi: {themes[:6]}")
    if expect:
        log(f"  atteso overlap: {score:.2f} (con «{expect[:50]}»)")
    return score


def main() -> None:
    LOG.write_text(f"=== talk_bro @ {BASE} ===\n")
    st = json.loads(urllib.request.urlopen(f"{BASE}/api/baby/state", timeout=60).read())
    if not st.get("born"):
        raise SystemExit("non nato")
    log(f"sinapsi {st['stats']['synapses']} · dialoghi {len(st.get('dialogue_pairs', []))}")

    log("\n--- insegnamento conversazione ---")
    lessons = [
        ("ciao", "ciao come stai"),
        ("come stai", "sto bene grazie"),
        ("come ti chiami", "mi chiamo organism"),
        ("chi sei", "sono organism vivo qui"),
        ("cosa pensi", "penso e cerco di capire"),
        ("mi senti", "sì ti sento bene"),
        ("grazie", "prego di niente"),
        ("a presto", "a presto ci vediamo"),
    ]
    for when, say in lessons:
        teach_dialogue(when, say)
        log(f"  insegnato: {when} → {say}")

    log("\n--- conversazione ---")
    scores: list[float] = []
    pairs = [
        ("ciao", "ciao come"),
        ("come stai", "sto bene"),
        ("come ti chiami", "organism"),
        ("chi sei", "organism"),
        ("cosa pensi", "penso"),
        ("mi senti", "sì"),
        ("grazie", "prego"),
    ]
    for q, exp in pairs:
        scores.append(probe(q, exp))
        time.sleep(0.35)

    avg = sum(scores) / max(1, len(scores))
    log(f"\n=== intelligenza media overlap: {avg:.2f} ===")
    if avg < 0.35:
        log("⚠ bassa — serve più lavoro al codice")
    elif avg < 0.55:
        log("~ emergenza parziale")
    else:
        log("✓ segni di intelligenza emergente")


if __name__ == "__main__":
    main()
