#!/usr/bin/env python3
"""Dialoga, corregge, verifica apprendimento."""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/organism"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def say(label: str, text: str) -> dict:
    m = post("/api/baby/sense", {"text": text})["moment"]
    spoke = m.get("spoke", "")
    emo = m.get("emotion", {}).get("label", "?")
    tone = m.get("social_tone", {})
    print(f"  [{label}] tu: «{text}»")
    print(f"         lui ({emo}): «{spoke[:120]}»")
    if tone.get("is_angry"):
        print("         ⚠ ha percepito rabbia")
    if tone.get("is_correction"):
        print("         ✎ ha percepito correzione")
    return m


def main() -> None:
    print(f"=== Probe dialogo @ {BASE} ===\n")
    post("/api/baby/birth", {})

    print("--- insegnamento ---")
    for _ in range(3):
        post("/api/baby/teach-dialogue", {"when": "come ti chiami", "say": "mi chiamo organism"})
    for _ in range(3):
        post("/api/baby/teach-dialogue", {"when": "quanti anni hai", "say": "sono appena nato"})

    print("\n--- domande ---")
    say("Q1", "come ti chiami")
    say("Q2", "quanti anni hai")
    say("Q3", "ciao come stai")

    print("\n--- simula errore + correzione ---")
    for _ in range(3):
        post("/api/baby/teach-dialogue", {"when": "dimmi il tuo nome", "say": "mi chiamo mario"})
    say("wrong", "dimmi il tuo nome")
    say("correction", "no sbagliato si dice mi chiamo organism")
    for _ in range(3):
        post("/api/baby/teach-dialogue", {"when": "dimmi il tuo nome", "say": "mi chiamo organism"})
    say("retry", "dimmi il tuo nome")

    print("\n--- tono arrabbiato ---")
    m = say("angry", "BASTA! hai sbagliato tutto!")
    emo = m.get("emotion", {})
    print(f"         emozione interna: {emo.get('dominant')} fear={emo.get('fear')}")

    print("\n--- tono positivo ---")
    say("praise", "bravo! molto bene!")

    st = json.loads(urllib.request.urlopen(f"{BASE}/api/baby/state", timeout=30).read())
    print(
        f"\n=== emozione: {st.get('affect', {}).get('state', {}).get('label')} | "
        f"correzioni: {st.get('corrections', {}).get('corrections', 0)} | "
        f"impulsi: {st.get('brain_pulse', {}).get('pulses', 0)} ==="
    )


if __name__ == "__main__":
    main()
