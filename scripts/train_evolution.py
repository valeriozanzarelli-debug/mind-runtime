#!/usr/bin/env python3
"""Training completo — linguaggio + coscienza + legami visivi (via API)."""

from __future__ import annotations

import json
import sys
import time
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


def teach_dialogue(when: str, say: str) -> bool:
    for _ in range(3):
        r = post("/api/baby/teach-dialogue", {"when": when, "say": say})
        if r.get("learned"):
            post("/api/baby/sense", {"text": when})
            return True
    return False


# Griglie sintetiche = "scene" per training cieco (luce / buio / contrasto)
SCENES = {
    "bright": [220] * (64 * 64),
    "dark": [25] * (64 * 64),
    "contrast": [240 if (i // 64 + i % 64) % 2 == 0 else 30 for i in range(64 * 64)],
}

VISUAL = [
    ("bright", "vedo la luce", "c'è molta luce davanti a me"),
    ("dark", "vedo il buio", "è buio, quasi non vedo nulla"),
    ("contrast", "vedo qualcosa", "c'è qualcosa con forme e contrasto"),
]

DIALOGUE = [
    ("ciao", "ciao"),
    ("come stai", "sto bene, sto imparando piano piano"),
    ("cosa vedi", "vedo con i miei occhi quello che mi mostri"),
    ("cosa pensi", "penso quando la coscienza si accende e le parole arrivano"),
    ("parlami", "parlo quando sono cosciente e ho qualcosa da dire"),
]

REFLECT = ["cosa pensi", "descrivi il mondo", "raccontami una storia incredibile"]


def main() -> None:
    print(f"=== Training evoluzione @ {BASE} ===\n")
    post("/api/baby/birth", {})

    print("--- dialogo base ---")
    for when, say in DIALOGUE:
        ok = teach_dialogue(when, say)
        print(f"  [{'✓' if ok else '✗'}] {when}")

    print("\n--- legami visivi (scene sintetiche) ---")
    for scene_key, when, say in VISUAL:
        gray = SCENES[scene_key]
        for n in range(3):
            r = post("/api/baby/teach", {
                "phrase": say,
                "image_gray": gray,
                "image_w": 64,
                "image_h": 64,
            })
            if r.get("learned"):
                post("/api/baby/sense", {"text": when, "image_gray": gray, "image_w": 64, "image_h": 64})
                print(f"  [✓] {scene_key} → «{say[:40]}…»")
                break
        else:
            print(f"  [✗] {scene_key}")

    print("\n--- flow coscienza ---")
    for _ in range(3):
        m = post("/api/baby/flow", {"image_gray": SCENES["bright"], "image_w": 64, "image_h": 64})["moment"]
        ws = m.get("consciousness", {})
        print(f"  ignition={ws.get('ignition')} conscious={ws.get('conscious')} spoke={m.get('spoke','')[:60]}")

    print("\n--- probe ---")
    for prompt in REFLECT:
        m = post("/api/baby/reflect", {"prompt": prompt})["moment"]
        print(f"  [{prompt}] conscious={m.get('consciousness',{}).get('conscious')} len={len(m.get('spoke',''))}")

    st = json.loads(urllib.request.urlopen(f"{BASE}/api/baby/state", timeout=30).read())
    print(f"\n=== {len(st.get('dialogue_pairs',[]))} dialoghi | "
          f"{len(st.get('learned_phrases',{}))} scene | "
          f"{st.get('consciousness',{}).get('ignitions',0)} ignitioni ===")


if __name__ == "__main__":
    main()
