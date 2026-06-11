#!/usr/bin/env python3
"""Training emergente — parla da solo, vede oggetti, niente frasi hardcoded."""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://inkconscius.eu/organism"
LOG = Path("/tmp/organism_train_emergent.log")


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


def circle_gray(size: int = 64) -> list[int]:
    grid = [[20] * size for _ in range(size)]
    cx = cy = size // 2
    r = size // 5
    for y in range(size):
        for x in range(size):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                grid[y][x] = 220
    return [grid[y][x] for y in range(size) for x in range(size)]


def square_gray(size: int = 64) -> list[int]:
    grid = [[15] * size for _ in range(size)]
    m = size // 4
    for y in range(m, size - m):
        for x in range(m, size - m):
            grid[y][x] = 230
    return [grid[y][x] for y in range(size) for x in range(size)]


def main() -> None:
    LOG.write_text(f"=== emergent @ {BASE} ===\n")
    st = json.loads(urllib.request.urlopen(f"{BASE}/api/baby/state", timeout=60).read())
    if not st.get("born"):
        raise SystemExit("non nato")
    log(f"riprende: {st.get('neurons')} neuroni, {len(st.get('dialogue_pairs',[]))} dialoghi")

    # Insegna concetti (pathway) — il parlato emerge dal compositore
    concepts = [
        ("ciao", "ciao come stai"),
        ("come stai", "sto bene grazie"),
        ("chi sei", "sono organism"),
        ("cosa vedi", "vedo qualcosa davanti"),
        ("cosa pensi", "penso e cerco capire"),
    ]
    for when, idea in concepts:
        for _ in range(3):
            post("/api/baby/teach-dialogue", {"when": when, "say": idea})
        post("/api/baby/sense", {"text": when})
        time.sleep(0.2)

    # Oggetti visivi
    log("--- oggetti ---")
    for name, gray in [("cerchio", circle_gray()), ("quadrato", square_gray())]:
        for _ in range(4):
            post(
                "/api/baby/teach-object",
                {"name": name, "image_gray": gray, "image_w": 64, "image_h": 64},
            )
        time.sleep(0.3)

    log("--- probe parlato emergente ---")
    for q in ["ciao", "come stai", "chi sei", "cosa pensi"]:
        m = post("/api/baby/sense", {"text": q})["moment"]
        log(f"Q: {q}")
        log(f"A: {m.get('spoke','')}")
        log(f"  temi: {m.get('thought',{}).get('themes',[])[:6]}")
        time.sleep(0.4)

    log("--- probe visione ---")
    for name, gray in [("cerchio", circle_gray()), ("quadrato", square_gray())]:
        m = post("/api/baby/sense", {"image_gray": gray, "image_w": 64, "image_h": 64, "text": "cosa vedi"})["moment"]
        spoke = m.get("spoke", "")
        syms = [s for s in m.get("symbols", []) if "OBJ" in s or name in str(s)]
        log(f"vista {name}: «{spoke[:80]}» syms={syms[:4]}")


if __name__ == "__main__":
    main()
