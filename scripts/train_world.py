#!/usr/bin/env python3
"""Insegna sì/no, colori, oggetti — risposte con senso prima della camera live."""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://inkconscius.eu/organism"
LOG = Path("/tmp/organism_train_world.log")


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


def shape_gray(size: int, shape: str, lum: int = 220) -> list[int]:
    grid = [[20] * size for _ in range(size)]
    m = size // 4
    if shape == "circle":
        cx = cy = size // 2
        r = size // 5
        for y in range(size):
            for x in range(size):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    grid[y][x] = lum
    else:
        for y in range(m, size - m):
            for x in range(m, size - m):
                grid[y][x] = lum
    return [grid[y][x] for y in range(size) for x in range(size)]


def teach_dialogue(when: str, say: str, *, rounds: int = 4) -> None:
    for _ in range(rounds):
        post("/api/baby/teach-dialogue", {"when": when, "say": say})
    post("/api/baby/sense", {"text": when})
    time.sleep(0.15)


def teach_object(name: str, gray: list[int], color_rgb: dict[str, float] | None = None) -> None:
    body: dict = {"name": name, "image_gray": gray, "image_w": 64, "image_h": 64}
    if color_rgb:
        body["color_rgb"] = color_rgb
    for _ in range(5):
        post("/api/baby/teach-object", body)
    time.sleep(0.2)


def probe(label: str, body: dict) -> None:
    m = post("/api/baby/sense", body)["moment"]
    spoke = m.get("spoke", "")
    themes = m.get("thought", {}).get("themes", [])[:8]
    log(f"{label}: «{spoke[:90]}» temi={themes}")


def main() -> None:
    LOG.write_text(f"=== train_world @ {BASE} ===\n")
    st = json.loads(urllib.request.urlopen(f"{BASE}/api/baby/state", timeout=60).read())
    if not st.get("born"):
        raise SystemExit("organismo non nato — non fare rebirth")
    log(f"riprende: {st.get('neurons')} neuroni")

    log("--- sì e no ---")
    pairs = [
        ("sì", "sì"),
        ("no", "no"),
        ("è vero", "sì"),
        ("è sbagliato", "no"),
        ("è giusto", "sì"),
        ("non è vero", "no"),
        ("è questo", "sì"),
        ("non è questo", "no"),
    ]
    for when, say in pairs:
        teach_dialogue(when, say)

    log("--- colori (parole) ---")
    color_words = [
        ("rosso", "rosso"),
        ("verde", "verde"),
        ("blu", "blu"),
        ("giallo", "giallo"),
        ("nero", "nero"),
        ("bianco", "bianco"),
        ("che colore è", "è rosso"),
        ("di che colore", "è verde"),
        ("cosa vedi", "vedo qualcosa"),
    ]
    for when, say in color_words:
        teach_dialogue(when, say)

    log("--- oggetti colorati ---")
    objects = [
        ("cerchio", shape_gray(64, "circle"), {"r": 220, "g": 40, "b": 40}),
        ("quadrato", shape_gray(64, "square"), {"r": 40, "g": 180, "b": 60}),
        ("rettangolo", shape_gray(64, "square", 200), {"r": 50, "g": 80, "b": 220}),
    ]
    for name, gray, rgb in objects:
        teach_object(name, gray, rgb)
        teach_dialogue(f"cosa vedi", f"vedo un {name}")
        teach_dialogue(f"che colore è", f"è {rgb_label(rgb)}")

    log("--- probe ---")
    for name, gray, rgb in objects:
        probe(f"vista {name}", {"image_gray": gray, "image_w": 64, "image_h": 64, "color_rgb": rgb, "text": "cosa vedi"})
        probe(f"colore {name}", {"image_gray": gray, "image_w": 64, "image_h": 64, "color_rgb": rgb, "text": "che colore è"})
        probe(f"conferma {name}", {"image_gray": gray, "image_w": 64, "image_h": 64, "color_rgb": rgb, "text": f"è un {name}?"})

    probe("saluto", {"text": "ciao"})
    probe("no semplice", {"text": "è sbagliato"})
    log("=== fine train_world ===")


def rgb_label(rgb: dict[str, float]) -> str:
    r, g, b = rgb.get("r", 0), rgb.get("g", 0), rgb.get("b", 0)
    if r > g and r > b:
        return "rosso"
    if g > r and g > b:
        return "verde"
    if b > r and b > g:
        return "blu"
    return "colore"


if __name__ == "__main__":
    main()
