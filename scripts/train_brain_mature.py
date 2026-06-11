#!/usr/bin/env python3
"""Training maturo — lessico ampio, visione, dialoghi guidati, probe diversità."""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://inkconscius.eu/organism"
LOG = Path("/tmp/organism_train_brain_mature.log")


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            last_err = e
            time.sleep(2**attempt)
    raise last_err  # type: ignore[misc]


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=120) as resp:
        return json.loads(resp.read().decode())


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


# Lessico bilanciato — non solo saluti
VOCAB_BLOCKS: list[list[str]] = [
    ["mondo", "cielo", "terra", "acqua", "fuoco", "luce", "ombra", "vento"],
    ["pensiero", "memoria", "voce", "parola", "domanda", "risposta", "silenzio"],
    ["organism", "cervello", "neurone", "sinapsi", "imparare", "capire", "sentire"],
    ["mare", "montagna", "albero", "fiore", "casa", "strada", "notte", "giorno"],
    ["felice", "curioso", "calmo", "attento", "sorpreso", "grato"],
    ["vedere", "ascoltare", "parlare", "guardare", "chiedere", "spiegare"],
    ["rosso", "verde", "blu", "giallo", "bianco", "nero"],
    ["libro", "mela", "palla", "cassa", "tavolo", "sedia"],
]

DIALOGUES: list[tuple[str, str]] = [
    ("chi sei", "sono organism sto imparando"),
    ("cosa pensi", "penso a quello che sento e imparo"),
    ("cosa vedi", "vedo con gli occhi e ricordo"),
    ("cos'è la gravità", "la gravità attira le cose verso terra"),
    ("dove sei", "sono qui nel mio spazio"),
    ("perché impari", "imparo per capire il mondo"),
    ("mi capisci", "cerco di capirti"),
    ("raccontami", "ti racconto quello che so"),
]

VISION_OBJECTS = [
    ("mare", "horizon", {"r": 40, "g": 120, "b": 200}, "questo è il mare"),
    ("mela", "circle", {"r": 200, "g": 50, "b": 40}, "questa è una mela rossa"),
    ("libro", "square", {"r": 50, "g": 80, "b": 200}, "questo è un libro blu"),
    ("albero", "square", {"r": 30, "g": 140, "b": 50}, "questo è un albero verde"),
]


def shape_gray(size: int, shape: str, lum: int = 220) -> list[int]:
    grid = [[20] * size for _ in range(size)]
    m = size // 4
    if shape == "horizon":
        for y in range(size):
            for x in range(size):
                grid[y][x] = (200 + (x % 12)) if y < size // 3 else 60 + (x + y) % 35
        return [grid[y][x] for y in range(size) for x in range(size)]
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


def rgba_from_gray(gray_flat: list[int], rgb: dict[str, float], size: int = 64) -> list[int]:
    out: list[int] = []
    r, g, b = int(rgb["r"]), int(rgb["g"]), int(rgb["b"])
    for lum in gray_flat:
        out.extend([r, g, b, 255] if lum > 100 else [20, 20, 20, 255])
    return out


def main() -> None:
    LOG.write_text(f"=== train_brain_mature @ {BASE} ===\n")
    st = get("/api/baby/state")
    if not st.get("born"):
        raise SystemExit("organismo non nato")
    stats = st.get("stats", {})
    log(f"partenza: {stats.get('neurons')} neuroni, {stats.get('synapses')} sinapsi")

    log("--- lessico bilanciato (hear) ---")
    for block in VOCAB_BLOCKS:
        for word in block:
            post("/api/baby/hear", {"phrase": word})
            time.sleep(0.08)
        post("/api/baby/hear", {"phrase": " ".join(block[:4])})
        time.sleep(0.15)

    log("--- dialoghi semantici (associazioni, non script) ---")
    for when, say in DIALOGUES:
        for _ in range(3):
            post("/api/baby/teach-dialogue", {"when": when, "say": say})
        time.sleep(0.1)

    log("--- visione oggetti ---")
    for name, shape, rgb, phrase in VISION_OBJECTS:
        gray = shape_gray(64, shape)
        body = {
            "phrase": phrase,
            "image_gray": gray,
            "image_rgba": rgba_from_gray(gray, rgb),
            "image_w": 64,
            "image_h": 64,
            "color_rgb": rgb,
        }
        for _ in range(3):
            post("/api/baby/teach-attention", body)
        post("/api/baby/hear", {"phrase": phrase})
        time.sleep(0.12)

    log("--- probe parlato ---")
    for q in [
        "ciao",
        "chi sei",
        "cosa pensi",
        "grazie",
        "cos'è la gravità",
        "questo è il mare",
    ]:
        spoke = post("/api/baby/sense", {"text": q})["moment"].get("spoke", "")
        log(f"  {q!r} → «{spoke[:80]}»")

    log("--- health check ---")
    try:
        h = get("/api/baby/health")
        log(f"  diversità={h.get('speech_diversity')} dom={h.get('lexicon_dominance')}")
        log(f"  neuroni={h.get('neurons')} sinapsi={h.get('synapses')}")
    except Exception as e:
        log(f"  health skip: {e}")

    log("=== maturo pronto ===")


if __name__ == "__main__":
    main()
