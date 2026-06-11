#!/usr/bin/env python3
"""Training adolescente — parla, vede, chiede, riconosce oggetti."""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://inkconscius.eu/organism"
LOG = Path("/tmp/organism_train_adolescent.log")


def post(path: str, body: dict) -> dict:
    import time as _time

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
            _time.sleep(2 ** attempt)
    raise last_err  # type: ignore[misc]


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a") as f:
        f.write(msg + "\n")


def teach_dialogue(when: str, say: str, *, rounds: int = 5) -> None:
    for _ in range(rounds):
        post("/api/baby/teach-dialogue", {"when": when, "say": say})
    time.sleep(0.06)


def shape_gray(size: int, shape: str, lum: int = 220) -> list[int]:
    grid = [[20] * size for _ in range(size)]
    m = size // 4
    if shape == "horizon":
        for y in range(size):
            for x in range(size):
                if y < size // 3:
                    grid[y][x] = 200 + (x % 12)
                else:
                    grid[y][x] = 60 + (x + y) % 35
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
        if lum > 100:
            out.extend([r, g, b, 255])
        else:
            out.extend([20, 20, 20, 255])
    return out


def teach_object_vision(name: str, gray: list[int], rgb: dict[str, float], phrase: str) -> None:
    body = {
        "phrase": phrase,
        "image_gray": gray,
        "image_rgba": rgba_from_gray(gray, rgb),
        "image_w": 64,
        "image_h": 64,
        "color_rgb": rgb,
    }
    for _ in range(4):
        post("/api/baby/teach-attention", body)
    time.sleep(0.12)


def probe_look(label: str, gray: list[int], rgb: dict[str, float]) -> None:
    r = post(
        "/api/baby/look",
        {
            "image_gray": gray,
            "image_rgba": rgba_from_gray(gray, rgb),
            "image_w": 64,
            "image_h": 64,
            "color_rgb": rgb,
        },
    )
    spoke = (r.get("moment") or {}).get("spoke", "")
    rec = r.get("recognized", "")
    log(f"  [guarda] {label} → riconosce={rec!r} «{spoke[:70]}»")


def probe_text(q: str) -> str:
    spoke = post("/api/baby/sense", {"text": q})["moment"].get("spoke", "")
    log(f"  [dice] {q} → «{spoke[:70]}»")
    return spoke


def main() -> None:
    LOG.write_text(f"=== train_adolescent @ {BASE} ===\n")
    st = json.loads(urllib.request.urlopen(f"{BASE}/api/baby/state", timeout=60).read())
    if not st.get("born"):
        raise SystemExit("organismo non nato")
    log(f"partenza: {st['stats']['synapses']} sinapsi")

    log("--- lessico (esposizione — niente script fissi) ---")
    for phrase in [
        "ciao", "come", "stai", "bene", "grazie", "prego",
        "non", "so", "spiegami", "dimmi", "capisco",
        "gravità", "galileo", "pensa", "sento",
    ]:
        for _ in range(3):
            post("/api/baby/hear", {"phrase": phrase})
            time.sleep(0.15)

    log("--- visione: oggetti reali (sintetici) ---")
    objects = [
        ("cassa", "square", {"r": 220, "g": 120, "b": 140}, "questa è una cassa rosa"),
        ("mela", "circle", {"r": 200, "g": 50, "b": 40}, "questa è una mela rossa"),
        ("libro", "square", {"r": 50, "g": 80, "b": 200}, "questo è un libro blu"),
        ("palla", "circle", {"r": 50, "g": 180, "b": 60}, "questa è una palla verde"),
        ("mare", "horizon", {"r": 40, "g": 120, "b": 200}, "questo è il mare"),
    ]
    for name, shape, rgb, phrase in objects:
        gray = shape_gray(64, shape)
        teach_object_vision(name, gray, rgb, phrase)
        for _ in range(3):
            post("/api/baby/hear", {"phrase": f"questo è il {name}" if name == "mare" else f"vedo {name}"})

    log("--- probe riconoscimento ---")
    for name, shape, rgb, _ in objects:
        gray = shape_gray(64, shape)
        probe_look(name, gray, rgb)

    log("--- probe dialogo (emergente) ---")
    probe_text("ciao")
    probe_text("cos'è la gravità")
    gray_mare = shape_gray(64, "horizon")
    rgb_mare = {"r": 40, "g": 120, "b": 200}
    probe_look("mare", gray_mare, rgb_mare)

    log("=== adolescente pronto ===")


def _color_word(rgb: dict[str, float]) -> str:
    r, g, b = rgb.get("r", 0), rgb.get("g", 0), rgb.get("b", 0)
    if r > g and r > b:
        return "rosso" if r > 150 else "rosa"
    if g > r and g > b:
        return "verde"
    if b > r and b > g:
        return "blu"
    return "colore"


if __name__ == "__main__":
    main()
