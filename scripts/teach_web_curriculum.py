#!/usr/bin/env python3
"""Insegna oggetti, volti ed emozioni da immagini Wikimedia."""

from __future__ import annotations

import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://inkconscius.eu/organism"


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    st = json.loads(urllib.request.urlopen(f"{BASE}/api/baby/state", timeout=60).read())
    if not st.get("born"):
        raise SystemExit("organismo non nato")
    print("=== curriculum web (oggetti + volti + emozioni) ===", flush=True)
    r = post("/api/baby/web-curriculum", {"objects": True, "faces": True, "emotions": True})
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print(
        f"oggetti={r.get('taught_objects')} volti={r.get('taught_faces')} "
        f"emozioni={r.get('taught_emotions')}",
        flush=True,
    )
    if r.get("errors"):
        print("errori:", r["errors"][:5], flush=True)


if __name__ == "__main__":
    main()
