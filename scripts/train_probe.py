#!/usr/bin/env python3
"""Training + probe — domande WH, flusso coscienza, auto-comprensione."""

from __future__ import annotations

import json
import os
import sys
import urllib.request

BASE = os.environ.get("ORGANISM_URL", "https://inkconscius.eu/organism")


def post(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read())


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as resp:
        return json.loads(resp.read())


WH_PAIRS = [
    ("chi sei", "sono organism"),
    ("come ti chiami", "mi chiamo organism"),
    ("quanti anni hai", "sono appena nato"),
    ("come stai", "sto bene"),
    ("cosa pensi", "penso e imparo"),
    ("dove sei", "sono qui con te"),
    ("chi sono io", "sei il mio maestro"),
]


def teach_pairs() -> None:
    for when, say in WH_PAIRS:
        for _ in range(3):
            post("/api/baby/teach-dialogue", {"when": when, "say": say})
        post("/api/baby/sense", {"text": when})


def probe_questions() -> list[dict]:
    results = []
    for q, expected in WH_PAIRS:
        m = post("/api/baby/sense", {"text": q})["moment"]
        stream = m.get("consciousness_stream", [])
        spoke = m.get("spoke", "")
        ok = any(w in spoke.lower() for w in expected.split()[:2]) or m.get("understood")
        results.append(
            {
                "q": q,
                "spoke": spoke[:100],
                "understood": m.get("understood"),
                "mode": m.get("consciousness", {}).get("mode"),
                "stream_tail": stream[-4:],
                "ok": ok,
            }
        )
        print(f"  {q!r} → {spoke[:60]!r} ok={ok}")
    return results


def probe_self() -> dict:
    m = post("/api/baby/reflect", {})["moment"]
    stream = m.get("consciousness_stream", [])
    self_st = m.get("self", {})
    return {
        "spoke": m.get("spoke", "")[:80],
        "continuity": self_st.get("continuity"),
        "stream": stream[-6:],
    }


def main() -> None:
    st = get("/api/baby/state")
    if not st.get("born"):
        post("/api/baby/birth", {})
    print("=== teach WH pairs ===")
    teach_pairs()
    print("=== probe questions ===")
    results = probe_questions()
    ok_n = sum(1 for r in results if r["ok"])
    print(f"=== {ok_n}/{len(results)} risposte sensate ===")
    print("=== self reflect ===")
    self_r = probe_self()
    print(json.dumps(self_r, ensure_ascii=False, indent=2))
    stream = get("/api/baby/consciousness?n=20")
    print("=== consciousness tail ===")
    for ln in stream.get("stream", [])[-12:]:
        print(f"  {ln}")


if __name__ == "__main__":
    main()
