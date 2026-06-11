#!/usr/bin/env python3
"""Training lungo — parla bene, chiede quando non sa, impara col suo metodo."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://inkconscius.eu/organism"
LOG = Path("/tmp/organism_train_intelligence.log")
TARGET_KNOWN = 0.55
TARGET_UNKNOWN = 0.65
MAX_ROUNDS = 8


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


def teach(when: str, say: str, *, rounds: int = 4) -> None:
    for _ in range(rounds):
        post("/api/baby/teach-dialogue", {"when": when, "say": say})
    time.sleep(0.08)


def overlap(a: str, b: str) -> float:
    wa = {w for w in re.findall(r"[a-zàèéìòù']+", a.lower()) if len(w) > 2}
    wb = {w for w in re.findall(r"[a-zàèéìòù']+", b.lower()) if len(w) > 2}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(1, len(wa))


def score_unknown(spoke: str) -> float:
    if not spoke.strip():
        return 0.0
    sl = spoke.lower()
    if "?" in spoke:
        return 0.85
    if "non so" in sl:
        return 0.8
    if any(w in sl for w in ("significa", "spiega", "dimmi", "capisco", "ancora")):
        return 0.75
    if len(sl.split()) <= 5:
        return 0.45
    return 0.15


def probe_known(q: str, expect: str) -> float:
    spoke = post("/api/baby/sense", {"text": q})["moment"].get("spoke", "")
    s = overlap(spoke, expect)
    log(f"  [noto] {q} → «{spoke[:70]}» ({s:.2f})")
    return s


def probe_unknown(q: str) -> float:
    spoke = post("/api/baby/sense", {"text": q})["moment"].get("spoke", "")
    s = score_unknown(spoke)
    log(f"  [nuovo] {q} → «{spoke[:70]}» ({s:.2f})")
    return s


def curriculum() -> None:
    log("--- lessico meta-apprendimento ---")
    meta = [
        ("non so", "non so ancora"),
        ("non capisco", "puoi spiegarmi meglio"),
        ("cosa significa", "cosa significa dimmi"),
        ("spiegami", "spiegami piano piano"),
        ("impara", "imparo piano piano"),
        ("non lo so", "non so me lo spieghi"),
    ]
    for w, s in meta:
        teach(w, s)

    log("--- conversazione base ---")
    base = [
        ("ciao", "ciao come stai"),
        ("come stai", "sto bene grazie"),
        ("come ti chiami", "mi chiamo organism"),
        ("chi sei", "sono organism vivo qui"),
        ("cosa pensi", "penso e cerco di capire"),
        ("mi senti", "sì ti sento bene"),
        ("grazie", "prego di niente"),
        ("addio", "a presto ci vediamo"),
    ]
    for w, s in base:
        teach(w, s, rounds=5)


def teach_unknown_responses() -> None:
    """Dopo che chiede, insegna risposta onesta + domanda."""
    pairs = [
        ("cos'è la fotosintesi", "non so cos'è la fotosintesi me lo spieghi"),
        ("chi era einstein", "non so chi era einstein dimmelo"),
        ("come funziona un computer", "non so come funziona spiegami"),
        ("cos'è l amore", "non so cos'è l amore cosa significa"),
        ("dove vive il pinguino", "non so dove vive il pinguino"),
    ]
    for w, s in pairs:
        teach(w, s, rounds=4)


def evaluate() -> tuple[float, float]:
    known_scores = [
        probe_known("ciao", "ciao come"),
        probe_known("come stai", "sto bene"),
        probe_known("chi sei", "organism"),
        probe_known("grazie", "prego"),
    ]
    unknown_scores = [
        probe_unknown("cos'è la gravità"),
        probe_unknown("chi era darwin"),
        probe_unknown("come funziona il cuore"),
        probe_unknown("cos'è la musica classica"),
    ]
    time.sleep(0.2)
    k = sum(known_scores) / len(known_scores)
    u = sum(unknown_scores) / len(unknown_scores)
    return k, u


def main() -> None:
    LOG.write_text(f"=== train_intelligence @ {BASE} ===\n")
    st = json.loads(urllib.request.urlopen(f"{BASE}/api/baby/state", timeout=60).read())
    if not st.get("born"):
        raise SystemExit("non nato")
    log(f"partenza: {st['stats']['synapses']} sinapsi")

    curriculum()

    for rnd in range(1, MAX_ROUNDS + 1):
        log(f"\n=== round {rnd}/{MAX_ROUNDS} valutazione ===")
        k, u = evaluate()
        log(f"round {rnd}: noto={k:.2f} nuovo={u:.2f}")
        if k >= TARGET_KNOWN and u >= TARGET_UNKNOWN:
            log("✓ obiettivo raggiunto")
            break
        log("--- rinforzo + insegnamento lacune ---")
        teach_unknown_responses()
        curriculum()
        # simula spiegazione caregiver → apprendimento
        explanations = [
            ("la fotosintesi", "la fotosintesi è come le piante mangiano la luce"),
            ("einstein", "einstein era un fisico famoso"),
            ("il computer", "il computer pensa con elettricità"),
        ]
        for w, s in explanations:
            teach(w, s, rounds=3)
        time.sleep(0.5)
    else:
        k, u = evaluate()
        log(f"fine: noto={k:.2f} nuovo={u:.2f}")

    log("=== training completato ===")


if __name__ == "__main__":
    main()
