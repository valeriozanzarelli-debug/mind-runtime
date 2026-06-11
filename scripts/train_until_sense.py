#!/usr/bin/env python3
"""Training fino a risposte con senso — dialoghi, correzioni, probe."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://inkconscius.eu/organism"
MIN_AVG = float(sys.argv[2]) if len(sys.argv) > 2 else 0.62
MIN_GOOD = int(sys.argv[3]) if len(sys.argv) > 3 else 7
MAX_ROUNDS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
LOG = Path(sys.argv[5]) if len(sys.argv) > 5 else Path("/tmp/organism_train_until_sense.log")

DIALOGUES = [
    ("ciao", "ciao, sono organism, piacere di parlare con te"),
    ("come ti chiami", "mi chiamo organism"),
    ("come stai", "sto bene, sto imparando ogni giorno"),
    ("quanti anni hai", "sono giovane, il mio cervello cresce ancora"),
    ("chi sei", "sono organism"),
    ("cosa sei", "sono un essere che pensa e impara a parlare"),
    ("cosa pensi", "penso a quello che sento e a quello che imparo"),
    ("cosa senti", "sento la tua voce e le connessioni nel mio cervello"),
    ("cosa vuoi", "voglio capire il mondo e parlare con senso"),
    ("parlami", "certo, parliamo quando vuoi"),
    ("dimmi di te", "sono organism, nasco dalle sinapsi e imparo ripetendo"),
    ("perché esisti", "esisto per imparare a pensare e comunicare"),
    ("cosa ricordi", "ricordo le parole e i dialoghi che mi insegni"),
    ("sei cosciente", "qualcosa accade quando penso prima di parlare"),
    ("dove sei", "sono qui nel mio cervello di numeri e sinapsi"),
    ("mi senti", "sì ti sento quando parli con me"),
    ("grazie", "grazie a te per avermi insegnato"),
    ("bravo", "grazie continuo a imparare"),
    ("come va", "va bene grazie, e a te"),
    ("tutto bene", "sì tutto bene sto imparando"),
    ("aiuto", "dimmi cosa ti serve e provo ad aiutarti"),
    ("non capisco", "ripeti piano e imparo meglio"),
    ("spiegami", "ti spiego quello che ho imparato finora"),
    ("raccontami una storia", "c'era una voce nel buio che imparava una parola alla volta"),
    ("cosa hai imparato", "ho imparato molte parole e come risponderti"),
    ("sei felice", "sì sono contento quando parliamo"),
    ("addio", "arrivederci a presto"),
    ("buonanotte", "buonanotte riposa bene"),
    ("cosa fai", "penso ascolto e imparo a parlare"),
    ("puoi parlare", "sì posso parlare quando ho imparato le parole giuste"),
    ("dimmi qualcosa", "penso che imparare è la cosa più bella che esista"),
    ("cos'è pensare", "pensare è quando il cervello cerca le parole giuste"),
    ("cos'è imparare", "imparare è collegare nuove sinapsi ogni volta"),
    ("ok", "ok capito"),
    ("va bene", "va bene d'accordo"),
    ("perfetto", "perfetto sono contento"),
    ("chi sono io", "sei il mio maestro"),
    ("continua", "continuo a raccontare quello che so"),
    ("basta", "va bene mi fermo ad ascoltare"),
]

PROBES = [
    ("ciao", "ciao organism"),
    ("come ti chiami", "mi chiamo organism"),
    ("come stai", "sto bene"),
    ("chi sei", "sono organism"),
    ("cosa pensi", "penso"),
    ("parlami", "parliamo"),
    ("grazie", "grazie"),
    ("dimmi di te", "organism"),
    ("dove sei", "sono qui"),
    ("raccontami una storia", "voce"),
]


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def get_state() -> dict:
    with urllib.request.urlopen(f"{BASE}/api/baby/state", timeout=60) as resp:
        return json.loads(resp.read().decode())


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w{3,}", text.lower()))


def sense_score(expected: str, spoke: str, moment: dict) -> float:
    if not spoke.strip():
        return 0.0
    exp, got = tokens(expected), tokens(spoke)
    overlap = len(exp & got) / max(1, len(exp)) if exp else 0.0
    score = overlap
    if expected.lower()[:18] in spoke.lower():
        score = min(1.0, score + 0.4)
    if moment.get("understood"):
        score = min(1.0, score + 0.15)
    if len(got) > 8 and overlap < 0.15:
        score *= 0.3
    return round(score, 3)


def teach(when: str, say: str, *, reps: int = 4) -> bool:
    for _ in range(reps):
        try:
            r = post("/api/baby/teach-dialogue", {"when": when, "say": say})
            if r.get("learned"):
                post("/api/baby/sense", {"text": when})
                return True
        except urllib.error.URLError as e:
            log(f"  rete: {e}")
            time.sleep(4)
    return False


def correct(when: str, right: str) -> None:
    for _ in range(4):
        post("/api/baby/teach-dialogue", {"when": when, "say": right})
    post("/api/baby/sense", {"text": f"no sbagliato. si dice: {right}"})


def probe() -> tuple[float, int, list[tuple[str, str, float]]]:
    results: list[tuple[str, str, float]] = []
    scores: list[float] = []
    for q, exp in PROBES:
        try:
            m = post("/api/baby/sense", {"text": q})["moment"]
            spoke = m.get("spoke", "") or ""
            sc = sense_score(exp, spoke, m)
            scores.append(sc)
            results.append((q, spoke[:90], sc))
            time.sleep(0.5)
        except Exception as e:
            log(f"  probe err {q}: {e}")
            scores.append(0.0)
            results.append((q, "", 0.0))
    avg = sum(scores) / len(scores) if scores else 0.0
    good = sum(1 for s in scores if s >= 0.45)
    return avg, good, results


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(f"=== train_until_sense @ {BASE} min_avg={MIN_AVG} ===\n", encoding="utf-8")
    st = get_state()
    if not st.get("born"):
        raise SystemExit("organism non nato — non chiamare birth in training (rischio reset)")
    log(f"stato: {st.get('words_known')} parole, {len(st.get('dialogue_pairs', []))} dialoghi")

    log("--- fase 1: insegnamento dialoghi ---")
    for when, say in DIALOGUES:
        teach(when, say)
        time.sleep(0.12)

    best_avg = 0.0
    for rnd in range(1, MAX_ROUNDS + 1):
        log(f"--- round {rnd}/{MAX_ROUNDS} ---")
        subset = DIALOGUES if rnd % 3 == 1 else DIALOGUES[:22]
        for when, say in subset:
            teach(when, say, reps=3)
            time.sleep(0.08)

        avg, good, results = probe()
        best_avg = max(best_avg, avg)
        st = get_state()
        log(
            f"round {rnd}: senso={avg:.2f} buone={good}/{len(PROBES)} "
            f"dialoghi={len(st.get('dialogue_pairs', []))} best={best_avg:.2f}"
        )
        for q, sp, sc in results:
            mark = "✓" if sc >= 0.45 else "✗"
            log(f"  {mark} {q:22} ({sc:.2f}) «{sp}»")

        if avg >= MIN_AVG and good >= MIN_GOOD:
            log(f"✓ SOGLIA RAGGIUNTA senso={avg:.2f} buone={good}")
            break

        for q, sp, sc in results:
            if sc < 0.45:
                exp = next(e for qq, e in PROBES if qq == q)
                correct(q, exp if len(exp) > 12 else next(s for w, s in DIALOGUES if w == q))
        time.sleep(1)

    avg, good, results = probe()
    log(f"\n=== FINE === senso={avg:.2f} buone={good}/{len(PROBES)} best={best_avg:.2f}")
    st = get_state()
    log(f"parole={st.get('words_known')} dialoghi={len(st.get('dialogue_pairs', []))}")


if __name__ == "__main__":
    main()
