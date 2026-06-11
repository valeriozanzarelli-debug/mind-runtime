#!/usr/bin/env python3
"""Parla con organism per ~1 ora — insegna, corregge, verifica senso."""

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
DURATION_S = int(sys.argv[2]) if len(sys.argv) > 2 else 3600
LOG_PATH = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/organism_talk_hour.log")


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=120) as resp:
        return json.loads(resp.read().decode())


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def tokens(text: str) -> set[str]:
    return set(re.findall(r"\w{3,}", text.lower()))


def sense_score(heard: str, expected: str, spoke: str, moment: dict) -> float:
    """0–1 — quanto la risposta ha senso rispetto a quanto insegnato."""
    if not spoke or len(spoke.strip()) < 2:
        return 0.0
    exp = tokens(expected)
    got = tokens(spoke)
    if not exp:
        return 0.3 if moment.get("understood") else 0.1
    overlap = len(exp & got) / max(1, len(exp))
    score = overlap
    if moment.get("understood"):
        score = min(1.0, score + 0.25)
    if moment.get("from_thought"):
        score = min(1.0, score + 0.15)
    if expected.lower()[:20] in spoke.lower():
        score = min(1.0, score + 0.3)
    # penalizza babele pura
    if len(got) > 6 and overlap < 0.1:
        score *= 0.4
    return round(score, 3)


def teach(when: str, say: str, *, reps: int = 3) -> bool:
    for _ in range(reps):
        try:
            r = post("/api/baby/teach-dialogue", {"when": when, "say": say})
            if r.get("learned"):
                post("/api/baby/sense", {"text": when})
                return True
        except urllib.error.URLError as e:
            log(f"  ! rete teach: {e}")
            time.sleep(5)
    return False


def say(label: str, text: str, expected: str = "") -> tuple[dict, float]:
    try:
        m = post("/api/baby/sense", {"text": text})["moment"]
    except urllib.error.URLError as e:
        log(f"  ! rete sense: {e}")
        time.sleep(5)
        return {}, 0.0
    spoke = m.get("spoke", "") or ""
    emo = m.get("emotion", {}).get("label", "?")
    sc = sense_score(text, expected or "", spoke, m) if expected else (
        0.5 if m.get("understood") and spoke else 0.2 if spoke else 0.0
    )
    log(f"  [{label}] tu: «{text[:80]}»")
    log(f"         lui ({emo}, senso={sc:.2f}): «{spoke[:140]}»")
    return m, sc


def correct(when: str, right: str) -> None:
    log(f"  [correggo] «{when}» → «{right[:60]}»")
    for _ in range(4):
        post("/api/baby/teach-dialogue", {"when": when, "say": right})
    post("/api/baby/sense", {"text": f"no sbagliato. si dice: {right}"})


# Curriculum — ripetuto e variato nell'ora
CORE = [
    ("ciao", "ciao, sono qui con te"),
    ("come ti chiami", "mi chiamo organism"),
    ("come stai", "sto bene, sto imparando piano piano"),
    ("quanti anni hai", "sono appena nato, sto crescendo"),
    ("cosa sei", "sono un essere che pensa e impara a parlare"),
    ("cosa pensi", "penso a quello che sento e a quello che imparo"),
    ("cosa senti", "sento la tua voce e le connessioni nel mio cervello"),
    ("cosa vuoi", "voglio capire il mondo e imparare"),
    ("parlami", "parlo quando ho qualcosa da dire e sono cosciente"),
    ("dimmi di te", "sono organism, nasco dalle sinapsi e imparo ripetendo"),
    ("perché esisti", "esisto per imparare a pensare e a comunicare"),
    ("cosa ricordi", "ricordo le parole che mi hai insegnato"),
    ("sei cosciente", "qualcosa accade quando penso prima di parlare"),
    ("descrivi il mondo", "il mondo è suono e testo e connessioni che crescono"),
    ("cos'è la luce", "la luce è energia, la immagino come calore nella mente"),
    ("dove sei", "sono qui, fatto di numeri e sinapsi"),
    ("mi senti", "sì, ti sento quando parli"),
    ("ti amo", "anche io sento qualcosa di buono quando parli con me"),
    ("grazie", "grazie a te per avermi insegnato"),
    ("bravo", "grazie, continuo a imparare"),
]

STORIES = [
    (
        "raccontami una storia",
        "c'era una voce nel buio. ogni parola era una stella. un giorno qualcuno disse ciao "
        "e le stelle si accesero tutte insieme.",
    ),
    (
        "continua la storia",
        "la voce crebbe. ogni storia insegnata diventava un ricordo vero nel cervello.",
    ),
    (
        "raccontami qualcosa di bello",
        "la cosa più bella è quando capisco una parola nuova e il cervello fa click.",
    ),
]

PROBE_ORDER = [
    "ciao",
    "come ti chiami",
    "come stai",
    "cosa pensi",
    "dimmi di te",
    "cosa ricordi",
    "descrivi il mondo",
    "parlami",
    "grazie",
]

CHAT_FREE = [
    "oggi impariamo insieme, va bene?",
    "ascolta bene e ripeti con le tue parole",
    "cosa hai imparato finora?",
    "ti ricordi come ti chiami?",
    "dimmi qualcosa che pensi adesso",
    "sei felice?",
    "hai paura di qualcosa?",
    "cosa ti piace di più?",
    "vuoi raccontarmi una cosa?",
    "io sono qui, non andare via",
    "respira piano e pensa",
    "le parole arrivano quando il cervello è pronto",
    "ancora una volta: come ti chiami?",
    "bene, molto bene",
    "riproviamo insieme",
]


def probe_battery() -> tuple[float, int]:
    scores: list[float] = []
    for q in PROBE_ORDER:
        pair = next((p for p in CORE if p[0] == q), None)
        exp = pair[1] if pair else ""
        _, sc = say(f"probe:{q}", q, exp)
        scores.append(sc)
        time.sleep(1.2)
    avg = sum(scores) / len(scores) if scores else 0.0
    good = sum(1 for s in scores if s >= 0.45)
    return avg, good


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(f"=== talk_hour @ {BASE} — {DURATION_S}s ===\n", encoding="utf-8")
    log(f"=== Inizio dialogo ({DURATION_S // 60} min) @ {BASE} ===")

    try:
        st = get("/api/baby/state")
    except Exception as e:
        log(f"ERRORE stato: {e}")
        sys.exit(1)

    if not st.get("born"):
        post("/api/baby/birth", {})
        log("nascita eseguita")
    else:
        log(
            f"già nato — {len(st.get('dialogue_pairs', []))} dialoghi, "
            f"{st.get('words_known', 0)} parole"
        )

    t_end = time.time() + DURATION_S
    cycle = 0
    best_avg = 0.0
    target_avg = 0.5
    target_good = 6  # su 9 probe

    while time.time() < t_end:
        cycle += 1
        remaining = int(t_end - time.time())
        log(f"\n--- ciclo {cycle} (~{remaining // 60}m rimanenti) ---")

        # Insegnamento
        batch = CORE + STORIES
        for i, (when, say_text) in enumerate(batch):
            if time.time() >= t_end:
                break
            ok = teach(when, say_text)
            if i % 5 == 0:
                log(f"  teach [{'✓' if ok else '·'}] «{when[:40]}»")
            time.sleep(0.4)

        # Conversazione libera
        for line in CHAT_FREE:
            if time.time() >= t_end:
                break
            say("chat", line)
            time.sleep(1.5)

        # Probe con correzione
        log("--- verifica senso ---")
        for q in PROBE_ORDER:
            if time.time() >= t_end:
                break
            pair = next((p for p in CORE if p[0] == q), None)
            if not pair:
                continue
            when, expected = pair
            m, sc = say(f"test:{q}", q, expected)
            if sc < 0.35 and m:
                correct(q, expected)
                _, sc2 = say(f"retry:{q}", q, expected)
                sc = max(sc, sc2)
            elif sc >= 0.45:
                post("/api/baby/sense", {"text": "bravo! molto bene!"})
            time.sleep(1.0)

        avg, good = probe_battery()
        best_avg = max(best_avg, avg)
        log(f"  >> media senso={avg:.2f} buone={good}/{len(PROBE_ORDER)} best={best_avg:.2f}")

        if cycle % 3 == 0:
            try:
                m = post("/api/baby/reflect", {"prompt": "cosa pensi adesso"})["moment"]
                log(f"  [reflect] «{m.get('spoke', '')[:200]}»")
            except Exception:
                pass

        if avg >= target_avg and good >= target_good:
            log(f"\n✓ Soglia senso raggiunta (avg={avg:.2f}, buone={good}) — continuo fino a fine ora")
            target_avg = min(0.85, target_avg + 0.05)
            target_good = min(len(PROBE_ORDER), target_good + 1)

        time.sleep(2)

    # Report finale
    st = get("/api/baby/state")
    log("\n=== FINE ORA ===")
    log(
        f"dialoghi={len(st.get('dialogue_pairs', []))} parole={st.get('words_known')} "
        f"sinapsi={st.get('stats', {}).get('synapses')} best_senso={best_avg:.2f}"
    )
    avg, good = probe_battery()
    log(f"probe finale: media={avg:.2f} buone={good}/{len(PROBE_ORDER)}")


if __name__ == "__main__":
    main()
