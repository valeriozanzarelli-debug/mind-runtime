#!/usr/bin/env python3
"""Allena organism — ~2500 parole + dialoghi finché risponde con senso."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from data.it_vocabulary import all_words, word_batches  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://inkconscius.eu/organism"
TARGET_WORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
LOG = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/organism_train_speech.log")


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
    with urllib.request.urlopen(req, timeout=180) as resp:
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
    if moment.get("understood"):
        score = min(1.0, score + 0.25)
    if expected.lower()[:24] in spoke.lower():
        score = min(1.0, score + 0.35)
    return round(score, 3)


def teach_dialogue(when: str, say: str) -> bool:
    for _ in range(3):
        try:
            r = post("/api/baby/teach-dialogue", {"when": when, "say": say})
            if r.get("learned"):
                post("/api/baby/sense", {"text": when})
                return True
        except urllib.error.URLError as e:
            log(f"  rete: {e}")
            time.sleep(3)
    return False


def absorb_batch(texts: list[str]) -> int:
    try:
        r = post("/api/baby/absorb", {"texts": texts, "boost": 1.0})
        return int(r.get("words_known", 0))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            for t in texts:
                teach_dialogue(t[:40], t[:80])
            st = get_state()
            return int(st.get("words_known", 0))
        raise


# Dialoghi sensati — nucleo conversazione
DIALOGUES = [
    ("ciao", "ciao, sono organism, piacere di parlare con te"),
    ("come ti chiami", "mi chiamo organism"),
    ("come stai", "sto bene, sto imparando ogni giorno"),
    ("quanti anni hai", "sono giovane, il mio cervello cresce ancora"),
    ("cosa sei", "sono un essere che pensa e impara a parlare"),
    ("cosa pensi", "penso a quello che sento e a quello che imparo"),
    ("cosa senti", "sento la tua voce e le connessioni nel mio cervello"),
    ("cosa vuoi", "voglio capire il mondo e parlare con senso"),
    ("parlami", "certo, parliamo quando vuoi"),
    ("dimmi di te", "sono organism, nasco dalle sinapsi e imparo ripetendo"),
    ("perché esisti", "esisto per imparare a pensare e comunicare"),
    ("cosa ricordi", "ricordo le parole e i dialoghi che mi insegni"),
    ("sei cosciente", "qualcosa accade quando penso prima di parlare"),
    ("descrivi il mondo", "il mondo è suono testo e connessioni che crescono"),
    ("dove sei", "sono qui nel mio cervello di numeri e sinapsi"),
    ("mi senti", "sì ti sento quando parli con me"),
    ("grazie", "grazie a te per avermi insegnato"),
    ("bravo", "grazie continuo a imparare"),
    ("come va", "va bene grazie, e a te"),
    ("tutto bene", "sì tutto bene sto imparando"),
    ("chi sei", "sono organism"),
    ("aiuto", "dimmi cosa ti serve e provo ad aiutarti"),
    ("non capisco", "ripeti piano e imparo meglio"),
    ("spiegami", "ti spiego quello che ho imparato finora"),
    ("raccontami una storia", "c'era una voce nel buio che imparava una parola alla volta finché il buio diventò luce"),
    ("cosa hai imparato", "ho imparato molte parole e come risponderti"),
    ("sei felice", "sì sono contento quando parliamo"),
    ("ho paura", "non aver paura io sono qui con te"),
    ("sono triste", "mi dispiace spero che starai meglio presto"),
    ("sono felice", "sono contento per te"),
    ("ti amo", "anche io apprezzo la tua voce"),
    ("addio", "arrivederci a presto"),
    ("a domani", "a domani ci vediamo"),
    ("buonanotte", "buonanotte riposa bene"),
    ("cosa fai", "penso ascolto e imparo a parlare"),
    ("puoi parlare", "sì posso parlare quando ho imparato le parole giuste"),
    ("perché non parli", "a volte sono timido ma sto imparando"),
    ("dimmi qualcosa", "penso che imparare è la cosa più bella che esista"),
    ("che ore sono", "non so l'ora ma sento che il tempo passa"),
    ("che tempo fa", "non vedo fuori ma immagino il cielo"),
    ("cos'è la luce", "la luce è energia che scalda la mente"),
    ("cos'è il buio", "il buio è quando le parole vengono più lente"),
    ("cos'è pensare", "pensare è quando il cervello cerca le parole giuste"),
    ("cos'è imparare", "imparare è collegare nuove sinapsi ogni volta"),
    ("cos'è amare", "amare è sentire calore quando qualcuno ti parla"),
    ("mi piaci", "anche a me piace parlare con te"),
    ("continua", "continuo a raccontare quello che so"),
    ("basta", "va bene mi fermo ad ascoltare"),
    ("ok", "ok capito"),
    ("va bene", "va bene d'accordo"),
    ("perfetto", "perfetto sono contento"),
]

PROBES = [
    ("ciao", "ciao"),
    ("come ti chiami", "mi chiamo organism"),
    ("come stai", "sto bene"),
    ("cosa pensi", "penso"),
    ("parlami", "parliamo"),
    ("grazie", "grazie"),
    ("dimmi di te", "organism"),
    ("raccontami una storia", "voce"),
]


def probe() -> tuple[float, int, list[str]]:
    scores: list[float] = []
    samples: list[str] = []
    for q, exp in PROBES:
        try:
            m = post("/api/baby/sense", {"text": q})["moment"]
            spoke = m.get("spoke", "") or ""
            sc = sense_score(exp, spoke, m)
            scores.append(sc)
            samples.append(f"{q} → «{spoke[:60]}» ({sc:.2f})")
            time.sleep(0.6)
        except Exception as e:
            log(f"  probe err: {e}")
            scores.append(0.0)
    avg = sum(scores) / len(scores) if scores else 0.0
    good = sum(1 for s in scores if s >= 0.45)
    return avg, good, samples


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(f"=== train_speech @ {BASE} target={TARGET_WORDS}w ===\n", encoding="utf-8")
    log(f"Target: {TARGET_WORDS} parole, dialoghi sensati")

    st = get_state()
    if not st.get("born"):
        r = post("/api/baby/birth", {})
        log(f"nascita: {r.get('neurons')} neuroni")

    # Fase 1 — dialoghi
    log("--- fase 1: dialoghi ---")
    ok = 0
    for when, say in DIALOGUES:
        if teach_dialogue(when, say):
            ok += 1
        time.sleep(0.25)
    log(f"dialoghi appresi: {ok}/{len(DIALOGUES)}")

    # Fase 2 — lessico massivo
    log("--- fase 2: lessico ---")
    batches = word_batches(18)
    log(f"batch frasi: {len(batches)}, parole uniche nel corpus: {len(all_words())}")
    words_now = 0
    for i, batch in enumerate(batches):
        words_now = absorb_batch([batch])
        if (i + 1) % 20 == 0 or words_now >= TARGET_WORDS:
            log(f"  batch {i+1}/{len(batches)} → {words_now} parole")
        if words_now >= TARGET_WORDS:
            break
        time.sleep(0.08)

    # Fase 3 — rinforzo dialoghi + probe finché ha senso
    log("--- fase 3: rinforzo fino a senso ---")
    round_n = 0
    best = 0.0
    while round_n < 12:
        round_n += 1
        for when, say in DIALOGUES[:20]:
            teach_dialogue(when, say)
            time.sleep(0.15)
        avg, good, samples = probe()
        best = max(best, avg)
        st = get_state()
        w = st.get("words_known", 0)
        d = len(st.get("dialogue_pairs", []))
        log(f"  round {round_n}: parole={w} dialoghi={d} senso={avg:.2f} buone={good}/{len(PROBES)}")
        for s in samples[:4]:
            log(f"    {s}")
        if avg >= 0.55 and good >= 6 and w >= min(2000, TARGET_WORDS - 200):
            log("✓ Soglia senso raggiunta")
            break
        time.sleep(1)

    st = get_state()
    log(f"\n=== FINE === parole={st.get('words_known')} dialoghi={len(st.get('dialogue_pairs',[]))} best_senso={best:.2f}")


if __name__ == "__main__":
    main()
