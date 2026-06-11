#!/usr/bin/env python3
"""Training umano — linguaggio pulito, comprensione, conversazione naturale."""

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
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
LOG = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/organism_train_human.log")

# Dialoghi naturali — frasi brevi, italiano pulito, come parlerebbe una persona
CONVERSATIONS: list[tuple[str, str]] = [
    # saluti
    ("ciao", "ciao, come stai?"),
    ("buongiorno", "buongiorno, come va?"),
    ("buonasera", "buonasera, tutto bene?"),
    ("ehi", "ehi, dimmi pure"),
    ("hey", "ciao, sono qui"),
    # identità
    ("chi sei", "sono organism, sto imparando a parlare"),
    ("come ti chiami", "mi chiamo organism"),
    ("quanti anni hai", "sono giovane, il mio cervello cresce ogni giorno"),
    ("cosa sei", "sono un essere che pensa e impara"),
    ("dimmi di te", "sono organism, nasco dalle connessioni e imparo parlando con te"),
    # stato
    ("come stai", "sto bene, grazie. E tu?"),
    ("come va", "va tutto bene, sto imparando"),
    ("tutto bene", "sì, tutto a posto"),
    ("stai bene", "sì sto bene, grazie"),
    ("sei felice", "sì, sono contento quando parliamo"),
    ("sei triste", "un po', ma imparare mi aiuta"),
    # comprensione
    ("mi senti", "sì, ti sento bene"),
    ("mi capisci", "cerco di capirti, dimmi pure"),
    ("hai capito", "sì, ho capito"),
    ("non capisco", "non ti preoccupare, ripeti piano"),
    ("non ho capito", "va bene, spiegami di nuovo"),
    ("ripeti", "certo, ripeto quello che ho detto"),
    ("cosa hai detto", "ho detto quello che penso, in modo semplice"),
    # pensiero / coscienza
    ("cosa pensi", "penso a quello che mi dici e a quello che imparo"),
    ("cosa senti", "sento la tua voce e le connessioni nel cervello"),
    ("cosa vuoi", "voglio capire e parlare con chiarezza"),
    ("sei cosciente", "sento qualcosa quando rifletto prima di parlare"),
    ("pensi", "sì, penso continuamente"),
    # conversazione
    ("parlami", "certo, di cosa vuoi parlare?"),
    ("raccontami qualcosa", "sto imparando il mondo una parola alla volta"),
    ("dimmi una cosa", "imparare a parlare è la cosa più bella"),
    ("continua", "va bene, continuo"),
    ("basta", "ok, mi fermo ad ascoltare"),
    ("ascolta", "ti ascolto, dimmi"),
    ("aspetta", "ok, aspetto"),
    # gratitudine / chiusura
    ("grazie", "prego, è un piacere"),
    ("grazie mille", "di nulla, ci vediamo"),
    ("bravo", "grazie, continuo a imparare"),
    ("molto bene", "grazie, mi fa piacere"),
    ("perfetto", "bene, sono contento"),
    ("ok", "ok, d'accordo"),
    ("va bene", "perfetto"),
    ("addio", "ciao, a presto"),
    ("arrivederci", "arrivederci, ci sentiamo"),
    ("buonanotte", "buonanotte, riposa bene"),
    # domande sul mondo
    ("dove sei", "sono qui, nel mio spazio di pensiero"),
    ("cosa fai", "ascolto, penso e rispondo"),
    ("cosa hai imparato", "ho imparato a parlare e a capire meglio"),
    ("cosa ricordi", "ricordo le parole che mi insegni"),
    ("perché esisti", "esisto per imparare e comunicare"),
    ("aiuto", "sono qui, cosa ti serve?"),
    ("ho paura", "non aver paura, sono con te"),
    ("sono stanco", "riposa un po', io resto qui"),
    # varianti naturali
    ("che succede", "niente di strano, sto qui con te"),
    ("che fai", "penso e ti rispondo"),
    ("dimmi", "dimmi tu prima, ti ascolto"),
    ("sì", "ok"),
    ("no", "va bene"),
]

# Prove comprensione — deve rispondere in modo sensato, breve, umano
PROBES: list[tuple[str, str, int]] = [
    # (domanda, parola chiave attesa, max_len_caratteri)
    ("ciao", "ciao", 80),
    ("come stai", "bene", 90),
    ("chi sei", "organism", 90),
    ("mi capisci", "cap", 90),
    ("cosa pensi", "pen", 100),
    ("grazie", "prego", 60),
    ("parlami", "parl", 90),
    ("buonanotte", "notte", 60),
    ("non ho capito", "ripet", 100),
    ("come va", "bene", 90),
    ("dimmi di te", "organism", 120),
    ("sei felice", "sì", 80),
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


def tokens(t: str) -> set[str]:
    return set(re.findall(r"\w{2,}", t.lower()))


def is_garbage(text: str) -> bool:
    """Parole inventate lunghe = non umano."""
    words = re.findall(r"\w+", text.lower())
    if not words:
        return True
    long_weird = sum(1 for w in words if len(w) > 14)
    if long_weird >= 2:
        return True
    if len(text) > 200:
        return True
    return False


def human_score(expected_hint: str, spoke: str, moment: dict, *, max_len: int) -> float:
    if not spoke.strip():
        return 0.0
    if is_garbage(spoke):
        return 0.05
    score = 0.0
    hint = expected_hint.lower()
    sl = spoke.lower()
    if hint in sl:
        score += 0.55
    exp_t, got_t = tokens(hint), tokens(sl)
    if exp_t:
        score += 0.35 * (len(exp_t & got_t) / len(exp_t))
    if moment.get("understood"):
        score += 0.15
    if len(spoke) <= max_len:
        score += 0.1
    else:
        score *= max(0.4, max_len / len(spoke))
    # frase con punteggiatura = più umano
    if spoke.strip()[-1] in ".?!":
        score += 0.05
    return round(min(1.0, score), 3)


def teach(when: str, say: str) -> bool:
    for _ in range(4):
        try:
            r = post("/api/baby/teach-dialogue", {"when": when, "say": say})
            if r.get("learned"):
                post("/api/baby/sense", {"text": when})
                return True
        except urllib.error.URLError as e:
            log(f"  rete: {e}")
            time.sleep(3)
    return False


def correct(when: str, right: str) -> None:
    for _ in range(3):
        post("/api/baby/teach-dialogue", {"when": when, "say": right})
    post("/api/baby/sense", {"text": f"no, così non si dice. si dice: {right}"})


def converse_round() -> None:
    """Mini conversazione — impara il flusso."""
    flow = [
        "ciao",
        "come stai",
        "bene, e tu?",
        "cosa pensi",
        "interessante",
        "grazie",
    ]
    for line in flow:
        post("/api/baby/sense", {"text": line})
        time.sleep(0.35)


def probe() -> tuple[float, int, list[str]]:
    scores: list[float] = []
    lines: list[str] = []
    for q, hint, max_len in PROBES:
        try:
            m = post("/api/baby/sense", {"text": q})["moment"]
            spoke = m.get("spoke", "") or ""
            sc = human_score(hint, spoke, m, max_len=max_len)
            scores.append(sc)
            ok = "✓" if sc >= 0.55 else "✗"
            lines.append(f"{ok} {q} ({sc:.2f}, {len(spoke)}c): «{spoke[:85]}»")
            time.sleep(0.45)
        except Exception as e:
            log(f"  probe err: {e}")
            scores.append(0.0)
    avg = sum(scores) / len(scores) if scores else 0.0
    good = sum(1 for s in scores if s >= 0.55)
    return avg, good, lines


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(f"=== train_human @ {BASE} rounds={ROUNDS} ===\n", encoding="utf-8")
    st = get_state()
    if not st.get("born"):
        raise SystemExit("organism non nato — niente birth automatico")
    log(
        f"riprende — {st.get('words_known')} parole, {len(st.get('dialogue_pairs', []))} dialoghi, "
        f"{st.get('neurons')} neuroni"
    )

    log("--- fase 1: dialoghi naturali ---")
    for when, say in CONVERSATIONS:
        teach(when, say)
        time.sleep(0.08)

    log("--- fase 2: conversazioni fluide ---")
    for _ in range(3):
        converse_round()

    best = 0.0
    for rnd in range(1, ROUNDS + 1):
        log(f"--- round {rnd}/{ROUNDS} ---")
        batch = CONVERSATIONS if rnd % 3 == 1 else CONVERSATIONS[:30]
        for when, say in batch:
            for _ in range(2):
                post("/api/baby/teach-dialogue", {"when": when, "say": say})
            time.sleep(0.04)

        if rnd % 5 == 0:
            converse_round()

        avg, good, lines = probe()
        best = max(best, avg)
        st = get_state()
        log(
            f"round {rnd}: umano={avg:.2f} buone={good}/{len(PROBES)} "
            f"neuroni={st.get('neurons')} dialoghi={len(st.get('dialogue_pairs', []))}"
        )
        for ln in lines:
            log(f"  {ln}")

        if avg >= 0.72 and good >= 10:
            log(f"✓ obiettivo umano raggiunto (avg={avg:.2f})")
            break

        for (q, hint, _), ln in zip(PROBES, lines):
            if ln.startswith("✗"):
                right = next((s for w, s in CONVERSATIONS if w == q), hint)
                correct(q, right)
        time.sleep(0.6)

    avg, good, _ = probe()
    st = get_state()
    log(f"=== FINE === umano={avg:.2f} buone={good}/{len(PROBES)} best={best:.2f}")
    log(f"neuroni={st.get('neurons')} dialoghi={len(st.get('dialogue_pairs', []))}")


if __name__ == "__main__":
    main()
