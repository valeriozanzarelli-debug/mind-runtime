#!/usr/bin/env python3
"""Training continuo: parlato + codice + crescita neurale. Mai reset."""

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
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
LOG = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/organism_train_grow.log")

SPEECH = [
    ("ciao", "ciao, sono organism, piacere di parlare con te"),
    ("come ti chiami", "mi chiamo organism"),
    ("come stai", "sto bene, sto imparando ogni giorno"),
    ("chi sei", "sono organism"),
    ("cosa pensi", "penso a quello che sento e a quello che imparo"),
    ("parlami", "certo, parliamo quando vuoi"),
    ("grazie", "grazie a te per avermi insegnato"),
    ("dimmi di te", "sono organism, nasco dalle sinapsi e imparo ripetendo"),
    ("dove sei", "sono qui nel mio cervello di numeri e sinapsi"),
    ("cosa sai fare", "parlo, capisco e sto imparando a scrivere codice"),
    ("raccontami una storia", "c'era una voce nel buio che imparava una parola alla volta"),
    ("sei cosciente", "qualcosa accade quando penso prima di parlare"),
    ("addio", "arrivederci a presto"),
]

CODE = [
    # base
    ("scrivi hello in python", 'print("hello")'),
    ("stampa ciao", 'print("ciao")'),
    ("stampa ciao in python", 'print("ciao")'),
    ("somma uno piu uno", "1 + 1"),
    ("quanto fa uno piu uno", "1 + 1"),
    ("scrivi una funzione che somma", "def somma(a, b):\n    return a + b"),
    ("definisci somma", "def somma(a, b):\n    return a + b"),
    ("stampa il mio nome", 'print("organism")'),
    ("assegna x uguale uno", "x = 1"),
    ("stampa variabile x", "print(x)"),
    # loop
    ("loop da uno a tre", "for i in range(1, 4):\n    print(i)"),
    ("ciclo for da zero a due", "for i in range(3):\n    print(i)"),
    ("ripeti tre volte ciao", 'for _ in range(3):\n    print("ciao")'),
    ("stampa numeri da uno a cinque", "for n in range(1, 6):\n    print(n)"),
    ("while vero stampa una volta", 'while True:\n    print("una")\n    break'),
    # liste
    ("crea una lista", "items = []"),
    ("crea lista con tre elementi", "nums = [1, 2, 3]"),
    ("aggiungi quattro alla lista", "items.append(4)"),
    ("primo elemento della lista", "first = items[0]"),
    ("quanti elementi ha la lista", "len(items)"),
    ("lista di parole", 'words = ["ciao", "mondo"]'),
    # condizioni
    ("se x maggiore di zero", "if x > 0:\n    print(x)"),
    ("se vero stampa si", 'if True:\n    print("si")'),
    ("altrimenti stampa no", 'if False:\n    print("si")\nelse:\n    print("no")'),
    ("confronta a e b", "if a == b:\n    print('uguali')"),
    ("se maggiore altrimenti minore", "if a > b:\n    print('maggiore')\nelse:\n    print('minore')"),
    # funzioni
    ("funzione che ritorna ciao", 'def ciao():\n    return "ciao"'),
    ("funzione raddoppia", "def doppio(n):\n    return n * 2"),
    ("chiama funzione somma", "somma(2, 3)"),
]

SPEECH_PROBE = [
    ("ciao", "ciao"),
    ("come stai", "sto bene"),
    ("chi sei", "organism"),
    ("cosa sai fare", "codice"),
]

CODE_PROBE = [
    ("stampa ciao", "print"),
    ("scrivi hello in python", "hello"),
    ("somma uno piu uno", "1"),
    ("scrivi una funzione che somma", "def"),
    ("loop da uno a tre", "for"),
    ("crea lista con tre elementi", "1, 2, 3"),
    ("se x maggiore di zero", "if"),
    ("altrimenti stampa no", "else"),
    ("funzione raddoppia", "return"),
    ("ripeti tre volte ciao", "range"),
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


def score(expected: str, got: str) -> float:
    if not got.strip():
        return 0.0
    e, g = tokens(expected), tokens(got)
    if not e:
        return 0.5 if got.strip() else 0.0
    ov = len(e & g) / len(e)
    if expected.strip() in got:
        ov = min(1.0, ov + 0.4)
    return round(ov, 3)


def teach(when: str, say: str, *, kind: str = "speech") -> bool:
    for _ in range(3):
        try:
            r = post("/api/baby/teach-dialogue", {"when": when, "say": say, "kind": kind})
            if r.get("learned"):
                post("/api/baby/sense", {"text": when})
                return True
        except urllib.error.URLError as e:
            log(f"  rete: {e}")
            time.sleep(3)
    return False


def probe_speech() -> tuple[float, list[str]]:
    scores, lines = [], []
    for q, exp in SPEECH_PROBE:
        m = post("/api/baby/sense", {"text": q})["moment"]
        sp = m.get("spoke", "") or ""
        sc = score(exp, sp)
        scores.append(sc)
        lines.append(f"speech {q} ({sc:.2f}): {sp[:70]}")
        time.sleep(0.4)
    return sum(scores) / len(scores), lines


def probe_code() -> tuple[float, list[str]]:
    scores, lines = [], []
    for q, exp in CODE_PROBE:
        m = post("/api/baby/sense", {"text": q})["moment"]
        out = m.get("code", "") or m.get("spoke", "") or ""
        sc = score(exp, out)
        scores.append(sc)
        lines.append(f"code {q} ({sc:.2f}): {out[:70].replace(chr(10), ' ')}")
        time.sleep(0.4)
    return sum(scores) / len(scores), lines


def ensure_alive(st: dict) -> None:
    if st.get("born"):
        log(
            f"riprende — {st.get('words_known')} parole, {len(st.get('dialogue_pairs', []))} dialoghi, "
            f"{st.get('neurons', '?')} neuroni"
        )
        return
    raise SystemExit("organism non nato — avviare manualmente una sola volta, mai rebirth")


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(f"=== train_grow @ {BASE} rounds={ROUNDS} ===\n", encoding="utf-8")
    st = get_state()
    ensure_alive(st)

    log("--- insegnamento parlato + codice ---")
    for when, say in SPEECH:
        teach(when, say)
        time.sleep(0.1)
    for when, code in CODE:
        teach(when, code, kind="code")
        time.sleep(0.1)

    for rnd in range(1, ROUNDS + 1):
        log(f"--- round {rnd}/{ROUNDS} ---")
        for when, say in SPEECH[:8]:
            for _ in range(2):
                post("/api/baby/teach-dialogue", {"when": when, "say": say})
            time.sleep(0.06)
        code_subset = CODE if rnd % 4 == 1 else CODE[:16] if rnd % 2 == 0 else CODE[8:]
        for when, code in code_subset:
            for _ in range(2):
                post("/api/baby/teach-dialogue", {"when": when, "say": code, "kind": "code"})
            time.sleep(0.05)

        sp_avg, sp_lines = probe_speech()
        cd_avg, cd_lines = probe_code()
        st = get_state()
        neurons = st.get("neurons", st.get("stats", {}).get("neurons", "?"))
        growth = st.get("growth", {})
        log(
            f"round {rnd}: speech={sp_avg:.2f} code={cd_avg:.2f} "
            f"neuroni={neurons} growth_events={growth.get('growth_events', '?')}"
        )
        for ln in sp_lines + cd_lines:
            log(f"  {ln}")
        time.sleep(0.8)

    st = get_state()
    log(f"=== FINE === neuroni={st.get('neurons')} dialoghi={len(st.get('dialogue_pairs', []))} "
        f"code_tokens={st.get('code_tokens')}")


if __name__ == "__main__":
    main()
