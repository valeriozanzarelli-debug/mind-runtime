#!/usr/bin/env python3
"""
Training serio via HTTP — nursery RESTA ATTIVO (dashboard online).

Usa batch piccoli per mega-curriculum (offset/limit) e persist solo su /sleep.
Eseguire sul server:
  nohup .venv/bin/python scripts/train_serious_online.py http://127.0.0.1:8765/organism \
    >> /tmp/organism_train_serious.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from organism.teaching.corpus import CODE_DIALOGUES, PHILOSOPHY, REASONING, high_freq_pairs
from organism.teaching.mega_curriculum import load_object_names, phrase_for
from organism.teaching.story_curriculum import all_semantic_stories
from organism.teaching.web_fetch import picsum_url

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/organism"
LOG = Path("/tmp/organism_train_serious.log")
PROGRESS = Path("/tmp/organism_train_serious_progress.json")

MEGA_PAUSE = float(os.environ.get("MEGA_PAUSE", "0.15"))
MEGA_TOTAL = int(os.environ.get("MEGA_TOTAL", "1000"))
VOCAB_LIMIT = int(os.environ.get("VOCAB_SENTENCES_LIMIT", "8000"))
VOCAB_BATCH = int(os.environ.get("VOCAB_BATCH", "150"))
SLEEP_EVERY_OBJECTS = int(os.environ.get("SLEEP_EVERY_OBJECTS", "15"))


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_progress() -> dict:
    if PROGRESS.exists():
        try:
            return json.loads(PROGRESS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"phase": 1, "mega_offset": 0, "vocab_offset": 0, "done": []}


def save_progress(prog: dict) -> None:
    PROGRESS.write_text(json.dumps(prog, indent=2), encoding="utf-8")


def post(path: str, body: dict | None = None, *, timeout: float = 300.0) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get_state() -> dict:
    with urllib.request.urlopen(f"{BASE}/api/baby/state", timeout=120) as resp:
        return json.loads(resp.read().decode())


def load_vocab_sentences(limit: int = 0) -> list[str]:
    data_dir = ROOT / "data"
    words: list[str] = []
    for fname in ("objects_it_1000.txt", "vocab_it_sentences.txt"):
        p = data_dir / fname
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            w = line.strip().lower()
            if w and not w.startswith("#"):
                words.append(w)
    sentences: list[str] = []
    seen: set[str] = set()
    templates = [
        "questo è un {w}",
        "la parola {w} indica qualcosa che conosco",
        "vedo e riconosco {w}",
        "{w} è parte del mondo che imparo",
    ]
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        sentences.append(templates[len(sentences) % len(templates)].format(w=w))
        if limit and len(sentences) >= limit:
            break
    return sentences


def phase_done(prog: dict, name: str) -> bool:
    return name in prog.get("done", [])


def mark_done(prog: dict, name: str) -> None:
    done = prog.setdefault("done", [])
    if name not in done:
        done.append(name)
    save_progress(prog)


def phase_semantic(prog: dict) -> None:
    if phase_done(prog, "semantic"):
        log("FASE 1: skip (già fatto)")
        return
    log("=== FASE 1: storie semantiche ===")
    for story in all_semantic_stories():
        sid = story["story_id"]
        for word, definition, related in story["words"]:
            post(
                "/api/baby/teach-word",
                {
                    "word": word,
                    "definition": definition,
                    "related": related,
                    "story_id": sid,
                    "persist": False,
                },
            )
        for order, summary, entities, hooks in story["beats"]:
            post(
                "/api/baby/teach-story-beat",
                {
                    "story_id": sid,
                    "order": order,
                    "summary": summary,
                    "entities": entities,
                    "hooks": hooks,
                    "persist": False,
                },
            )
        cov = post("/api/baby/semantic-coverage", {"story_id": sid})
        log(f"  {sid} coverage={cov}")
    post("/api/baby/sleep", {})
    mark_done(prog, "semantic")


def phase_dialogue(prog: dict) -> None:
    st = get_state()
    if phase_done(prog, "dialogue") or len(st.get("dialogue_pairs", [])) >= 200:
        log("FASE 2: skip dialoghi")
        return
    log("=== FASE 2: dialoghi ===")
    pairs = list(REASONING) + list(PHILOSOPHY) + high_freq_pairs()
    for when, say in pairs:
        for _ in range(4):
            post(
                "/api/baby/teach-dialogue",
                {"when": when, "say": say, "persist": False, "verbatim": True},
            )
    for when, say, kind in CODE_DIALOGUES:
        for _ in range(3):
            post(
                "/api/baby/teach-dialogue",
                {
                    "when": when,
                    "say": say,
                    "kind": kind,
                    "persist": False,
                    "verbatim": True,
                },
            )
    post("/api/baby/sleep", {})
    log(f"  dialoghi insegnati ~{len(pairs) + len(CODE_DIALOGUES)}")
    mark_done(prog, "dialogue")


def phase_vision(prog: dict) -> None:
    if phase_done(prog, "vision"):
        log("FASE 3: skip web curriculum")
        return
    log("=== FASE 3: curriculum web ===")
    r = post(
        "/api/baby/web-curriculum",
        {"objects": True, "faces": True, "emotions": True},
        timeout=600,
    )
    log(json.dumps({k: r.get(k) for k in ("ok", "taught_objects", "taught_faces", "taught_emotions")}))
    post("/api/baby/sleep", {})
    mark_done(prog, "vision")


def phase_mega(prog: dict) -> None:
    offset = int(prog.get("mega_offset", 0))
    if phase_done(prog, "mega"):
        log("FASE 4: skip mega (completato)")
        return
    names = load_object_names(limit=MEGA_TOTAL)
    if offset >= len(names):
        log("FASE 4: offset oltre fine lista")
        mark_done(prog, "mega")
        return
    log(f"=== FASE 4: oggetti HD uno-a-uno ({offset}/{len(names)}) ===")
    taught = 0
    for i in range(offset, len(names)):
        name = names[i]
        url = picsum_url(name, size=256)
        try:
            r = post(
                "/api/baby/teach-url",
                {
                    "url": url,
                    "name": name,
                    "phrase": phrase_for(name),
                    "kind": "object",
                    "persist": False,
                },
                timeout=90,
            )
            if r.get("ok"):
                taught += 1
        except Exception as e:
            log(f"  skip {name} @ {i}: {e!s}"[:70])
        prog["mega_offset"] = i + 1
        if (i - offset) % 5 == 0:
            save_progress(prog)
        if (i + 1) % SLEEP_EVERY_OBJECTS == 0:
            post("/api/baby/sleep", {})
            log(f"  checkpoint {i + 1}/{len(names)} taught={taught}")
        if i % 20 == 0:
            log(f"  … {name} ({i + 1}/{len(names)})")
        time.sleep(MEGA_PAUSE)

    save_progress(prog)
    post("/api/baby/sleep", {})
    post("/api/baby/rebalance-lexicon", {})
    log(f"  mega fine: taught={taught} total={len(names)}")
    mark_done(prog, "mega")


def phase_vocab(prog: dict) -> None:
    if phase_done(prog, "vocab"):
        log("FASE 5: skip vocab")
        return
    log(f"=== FASE 5: lessico (fino a {VOCAB_LIMIT} frasi) ===")
    sentences = load_vocab_sentences(limit=VOCAB_LIMIT)
    start = int(prog.get("vocab_offset", 0))
    st = get_state()
    before = int(st.get("words_known", 0))
    for i in range(start, len(sentences), VOCAB_BATCH):
        chunk = sentences[i : i + VOCAB_BATCH]
        post("/api/baby/absorb", {"texts": chunk, "boost": 0.55, "persist": False})
        prog["vocab_offset"] = i + len(chunk)
        save_progress(prog)
        if ((i - start) // VOCAB_BATCH) % 5 == 4:
            post("/api/baby/sleep", {})
            st = get_state()
            log(f"  absorb {prog['vocab_offset']}/{len(sentences)} parole={st.get('words_known')}")
    st = get_state()
    log(f"  lessico {before} → {st.get('words_known')}")
    post("/api/baby/sleep", {})
    mark_done(prog, "vocab")


def phase_reading(prog: dict) -> None:
    if phase_done(prog, "reading"):
        log("FASE 6: skip lettura")
        return
    topics = [
        "https://it.wikipedia.org/wiki/Cane",
        "https://it.wikipedia.org/wiki/Gatto",
        "https://it.wikipedia.org/wiki/Albero",
        "https://it.wikipedia.org/wiki/Acqua",
        "https://it.wikipedia.org/wiki/Legno",
        "https://it.wikipedia.org/wiki/Pioggia",
        "https://it.wikipedia.org/wiki/Pinocchio",
        "https://it.wikipedia.org/wiki/Italia",
    ]
    log("=== FASE 6: lettura web ===")
    for url in topics:
        try:
            br = post("/api/baby/browse", {"url": url, "persist": False}, timeout=120)
            log(f"  {url.split('/')[-1]}: ok={br.get('ok')} kw={br.get('absorbed_words', 0)}")
        except Exception as e:
            log(f"  skip {url}: {e!s}"[:60])
        time.sleep(0.5)
    post("/api/baby/sleep", {})
    mark_done(prog, "reading")


def phase_code(prog: dict) -> None:
    if phase_done(prog, "code"):
        log("FASE 7: skip codice")
        return
    log("=== FASE 7: codice ===")
    cr = post("/api/baby/code-curriculum", {}, timeout=300)
    log(json.dumps({k: cr.get(k) for k in ("ok", "taught")}))
    post("/api/baby/sleep", {})
    mark_done(prog, "code")


def phase_consolidate(prog: dict) -> None:
    log("=== FASE 8: consolidamento + probe ===")
    post("/api/baby/rebalance-lexicon", {})
    for _ in range(3):
        post("/api/baby/sleep", {})
    probes = [
        "perché piove",
        "raccontami pinocchio",
        "chi è pinocchio",
        "cosa vedi",
        "cos'è un cane",
    ]
    for q in probes:
        m = post("/api/baby/sense", {"text": q}, timeout=180).get("moment") or {}
        spoke = str(m.get("spoke", ""))
        log(f"  PROBE {q} → ({len(spoke.split())}w) {spoke[:120]}")
    st = get_state()
    log(
        json.dumps(
            {
                "neurons": st.get("neurons"),
                "words_known": st.get("words_known"),
                "dialogues": len(st.get("dialogue_pairs", [])),
            },
            ensure_ascii=False,
        )
    )
    mark_done(prog, "consolidate")


def main() -> None:
    log(f"=== train_serious_online @ {BASE} ===")
    prog = load_progress()
    st = get_state()
    if not st.get("born"):
        post("/api/baby/birth", {})
        st = get_state()
    log(
        f"stato: neuroni={st.get('neurons')} parole={st.get('words_known')} "
        f"dialoghi={len(st.get('dialogue_pairs', []))} progress={prog}"
    )
    post("/api/baby/plasticity", {"rate": float(os.environ.get("TRAIN_PLASTICITY", "0.025"))})

    for fn in (
        phase_semantic,
        phase_dialogue,
        phase_vision,
        phase_mega,
        phase_vocab,
        phase_reading,
        phase_code,
        phase_consolidate,
    ):
        try:
            fn(prog)
        except Exception as e:
            log(f"ERRORE in {fn.__name__}: {e!s}")
            save_progress(prog)
            raise

    post("/api/baby/plasticity", {"rate": 0.015})
    log("=== TRAINING SERIO ONLINE COMPLETATO ===")


if __name__ == "__main__":
    main()
