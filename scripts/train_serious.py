#!/usr/bin/env python3
"""
Training serio — locale (no HTTP), batch persist, immagini, oggetti, lessico, storie.

Eseguire con nursery FERMO per evitare corruzione stato:
  systemctl stop organism-nursery
  python scripts/train_serious.py
  systemctl start organism-nursery

Oppure: bash scripts/run_serious_training.sh
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG = Path("/tmp/organism_train_serious.log")
os.environ.setdefault("ORGANISM_PERSIST_EVERY", "999999")
os.environ.setdefault("ORGANISM_STATE_COMPACT", "1")


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_vocab_sentences(limit: int = 0) -> list[str]:
    """Frasi per absorb — oggetti + definizioni sintetiche."""
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
    # frasi definizione — grounding leggero per absorb
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


def phase_semantic(agent) -> None:
    from organism.teaching.story_curriculum import all_semantic_stories

    log("=== FASE 1: storie semantiche (parole + beat) ===")
    for story in all_semantic_stories():
        sid = story["story_id"]
        for word, definition, related in story["words"]:
            agent.teach_word(word, definition, related=related, story_id=sid, persist=False)
        for order, summary, entities, hooks in story["beats"]:
            agent.teach_story_beat(
                sid, order, summary, entities=entities, hooks=hooks, persist=False
            )
        cov = agent.semantic.coverage(sid)
        log(f"  {sid} coverage={cov}")


def phase_dialogue(agent) -> None:
    from organism.teaching.corpus import (
        CODE_DIALOGUES,
        PHILOSOPHY,
        REASONING,
        high_freq_pairs,
    )

    log("=== FASE 2: dialoghi causali e vocab semantico ===")
    pairs = list(REASONING) + list(PHILOSOPHY) + high_freq_pairs()
    for when, say in pairs:
        for _ in range(4):
            agent.teach_dialogue(when, say, persist=False)
    for when, say, kind in CODE_DIALOGUES:
        for _ in range(3):
            agent.teach_dialogue(when, say, kind=kind, persist=False)
    log(f"  dialoghi insegnati: {len(pairs) + len(CODE_DIALOGUES)}")


def phase_vision(agent) -> None:
    log("=== FASE 3: curriculum web (volti, emozioni, oggetti base) ===")
    wr = agent.run_web_curriculum(objects=True, faces=True, emotions=True)
    log(json.dumps({k: wr.get(k) for k in ("ok", "taught_objects", "taught_faces", "taught_emotions")}))


def phase_mega(agent) -> None:
    limit = int(os.environ.get("MEGA_CURRICULUM_LIMIT", "1000"))
    pause = float(os.environ.get("MEGA_CURRICULUM_PAUSE", "0.25"))
    log(f"=== FASE 4: mega curriculum {limit} oggetti HD ===")
    mr = agent.run_mega_curriculum(limit=limit, pause_s=pause)
    log(json.dumps({k: mr.get(k) for k in ("ok", "taught", "total", "errors")}, ensure_ascii=False)[:500])


def phase_vocab(agent) -> None:
    limit = int(os.environ.get("VOCAB_SENTENCES_LIMIT", "8000"))
    batch = int(os.environ.get("VOCAB_BATCH", "200"))
    log(f"=== FASE 5: lessico massivo (fino a {limit} frasi) ===")
    sentences = load_vocab_sentences(limit=limit)
    before = agent.composer.lexicon.count
    for i in range(0, len(sentences), batch):
        chunk = sentences[i : i + batch]
        agent.absorb_vocabulary(chunk, boost=0.55, persist=False)
        if (i // batch) % 5 == 0:
            agent.sleep_cycle()
            log(f"  absorb {i + len(chunk)}/{len(sentences)} parole={agent.composer.lexicon.count}")
    log(f"  lessico {before} → {agent.composer.lexicon.count}")


def phase_reading(agent) -> None:
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
            br = agent.browse_web(url, persist=False)
            log(f"  {url.split('/')[-1]}: ok={br.get('ok')} kw={br.get('absorbed_words', 0)}")
        except Exception as e:
            log(f"  skip {url}: {e!s}"[:60])
        time.sleep(0.5)


def phase_code(agent) -> None:
    log("=== FASE 7: codice ===")
    cr = agent.run_code_curriculum()
    log(json.dumps({k: cr.get(k) for k in ("ok", "taught")}))


def phase_consolidate(agent) -> None:
    log("=== FASE 8: sonno + rebalance + probe ===")
    agent.composer.lexicon.squash_overexposed()
    for _ in range(3):
        agent.sleep_cycle()
    agent._persist()

    probes = [
        "perché piove",
        "raccontami pinocchio",
        "chi è pinocchio",
        "cosa vedi",
        "cos'è un cane",
    ]
    for q in probes:
        m = agent.sense(text=q).get("moment") or {}
        spoke = str(m.get("spoke", ""))
        log(f"  PROBE {q} → ({len(spoke.split())}w) {spoke[:120]}")

    h = agent.health()
    log(
        json.dumps(
            {
                "neurons": h.get("neurons"),
                "synapses": h.get("synapses"),
                "words_known": h.get("words_known"),
                "pinocchio": agent.semantic.coverage("pinocchio"),
                "speech_diversity": h.get("speech_diversity"),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    LOG.write_text(f"=== train_serious @ {time.strftime('%Y-%m-%d %H:%M')} ===\n")
    from organism.autonomous.baby_agent import BabyAgent
    from organism.autonomous.baby_store import baby_state_path

    agent = BabyAgent(store_path=str(baby_state_path()))
    if not agent._born:
        agent.birth()
    org = agent.org
    assert org is not None
    log(f"start neurons={org.brain.neuron_count} synapses={org.brain.synapse_count} words={agent.composer.lexicon.count}")

    if org.brain.plasticity:
        org.brain.plasticity.hebbian["rate"] = float(os.environ.get("TRAIN_PLASTICITY", "0.025"))

    for fn in (
        phase_semantic,
        phase_dialogue,
        phase_vision,
        phase_mega,
        phase_vocab,
        phase_reading,
        phase_code,
    ):
        fn(agent)
        agent.composer.lexicon.squash_overexposed()
        agent._persist()
        log(f"  checkpoint dopo {fn.__name__}")
    phase_consolidate(agent)
    log("=== TRAINING SERIO COMPLETATO ===")


if __name__ == "__main__":
    main()
