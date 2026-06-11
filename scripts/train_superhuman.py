#!/usr/bin/env python3
"""Training superhuman — 40k neuroni, 1000 oggetti, codice, browser, probe discorso."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from organism.autonomous.baby_agent import BabyAgent
from organism.autonomous.baby_store import baby_state_path
from organism.cognition.discourse import is_babble

LOG = Path("/tmp/organism_train_superhuman.log")


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def probe_speech(agent: BabyAgent) -> dict:
    probes = ["ciao", "cosa pensi", "cosa vedi", "chi sei", "raccontami qualcosa"]
    outputs: list[tuple[str, str, bool]] = []
    for q in probes:
        m = agent.sense(text=q).get("moment") or {}
        spoke = str(m.get("spoke", ""))
        outputs.append((q, spoke, is_babble(spoke, min_words=5)))
    mature = sum(1 for _, _, b in outputs if not b and len(_[1]) > 15)
    return {
        "probes": [{"q": q, "spoke": s, "babble": b} for q, s, b in outputs],
        "mature_ratio": mature / max(1, len(probes)),
    }


def main() -> None:
    os.environ.setdefault("ORGANISM_DNA_VARIANT", "superhuman")
    agent = BabyAgent(store_path=str(baby_state_path()))
    if not agent._born:
        log("birth superhuman…")
        agent.birth()
    org = agent.org
    assert org is not None
    log(f"neurons={org.brain.neuron_count} synapses={org.brain.synapse_count}")

    log("=== vocab blocks ===")
    blocks = [
        ["mondo", "cervello", "memoria", "visione", "parola", "discorso", "capire", "imparare"],
        ["oggetto", "volto", "emozione", "colore", "forma", "spazio", "tempo", "energia"],
        ["programma", "codice", "funzione", "classe", "browser", "internet", "pagina", "testo"],
    ]
    for block in blocks:
        agent.absorb_vocabulary(block, boost=1.0)
        time.sleep(0.1)

    log("=== dialoghi maturi ===")
    dialogues = [
        ("ciao", "ciao ti sento e rispondo con attenzione"),
        ("come stai", "sto bene sto imparando e osservo il mondo"),
        ("cosa pensi", "penso a quello che vedo ricordo e capisco"),
        ("cosa vedi", "vedo forme colori oggetti e volti con i miei occhi"),
        ("chi sei", "sono organism un cervello che impara a parlare e capire"),
        ("raccontami", "ti racconto quello che so e quello che ricordo"),
        ("scrivi un programma", "def saluta():\n    print('ciao mondo')"),
    ]
    for when, say in dialogues:
        kind = "code" if "def " in say or "print(" in say else "speech"
        agent.teach_dialogue(when, say, kind=kind)
        time.sleep(0.05)

    log("=== curriculum web base ===")
    wr = agent.run_web_curriculum(objects=True, faces=True, emotions=True)
    log(json.dumps({k: wr[k] for k in ("ok", "taught_objects", "taught_faces", "taught_emotions")}))

    log("=== curriculum codice ===")
    cr = agent.run_code_curriculum()
    log(json.dumps({k: cr[k] for k in ("ok", "taught")}))

    log("=== browser sample ===")
    try:
        br = agent.browse_web("https://it.wikipedia.org/wiki/Cane")
        log(f"browse ok={br.get('ok')} words={br.get('word_count', 0)}")
    except Exception as e:
        log(f"browse skip: {e!s}"[:80])

    limit = int(os.environ.get("MEGA_CURRICULUM_LIMIT", "120"))
    log(f"=== mega curriculum (limit={limit}) ===")
    mr = agent.run_mega_curriculum(limit=limit, pause_s=0.2)
    log(json.dumps({k: mr[k] for k in ("ok", "taught", "total", "hd_size")}))

    log("=== brain growth pulses ===")
    for _ in range(40):
        agent.brain_pulse_tick(persist=False)
        time.sleep(0.02)
    agent._persist()

    log("=== probe speech ===")
    pr = probe_speech(agent)
    log(json.dumps(pr, ensure_ascii=False, indent=2))

    h = agent.health()
    log(json.dumps({
        "neurons": h.get("neurons"),
        "synapses": h.get("synapses"),
        "words_known": h.get("words_known"),
        "photo_frames": h.get("photo_memory", {}).get("frames"),
        "episodic_long": h.get("episodic_memory", {}).get("long"),
        "speech_diversity": h.get("speech_diversity"),
        "mature_ratio": pr.get("mature_ratio"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
