#!/usr/bin/env python3
"""Training semantico — parole spiegate, beat, narrazione emergente (no monologo)."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from organism.teaching.story_curriculum import all_semantic_stories

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/organism"
LOG = Path("/tmp/organism_train_semantic.log")
REPEATS = int(os.environ.get("SEMANTIC_REPEATS", "3"))


def post(path: str, body: dict, *, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main() -> None:
    LOG.write_text(f"=== train_semantic_stories @ {BASE} ===\n")
    post("/api/baby/rebalance-lexicon", {})

    for story in all_semantic_stories():
        sid = story["story_id"]
        log(f"--- {sid}: parole ---")
        for word, definition, related in story["words"]:
            for i in range(REPEATS):
                post(
                    "/api/baby/teach-word",
                    {
                        "word": word,
                        "definition": definition,
                        "related": related,
                        "story_id": sid,
                        "persist": i == REPEATS - 1,
                    },
                )
            log(f"  {word} ok")

        log(f"--- {sid}: beat ---")
        for order, summary, entities, hooks in story["beats"]:
            for i in range(REPEATS):
                post(
                    "/api/baby/teach-story-beat",
                    {
                        "story_id": sid,
                        "order": order,
                        "summary": summary,
                        "entities": entities,
                        "hooks": hooks,
                        "persist": i == REPEATS - 1,
                    },
                )
            log(f"  beat {order} ok")

        cov = post("/api/baby/semantic-coverage", {"story_id": sid})
        log(f"coverage {sid}: {json.dumps(cov)}")

    probes = [
        "raccontami pinocchio",
        "chi è pinocchio",
        "raccontami la favola del corvo",
        "perché piove",
    ]
    log("--- probe narrazione ---")
    for q in probes:
        m = post("/api/baby/sense", {"text": q}, timeout=180).get("moment") or {}
        spoke = str(m.get("spoke", ""))
        log(f"  {q} → ({len(spoke.split())}w) {spoke[:140]}")

    log("=== fine ===")


if __name__ == "__main__":
    main()
