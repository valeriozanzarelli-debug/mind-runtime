#!/usr/bin/env python3
"""Mega curriculum 1000 oggetti — eseguire sul server (no timeout HTTP)."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from organism.autonomous.baby_agent import BabyAgent
from organism.autonomous.baby_store import baby_state_path


def main() -> None:
    limit = int(os.environ.get("MEGA_CURRICULUM_LIMIT", "1000"))
    pause = float(os.environ.get("MEGA_CURRICULUM_PAUSE", "0.35"))
    agent = BabyAgent(store_path=str(baby_state_path()))
    if not agent._born:
        agent.birth()
    print(f"=== mega curriculum limit={limit} ===", flush=True)
    r = agent.run_mega_curriculum(limit=limit, pause_s=pause)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
