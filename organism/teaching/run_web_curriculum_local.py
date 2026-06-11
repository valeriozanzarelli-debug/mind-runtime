#!/usr/bin/env python3
"""Curriculum web senza HTTP — eseguire sul server."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from organism.autonomous.baby_agent import BabyAgent
from organism.autonomous.baby_store import baby_state_path


def main() -> None:
    agent = BabyAgent(store_path=str(baby_state_path()))
    if not agent._born:
        agent.birth()
    print("=== curriculum locale ===", flush=True)
    r = agent.run_web_curriculum(objects=True, faces=True, emotions=True)
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
