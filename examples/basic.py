#!/usr/bin/env python3
"""Esempio minimo — esegui: python examples/basic.py"""

from mind.runtime import MindRuntime
from mind.types import Cue, CueKind

rt = MindRuntime.load_seed()

scenarios = [
    ("Puzzle forme", Cue(kind=CueKind.VISUAL, value="quadrato+cerchio,triangolo+cerchio,rettangolo+")),
    ("Lampadina", Cue(kind=CueKind.TEXT, value="lampadina non si accende")),
]

for name, cue in scenarios:
    result = rt.think(cue)
    print(f"\n=== {name} ===")
    import json
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
