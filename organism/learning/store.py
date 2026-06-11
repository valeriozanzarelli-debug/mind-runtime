"""Persist organism state — topologia completa + memoria."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mind.memory import MemoryGraph
from mind.types import Fragment, HiddenDetail, Outcome, CostLevel
from organism.brain.topology import NeuralTopology
from organism.learning.live import LiveLearner


DEFAULT_STORE = Path.home() / ".organism" / "state.json"


def build_organism_payload(
    brain: NeuralTopology,
    memory: MemoryGraph,
    learner: LiveLearner,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": 2,
        "meta": meta or {},
        "brain": brain.snapshot(),
        "memory": {
            "fragments": [_fragment_to_dict(f) for f in memory.all_fragments()],
        },
        "learner": {
            "total_cycles": learner.total_cycles,
            "episode_counts": learner._episode_counts,
        },
    }


def restore_organism_payload(
    payload: dict[str, Any],
    brain: NeuralTopology,
    memory: MemoryGraph,
    learner: LiveLearner,
) -> dict[str, Any]:
    version = int(payload.get("version", 1))
    brain_data = payload.get("brain", {})
    topo_stats: dict[str, int] = {}

    if version >= 2 and brain_data.get("neurons"):
        topo_stats = brain.restore_snapshot(brain_data)
    else:
        saved_syn = brain_data.get("synapses", [])
        updated = 0
        if len(saved_syn) == len(brain.synapses):
            for syn, row in zip(brain.synapses, saved_syn):
                syn.weight = float(row["w"])
            updated = len(saved_syn)
        else:
            syn_map = {(int(s["pre"]), int(s["post"])): float(s["w"]) for s in saved_syn}
            for syn in brain.synapses:
                key = (syn.pre_id, syn.post_id)
                if key in syn_map:
                    syn.weight = syn_map[key]
                    updated += 1
        topo_stats = {"updated_weights": updated}

    brain.tick = float(brain_data.get("tick", brain.tick))

    for fd in payload.get("memory", {}).get("fragments", []):
        frag = _fragment_from_dict(fd)
        memory.add(frag)

    lr = payload.get("learner", {})
    learner._total_cycles = int(lr.get("total_cycles", 0))
    learner._episode_counts = dict(lr.get("episode_counts", {}))

    updated = topo_stats.get("updated_weights", 0) + topo_stats.get("added_synapses", 0)
    return {
        "loaded": True,
        "version": version,
        "synapses_updated": updated,
        **topo_stats,
        "fragments": len(payload.get("memory", {}).get("fragments", [])),
        "cycles": learner.total_cycles,
    }


class OrganismStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_STORE

    def save(
        self,
        brain: NeuralTopology,
        memory: MemoryGraph,
        learner: LiveLearner,
        *,
        meta: dict[str, Any] | None = None,
    ) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = build_organism_payload(brain, memory, learner, meta=meta)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.path

    def load(
        self,
        brain: NeuralTopology,
        memory: MemoryGraph,
        learner: LiveLearner,
    ) -> dict[str, Any]:
        if not self.path.exists():
            return {"loaded": False, "reason": "no file"}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return restore_organism_payload(payload, brain, memory, learner)


def _fragment_to_dict(f: Fragment) -> dict:
    return {
        "id": f.id,
        "title": f.title,
        "weight": f.weight,
        "sensation_id": f.sensation_id,
        "hooks": f.hooks,
        "links": f.links,
        "functions": f.functions,
        "hidden_details": [
            {"key": h.key, "retrieve_only_if_cue_contains": h.retrieve_only_if_cue_contains}
            for h in f.hidden_details
        ],
        "outcomes": [
            {"action": o.action, "success": o.success, "cost": o.cost.value}
            for o in f.outcomes
        ],
    }


def _fragment_from_dict(d: dict) -> Fragment:
    return Fragment(
        id=d["id"],
        title=d["title"],
        weight=float(d["weight"]),
        sensation_id=d["sensation_id"],
        hooks=list(d.get("hooks", [])),
        links=list(d.get("links", [])),
        functions=list(d.get("functions", [])),
        hidden_details=[
            HiddenDetail(key=h["key"], retrieve_only_if_cue_contains=h["retrieve_only_if_cue_contains"])
            for h in d.get("hidden_details", [])
        ],
        outcomes=[
            Outcome(action=o["action"], success=o["success"], cost=CostLevel(o["cost"]))
            for o in d.get("outcomes", [])
        ],
    )
