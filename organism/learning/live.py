"""Live learning — reinforce synapses + MIND fragments after each cycle."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from mind.memory import MemoryGraph
from mind.types import Fragment, Outcome, CostLevel
from organism.brain.topology import NeuralTopology
from organism.mind_bridge import OrganismThought, SensoryBundle


@dataclass
class LearningReport:
    synapses_strengthened: int = 0
    hebbian_updates: int = 0
    fragments_reinforced: list[str] = field(default_factory=list)
    new_fragment_id: str | None = None
    episode_count: int = 0
    symbols: list[str] = field(default_factory=list)


class LiveLearner:
    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.fragment_boost = float(cfg.get("fragment_weight_boost", 0.04))
        self.fragment_boost_fail = float(cfg.get("fragment_weight_penalty", 0.02))
        self.episode_compress_every = int(cfg.get("episode_compress_every", 5))
        self.pathway_boost = float(cfg.get("pathway_boost", 0.03))
        self._episode_counts: dict[str, int] = {}
        self._total_cycles = 0

    @property
    def total_cycles(self) -> int:
        return self._total_cycles

    def consolidate(
        self,
        brain: NeuralTopology,
        memory: MemoryGraph,
        sensory: SensoryBundle,
        thought: OrganismThought,
        expression: object | None = None,
        *,
        outcome_success: bool = True,
    ) -> LearningReport:
        self._total_cycles += 1
        report = LearningReport(episode_count=self._total_cycles)

        # 1. Brain: Hebbian on active neurons from this cycle
        if brain.plasticity:
            report.hebbian_updates = brain.plasticity.apply_hebbian(brain, brain.tick)
            report.hebbian_updates += brain.plasticity.apply_stdp(brain, brain.tick)

        # 2. Brain: strengthen synapses along sensory → associative → motor pathway
        report.synapses_strengthened = brain.reinforce_active_pathway(boost=self.pathway_boost)

        # 3. MIND: reinforce retrieved fragments
        delta = self.fragment_boost if outcome_success else -self.fragment_boost_fail
        for frag in thought.mind_result.fragments:
            new_w = max(0.05, min(1.0, frag.weight + delta))
            frag.weight = round(new_w, 4)
            report.fragments_reinforced.append(frag.id)

        # 4. Episodic consolidation — repeated cue+action → new compressed fragment
        action_id = thought.mind_result.action.id if thought.mind_result.action else "none"
        cue_key = thought.fused_cue.value[:80]
        ep_key = hashlib.sha256(f"{cue_key}|{action_id}".encode()).hexdigest()[:12]
        self._episode_counts[ep_key] = self._episode_counts.get(ep_key, 0) + 1
        count = self._episode_counts[ep_key]

        if count >= self.episode_compress_every and outcome_success:
            fid = f"learned_{ep_key}"
            if memory.get(fid) is None:
                hooks = _hooks_from_cue(thought.fused_cue.value)
                fn = _function_from_action(action_id)
                memory.add(
                    Fragment(
                        id=fid,
                        title=f"esperienza: {cue_key[:40]} → {action_id}",
                        weight=0.55 + min(0.35, count * 0.03),
                        sensation_id=thought.mind_result.sensation_ids[0]
                        if thought.mind_result.sensation_ids
                        else "learned",
                        hooks=hooks,
                        functions=[fn] if fn else [],
                        outcomes=[
                            Outcome(
                                action=action_id,
                                success=True,
                                cost=CostLevel.LOW,
                            )
                        ],
                    )
                )
                report.new_fragment_id = fid
                self._episode_counts[ep_key] = 0

        report.symbols = [
            f"LEARN:synapses+={report.synapses_strengthened}",
            f"LEARN:hebbian={report.hebbian_updates}",
            f"LEARN:fragments={len(report.fragments_reinforced)}",
        ]
        if report.new_fragment_id:
            report.symbols.append(f"LEARN:new_fragment={report.new_fragment_id}")
        return report


def _hooks_from_cue(text: str) -> list[str]:
    tokens = [t for t in text.lower().split() if len(t) > 3]
    return tokens[:8]


def _function_from_action(action_id: str) -> str:
    mapping = {
        "replace_bulb": "solve_lamp",
        "test_bulb": "solve_lamp",
        "ask_placement": "wa_quote",
        "ask_one_more": "wa_resonate",
        "send_quote": "wa_quote",
        "block_client": "wa_quote",
    }
    return mapping.get(action_id, "learned")
