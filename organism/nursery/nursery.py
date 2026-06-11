"""Nursery — osserva l'organismo nascere, pensare, crescere."""

from __future__ import annotations

import time
from typing import Any

from organism.nursery.growth import GrowthSnapshot, GrowthTracker
from organism.nursery.journal import ThoughtJournal
from organism.nursery.phases import CURRICULUM, Lesson, get_phase
from organism.runtime import OrganismRuntime


class Nursery:
    """
    Wrapper osservabile attorno a OrganismRuntime.
    Ogni teach() = un momento osservabile nel journal + growth timeline.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.org: OrganismRuntime | None = None
        self.journal = ThoughtJournal()
        self.growth = GrowthTracker()
        self.phase = "unborn"
        self._lesson_index: dict[str, int] = {}

    def birth(self) -> dict[str, Any]:
        """DNA → cervello. Il 'parto'."""
        self.org = OrganismRuntime.studio_assistant(seed=self.seed)
        self.phase = "birth"
        snap = self._brain_snapshot(0, "birth", event="birth")
        self.journal.log_birth(snap)
        self.growth.record(
            GrowthSnapshot(
                cycle=0,
                phase="birth",
                neurons=snap["neurons"],
                synapses=snap["synapses"],
                mean_weight=snap["mean_weight"],
                learning_cycles=0,
                fragment_count=snap["fragment_count"],
                learned_fragments=0,
                layer_activation=snap["layer_activation"],
                event="🌱 Nascita — DNA dispiegato",
            )
        )
        return {
            "event": "birth",
            "message": "Struttura neurale generata dal DNA",
            "stats": self.org.stats,
            "graph": self.org.brain.export_active_subgraph(),
            "birth": self.journal.birth_record,
        }

    def teach(
        self,
        input_data: dict[str, Any],
        *,
        phase: str | None = None,
        modality: str = "speech",
        **kwargs,
    ) -> dict[str, Any]:
        if self.org is None:
            self.birth()
        assert self.org is not None
        if phase:
            self.phase = phase

        thought, expr, learn = self.org.live(input_data, output_modality=modality, **kwargs)  # type: ignore

        expr_text = ""
        if expr.speech:
            expr_text = expr.speech.text
        elif expr.text:
            expr_text = expr.text.text

        symbols = thought.symbols + (learn.symbols if learn else [])
        cycle = self.org.learner.total_cycles
        brain_snap = self._brain_snapshot(cycle, self.phase)

        entry = self.journal.record(
            cycle=cycle,
            phase=self.phase,
            input_data=input_data,
            thought_symbols=symbols,
            mind_action=thought.mind_result.action.id if thought.mind_result.action else None,
            mind_fragments=[f.title for f in thought.mind_result.fragments[:5]],
            expression_text=expr_text,
            learning=learn.__dict__ if learn else None,
            brain_snapshot=brain_snap,
            new_fragment=learn.new_fragment_id if learn else None,
        )

        self.growth.record(
            GrowthSnapshot(
                cycle=cycle,
                phase=self.phase,
                neurons=brain_snap["neurons"],
                synapses=brain_snap["synapses"],
                mean_weight=brain_snap["mean_weight"],
                learning_cycles=cycle,
                fragment_count=brain_snap["fragment_count"],
                learned_fragments=brain_snap["learned_fragments"],
                layer_activation=brain_snap["layer_activation"],
                event=f"cycle {cycle}",
            )
        )

        return {
            "entry": entry.to_dict(),
            "graph": self.org.brain.export_active_subgraph(),
            "stats": self.org.stats,
        }

    def run_lesson(self, phase_id: str, lesson_id: str) -> dict[str, Any]:
        phase = get_phase(phase_id)
        if phase is None:
            return {"error": f"phase unknown: {phase_id}"}
        lesson = next((l for l in phase.lessons if l.id == lesson_id), None)
        if lesson is None:
            return {"error": f"lesson unknown: {lesson_id}"}
        self.phase = phase_id
        return self.teach(
            lesson.input_data,
            phase=phase_id,
            modality=lesson.modality,
            **lesson.kwargs,
        )

    def run_phase(self, phase_id: str) -> list[dict[str, Any]]:
        phase = get_phase(phase_id)
        if phase is None:
            return [{"error": f"phase unknown: {phase_id}"}]
        if phase_id == "birth":
            return [self.birth()]
        results = []
        for lesson in phase.lessons:
            results.append(self.run_lesson(phase_id, lesson.id))
            time.sleep(0.05)  # tiny pause so UI sees steps
        return results

    def run_full_curriculum(self) -> dict[str, Any]:
        """Tutta la 'crescita' — nascita → sensi → linguaggio → mondo."""
        self.birth()
        for phase in CURRICULUM:
            if phase.id == "birth":
                continue
            self.run_phase(phase.id)
        return self.state()

    def sleep(self) -> dict[str, Any]:
        if self.org is None:
            return {"error": "not born"}
        result = self.org.sleep()
        snap = self._brain_snapshot(self.org.learner.total_cycles, self.phase, event="sleep")
        self.growth.record(
            GrowthSnapshot(
                cycle=snap["cycle"],
                phase=self.phase,
                neurons=snap["neurons"],
                synapses=snap["synapses"],
                mean_weight=snap["mean_weight"],
                learning_cycles=snap["cycle"],
                fragment_count=snap["fragment_count"],
                learned_fragments=snap["learned_fragments"],
                layer_activation=snap["layer_activation"],
                event="😴 Sonno — pruning",
            )
        )
        return {**result, "stats": self.org.stats}

    def verify(self) -> dict[str, Any]:
        return self.growth.verify_auto_development().to_dict()

    def state(self) -> dict[str, Any]:
        org = self.org
        return {
            "born": org is not None,
            "phase": self.phase,
            "stats": org.stats if org else {},
            "birth": self.journal.birth_record,
            "thoughts": self.journal.recent(40),
            "thought_stream": self.journal.stream_text(20),
            "growth": self.growth.timeline_dict(),
            "verification": self.verify(),
            "graph": org.brain.export_active_subgraph() if org else {"nodes": [], "edges": []},
        }

    def _brain_snapshot(self, cycle: int, phase: str, event: str = "") -> dict[str, Any]:
        org = self.org
        if org is None:
            return {}
        frags = org.memory.all_fragments()
        learned = sum(1 for f in frags if f.id.startswith("learned_"))
        return {
            "cycle": cycle,
            "phase": phase,
            "event": event,
            "neurons": org.brain.neuron_count,
            "synapses": org.brain.synapse_count,
            "mean_weight": round(org.brain.mean_synapse_weight(), 6),
            "fragment_count": len(frags),
            "learned_fragments": learned,
            "layer_activation": org.brain.layer_activation_summary(),
            "tick": org.brain.tick,
        }
