"""Continuous thought loop — pensiero spontaneo senza stimoli esterni."""

from __future__ import annotations

import random
import threading
import time
from typing import Any, Callable

from organism.brain.topology import NeuralTopology
from organism.cognition.goal_stack import GoalStack
from organism.cognition.hypothesis_reasoning import HypothesisEngine
from organism.cognition.working_memory import WorkingMemory


class ThoughtGenerator:
    """Background thread: attiva neuroni weighted by Hebbian, popola working memory."""

    def __init__(
        self,
        *,
        interval_s: tuple[float, float] = (2.0, 5.0),
        seed: int = 0,
    ) -> None:
        self._interval = interval_s
        self.rng = random.Random(seed)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._brain: NeuralTopology | None = None
        self._wm: WorkingMemory | None = None
        self._goals: GoalStack | None = None
        self._hypotheses: HypothesisEngine | None = None
        self._on_thought: Callable[[list[str]], None] | None = None
        self._curiosity_fn: Callable[[], float] | None = None
        self.pulses = 0
        self._last_themes: list[str] = []
        self._last_t = 0.0

    def bind(
        self,
        brain: NeuralTopology,
        working_memory: WorkingMemory,
        *,
        goals: GoalStack | None = None,
        hypotheses: HypothesisEngine | None = None,
        on_thought: Callable[[list[str]], None] | None = None,
        curiosity_fn: Callable[[], float] | None = None,
    ) -> None:
        self._brain = brain
        self._wm = working_memory
        self._goals = goals
        self._hypotheses = hypotheses
        self._on_thought = on_thought
        self._curiosity_fn = curiosity_fn

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="organism-thought-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def tick_once(self) -> list[str]:
        """Un impulso di pensiero — usabile anche senza thread."""
        themes = self._spontaneous_thought()
        if themes and self._wm:
            self._wm.activate(themes, weight=0.6, link_to=themes[:2])
        if themes and self._on_thought:
            self._on_thought(themes)
        self.pulses += 1
        self._last_themes = themes
        self._last_t = time.time()
        return themes

    def _run(self) -> None:
        while not self._stop.is_set():
            lo, hi = self._interval
            wait = self.rng.uniform(lo, hi)
            if self._stop.wait(wait):
                break
            try:
                self.tick_once()
            except Exception:
                pass

    def _spontaneous_thought(self) -> list[str]:
        brain = self._brain
        if brain is None:
            return []

        curiosity = self._curiosity_fn() if self._curiosity_fn else 0.4
        themes: list[str] = []

        # Weighted activation — sinapsi forti guidano il salto
        if brain.synapses:
            sample = self.rng.sample(brain.synapses, min(40, len(brain.synapses)))
            weighted = sorted(sample, key=lambda s: -s.weight)[:12]
            for syn in weighted:
                pre = brain.neurons.get(syn.pre_id)
                post = brain.neurons.get(syn.post_id)
                if not pre or not post:
                    continue
                boost = syn.weight * (0.5 + curiosity * 0.5)
                pre.activation = min(1.0, pre.activation + boost * 0.08)
                post.activation = min(1.0, post.activation + boost * 0.06)
                tag = post.subtype.replace("_", " ")[:16]
                if tag and tag not in themes and post.layer == "associative":
                    themes.append(tag)

        brain.propagate(steps=1)

        if self._goals:
            themes = list(dict.fromkeys(self._goals.attention_bias_words() + themes))[:8]
        if self._hypotheses:
            themes = list(dict.fromkeys(self._hypotheses.theme_hints() + themes))[:10]

        # Curiosity-driven: ultimi temi WM come ponti
        if self._wm and curiosity > 0.35:
            ctx = self._wm.context_words(limit=6)
            themes = list(dict.fromkeys(ctx + themes))[:10]

        return [t for t in themes if t and len(t) > 2][:8]

    def stats(self) -> dict[str, Any]:
        return {
            "pulses": self.pulses,
            "last_themes": self._last_themes,
            "last_t": self._last_t,
            "running": bool(self._thread and self._thread.is_alive()),
        }
