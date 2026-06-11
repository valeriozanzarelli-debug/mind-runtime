"""Senso di sé — emerge da memoria + onde + emozione, non da stringhe fisse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mind.memory import MemoryGraph
from organism.brain.topology import NeuralTopology
from organism.cognition.affect import AffectState
from organism.cognition.waves import WaveState


@dataclass
class SelfState:
    continuity: float = 0.0
    stream: list[str] = field(default_factory=list)
    sense: str = "io"
    awake: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "continuity": round(self.continuity, 3),
            "stream": self.stream[:12],
            "sense": self.sense,
            "awake": self.awake,
        }


class SelfModel:
    """Continuità del sé — senza corpo, senza mondo esterno assunto."""

    def __init__(self) -> None:
        self.state = SelfState()
        self._episodes: list[str] = []
        self._pulse_count = 0

    def update(
        self,
        brain: NeuralTopology,
        memory: MemoryGraph,
        *,
        wave: WaveState,
        affect: AffectState,
        dream_fragment: str = "",
        saw: str = "",
        heard: str = "",
        spoke: str = "",
    ) -> SelfState:
        self._pulse_count += 1
        mem_act = _layer_mean(brain, "associative", "memory_consolidator")
        emo_act = _layer_mean(brain, "associative", "emotion_modulator")
        n_frags = len(memory.all_fragments())

        self.state.continuity = min(
            1.0,
            0.15 * n_frags / 8
            + 0.25 * mem_act
            + 0.2 * emo_act
            + 0.15 * affect.trust
            + 0.1 * (self._pulse_count / 200),
        )
        self.state.awake = wave.phase != "dream"

        if spoke:
            self._episodes.append(f"espresse:{spoke[:40]}")
        if heard:
            self._episodes.append(f"udì:{heard[:40]}")
        if saw:
            self._episodes.append(f"vide:{saw[:24]}")
        if dream_fragment:
            self._episodes.append(f"sognò:{dream_fragment[:50]}")
        self._episodes = self._episodes[-24:]

        stream: list[str] = [f"io:{wave.label}"]
        for ep in self._episodes[-5:]:
            stream.append(ep)
        if affect.label:
            stream.append(f"sentimento:{affect.label}")
        self.state.stream = stream
        return self.state

    def note_correction(self, wrong: str, right: str) -> None:
        """Impara dal proprio errore — coscienza come auto-percezione."""
        self._episodes.append(f"corretto:{wrong[:24]}→{right[:24]}")
        self._episodes = self._episodes[-24:]
        self.state.continuity = min(1.0, self.state.continuity + 0.04)
        self.state.stream = ["io:imparo"] + self.state.stream[:8]

    def stats(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "episodes": list(self._episodes),
            "pulses": self._pulse_count,
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        s = data.get("state", data)
        self.state = SelfState(
            continuity=float(s.get("continuity", 0)),
            stream=list(s.get("stream", [])),
            sense=str(s.get("sense", "io")),
            awake=bool(s.get("awake", True)),
        )
        raw_eps = data.get("episodes", [])
        if isinstance(raw_eps, list):
            self._episodes = list(raw_eps)
        else:
            self._episodes = []
        self._pulse_count = int(data.get("pulses", 0))


def _layer_mean(brain: NeuralTopology, layer: str, subtype: str) -> float:
    ns = brain.get_neurons(layer, subtype)
    if not ns:
        return 0.0
    return sum(n.activation for n in ns) / len(ns)
