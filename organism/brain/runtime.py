"""Brain runtime — always-on consciousness loop with sensory → output flow."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from organism.brain.architect import BrainArchitect
from organism.brain.consciousness import ConsciousnessState, PhiCalculator
from organism.brain.dopamine import DopamineSystem
from organism.brain.gpu_engine import GPUBrainEngine
from organism.brain.regions import REGIONS, total_neurons
from organism.brain.topology import NeuralTopology, Spike


@dataclass
class BrainRuntime:
    """Operational brain — perception, integration, consciousness, output."""

    brain: NeuralTopology
    dopamine: DopamineSystem = field(default_factory=DopamineSystem)
    phi_calc: PhiCalculator = field(default_factory=PhiCalculator)
    consciousness: ConsciousnessState = field(default_factory=ConsciousnessState)
    gpu: GPUBrainEngine = field(default_factory=GPUBrainEngine)
    _born: bool = False
    _pulse_count: int = 0
    _last_tick_ms: float = 0.0
    _gpu_weights: Any = None
    _use_gpu_propagate: bool = False

    @classmethod
    def create(cls, seed: int = 42, *, use_gpu: bool = False) -> BrainRuntime:
        architect = BrainArchitect(seed=seed)
        brain = architect.build()
        runtime = cls(brain=brain, gpu=GPUBrainEngine(use_gpu=use_gpu))
        runtime._use_gpu_propagate = use_gpu and runtime.gpu.gpu_available
        if runtime._use_gpu_propagate:
            row_ptr, cols, weights = brain.to_csr_arrays()
            runtime._gpu_weights = runtime.gpu.build_sparse_weights(
                brain.neuron_count, row_ptr, cols, weights
            )
        return runtime

    def birth(self) -> dict[str, Any]:
        self._born = True
        self._pulse_count = 0
        for region in ("interoceptive", "proprioceptive"):
            self.brain.inject_region(region, 0.2, self.brain.tick, fraction=0.1)
        return {"born": True, "neurons": self.brain.neuron_count, "synapses": self.brain.synapse_count}

    def perceive_text(self, text: str) -> dict[str, Any]:
        """Auditory cortex: text → phoneme-pattern activation."""
        if not text:
            return {"activated": 0}
        t = self.brain.tick
        intensity = min(1.0, 0.3 + len(text) * 0.02)
        n = self.brain.inject_region("auditory_cortex", intensity, t, fraction=0.25)
        if any(c.isupper() for c in text):
            self.brain.inject_region("visual_cortex", 0.15, t, fraction=0.05)
        return {"activated": n, "modality": "auditory", "intensity": intensity}

    def perceive_vision(self, *, intensity: float = 0.5) -> dict[str, Any]:
        n = self.brain.inject_region("visual_cortex", intensity, self.brain.tick, fraction=0.2)
        return {"activated": n, "modality": "visual"}

    def tick(self, *, task_complexity: float = 0.5) -> dict[str, Any]:
        """One brain cycle: propagate → dopamine → plasticity → Φ."""
        if not self._born:
            return {"alive": False}

        t0 = time.perf_counter()
        self._pulse_count += 1

        prop = self.brain.propagate()
        self._update_dopamine()
        plastic = self.brain.plasticity_tick(self.dopamine.level)
        self._update_consciousness(task_complexity)

        self._last_tick_ms = (time.perf_counter() - t0) * 1000
        return {
            "alive": True,
            "pulse": self._pulse_count,
            "propagate": prop,
            "plasticity": plastic,
            "dopamine": self.dopamine.to_dict(),
            "consciousness": self.consciousness.to_dict(),
            "tick_ms": round(self._last_tick_ms, 2),
        }

    def brain_pulse_tick(self) -> dict[str, Any]:
        """Background pulse — keeps consciousness loop alive."""
        complexity = min(1.0, 0.3 + self._pulse_count * 0.001)
        return self.tick(task_complexity=complexity)

    def chat(self, text: str) -> dict[str, Any]:
        """Full sensory → language → output cycle."""
        self.perceive_text(text)
        for _ in range(3):
            self.tick(task_complexity=min(1.0, len(text) * 0.05))
        output = self._compose_output(text)
        self.phi_calc.record_thought(self.consciousness, f"heard: {text[:40]}")
        return {
            "input": text,
            "output": output,
            "consciousness": self.consciousness.to_dict(),
            "dopamine": self.dopamine.to_dict(),
            "active_neurons": self.brain.active_count,
        }

    def _update_dopamine(self) -> None:
        pfc = self.brain.region_activation("prefrontal_cortex")
        sensory = (
            self.brain.region_activation("auditory_cortex")
            + self.brain.region_activation("visual_cortex")
        ) / 2
        da_act = self.brain.region_activation("dopamine_neurons")
        pe = self.dopamine.compute_prediction_error(pfc, sensory)
        self.dopamine.update(pe, da_act)

    def _update_consciousness(self, task_complexity: float) -> None:
        activations = self.brain.all_region_activations()
        self.consciousness = self.phi_calc.compute(activations, task_complexity)
        if self.consciousness.ignition and self._pulse_count % 5 == 0:
            self.phi_calc.record_thought(
                self.consciousness,
                f"integrating {self.consciousness.focus_region}",
            )

    def _compose_output(self, text: str) -> str:
        broca_act = self.brain.region_activation("broca")
        wernicke_act = self.brain.region_activation("wernicke")
        motor_act = self.brain.region_activation("motor_cortex")
        if broca_act < 0.1 and wernicke_act < 0.1:
            return ""
        strength = (broca_act + wernicke_act + motor_act) / 3
        if strength < 0.15:
            return "…"
        phi_tag = f"[Φ={self.consciousness.phi:.2f}]"
        return f"{phi_tag} processing: {text[:60]}"

    def health(self) -> dict[str, Any]:
        cap = self.gpu.estimate_capacity_for_brain(
            self.brain.neuron_count,
            self.brain.synapse_count,
        )
        return {
            "ok": True,
            "born": self._born,
            "neurons": self.brain.neuron_count,
            "synapses": self.brain.synapse_count,
            "active": self.brain.active_count,
            "architecture": {
                "version": "biological_22k",
                "regions": len(REGIONS),
                "total_neurons_target": total_neurons(),
                "capacity": {
                    "gpu_backend": cap.backend,
                    "gpu_device": cap.device,
                    "ticks_per_second": cap.ticks_per_second,
                    "max_neurons_estimate": cap.max_neurons_estimate,
                    "max_synapses_estimate": cap.max_synapses_estimate,
                    "impulse_mode": cap.backend,
                },
                "score": {
                    "overall": round(self.consciousness.phi * 100, 1),
                    "integration": round(self.consciousness.integration * 100, 1),
                },
            },
            "dopamine": self.dopamine.to_dict(),
            "consciousness": self.consciousness.to_dict(),
        }

    def state_lite(self) -> dict[str, Any]:
        return {
            "born": self._born,
            "neurons": self.brain.neuron_count,
            "synapses": self.brain.synapse_count,
            "active": self.brain.active_count,
            "pulse": self._pulse_count,
            "architecture": self.health()["architecture"],
            "emotion": {
                "label": "curious" if self.dopamine.level > 0.4 else "calm",
                "curiosity": round(self.dopamine.level, 3),
            },
            "waves": {
                "phase": "gamma" if self.consciousness.ignition else "theta",
                "tick": self._pulse_count,
            },
            "consciousness_stream": list(self.consciousness.stream),
            "body_schema": {
                "navigate_mode": "float",
                "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "velocity": 0.0, "vertical_velocity": 0.0, "heading_deg": 0.0},
                "place_cells_active": len(self.brain.region_ids("proprioceptive")) // 10,
            },
            "dopamine": self.dopamine.to_dict(),
            "phi": self.consciousness.phi,
        }

    def consciousness_events(self, *, since_seq: int = 0, limit: int = 48) -> dict[str, Any]:
        events = [
            {"seq": i + 1, "text": s}
            for i, s in enumerate(self.consciousness.stream)
            if i + 1 > since_seq
        ][-limit:]
        return {"events": events, "seq": self.consciousness._seq}

    def consciousness_recent(self, limit: int = 24) -> list[str]:
        return self.consciousness.stream[-limit:]
