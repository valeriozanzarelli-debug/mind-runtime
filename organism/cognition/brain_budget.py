"""Budget neurale — quanti neuroni servono al pensiero vs corpo/motorio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Cervello umano (ordine di grandezza, letteratura)
HUMAN_TOTAL_NEURONS = 86_000_000_000
HUMAN_CEREBELLUM = 69_000_000_000  # ~80% — coordinazione motoria, non pensiero
HUMAN_CORTEX = 16_000_000_000
HUMAN_MOTOR_SOMATOSENSORY = 2_500_000_000  # aree primarie motorie/somatiche ~15% corteccia
HUMAN_THINKING_CORTEX = 10_000_000_000  # associativa + PFC + limbica cognitiva (stima)

# Subtipi motori «corpo» vs «linguaggio» nel nostro DNA
BODY_MOTOR_SUBTYPES = frozenset({"motion_gesture_emitter", "song_melody_composer"})
SPEECH_MOTOR_SUBTYPES = frozenset({"speech_phoneme_generator", "text_syntax_formatter"})
THINK_SUBTYPES = frozenset({"pattern_matcher", "memory_consolidator", "emotion_modulator"})
SENSORY_SUBTYPES = frozenset(
    {"vision_edge_detector", "audio_frequency_analyzer", "text_semantic_encoder"}
)


@dataclass
class LayerBudget:
    sensory: int = 0
    associative: int = 0
    motor: int = 0
    motor_body: int = 0
    motor_speech: int = 0
    think_subtypes: int = 0
    total: int = 0
    synapses: int = 0

    @property
    def thinking(self) -> int:
        """Neuroni dedicati al pensiero (associativa + emozione/memoria)."""
        return self.associative

    @property
    def thinking_ratio(self) -> float:
        return self.thinking / self.total if self.total else 0.0

    @property
    def body_motor_ratio(self) -> float:
        return self.motor_body / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "synapses": self.synapses,
            "sensory": self.sensory,
            "thinking_associative": self.thinking,
            "thinking_pct": round(100 * self.thinking_ratio, 1),
            "motor_total": self.motor,
            "motor_body": self.motor_body,
            "motor_speech": self.motor_speech,
            "body_motor_pct": round(100 * self.body_motor_ratio, 1),
            "think_subtypes": self.think_subtypes,
        }


@dataclass
class CapacityPlan:
    """Piano capacità per architettura delocalizzata server RAM + GPU locale."""

    graph_neurons: int
    graph_thinking: int
    graph_ram_mb: float
    gpu_pixels: int
    gpu_resolution: str
    gpu_ram_mb: float
    total_effective_neurons: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_neurons": self.graph_neurons,
            "graph_thinking_neurons": self.graph_thinking,
            "graph_ram_mb": round(self.graph_ram_mb, 0),
            "gpu_pixel_neurons": self.gpu_pixels,
            "gpu_resolution": self.gpu_resolution,
            "gpu_ram_mb": round(self.gpu_ram_mb, 0),
            "total_effective_neurons": self.total_effective_neurons,
            "notes": self.notes,
        }


def analyze_brain(brain) -> LayerBudget:
    b = LayerBudget(synapses=brain.synapse_count)
    subtypes: dict[str, int] = {}
    for n in brain.neurons.values():
        subtypes[n.subtype] = subtypes.get(n.subtype, 0) + 1
        if n.layer == "sensory":
            b.sensory += 1
        elif n.layer == "associative":
            b.associative += 1
        elif n.layer == "motor":
            b.motor += 1
            if n.subtype in BODY_MOTOR_SUBTYPES:
                b.motor_body += 1
            elif n.subtype in SPEECH_MOTOR_SUBTYPES:
                b.motor_speech += 1
    b.think_subtypes = sum(subtypes.get(s, 0) for s in THINK_SUBTYPES)
    b.total = brain.neuron_count
    return b


def estimate_graph_ram_mb(neurons: int, synapses: int) -> float:
    """Empirico da benchmark mega (~2.65 KB/neurone incl. sinapsi)."""
    return 45 + neurons * 0.00265 + synapses * 0.00000035


def estimate_gpu_field_mb(width: int, height: int, *, temporal: bool = True) -> float:
    px = width * height
    tensors = 5
    mb = px * 4 * tensors / (1024 * 1024)
    if temporal:
        mb += px * 4 * 2 / (1024 * 1024)
    mb *= 1.35
    return mb


def recommend_gpu_resolution(
    gpu_vram_mb: int = 8192,
    *,
    reserve_mb: int = 2500,
    max_side: int = 4096,
) -> tuple[int, int, float]:
    """Risoluzione campo impulsi — conservativa per 8 GB (compute + doppio buffer)."""
    budget = max(512, gpu_vram_mb - reserve_mb)
    px_budget = budget * 1024 * 1024 / (4 * 7 * 1.35)
    side = int(min(max_side, max(512, px_budget**0.5)))
    side = (side // 64) * 64
    w = side
    h = max(384, int(side * 0.75))
    h = (h // 64) * 64
    return w, h, estimate_gpu_field_mb(w, h)


def recommend_graph_tier(server_ram_gb: int = 16, *, reserve_gb: float = 3.5) -> dict[str, Any]:
    """Profilo DNA consigliato per RAM server."""
    budget_mb = max(512, (server_ram_gb - reserve_gb) * 1024)
    tiers = {
        "baby": (1_500, 63_000, 50),
        "mind": (2_100, 25_000, 55),
        "mega": (1_584_000, 13_555_599, 4200),
        "giga": (2_912_000, 15_447_000, 9400),
        "mind_giga": (4_200_000, 16_000_000, 11_500),
        "ultra": (5_000_000, 18_000_000, 14_000),
    }
    best = "baby"
    for name, (n, s, ram) in tiers.items():
        if ram <= budget_mb:
            best = name
    return {
        "server_ram_gb": server_ram_gb,
        "budget_mb": budget_mb,
        "recommended_variant": best,
        "tiers": {k: {"neurons": v[0], "ram_mb": v[2]} for k, v in tiers.items()},
    }


def build_capacity_plan(
    *,
    graph_neurons: int,
    graph_thinking: int,
    graph_synapses: int,
    gpu_w: int,
    gpu_h: int,
) -> CapacityPlan:
    gpu_px = gpu_w * gpu_h
    eff = graph_neurons + gpu_px
    notes = [
        "Grafo DNA: neuroni funzionali sparse (pensiero + linguaggio).",
        "GPU: neuroni-pixel con sinapsi virtuali (kernel conv).",
        f"Rapporto pensiero: {100*graph_thinking/max(1,graph_neurons):.1f}% del grafo.",
        "Cervello umano: ~80% neuroni nel cerebellum (motorio) — noi non lo simuliamo.",
    ]
    return CapacityPlan(
        graph_neurons=graph_neurons,
        graph_thinking=graph_thinking,
        graph_ram_mb=estimate_graph_ram_mb(graph_neurons, graph_synapses),
        gpu_pixels=gpu_px,
        gpu_resolution=f"{gpu_w}x{gpu_h}",
        gpu_ram_mb=estimate_gpu_field_mb(gpu_w, gpu_h),
        total_effective_neurons=eff,
        notes=notes,
    )


def human_comparison(our_thinking: int, our_total: int) -> dict[str, Any]:
    return {
        "human_total_neurons": HUMAN_TOTAL_NEURONS,
        "human_cerebellum_skipped": HUMAN_CEREBELLUM,
        "human_thinking_estimate": HUMAN_THINKING_CORTEX,
        "our_graph_neurons": our_total,
        "our_thinking_neurons": our_thinking,
        "our_vs_human_thinking_pct": round(100 * our_thinking / HUMAN_THINKING_CORTEX, 6),
        "efficiency_note": (
            "Un neurone digitale associative ≈ circuito mesoscopico; "
            "sinapsi virtuali + GPU pixel ampliano capacità oltre il conteggio grezzo."
        ),
    }
