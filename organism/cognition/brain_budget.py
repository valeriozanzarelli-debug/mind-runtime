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
    if getattr(brain, "compact", None) and brain.compact.count:
        b.associative += brain.compact.count
    b.think_subtypes = sum(subtypes.get(s, 0) for s in THINK_SUBTYPES)
    b.total = brain.neuron_count
    return b


def estimate_graph_ram_mb(neurons: int, synapses: int, *, compact: bool = False) -> float:
    """RAM grafo — compact ~48 B/neurone vs Python ~2.65 KB."""
    if compact:
        return 45 + neurons * 0.000048 + synapses * 0.00000012
    return 45 + neurons * 0.00265 + synapses * 0.00000035


def estimate_gpu_field_mb(
    width: int,
    height: int,
    *,
    depth: int = 1,
    temporal: bool = True,
) -> float:
    voxels = width * height * max(1, depth)
    tensors = 6 if depth > 1 else 5
    mb = voxels * 4 * tensors / (1024 * 1024)
    if temporal and depth <= 1:
        mb += voxels * 4 * 2 / (1024 * 1024)
    if depth > 1:
        mb += voxels * 4 * 1 / (1024 * 1024)  # vz + extra trace
    mb *= 1.35
    return mb


def recommend_gpu_resolution(
    gpu_vram_mb: int = 8192,
    *,
    reserve_mb: int = 2500,
    max_side: int = 512,
    depth: int | None = None,
) -> tuple[int, int, int, float]:
    """Risoluzione 3D — W×H×D voxel-neuroni per 8 GB VRAM."""
    import os

    budget = max(512, gpu_vram_mb - reserve_mb)
    d = depth if depth is not None else int(os.environ.get("ORGANISM_IMPULSE_D", "128"))
    d = max(8, min(256, d))
    # ~7 float32 tensors × 1.35 overhead per voxel
    vox_budget = budget * 1024 * 1024 / (4 * 7 * 1.35)
    area = vox_budget / d
    side = int(min(max_side, max(128, area**0.5)))
    side = (side // 64) * 64
    w = side
    h = max(128, int(side * 0.75))
    h = (h // 64) * 64
    return w, h, d, estimate_gpu_field_mb(w, h, depth=d)


def recommend_gpu_resolution_2d(
    gpu_vram_mb: int = 8192,
    *,
    reserve_mb: int = 2500,
    max_side: int = 4096,
) -> tuple[int, int, float]:
    """Legacy 2D — mantenuto per compatibilità."""
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
        "mind_compact": (80_000_000, 40_000_000, 4_200),
        "ultra": (5_000_000, 18_000_000, 14_000),
        "ultra_compact": (200_000_000, 80_000_000, 10_500),
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
    gpu_d: int = 1,
    compact: bool = False,
) -> CapacityPlan:
    gpu_px = gpu_w * gpu_h * max(1, gpu_d)
    eff = graph_neurons + gpu_px
    res = f"{gpu_w}x{gpu_h}x{gpu_d}" if gpu_d > 1 else f"{gpu_w}x{gpu_h}"
    notes = [
        "Grafo DNA: neuroni funzionali sparse (pensiero + linguaggio).",
        "GPU: neuroni-voxel 3D con sinapsi virtuali (conv3d)." if gpu_d > 1 else "GPU: neuroni-pixel con sinapsi virtuali (kernel conv).",
        f"Rapporto pensiero: {100*graph_thinking/max(1,graph_neurons):.1f}% del grafo.",
        "Compact numpy: ~48 B/neurone vs ~2.65 KB Python." if compact else "Python objects: ~2.65 KB/neurone.",
        "Cervello umano: ~80% neuroni nel cerebellum (motorio) — noi non lo simuliamo.",
    ]
    return CapacityPlan(
        graph_neurons=graph_neurons,
        graph_thinking=graph_thinking,
        graph_ram_mb=estimate_graph_ram_mb(graph_neurons, graph_synapses, compact=compact),
        gpu_pixels=gpu_px,
        gpu_resolution=res,
        gpu_ram_mb=estimate_gpu_field_mb(gpu_w, gpu_h, depth=gpu_d),
        total_effective_neurons=eff,
        notes=notes,
    )


def silicon_comparison(our_total: int) -> dict[str, Any]:
    """Perché i chip hanno miliardi di transistor ma noi partiamo da milioni."""
    return {
        "human_neurons": HUMAN_TOTAL_NEURONS,
        "human_thinking_estimate": HUMAN_THINKING_CORTEX,
        "chip_transistors_2024": 19_000_000_000,
        "our_compute_units": our_total,
        "transistor_vs_our_neuron_ratio": round(19_000_000_000 / max(1, our_total), 1),
        "key_insight": (
            "Un transistor è un interruttore; un nostro «neurone» è un nodo computazionale "
            "con stato, meta, sinapsi e propagazione — più vicino a un microcircuito. "
            "Il cervello biologico ha ~7000 sinapsi/neurone ma ogni neurone è analogico "
            "e massively parallel — non esegue istruzioni sequenziali come CPU/GPU."
        ),
        "path_to_billions": [
            "Neuroni compact numpy (~48 B) → 200M+ su 16 GB RAM",
            "Campo 3D GPU 512×384×128 → 25M voxel-neuroni",
            "Vault disco illimitato per memoria episodica",
            "Prossimo: mmap + quantizzazione 8-bit → miliardi su NVMe",
        ],
    }


def human_comparison(our_thinking: int, our_total: int) -> dict[str, Any]:
    base = silicon_comparison(our_total)
    base["our_graph_neurons"] = our_total
    base["our_thinking_neurons"] = our_thinking
    base["our_vs_human_thinking_pct"] = round(100 * our_thinking / HUMAN_THINKING_CORTEX, 4)
    return base
