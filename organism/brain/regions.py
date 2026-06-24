"""Biological brain regions — 7 systems, ~22.8k neurons."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BrainSystem(str, Enum):
    SENSORY = "sensory"
    INTEGRATION = "integration"
    PREFRONTAL_LIMBIC = "prefrontal_limbic"
    LANGUAGE = "language"
    CONSCIOUSNESS = "consciousness"


@dataclass(frozen=True, slots=True)
class RegionSpec:
    name: str
    system: BrainSystem
    neuron_count: int
    role: str


# Totale: 23_800 neuroni (somma esatta delle 18 regioni)
REGIONS: tuple[RegionSpec, ...] = (
    # 1. Sensory Input System (4_300)
    RegionSpec("auditory_cortex", BrainSystem.SENSORY, 2000, "phoneme patterns from text/audio"),
    RegionSpec("visual_cortex", BrainSystem.SENSORY, 1500, "edge/object features"),
    RegionSpec("proprioceptive", BrainSystem.SENSORY, 500, "internal spatial awareness"),
    RegionSpec("interoceptive", BrainSystem.SENSORY, 300, "hunger, fatigue, attention"),
    # 2. Integration Core (9_000)
    RegionSpec("association_cortex", BrainSystem.INTEGRATION, 5000, "cross-modal meaning"),
    RegionSpec("temporal_lobe", BrainSystem.INTEGRATION, 3000, "sequences, episodic context"),
    RegionSpec("insula", BrainSystem.INTEGRATION, 1000, "body-emotion integration"),
    # 3. Prefrontal + Limbic (4_300)
    RegionSpec("prefrontal_cortex", BrainSystem.PREFRONTAL_LIMBIC, 3000, "working memory, planning"),
    RegionSpec("anterior_cingulate", BrainSystem.PREFRONTAL_LIMBIC, 500, "conflict, motivation"),
    RegionSpec("amygdala", BrainSystem.PREFRONTAL_LIMBIC, 300, "threat/reward urgency"),
    RegionSpec("dopamine_neurons", BrainSystem.PREFRONTAL_LIMBIC, 500, "prediction error signal"),
    # 4. Language Specialization (3_000)
    RegionSpec("wernicke", BrainSystem.LANGUAGE, 1500, "language comprehension"),
    RegionSpec("broca", BrainSystem.LANGUAGE, 1000, "language production"),
    RegionSpec("motor_cortex", BrainSystem.LANGUAGE, 500, "articulation output"),
    # 5. Consciousness Loop (3_200)
    RegionSpec("thalamic_relay", BrainSystem.CONSCIOUSNESS, 2000, "attention gating"),
    RegionSpec("posterior_cingulate", BrainSystem.CONSCIOUSNESS, 400, "self-referential thought"),
    RegionSpec("temporo_parietal_junction", BrainSystem.CONSCIOUSNESS, 300, "theory of mind"),
    RegionSpec("consciousness_integrator", BrainSystem.CONSCIOUSNESS, 500, "integrated information Φ"),
)

REGION_BY_NAME: dict[str, RegionSpec] = {r.name: r for r in REGIONS}


def total_neurons() -> int:
    return sum(r.neuron_count for r in REGIONS)


def regions_by_system(system: BrainSystem) -> list[RegionSpec]:
    return [r for r in REGIONS if r.system == system]
