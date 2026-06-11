"""Vision sensory module — edges → spikes → pattern activation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from organism.brain.topology import ActivePattern, NeuralTopology, Spike
from organism.sensory._array import grayscale_grid, sobel_edges


@dataclass
class VisionResult:
    spikes: list[Spike]
    patterns: list[ActivePattern]
    edges_density: float = 0.0
    symbols: list[str] = field(default_factory=list)


class VisionModule:
    def __init__(self, brain: NeuralTopology) -> None:
        self.brain = brain
        self.detectors = brain.get_neurons("sensory", "vision_edge_detector")

    def perceive(self, image: list[list[int]] | bytes, *, width: int = 16, height: int = 16) -> VisionResult:
        t = time.time()
        grid = grayscale_grid(image, width, height)
        edges = sobel_edges(grid)
        spikes: list[Spike] = []
        active_cells = 0
        side = max(1, int(len(self.detectors) ** 0.5))
        eh, ew = len(edges), len(edges[0]) if edges else 0
        for det in self.detectors:
            rf = det.receptive_field()
            if rf is None:
                continue
            rx, ry = rf
            x = int(rx * ew / side) if ew else rx
            y = int(ry * eh / side) if eh else ry
            if y < eh and x < ew and edges[y][x] > 0.25:
                spikes.append(Spike(neuron_id=det.id, timestamp=t, intensity=edges[y][x]))
                active_cells += 1

        self.brain.inject_spikes(spikes)
        self.brain.propagate(steps=2)
        patterns = self.brain.get_active_patterns(threshold=0.35, modality="pattern")

        total = max(1, width * height)
        density = active_cells / total
        symbols = [f"VIS:edges={density:.2f}", f"VIS:spikes={len(spikes)}"]

        return VisionResult(
            spikes=spikes,
            patterns=patterns,
            edges_density=density,
            symbols=symbols,
        )

    def perceive_shapes_ascii(self, shapes_csv: str) -> VisionResult:
        """Bridge to MIND visual cue format: quadrato+cerchio,..."""
        # synthetic high activation for pattern path
        t = time.time()
        spikes = [
            Spike(neuron_id=det.id, timestamp=t, intensity=0.8)
            for det in self.detectors[: min(12, len(self.detectors))]
        ]
        self.brain.inject_spikes(spikes)
        self.brain.propagate(steps=1)
        patterns = self.brain.get_active_patterns(threshold=0.2)
        return VisionResult(
            spikes=spikes,
            patterns=patterns,
            edges_density=0.5,
            symbols=[f"VIS:shapes={shapes_csv}"],
        )
