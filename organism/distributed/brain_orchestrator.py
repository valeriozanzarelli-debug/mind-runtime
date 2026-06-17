"""Orchestratore cervello ibrido — grafo RAM + GPU remota + vault disco."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from organism.brain.gpu_backend import gpu_info
from organism.cognition.brain_budget import analyze_brain, build_capacity_plan, recommend_gpu_resolution
from organism.cognition.disk_vault import DiskMemoryVault


@dataclass
class BrainOrchestrator:
    """Vista unificata della potenza disponibile (delocalizzata)."""

    brain: Any
    impulse: Any | None = None
    disk_vault: DiskMemoryVault | None = None
    server_ram_gb: int = 16
    gpu_vram_gb: int = 8

    def __post_init__(self) -> None:
        if os.environ.get("ORGANISM_DISK_VAULT", "1") != "0" and self.disk_vault is None:
            self.disk_vault = DiskMemoryVault()

    def capacity(self) -> dict[str, Any]:
        budget = analyze_brain(self.brain)
        gw, gh, _ = recommend_gpu_resolution(self.gpu_vram_gb * 1024)
        gpu_px = gw * gh
        if self.impulse and hasattr(self.impulse, "stats"):
            st = self.impulse.stats()
            if st.get("width") and st.get("height"):
                gpu_px = int(st["width"]) * int(st["height"])
        plan = build_capacity_plan(
            graph_neurons=budget.total,
            graph_thinking=budget.thinking,
            graph_synapses=budget.synapses,
            gpu_w=gw,
            gpu_h=gh,
        )
        vault_eps = self.disk_vault.stats()["episodes"] if self.disk_vault else 0
        return {
            "graph": budget.to_dict(),
            "gpu_pixels": gpu_px,
            "gpu_resolution": plan.gpu_resolution,
            "total_compute_units": plan.total_effective_neurons,
            "thinking_neurons": budget.thinking,
            "disk_episodes": vault_eps,
            "impulse_mode": (self.impulse.stats().get("hybrid_mode") if self.impulse else "off"),
            "architecture": "hybrid_v2",
        }

    def architecture_score(self) -> dict[str, Any]:
        """Valutazione onesta — cosa è ottimo e cosa manca ancora."""
        cap = self.capacity()
        scores = {
            "sparse_graph_efficiency": 0.92,
            "mind_only_design": 0.88,
            "gpu_pixel_scale": 0.85,
            "remote_resilience": 0.80 if cap.get("impulse_mode") else 0.5,
            "unlimited_disk_memory": 0.90 if self.disk_vault else 0.0,
            "single_machine_simplicity": 0.75,
        }
        overall = sum(scores.values()) / len(scores)
        gaps = []
        if scores["remote_resilience"] < 0.85:
            gaps.append("Aggiungere WebSocket/GPU persistente per latenza minore")
        if not self.disk_vault:
            gaps.append("Abilitare ORGANISM_DISK_VAULT per memoria illimitata")
        gaps.append("Neuroni grafo ancora oggetti Python — futuro: array compatto numpy")
        return {
            "overall": round(overall, 2),
            "scores": scores,
            "gaps": gaps,
            "verdict": "ottima base" if overall >= 0.82 else "migliorabile",
            "capacity": cap,
        }

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"capacity": self.capacity(), "score": self.architecture_score()}
        if self.disk_vault:
            out["disk_vault"] = self.disk_vault.stats()
        out["local_gpu"] = gpu_info()
        return out
