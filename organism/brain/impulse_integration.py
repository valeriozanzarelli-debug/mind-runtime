"""Integrazione mare impulsi GPU con Baby — impalcatura + coscienza."""

from __future__ import annotations

import os
from typing import Any

from organism.brain.impulse_scaffold import ImpulseScaffold
from organism.cognition.workspace import WorkspaceState


def impulse_enabled() -> bool:
    return os.environ.get("ORGANISM_IMPULSE", "1") != "0"


def temporal_enabled() -> bool:
    return os.environ.get("ORGANISM_TEMPORAL", "1") != "0"


def create_impulse_scaffold() -> ImpulseScaffold | None:
    if not impulse_enabled():
        return None
    remote = os.environ.get("ORGANISM_GPU_REMOTE", "").strip()
    if remote or os.environ.get("ORGANISM_HYBRID_GPU", "1") != "0":
        from organism.distributed.hybrid_impulse import HybridImpulseScaffold

        return HybridImpulseScaffold(remote)  # type: ignore[return-value]
    device = os.environ.get("ORGANISM_IMPULSE_DEVICE", "auto")
    try:
        return ImpulseScaffold(device=device)
    except Exception:
        return ImpulseScaffold(device="cpu")


def flatten_gray(grid: list[list[int]] | list[list[float]] | list[int], w: int, h: int) -> list[int] | list[list[float]]:
    if grid and isinstance(grid[0], (int, float)):
        return grid  # type: ignore[return-value]
    return grid  # type: ignore[return-value]


def merge_workspace(ws: WorkspaceState, impulse: ImpulseScaffold | None) -> WorkspaceState:
    """La coscienza a impulsi arricchisce (non sostituisce) il workspace simbolico."""
    if impulse is None:
        return ws
    reading = impulse.last_reading
    if not reading:
        return ws
    ws.ignition = max(ws.ignition, reading.ignition * 0.92)
    ws.conscious = ws.conscious or reading.conscious
    merged = list(dict.fromkeys(list(reading.broadcast) + list(ws.broadcast)))
    ws.broadcast = merged[:12]
    if reading.focus_region:
        ws.focus = f"IMP:{reading.focus_region}"
    if reading.mode in ("speak", "reflect", "flow"):
        ws.mode = reading.mode  # type: ignore[assignment]
    ws.self_signal = max(ws.self_signal, reading.self_signal)
    return ws


def impulse_consciousness_lines(impulse: ImpulseScaffold | None) -> list[str]:
    if impulse is None or impulse.last_reading is None:
        return []
    r = impulse.last_reading
    lines: list[str] = []
    if r.conscious:
        lines.append(f"impulso · coscienza {r.ignition:.2f} · {r.focus_region}")
    for s in r.sensations[:3]:
        lines.append(f"  ↳ {s}")
    for t in r.thoughts[:3]:
        lines.append(f"  ◉ {t}")
    for m in r.memories_recalled[:2]:
        lines.append(f"  mem · {m}")
    return lines


def impulse_state_dict(impulse: ImpulseScaffold | None) -> dict[str, Any]:
    if impulse is None:
        return {"enabled": False}
    out = {"enabled": True, "temporal": temporal_enabled(), **impulse.stats()}
    return out
