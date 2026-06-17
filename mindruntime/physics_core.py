"""Fisica emergente — Turing, SOC, transizione di fase, lock-in armonico.

Quattro principi:
  1. Ordine globale (Kuramoto) → transizione di fase / coscienza
  2. Reazione-diffusione Turing → memoria geometrica
  3. Self-organized criticality → auto-tuning verso il margine del caos
  4. Biforcazione armonica → riconoscimento per lock-in di fase
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mindruntime.dendritic_core import CH_IMP, CH_PH, CH_W


@dataclass
class PhysicsState:
    order_parameter: float = 0.0
    conscious: bool = False
    phase_transition: str = "subcritical"  # subcritical | critical | supercritical
    avalanche_size: float = 0.0
    branch_ratio: float = 1.0
    soc_coupling: float = 0.12
    turing_energy: float = 0.0
    lock_in: float = 0.0
    locked_symbol: str = ""
    history_order: list[float] = field(default_factory=list)


def laplacian(field: np.ndarray) -> np.ndarray:
    h, w = field.shape
    out = np.zeros_like(field)
    for y in range(h):
        for x in range(w):
            s = -4.0 * field[y, x]
            if y > 0:
                s += field[y - 1, x]
            if y + 1 < h:
                s += field[y + 1, x]
            if x > 0:
                s += field[y, x - 1]
            if x + 1 < w:
                s += field[y, x + 1]
            out[y, x] = s
    return out


def kuramoto_order(phase: np.ndarray) -> float:
    """Parametro d'ordine R ∈ [0,1] — magnetizzazione di fase."""
    c = np.cos(phase).mean()
    s = np.sin(phase).mean()
    return float(np.sqrt(c * c + s * s))


def classify_phase(R: float, dR: float) -> str:
    if R < 0.35:
        return "subcritical"
    if R < 0.62 and abs(dR) < 0.08:
        return "critical"
    return "supercritical"


def turing_step(
    u: np.ndarray,
    v: np.ndarray,
    *,
    feed: float = 0.036,
    kill: float = 0.058,
    du: float = 0.16,
    dv: float = 0.08,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Gray-Scott reazione-diffusione — pattern di Turing come memoria."""
    lu = laplacian(u)
    lv = laplacian(v)
    uvv = u * v * v
    u_new = u + (du * lu - uvv + feed * (1.0 - u)) * dt
    v_new = v + (dv * lv + uvv - (feed + kill) * v) * dt
    return np.clip(u_new, 0, 1), np.clip(v_new, 0, 1)


def inject_turing_from_impulse(
    u: np.ndarray,
    v: np.ndarray,
    impulse: np.ndarray,
    *,
    gain: float = 0.15,
) -> None:
    """Stimolo visivo → attivatore Turing."""
    u[:] = np.clip(u + impulse * gain, 0, 1)
    v[:] = np.clip(v * (1.0 - impulse * gain * 0.3), 0, 1)


def apply_turing_memory(state: np.ndarray, u: np.ndarray, v: np.ndarray, *, gain: float = 0.08) -> None:
    """Pattern Turing → peso gravitazionale (attrattori geometrici)."""
    pattern = np.abs(u - v)
    state[:, :, CH_W] = np.clip(state[:, :, CH_W] + pattern * gain, 0.02, 3.0)
    state[:, :, CH_PH] = (state[:, :, CH_PH] + pattern * np.pi * 0.12) % (2 * np.pi)


def avalanche_size(prev_imp: np.ndarray, cur_imp: np.ndarray, *, thresh: float = 0.06) -> float:
    """SOC: dimensione valanga da perturbazione locale."""
    delta = np.abs(cur_imp - prev_imp)
    active = delta > thresh
    if not active.any():
        return 0.0
    return float(active.sum()) / max(1, active.size)


def soc_tune(coupling: float, branch_ratio: float, *, target: float = 1.0, rate: float = 0.02) -> float:
    """Auto-tuning verso criticità (rapporto rami ≈ 1)."""
    if branch_ratio < 1e-6:
        return coupling
    err = target - branch_ratio
    return float(np.clip(coupling + rate * err, 0.04, 0.28))


def phase_lock_scores(phase: np.ndarray, template_phases: np.ndarray) -> np.ndarray:
    """Biforcazione armonica: lock-in di fase vs template (N template)."""
    h, w = phase.shape
    n_tpl, th, tw = template_phases.shape
    scores = np.zeros(n_tpl, dtype=np.float32)
    R = kuramoto_order(phase)
    y0, x0 = (h - th) // 2, (w - tw) // 2
    patch_ph = phase[y0 : y0 + th, x0 : x0 + tw]
    for i in range(n_tpl):
        tpl = template_phases[i]
        dph = patch_ph - tpl
        lock = float(np.mean(np.cos(dph)))
        scores[i] = lock * R
    return scores


def build_phase_templates(spatial_stack: np.ndarray) -> np.ndarray:
    """Template spaziali → pattern di fase (riconoscimento fisico)."""
    n, h, w = spatial_stack.shape
    out = np.zeros((n, h, w), dtype=np.float32)
    cx, cy = w / 2, h / 2
    for i in range(n):
        for y in range(h):
            for x in range(w):
                dx, dy = x - cx, y - cy
                freq = 0.35 + spatial_stack[i, y, x] * 1.2
                out[i, y, x] = (freq * dx + freq * dy * 0.7) % (2 * math.pi)
    return out


def physics_tick(
    state: np.ndarray,
    prev_imp: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    phys: PhysicsState,
    template_phases: np.ndarray,
    template_names: list[str],
    *,
    conscious_threshold: float = 0.52,
) -> dict[str, object]:
    """Un passo fisica macro — Turing + SOC + transizione + lock-in."""
    impulse = state[:, :, CH_IMP]
    phase = state[:, :, CH_PH]

    inject_turing_from_impulse(u, v, impulse, gain=0.12 * phys.soc_coupling)
    u[:], v[:] = turing_step(u, v, dt=phys.soc_coupling * 8)
    apply_turing_memory(state, u, v, gain=0.06)

    R = kuramoto_order(phase)
    dR = R - (phys.history_order[-1] if phys.history_order else 0.0)
    phys.history_order.append(R)
    if len(phys.history_order) > 64:
        phys.history_order.pop(0)

    av = avalanche_size(prev_imp, impulse)
    prev_av = phys.avalanche_size if phys.avalanche_size > 1e-6 else av
    branch = av / max(1e-6, prev_av) if av > 0 else 1.0
    phys.avalanche_size = av
    phys.branch_ratio = float(np.clip(branch, 0.1, 4.0))
    phys.soc_coupling = soc_tune(phys.soc_coupling, phys.branch_ratio)

    phys.order_parameter = R
    phys.phase_transition = classify_phase(R, dR)
    phys.conscious = R >= conscious_threshold and phys.phase_transition in ("critical", "supercritical")
    phys.turing_energy = float(np.mean(np.abs(u - v)))

    scores = phase_lock_scores(phase, template_phases)
    best_i = int(np.argmax(scores))
    phys.lock_in = float(scores[best_i])
    phys.locked_symbol = template_names[best_i] if phys.lock_in > 0.25 else ""

    recognition: list[tuple[str, float]] = []
    order = np.argsort(scores)[::-1]
    for idx in order[:5]:
        sc = float(scores[idx])
        if sc > 0.18:
            recognition.append((template_names[int(idx)], sc))

    return {
        "order": round(R, 4),
        "conscious": phys.conscious,
        "phase": phys.phase_transition,
        "avalanche": round(av, 4),
        "coupling": round(phys.soc_coupling, 4),
        "lock_in": round(phys.lock_in, 4),
        "symbol": phys.locked_symbol,
        "recognition": recognition,
        "turing_energy": round(phys.turing_energy, 4),
    }
