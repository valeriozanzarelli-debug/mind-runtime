"""Experience-weighted policy — same sensation, different execution."""

from __future__ import annotations

from mind.types import Action, CostLevel, Cue, CueKind, Fragment


# Candidate actions per function with base utility from experience weight
_ACTION_CATALOG: dict[str, list[dict]] = {
    "solve_lamp": [
        {"id": "replace_bulb", "label": "Cambia lampadina", "base": 0.9, "cost": CostLevel.LOW},
        {"id": "test_bulb", "label": "Testa lampadina prima", "base": 0.5, "cost": CostLevel.LOW},
        {"id": "check_cable", "label": "Controlla cavo", "base": 0.15, "cost": CostLevel.MEDIUM},
    ],
    "wa_quote": [
        {"id": "ask_placement", "label": "Chiedi placement", "base": 0.85, "cost": CostLevel.LOW},
        {"id": "send_quote", "label": "Invia preventivo", "base": 0.4, "cost": CostLevel.LOW},
        {"id": "block_client", "label": "Ignora cliente", "base": 0.1, "cost": CostLevel.LOW},
    ],
    "wa_resonate": [
        {"id": "ask_one_more", "label": "Fai una domanda in più (non chiudere)", "base": 0.75, "cost": CostLevel.LOW},
        {"id": "block_client", "label": "Ignora cliente", "base": 0.3, "cost": CostLevel.LOW},
    ],
}


class ExperiencePolicy:
    def decide(
        self,
        cue: Cue,
        fragments: list[Fragment],
        *,
        sensation_ids: list[str],
        cost_override: CostLevel | None = None,
        resonate_mode: bool = False,
    ) -> Action | None:
        if not fragments:
            return None

        functions: list[str] = []
        for f in fragments:
            functions.extend(f.functions)
        if not functions:
            return None

        fn = functions[0]
        if resonate_mode and "wa_quote" in functions:
            fn = "wa_resonate"

        candidates = _ACTION_CATALOG.get(fn, [])
        if not candidates:
            return None

        # aggregate weight from fragments for this function
        relevant = [f for f in fragments if fn in f.functions or f.functions]
        exp_weight = max((f.weight for f in relevant), default=0.5)

        best: Action | None = None
        best_u = -1.0

        for c in candidates:
            cost = cost_override or c["cost"]
            u = self._utility(c["base"], exp_weight, cost, cue, c["id"])
            if resonate_mode and c["id"] == "block_client":
                # experience: same sensation but don't repeat mistake
                u *= 0.35
            if resonate_mode and c["id"] == "ask_one_more":
                u *= 1.25

            if u > best_u:
                best_u = u
                best = Action(
                    id=c["id"],
                    label=c["label"],
                    cost=cost,
                    utility=u,
                    reason=self._reason(fn, c["id"], exp_weight, resonate_mode),
                )
        return best

    def _utility(
        self, base: float, exp_weight: float, cost: CostLevel, cue: Cue, action_id: str
    ) -> float:
        cost_penalty = {CostLevel.LOW: 0.0, CostLevel.MEDIUM: 0.12, CostLevel.HIGH: 0.35}[cost]
        u = base * (0.4 + 0.6 * exp_weight) - cost_penalty
        v = cue.value.lower()
        is_lamp = any(k in v for k in ("lampadina", "luce", "illuminazione"))
        # costo alto → prova prima di sostituire (esperienza modula escuzione)
        if cost == CostLevel.HIGH and is_lamp:
            if action_id == "test_bulb":
                u += 0.45
            elif action_id == "replace_bulb":
                u -= 0.4
        return u

    def _reason(self, fn: str, action_id: str, weight: float, resonate: bool) -> str:
        parts = [f"exp_weight={weight:.2f}"]
        if resonate:
            parts.append("resonance:same_circuit≠same_action")
        parts.append(f"function={fn}")
        parts.append(f"action={action_id}")
        return " | ".join(parts)


def is_human_interface(cue: Cue) -> bool:
    if cue.kind == CueKind.AUDIO:
        return True
    if cue.meta.get("human"):
        return True
    v = cue.value.lower()
    human_markers = (
        "cliente",
        "whatsapp",
        "wa",
        "messaggio",
        "preventivo",
        "ciao",
        "fratello",
        "matrimonio",
    )
    return any(m in v for m in human_markers)
