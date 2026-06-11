"""Seed graph — lampadina, forme, WA resonance, hidden wedding memory."""

from __future__ import annotations

from mind.memory import MemoryGraph
from mind.pattern import PatternEngine
from mind.sensation import SensationRegistry
from mind.types import Circuit, CostLevel, Fragment, HiddenDetail, Outcome, Schema, VisualShape


def build_seed_memory() -> MemoryGraph:
    # --- Lampadina: 30 episodi compressi in un titolo ---
    episodes = []
    for i in range(30):
        fid = f"lamp_ep_{i}"
        hidden = []
        if i == 12:
            # stesso giorno matrimonio fratello — nascosto da cue lampadina
            hidden.append(
                HiddenDetail(
                    key="matrimonio_fratello",
                    retrieve_only_if_cue_contains=["matrimonio", "fratello", "festa"],
                )
            )
        episodes.append(
            Fragment(
                id=fid,
                title=f"cambio lampadina episodio {i}",
                weight=0.7,
                sensation_id="something_broken",
                hooks=["lampadina", "luce", "non si accende"],
                functions=["solve_lamp"],
                hidden_details=hidden,
                outcomes=[Outcome(action="replace_bulb", success=True, cost=CostLevel.LOW)],
            )
        )

    compressed = Fragment(
        id="lamp_compressed",
        title="ho cambiato la lampadina → ha funzionato",
        weight=0.9,
        sensation_id="something_broken",
        hooks=["lampadina", "luce", "non si accende", "spegne", "illuminazione"],
        functions=["solve_lamp"],
        links=[e.id for e in episodes],
        outcomes=[Outcome(action="replace_bulb", success=True, cost=CostLevel.LOW)],
    )

    cable_knowledge = Fragment(
        id="electricity_cable",
        title="la corrente passa nei cavi — potrebbe essere interrotto",
        weight=0.15,
        sensation_id="something_broken",
        hooks=["elettricità", "cavo", "corrente", "lampadina"],
        functions=["solve_lamp"],
    )

    wedding = Fragment(
        id="brother_wedding",
        title="matrimonio di mio fratello",
        weight=0.85,
        sensation_id="family_joy",
        hooks=["matrimonio", "fratello", "festa", "famiglia"],
        functions=["family"],
        links=["lamp_ep_12"],
        hidden_details=[
            HiddenDetail(
                key="quel giorno cambiai lampadina",
                retrieve_only_if_cue_contains=["matrimonio", "fratello"],
            )
        ],
    )

    # --- WA clienti ---
    client_march = Fragment(
        id="wa_client_march",
        title="cliente marzo — perso tempo, chiuso troppo presto",
        weight=0.8,
        sensation_id="social_distrust",
        hooks=["cliente", "marzo", "whatsapp", "diffidente"],
        functions=["wa_quote"],
        outcomes=[Outcome(action="block_client", success=False, cost=CostLevel.LOW)],
    )

    client_march_lesson = Fragment(
        id="wa_lesson_march",
        title="lezione: non chiudere subito, chiedi una cosa in più",
        weight=0.85,
        sensation_id="social_distrust",
        hooks=["cliente", "marzo", "lezione", "diffidente"],
        functions=["wa_resonate", "wa_quote"],
        links=["wa_client_march"],
    )

    graph = MemoryGraph()
    for f in [*episodes, compressed, cable_knowledge, wedding, client_march, client_march_lesson]:
        graph.add(f)
    return graph


def build_seed_circuits() -> SensationRegistry:
    reg = SensationRegistry()
    reg.add(
        Circuit(
            id="something_broken",
            label="qualcosa non funziona",
            fragment_ids=["lamp_compressed", "electricity_cable"],
        )
    )
    reg.add(
        Circuit(
            id="social_distrust",
            label="diffidenza sociale",
            fragment_ids=["wa_client_march", "wa_lesson_march"],
        )
    )
    reg.add(Circuit(id="family_joy", label="gioia familiare", fragment_ids=["brother_wedding"]))
    return reg


def build_seed_patterns() -> PatternEngine:
    engine = PatternEngine()
    engine.add_schema(
        Schema(
            id="shape_inner_circle",
            inner_invariant="cerchio",
            examples=[
                VisualShape("quadrato", "cerchio"),
                VisualShape("triangolo", "cerchio"),
                VisualShape("rettangolo", "cerchio"),
            ],
            confidence=0.95,
        )
    )
    return engine
