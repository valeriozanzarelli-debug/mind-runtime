"""Consolidamento nel sonno — episodi → semantica, dialoghi → causale."""

from __future__ import annotations

from typing import Any


def consolidate_on_sleep(
    agent: Any,
    *,
    min_spoke_len: int = 24,
) -> dict[str, Any]:
    """Ciclo REM digitale: memoria episodica alimenta schemi e regole."""
    report: dict[str, Any] = {
        "episodes_to_long": 0,
        "causal_links_added": 0,
        "semantic_words_reinforced": 0,
        "superego_norms": 0,
    }

    report["episodes_to_long"] = agent.episodic_memory.consolidate_short_to_long(
        min_spoke_len=min_spoke_len
    )

    psyche = getattr(agent, "psyche", None)
    if psyche:
        psyche.ensure_seeded()
        before = len(psyche.causal._links)
        for pair in agent.dialogue.all_pairs():
            when = str(pair.get("when", ""))
            say = str(pair.get("say", ""))
            if not when or not say:
                continue
            wl = when.lower()
            if any(c in wl for c in ("perché", "perche", "se ", "cosa succede", "cosa accade")):
                psyche.causal.teach(when, say)
        report["causal_links_added"] = max(0, len(psyche.causal._links) - before)

    semantic = getattr(agent, "semantic", None)
    if semantic:
        reinforced = 0
        for ep in agent.episodic_memory.recall_context(limit=12):
            spoke = str(ep.get("spoke", ""))
            if len(spoke) < min_spoke_len:
                continue
            for word in semantic.grounded_words():
                if word in spoke.lower() and semantic.definition(word):
                    reinforced += 1
        report["semantic_words_reinforced"] = reinforced

    superego = getattr(agent, "superego", None)
    corrections = getattr(agent, "corrections", None)
    if superego and corrections:
        for ev in corrections.to_dict().get("events", [])[-10:]:
            wrong = str(ev.get("wrong", ""))
            right = str(ev.get("right", ""))
            if wrong and right:
                superego.internalize(wrong, right)
                report["superego_norms"] += 1

    schema = agent.working_memory.active_schema()
    if schema:
        thread = agent.episodic_memory.recall_story_thread(schema, limit=3)
        for line in thread:
            agent.working_memory.push_thread(line)

    return report
