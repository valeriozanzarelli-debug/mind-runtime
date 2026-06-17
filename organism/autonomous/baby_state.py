"""Diagnostica e stato — metodi di lettura per il baby agent."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from organism.cognition.consciousness_stream import consciousness_events_from_log
from organism.drives.brain_readout import inject_circadian, read_brain_mood
from organism.autonomous.baby_types import TZ
from organism.brain.impulse_integration import impulse_state_dict


class BabyStateMixin:
    def health(self) -> dict[str, Any]:
        """Controlli vitali — diversità lessico, cervello, ripetizioni."""
        org = self._ensure()
        lex = self.composer.lexicon
        exposures = sorted(
            ((w, lex._exposure.get(w, 0.0)) for w in lex.known_words(40)),
            key=lambda x: -x[1],
        )
        top = exposures[:8]
        dom = top[0][1] if top else 0.0
        avg_top3 = sum(e for _, e in top[:3]) / max(1, len(top[:3]))
        recent = self._recent_spokes[-6:]
        unique_recent = len({s.strip() for s in recent if s})
        metrics = self._probe_metrics()
        outputs = [o for _, o in metrics["probe_outputs"]]
        diversity = metrics["speech_diversity"]
        evolve_hint = self.dna_evolver.evaluate_metrics(metrics)
        return {
            "ok": org.brain.neuron_count > 1000
            and (
                lex.count >= 20
                or bool(org.dna.genome.get("baby", {}).get("genesis"))
            ),
            "neurons": org.brain.neuron_count,
            "synapses": org.brain.synapse_count,
            "synapses_grown": org.brain.synapse_count - self._synapses_at_birth,
            "species": org.dna.genome.get("species", ""),
            "words_known": lex.count,
            "lexicon_top": [{"word": w, "exposure": round(e, 1)} for w, e in top],
            "lexicon_dominance": round(dom, 1),
            "lexicon_avg_top3": round(avg_top3, 1),
            "squash_needed": dom > 100,
            "recent_unique_speeches": unique_recent,
            "probe_outputs": metrics["probe_outputs"],
            "speech_diversity": round(diversity, 2),
            "thought_coherence": round(self._last_thought_coherence, 2),
            "inner_voice_sample": self._last_inner_voice,
            "dna_evolution": self.dna_evolver.stats(),
            "evolution_score": evolve_hint.get("score"),
            "working_memory": self.working_memory.stats(),
            "photo_memory": self.photo_memory.stats(),
            "episodic_memory": self.episodic_memory.stats(),
            "disk_vault": self.disk_vault.stats() if getattr(self, "disk_vault", None) else None,
            "architecture": (
                self.brain_orchestrator.architecture_score()
                if getattr(self, "brain_orchestrator", None)
                else None
            ),
            "growth": self.brain_growth.stats(),
            "faces": self.face_binder.stats(),
        }

    def state_lite(self) -> dict[str, Any]:
        """Stato leggero per boot UI — niente liste pesanti."""
        org = self.org
        brain_mood: dict[str, Any] = {}
        if org and self.speech:
            em = self.speech.readout()
            brain_mood = read_brain_mood(
                org.brain,
                synapses_at_birth=self._synapses_at_birth,
                motor_pressure=em.motor_pressure,
                inhibition=em.inhibition,
                wants_voice=em.will_speak,
                arousal=inject_circadian(org.brain),
            ).to_dict()
        return {
            "born": self._born,
            "dormant": self._dormant,
            "resumed_from_disk": self._resume_meta is not None,
            "brain": brain_mood,
            "synapses_at_birth": self._synapses_at_birth,
            "syllables_known": self.speech.phonemes.count if self.speech else 0,
            "words_known": self.composer.lexicon.count,
            "dialogue_pairs": [],
            "dialogue_count": len(self.dialogue.all_pairs()),
            "visual_binder": {"object_names": self.visual_binder.to_dict().get("object_names", {})},
            "last_moment": self._last_moment.to_dict() if self._last_moment else None,
            "consciousness_stream": self.consciousness_recent(32),
            "dialogue": self.dialogue_recent(40),
            "neurons": org.brain.neuron_count if org else 0,
            "impulse": impulse_state_dict(getattr(self, "impulse", None)),
            "architecture": (
                self.brain_orchestrator.capacity()
                if getattr(self, "brain_orchestrator", None)
                else None
            ),
        }

    def state(self) -> dict[str, Any]:
        org = self.org
        brain_mood: dict[str, Any] = {}
        if org and self.speech:
            em = self.speech.readout()
            brain_mood = read_brain_mood(
                org.brain,
                synapses_at_birth=self._synapses_at_birth,
                motor_pressure=em.motor_pressure,
                inhibition=em.inhibition,
                wants_voice=em.will_speak,
                arousal=inject_circadian(org.brain),
            ).to_dict()
        return {
            "born": self._born,
            "resumed_from_disk": self._resume_meta is not None,
            "state_file": str(self.store.path),
            "stats": org.stats if org else {},
            "brain": brain_mood,
            "synapses_at_birth": self._synapses_at_birth,
            "curiosity": self.curiosity.state.to_dict(),
            "learned_phrases": self.teacher.all_learned(),
            "dialogue_pairs": self.dialogue.all_pairs(),
            "code_tokens": self.code.tokens.count,
            "words_known": self.composer.lexicon.count,
            "syllables_known": self.speech.phonemes.count if self.speech else 0,
            "last_moment": self._last_moment.to_dict() if self._last_moment else None,
            "researched": self._researched,
            "locale": org.dna.genome.get("baby", {}).get("locale", "it-IT") if org else "it-IT",
            "schedule_hour": datetime.now(TZ).hour,
            "speech_loop": self.speech_loop.stats(),
            "consciousness": self.workspace.stats(),
            "affect": self.affect.stats(),
            "amygdala": self.amygdala.stats(),
            "presence": self.presence.to_dict(),
            "corrections": self.corrections.stats(),
            "self_model": self.self_model.stats(),
            "dream": self.dream_engine.stats(),
            "waves": self.waves.stats(),
            "tasks": self.tasks.to_dict(),
            "growth": self.brain_growth.stats(),
            "consciousness_stream": self.consciousness_recent(32),
            "neurons": org.brain.neuron_count if org else 0,
            "last_vision_hash": self._last_vision_hash,
            "visual_features": self._last_visual_features,
            "face_binder": self.face_binder.stats(),
            "last_face": self._face_context(),
            "photo_memory": self.photo_memory.stats(),
            "episodic_memory": self.episodic_memory.stats(),
            "dna_evolution": self.dna_evolver.stats(),
            "inner_voice": self._last_inner_voice,
            "thought_coherence": round(self._last_thought_coherence, 2),
        }

    def debug_thoughts(self, n: int = 20) -> list[str]:
        return self.journal.stream_text(n)

    def consciousness_recent(self, n: int = 48) -> list[str]:
        """Flusso di coscienza — pensieri anche quando non parla."""
        return list(self._consciousness_log[-n:])

    def consciousness_events(self, *, since_seq: int = 0, limit: int = 48) -> dict[str, Any]:
        return consciousness_events_from_log(
            self._consciousness_log,
            since_seq=since_seq,
            limit=limit,
            include_noise=False,
        )

    def dialogue_recent(self, n: int = 60) -> list[dict[str, Any]]:
        return list(self._dialogue_log[-n:])

    def chat_text(self, text: str) -> dict[str, Any]:
        """Dialogo scritto — tu scrivi, lui risponde."""
        phrase = text.strip()
        if not phrase:
            return {"ok": False, "reason": "empty"}
        result = self.hear_spoken(phrase, source="caregiver")
        moment = result.get("moment") or {}
        reply = ""
        if isinstance(moment, dict):
            reply = str(moment.get("spoke") or "")
        if not reply and self._last_moment:
            reply = self._last_moment.spoke or ""
        return {
            "ok": True,
            "heard": phrase,
            "reply": reply,
            "moment": moment,
            "dialogue": self.dialogue_recent(60),
            "consciousness": self.consciousness_events(limit=24),
        }
