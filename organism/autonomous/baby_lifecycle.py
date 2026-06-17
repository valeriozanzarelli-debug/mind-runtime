"""Ciclo di vita del baby agent — nascita, persistenza, sonno."""
from __future__ import annotations

import time
from typing import Any

from organism.runtime import OrganismRuntime
from organism.cognition.self_learn import SelfLearner
from organism.cognition.working_memory import WorkingMemory
from organism.cognition.photographic_memory import PhotographicMemory
from organism.cognition.episodic_memory import EpisodicMemory
from organism.cognition.linguistic_narrator import LinguisticNarrator
from organism.motor.compose_speech import SpeechComposer
from organism.motor.emergent_speech import EmergentSpeechMotor
from organism.motor.motion import MotionModule
from organism.cognition.visual_imagination import VisualImagination
from organism.cognition.goal_stack import GoalStack
from organism.cognition.hypothesis_reasoning import HypothesisEngine
from organism.autonomous.thought_generator import ThoughtGenerator
from organism.sensory.vision_sense import VisionSense
from organism.cognition.face_bind import FaceEmotionBinder
from organism.cognition.visual_bind import VisualBinder
from organism.teaching.correction import CorrectionLearner
from organism.teaching.repetition import RepetitionTeacher
from organism.teaching.dialogue import DialogueTeacher
from organism.dna.evolution import DNAEvolver


class BabyLifecycleMixin:
    def birth(self) -> dict[str, Any]:
        if self._born and self.org is not None and self.speech is not None:
            return self._resume_response()
        if not self._born and self.store.exists():
            loaded = self.store.load(self)
            if loaded.get("loaded") and self.org is not None and self.speech is not None:
                self._born = True
                return self._resume_response()
        self.org = OrganismRuntime.baby(seed=self.seed)
        self.speech = EmergentSpeechMotor(self.org.brain, seed=self.seed)
        self.composer.bind(self.org.brain, self.speech)
        self.motion = MotionModule(self.org.brain)
        self._apply_cognition_config()
        layer2 = self._bootstrap_layer2()
        self._synapses_at_birth = self.org.brain.synapse_count
        self._born = True
        self._last_sense_t = time.time()
        self._start_thought_loop()
        snap = {
            "neurons": self.org.brain.neuron_count,
            "synapses": self.org.brain.synapse_count,
            "locale": self.org.dna.genome.get("baby", {}).get("locale", "it-IT"),
            "layer2": layer2,
        }
        self.journal.log_birth(snap)
        self._persist()
        return {"born": True, "resumed": False, **snap, "stats": self.org.stats}

    def rebirth(self, *, keep_dialogue: bool = True) -> dict[str, Any]:
        """Nuovo cervello (es. scala giga) — opzionalmente conserva dialoghi e parole."""
        saved = {}
        if keep_dialogue and self._born:
            saved = {
                "dialogue": self.dialogue.to_dict(),
                "neural_lexicon": self.composer.lexicon.to_dict(),
                "phonemes": self.speech.phonemes.to_dict() if self.speech else {},
                "visual_binder": self.visual_binder.to_dict(),
                "face_binder": self.face_binder.to_dict(),
                "teacher": self.teacher.to_dict(),
                "self_learner": self.self_learner.to_dict(),
                "corrections": self.corrections.to_dict(),
                "working_memory": self.working_memory.to_dict(),
                "photo_memory": self.photo_memory.to_dict(),
                "episodic_memory": self.episodic_memory.to_dict(),
                "narrator": self.narrator.to_dict(),
                "goal_stack": self.goal_stack.to_dict(),
                "hypothesis_engine": self.hypothesis_engine.to_dict(),
            }
        if self.store.exists():
            self.store.path.unlink()
        self.thought_generator.stop()
        self._born = False
        self.org = None
        self.speech = None
        self._resume_meta = None
        self._synapses_at_birth = 0
        r = self.birth()
        if saved:
            self.dialogue.load_dict(saved.get("dialogue", {}))
            self.composer.lexicon.load_dict(
                saved.get("neural_lexicon") or saved.get("word_weights", {})
            )
            self.composer.lexicon.squash_overexposed()
            if self.speech and saved.get("phonemes"):
                self.speech.phonemes.load_dict(saved["phonemes"])
            if saved.get("visual_binder"):
                self.visual_binder.load_dict(saved["visual_binder"])
            if saved.get("face_binder"):
                self.face_binder.load_dict(saved["face_binder"])
            if saved.get("teacher"):
                self.teacher.load_dict(saved["teacher"])
            if saved.get("self_learner"):
                self.self_learner.load_dict(saved["self_learner"])
            if saved.get("corrections"):
                self.corrections.load_dict(saved["corrections"])
            if saved.get("working_memory"):
                self.working_memory.load_dict(saved["working_memory"])
            if saved.get("photo_memory"):
                self.photo_memory.load_dict(saved["photo_memory"])
            if saved.get("episodic_memory"):
                self.episodic_memory.load_dict(saved["episodic_memory"])
            if saved.get("narrator"):
                self.narrator.load_dict(saved["narrator"])
            if saved.get("goal_stack"):
                self.goal_stack.load_dict(saved["goal_stack"])
            if saved.get("hypothesis_engine"):
                self.hypothesis_engine.load_dict(saved["hypothesis_engine"])
            self._persist()
        r["rebirth"] = True
        r["kept_dialogue"] = keep_dialogue
        return r

    def _reset_learned_state(self) -> None:
        """Azzera tutta la conoscenza — mantiene solo capacità strutturali."""
        from organism.cognition.face_bind import FaceEmotionBinder
        from organism.cognition.visual_bind import VisualBinder
        from organism.motor.compose_speech import SpeechComposer
        from organism.motor.emergent_speech import EmergentSpeechMotor
        from organism.teaching.correction import CorrectionLearner
        from organism.teaching.repetition import RepetitionTeacher
        from organism.teaching.dialogue import DialogueTeacher

        self.teacher = RepetitionTeacher(consolidate_at=3)
        self.dialogue = DialogueTeacher(consolidate_at=3)
        self.visual_binder = VisualBinder()
        self.face_binder = FaceEmotionBinder()
        self.corrections = CorrectionLearner()
        self.self_learner = SelfLearner()
        self.working_memory = WorkingMemory(capacity=7)
        self.photo_memory = PhotographicMemory(capacity=2000)
        self.episodic_memory = EpisodicMemory(short_slots=32, long_capacity=1200)
        self.narrator = LinguisticNarrator(seed=self.seed)
        self.composer = SpeechComposer(seed=self.seed, narrator=self.narrator)
        self.visual_imagination = VisualImagination(self.photo_memory)
        self.goal_stack = GoalStack()
        self.hypothesis_engine = HypothesisEngine()
        self.thought_generator = ThoughtGenerator(seed=self.seed)
        self.vision_sense = VisionSense()
        self._consciousness_log = []
        self._dialogue_log = []
        self._consciousness_seq = 0
        self._recent_spokes = []
        self._last_baby_spoke = ""
        self._researched = False
        if self.org and self.speech:
            self.composer.bind(self.org.brain, self.speech)

    def birth_genesis(self, *, seed: int | None = None) -> dict[str, Any]:
        """Nascita genesis — sa parlare e pensare, zero informazioni sul mondo."""
        from organism.teaching.bootstrap import wire_genesis_capacity

        if seed is not None:
            self.seed = seed
        if self.store.exists():
            self.store.path.unlink()
        self._born = False
        self.org = None
        self.speech = None
        self._resume_meta = None
        self._synapses_at_birth = 0
        self._reset_learned_state()

        self.org = OrganismRuntime.genesis(seed=self.seed)
        self.speech = EmergentSpeechMotor(self.org.brain, seed=self.seed)
        self.composer.bind(self.org.brain, self.speech)
        boot = wire_genesis_capacity(self.org.brain, self.speech)
        self._apply_cognition_config()
        layer2 = self._bootstrap_layer2()
        self._synapses_at_birth = self.org.brain.synapse_count
        self._born = True
        self._last_sense_t = time.time()
        self._start_thought_loop()
        snap = {
            "neurons": self.org.brain.neuron_count,
            "synapses": self.org.brain.synapse_count,
            "locale": self.org.dna.genome.get("baby", {}).get("locale", "it-IT"),
            "genesis": True,
            "bootstrap": boot,
            "layer2": layer2,
            "words_known": self.composer.lexicon.count,
            "dialogue_pairs": len(self.dialogue.all_pairs()),
        }
        self.journal.log_birth(snap)
        self._persist()
        return {"born": True, "genesis": True, "resumed": False, **snap, "stats": self.org.stats}

    def evolve_dna(self, *, force: bool = False) -> dict[str, Any]:
        """Aggiorna DNA da metriche — separato dalla memoria appresa."""
        metrics = self._probe_metrics()
        self.dna_evolver = DNAEvolver.load()
        result = self.dna_evolver.maybe_evolve(metrics, force=force)
        return {**result, "dna": self.dna_evolver.stats()}

    def export_genesis_dna(self) -> dict[str, Any]:
        """Pacchetto DNA installabile altrove — senza informazioni apprese."""
        import yaml
        from organism.dna.evolution import evolved_genesis_path

        evolver = DNAEvolver.load()
        clean = evolver.export_clean()
        path = evolved_genesis_path()
        return {
            "ok": True,
            "path": str(path),
            "generation": evolver.generation,
            "species": clean.get("species"),
            "version": clean.get("genome_version"),
            "yaml": yaml.dump(clean, allow_unicode=True, default_flow_style=False, sort_keys=False),
            "words_in_memory": self.composer.lexicon.count,
            "note": "Solo DNA strutturale — nessun lessico o dialogo incluso",
        }

    def _probe_metrics(self) -> dict[str, Any]:
        probes = ["ciao", "cosa pensi", "chi sei", "cosa vedi", "raccontami qualcosa"]
        outputs: list[tuple[str, str]] = []
        for q in probes:
            m = self.sense(text=q).get("moment") or {}
            outputs.append((q, str(m.get("spoke", ""))))
        unique_out = len({o.lower().strip() for _, o in outputs if o})
        return {
            "probe_outputs": outputs,
            "speech_diversity": unique_out / max(1, len(probes)),
            "thought_coherence": self._last_thought_coherence,
        }

    def _resume_response(self) -> dict[str, Any]:
        org = self._ensure()
        return {
            "born": True,
            "resumed": True,
            "neurons": org.brain.neuron_count,
            "synapses": org.brain.synapse_count,
            "synapses_grown": org.brain.synapse_count - self._synapses_at_birth,
            "syllables_known": self.speech.phonemes.count if self.speech else 0,
            "learned_phrases": len(self.teacher.all_learned()),
            "stats": org.stats,
        }

    def _persist(self) -> None:
        if self._born:
            self.store.save(self)

    def _maybe_persist(self) -> None:
        self._sense_count += 1
        if self._sense_count % 5 == 0:
            self._persist()

    def _ensure(self) -> OrganismRuntime:
        if not self._born or self.org is None:
            self.birth()
        assert self.org is not None and self.speech is not None
        return self.org

    def set_training_plasticity(self, *, rate: float | None = None, decay: float | None = None) -> dict[str, Any]:
        """Alza/abbassa Hebbian durante training intensivo."""
        org = self._ensure()
        pl = org.brain.plasticity
        if pl is None:
            return {"ok": False}
        before = dict(pl.hebbian)
        if rate is not None:
            pl.hebbian["rate"] = float(rate)
        if decay is not None:
            pl.hebbian["decay"] = float(decay)
        return {"ok": True, "before": before, "after": dict(pl.hebbian)}

    def sleep_cycle(self) -> dict[str, Any]:
        """Sonno — pruning sinapsi deboli + consolidamento."""
        org = self._ensure()
        result = org.sleep()
        consolidated = self.episodic_memory.consolidate_short_to_long()
        self.working_memory.clear()
        self._persist()
        return {**result, "episodes_consolidated": consolidated}

    def enter_dormant(self) -> dict[str, Any]:
        """Riposo — niente crescita né speech spontaneo."""
        self._dormant = True
        self.thought_generator.stop()
        self._persist()
        return {"dormant": True}

    def wake(self) -> dict[str, Any]:
        """Esce dal riposo — riprende il loop di pensiero interno."""
        self._dormant = False
        if self.org:
            self._start_thought_loop()
        self._persist()
        return {"dormant": False, "awake": True}

    def stabilize(self, *, aggressive: bool = True) -> dict[str, Any]:
        """Sonno profondo — pota sinapsi, consolida training, blocca output casuale."""
        org = self._ensure()
        self._dormant = True
        self.thought_generator.stop()
        cfg = org.dna.pruning_config()
        threshold = float(cfg.get("weak_synapse_threshold", 0.05))
        keep = float(cfg.get("keep_percentage", 0.85))
        if aggressive:
            threshold = max(threshold, 0.07)
            keep = min(keep, 0.82)
        removed = org.brain.prune_weak_synapses(threshold=threshold, keep_percentage=keep)
        consolidated = org.brain.consolidate_memory()
        episodes = self.episodic_memory.consolidate_short_to_long()
        self.working_memory.clear()
        self._append_consciousness(["sonno · consolidamento sinapsi e memoria"])
        self._persist()
        return {
            "dormant": True,
            "stabilized": True,
            "pruned_synapses": removed,
            "synapses_after": org.brain.synapse_count,
            "synapses_grown": org.brain.synapse_count - self._synapses_at_birth,
            "episodes_consolidated": episodes,
            **consolidated,
        }
