"""Baby agent — voce e crescita emergono dal grafo neurale."""

from __future__ import annotations

import os
import time
from typing import Any

from organism.autonomous.baby_store import BabyStore, baby_state_path
from organism.autonomous.baby_types import BabyMoment, TZ, _COLOR_WORDS, _tokens_from_heard
from organism.autonomous.baby_lifecycle import BabyLifecycleMixin
from organism.autonomous.baby_vision import BabyVisionMixin
from organism.autonomous.baby_hearing import BabyHearingMixin
from organism.autonomous.baby_teaching import BabyTeachingMixin
from organism.autonomous.baby_state import BabyStateMixin
from organism.brain.growth import (
    active_ids,
    wire_coactive,
    wire_cross_modal,
    wire_self_recurrence,
    wire_teach_burst,
)
from organism.brain.oscillation import inject_wave
from organism.cognition.dream import DreamEngine
from organism.cognition.pathway import (
    prime_curiosity_pathway,
    prime_dialogue_pathway,
)
from organism.cognition.visual_pathway import prime_visual_percept
from organism.cognition.self_model import SelfModel
from organism.cognition.consciousness_stream import (
    build_consciousness_stream,
    is_question,
)
from organism.cognition.theme_coherence import coherent_themes
from organism.cognition.self_learn import SelfLearner
from organism.cognition.growth import BrainGrowth
from organism.cognition.tasks import TaskRunner
from organism.cognition.face_bind import FaceEmotionBinder
from organism.cognition.visual_bind import VisualBinder
from organism.cognition.waves import BrainWaveCycle
from organism.cognition.workspace import GlobalWorkspace
from organism.sensory.reading import ReadingChannel
from organism.drives.brain_readout import BrainMood, inject_circadian, read_brain_mood
from organism.drives.presence import HumanPresence
from organism.drives.curiosity import (
    CuriosityDrive,
    stimulus_key_from_sensory,
    stimulus_key_visual_context,
)
from organism.cognition.affect import AffectiveEngine
from organism.cognition.amygdala import AmygdalaEngine
from organism.cognition.basal_ganglia import BasalGanglia
from organism.cognition.working_memory import WorkingMemory
from organism.cognition.photographic_memory import PhotographicMemory
from organism.cognition.episodic_memory import EpisodicMemory
from organism.cognition.linguistic_narrator import LinguisticNarrator
from organism.cognition.visual_imagination import VisualImagination
from organism.cognition.goal_stack import GoalStack
from organism.cognition.hypothesis_reasoning import HypothesisEngine
from organism.cognition.comprehension import PsycheEngine, ComprehensionFrame
from organism.cognition.superego import SuperegoEngine
from organism.cognition.neurochemistry import NeurochemistryEngine
from organism.cognition.endocrine import EndocrineSystem
from organism.cognition.interoception import InteroceptionEngine
from organism.cognition.spatial_body import VirtualBodySchema
from organism.brain.quantum_microtubules import QuantumMicrotubuleLayer
from organism.teaching.semantic_knowledge import SemanticKnowledge
from organism.motor.motion import MotionModule, MotionResult
from organism.cognition.disk_vault import DiskMemoryVault
from organism.distributed.brain_orchestrator import BrainOrchestrator
from organism.autonomous.thought_generator import ThoughtGenerator
from datetime import datetime
from organism.sensory.vision_sense import VisionSense
from organism.cognition.human_thought import (
    compose_inner_voice,
    enrich_consciousness_lines,
    thought_coherence,
)
from organism.dna.evolution import DNAEvolver
from organism.brain.impulse_integration import (
    create_impulse_scaffold,
    impulse_consciousness_lines,
    impulse_state_dict,
    merge_workspace,
)
from organism.cognition.motor_plan import MotorPlan
from organism.cognition.speech_loop import SpeechSensorimotorLoop
from organism.cognition.thought import ThoughtEngine
from organism.sensory.social_tone import SocialTone, analyze_social_tone
from organism.teaching.correction import CorrectionLearner
from organism.motor.code_emergent import EmergentCodeMotor
from organism.motor.compose_speech import SpeechComposer
from organism.motor.emergent_speech import EmergentSpeechMotor
from organism.nursery.journal import ThoughtJournal
from organism.runtime import OrganismRuntime
from organism.sensory.web_media import (
    audio_hash,
    decode_audio_b64,
    vision_hash,
)
from organism.teaching.dialogue import DialogueTeacher
from organism.teaching.repetition import RepetitionTeacher


class BabyAgent(BabyLifecycleMixin, BabyVisionMixin, BabyHearingMixin, BabyTeachingMixin, BabyStateMixin):
    def __init__(self, seed: int = 42, *, store_path: str | None = None) -> None:
        self.seed = seed
        self.store = BabyStore(store_path or baby_state_path())
        self.org: OrganismRuntime | None = None
        self.curiosity = CuriosityDrive()
        self.teacher = RepetitionTeacher(consolidate_at=3)
        self.dialogue = DialogueTeacher(consolidate_at=3)
        self.journal = ThoughtJournal()
        self.speech: EmergentSpeechMotor | None = None
        self.code = EmergentCodeMotor()
        self.thought_engine = ThoughtEngine()
        self.narrator = LinguisticNarrator(seed=seed)
        self.composer = SpeechComposer(seed=seed, narrator=self.narrator)
        self.speech_loop = SpeechSensorimotorLoop()
        self.workspace = GlobalWorkspace()
        self.affect = AffectiveEngine()
        self.amygdala = AmygdalaEngine()
        self.presence = HumanPresence()
        self.corrections = CorrectionLearner()
        self.self_model = SelfModel()
        self.dream_engine = DreamEngine(seed=seed)
        self.waves = BrainWaveCycle()
        self.visual_binder = VisualBinder()
        self.face_binder = FaceEmotionBinder()
        self.self_learner = SelfLearner()
        self.tasks = TaskRunner()
        self.brain_growth = BrainGrowth()
        self.working_memory = WorkingMemory(capacity=24)
        self.photo_memory = PhotographicMemory(capacity=2000)
        self.episodic_memory = EpisodicMemory(short_slots=32, long_capacity=1200)
        self.visual_imagination = VisualImagination(self.photo_memory)
        self.goal_stack = GoalStack()
        self.hypothesis_engine = HypothesisEngine()
        self.psyche = PsycheEngine()
        self.superego = SuperegoEngine()
        self.semantic = SemanticKnowledge()
        self._semantic_seeded = False
        self.neurochemistry = NeurochemistryEngine()
        self.endocrine = EndocrineSystem()
        self.interoception = InteroceptionEngine()
        self.body_schema = VirtualBodySchema()
        self.quantum_layer = QuantumMicrotubuleLayer()
        self.motion: MotionModule | None = None
        self._last_motion: MotionResult | None = None
        self._last_comp_frame: ComprehensionFrame | None = None
        self.thought_generator = ThoughtGenerator(seed=seed)
        self.vision_sense = VisionSense()
        self.dna_evolver = DNAEvolver.load()
        self.impulse = create_impulse_scaffold()
        self.disk_vault: DiskMemoryVault | None = None
        if os.environ.get("ORGANISM_DISK_VAULT", "1") != "0":
            self.disk_vault = DiskMemoryVault()
        self.brain_orchestrator: BrainOrchestrator | None = None
        self._last_inner_voice = ""
        self._last_thought_coherence = 0.0
        self.basal_ganglia = BasalGanglia()
        self._reading: ReadingChannel | None = None
        self._last_baby_spoke = ""
        self._last_dream_content = ""
        self._born = False
        self._researched = False
        self._synapses_at_birth = 0
        self._last_moment: BabyMoment | None = None
        self._last_sense_t = time.time()
        self._last_vision_hash = ""
        self._last_glance_sig = ""
        self._last_glance_t = 0.0
        self._last_audio_hash = ""
        self._sense_count = 0
        self._last_visual_features: dict[str, Any] = {}
        self._vision_fresh = False
        self._context_words: list[str] = []
        self._last_presence: dict[str, Any] = {}
        self._last_tone: dict[str, Any] = {}
        self._consciousness_log: list[str] = []
        self._dialogue_log: list[dict[str, Any]] = []
        self._consciousness_seq = 0
        self._last_ws: dict[str, Any] = {}
        self._recent_spokes: list[str] = []
        self._last_spoke_wall_t: float = 0.0
        self._dormant: bool = False
        loaded = self.store.load(self)
        self._resume_meta = loaded if loaded.get("loaded") else None
        if loaded.get("loaded") and self.org:
            grown = self.org.brain.synapse_count - self._synapses_at_birth
            if grown > 1500:
                self._dormant = True
                self.thought_generator.stop()

    def _dialogue_for_psyche(self, heard: str) -> tuple[str | None, str | None, bool]:
        say, kind = self.dialogue.respond(heard)
        verbatim = bool(say and kind in ("speech", "code"))
        return say, kind, verbatim

    def _ensure_semantic(self) -> None:
        if self._semantic_seeded:
            return
        from organism.teaching.story_curriculum import pinocchio_semantic_lessons

        lessons = pinocchio_semantic_lessons()
        for word, definition, related in lessons.get("words", []):
            self.semantic.teach_word(word, definition, related=related, story_id="pinocchio")
        for order, summary, entities, hooks in lessons.get("beats", []):
            self.semantic.teach_beat("pinocchio", order, summary, entities=entities, hooks=hooks)
        for pair in self.dialogue.all_pairs():
            when = str(pair.get("when", ""))
            say = str(pair.get("say", ""))
            if when and say and len(say.split()) <= 6:
                for w in _tokens_from_heard(when):
                    if len(w) > 3 and not self.semantic.is_grounded(w):
                        self.semantic.teach_word(w, say[:80], related=[when], story_id="dialogue")
        self._semantic_seeded = True

    def _tick_subcortex(
        self,
        org: OrganismRuntime,
        *,
        idle_s: float = 0.0,
        social_warm: bool = False,
        learned: bool = False,
        wave_phase: str = "think",
    ) -> None:
        """Ghiandole, neurotrasmettitori, interocezione — ogni ciclo cognitivo."""
        aff = self.affect.state
        stress = self.endocrine.stress_from_affect(
            fear=aff.fear, anger=aff.anger, shame=aff.shame
        )
        hour = datetime.now(TZ).hour
        self.neurochemistry.tick(
            joy=aff.joy,
            fear=aff.fear,
            anger=aff.anger,
            trust=aff.trust,
            curiosity=aff.curiosity,
            stress=stress,
            social_warm=social_warm,
            idle_s=idle_s,
            learned=learned,
        )
        self.endocrine.tick(
            hour=hour,
            wave_phase=wave_phase,
            stress=stress,
            social_bond=aff.trust,
            fatigue=self.interoception.state.fatigue,
        )
        j, f, sa, an, tr, cu = self.neurochemistry.modulate_affect_dims(
            joy=aff.joy,
            fear=aff.fear,
            sadness=aff.sadness,
            anger=aff.anger,
            trust=aff.trust,
            curiosity=aff.curiosity,
        )
        aff.joy, aff.fear, aff.sadness = j, f, sa
        aff.anger, aff.trust, aff.curiosity = an, tr, cu
        self.affect._set_dominant()
        self.neurochemistry.inject_into_brain(org.brain)
        self.interoception.update(
            self.neurochemistry.state,
            self.endocrine.hormones,
            joy=aff.joy,
            fear=aff.fear,
            shame=aff.shame,
            idle_s=idle_s,
        )

    def _update_body_schema(self) -> None:
        features = self._last_visual_features or {}
        flow = float(features.get("motion", 0.0))
        bearing = self.body_schema.salient_bearing_from_features(features)
        gesture_hint = self.body_schema.gesture_from_state(
            joy=self.affect.state.joy,
            fear=self.affect.state.fear,
        )
        self.body_schema.tick(
            optical_flow=flow,
            flow_direction=0.0,
            salient_bearing=bearing,
            curiosity=self.affect.state.curiosity,
            fear=self.affect.state.fear,
            motion_gesture=gesture_hint,
        )

    def _apply_superego(self, spoke: str, heard: str, frame: ComprehensionFrame | None) -> str:
        if not spoke.strip():
            return spoke
        self._ensure_semantic()
        verdict = self.superego.review(
            spoke,
            heard=heard,
            frame=frame,
            semantic=self.semantic,
            articulable=lambda w: self.composer.lexicon.is_articulable(w, min_exposure=0.25),
        )
        if verdict.action == "allow":
            return spoke
        text = verdict.text or spoke
        if (
            frame
            and frame.intent == "social"
            and frame.inhibit_lexicon_dump
            and text.strip().lower().rstrip(".") == (frame.taught_say or "").strip().lower().rstrip(".")
        ):
            text = "ciao, ti ascolto. dimmi pure."
        if verdict.action == "substitute" and text:
            return text
        if verdict.action == "block":
            return text or ""
        return spoke

    def _apply_cognition_config(self) -> None:
        if not self.org:
            return
        cfg = self.org.dna.genome.get("cognition", {})
        brain = self.org.brain
        if "spreading_activation_threshold" in cfg:
            brain.spreading_activation_threshold = float(cfg["spreading_activation_threshold"])
        if "episodic_decay" in cfg:
            self.episodic_memory.episodic_decay = float(cfg["episodic_decay"])
        interval = cfg.get("thought_loop_interval_s")
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            self.thought_generator._interval = (float(interval[0]), float(interval[1]))
        self._sync_brain_orchestrator()

    def _sync_brain_orchestrator(self) -> None:
        if not self.org:
            return
        ram_gb = int(os.environ.get("ORGANISM_SERVER_RAM_GB", "16"))
        vram_gb = int(os.environ.get("ORGANISM_GPU_VRAM_GB", "8"))
        self.brain_orchestrator = BrainOrchestrator(
            brain=self.org.brain,
            impulse=self.impulse,
            disk_vault=self.disk_vault,
            server_ram_gb=ram_gb,
            gpu_vram_gb=vram_gb,
        )

    def _episodic_recall_merged(self, query: str = "", *, limit: int = 3) -> list[dict[str, Any]]:
        hits = self.episodic_memory.recall_context(query, limit=limit)
        if self.disk_vault and query.strip():
            disk_hits = self.disk_vault.search(query, limit=limit)
            seen = {f"{h.get('heard','')}|{h.get('spoke','')}" for h in hits}
            for row in disk_hits:
                key = f"{row.get('heard','')}|{row.get('spoke','')}"
                if key not in seen:
                    hits.append(row)
                    seen.add(key)
                if len(hits) >= limit:
                    break
        return hits[:limit]

    def _start_thought_loop(self) -> None:
        if not self.org:
            return
        self.thought_generator.bind(
            self.org.brain,
            self.working_memory,
            goals=self.goal_stack,
            hypotheses=self.hypothesis_engine,
            curiosity_fn=lambda: self.curiosity.state.level,
            on_thought=self._on_spontaneous_thought,
            mind=self.org.mind_bridge.mind,
        )
        self.thought_generator.start()

    def _append_consciousness(self, lines: list[str]) -> None:
        for ln in lines:
            s = str(ln).strip()
            if not s:
                continue
            if self._consciousness_log and self._consciousness_log[-1] == s:
                continue
            self._consciousness_log.append(s)
            self._consciousness_seq += 1
        self._consciousness_log = self._consciousness_log[-240:]

    def _log_dialogue(self, role: str, text: str, **meta: Any) -> None:
        t = (text or "").strip()
        if not t:
            return
        now = time.time()
        if self._dialogue_log:
            last = self._dialogue_log[-1]
            if (
                last.get("role") == role
                and last.get("text") == t
                and now - float(last.get("t", 0)) < 3.0
            ):
                return
        self._dialogue_log.append({"t": now, "role": role, "text": t, **meta})
        self._dialogue_log = self._dialogue_log[-120:]

    def _on_spontaneous_thought(self, themes: list[str]) -> None:
        if not themes or self._dormant:
            return
        self.working_memory.push(themes[:6], heard="")
        img = self.visual_imagination.flash(
            features=self._last_visual_features,
            labels=themes[:4],
            neuron_activation=self.curiosity.state.level,
        )
        if img and img.labels:
            for lb in img.labels:
                if lb not in themes:
                    themes.append(lb)

    def _bootstrap_layer2(self) -> dict[str, Any]:
        """Inizializzazione rapida al boot — solo capacità linguistica strutturale.

        Volontariamente leggera: crea la struttura sintattica ma non popola
        il vocabolario semantico. Usa train_foundation() per il training completo.
        """
        stats = self.narrator.bootstrap_curriculum(repeats=3)
        return {"syntax": stats, "narrator": self.narrator.stats()}

    def _context_stimulus_key(self) -> str:
        if self._last_vision_hash or self._last_audio_hash:
            return stimulus_key_visual_context(
                vision_hash=self._last_vision_hash,
                audio_hash=self._last_audio_hash,
            )
        return self.curiosity.state.last_stimulus_key

    def _prime_learned_pathways(
        self,
        org: OrganismRuntime,
        skey: str,
        text: str | None,
        *,
        vision_fresh: bool = False,
    ) -> tuple[bool, str | None, list[str], str]:
        """Attiva percorsi sinaptici appresi — hint lessicali, non copia scriptata.

        Ritorna (has_path, code_out, pathway_words, dialogue_text).
        dialogue_text è la risposta completa appresa (stringa intera), usabile
        direttamente se il lessico può articolarla.
        """
        has_path = False
        code_out: str | None = None
        pathway_words: list[str] = []
        dialogue_text = ""
        if text and self.speech:
            say, kind = self.dialogue.respond(text)
            if say and kind == "code":
                return True, say, [], ""
            if say:
                for w in _tokens_from_heard(text):
                    self.composer.absorb(w, boost=0.35)
                self.speech.hear(text, boost=0.45)
                pathway_words = _tokens_from_heard(say)
                dialogue_text = say
                has_path = True
            else:
                scene = self._match_scene_phrase(text)
                if scene:
                    for w in _tokens_from_heard(text):
                        self.composer.absorb(w, boost=0.4)
                    self.speech.hear(text, boost=0.45)
                    pathway_words = _tokens_from_heard(scene)
                    has_path = True
        if not text:
            phrase = self.teacher.respond(skey)
            if phrase and self.speech:
                prime_dialogue_pathway(org.brain, self.speech, when=skey, say=phrase)
                self.composer.absorb(phrase, boost=0.5)
                has_path = True
        if vision_fresh and self._last_vision_hash and self.speech:
            rec = self.visual_binder.recognize_object(self._last_visual_features, min_sim=0.38)
            if rec:
                self.composer.lexicon.prime_word(rec, boost=0.65)
                vis_themes = self.visual_binder.speech_themes(
                    self._last_visual_features,
                    articulable=lambda w: self.composer.lexicon.is_articulable(w, min_exposure=0.2),
                )
                prime_visual_percept(
                    org.brain, self.speech, self.composer.lexicon, themes=vis_themes, boost=0.55
                )
                has_path = True
            else:
                for w in self.visual_binder.themes(
                    self._last_vision_hash, self._last_visual_features, limit=5
                ):
                    if not w.startswith(("VIS:", "OBJ:", "COL:", "SCENE:")):
                        self.composer.lexicon.prime_word(w, boost=0.35)
                        has_path = True
        return has_path, code_out, pathway_words, dialogue_text

    def _wire_from_input(
        self,
        org: OrganismRuntime,
        *,
        had_text: bool,
        had_vision: bool,
        teach: bool = False,
    ) -> int:
        brain = org.brain
        if teach:
            return wire_teach_burst(brain, had_vision=had_vision, had_text=had_text)
        grown = wire_cross_modal(brain, had_vision=had_vision, had_text=had_text, max_new=10)
        pre: list[int] = []
        if had_text:
            pre.extend(active_ids(brain, "sensory", "text_semantic_encoder"))
        if had_vision:
            pre.extend(active_ids(brain, "sensory", "vision_edge_detector", min_act=0.15))
        post = [n.id for n in brain.get_neurons("motor", "speech_phoneme_generator")]
        assoc = active_ids(brain, "associative", "pattern_matcher", min_act=0.12)
        if not pre:
            pre = assoc
        grown += wire_coactive(brain, pre[:16], post[:14], max_new=10)
        if had_vision:
            grown += wire_coactive(brain, assoc[:12], post[:10], max_new=6, weight=0.17)
        return grown

    def _long_form_trigger(self, text: str) -> bool:
        tl = text.strip().lower()
        if len(tl) < 12:
            return False
        say, _ = self.dialogue.respond(tl)
        return bool(say and len(say) > 60)

    def _act(
        self,
        org: OrganismRuntime,
        *,
        heard: str | None,
        impulse: str,
        long_form: bool = False,
        skey: str = "",
        sensory_symbols: list[str] | None = None,
        social_tone: SocialTone | None = None,
    ) -> tuple[str, str, dict, dict, BrainMood, bool, bool, MotorPlan | None, list[str]]:
        idle_s = time.time() - self._last_sense_t
        wave_phase = self.waves.last.phase
        tone_pre = social_tone or SocialTone(0.0, 0.1, False, False, False, False)
        self._tick_subcortex(
            org,
            idle_s=idle_s,
            social_warm=tone_pre.is_warm or tone_pre.is_praise,
            wave_phase=wave_phase,
        )
        if self._vision_fresh:
            self._update_body_schema()
        if self.motion is None and org.brain:
            self.motion = MotionModule(org.brain)

        arousal = inject_circadian(org.brain)
        if impulse == "vocalize":
            for n in self.speech.phoneme_neurons[:6]:
                n.activation = min(1.0, n.activation + 0.08 * arousal)
        org.brain.propagate(steps=1)

        grown = org.brain.synapse_count - self._synapses_at_birth
        vis_themes = self._visual_themes()
        has_path, code_out, pathway_words, dialogue_text = self._prime_learned_pathways(
            org, skey, heard, vision_fresh=self._vision_fresh
        )
        comp_frame: ComprehensionFrame | None = None
        if heard:
            self._ensure_semantic()
            comp_frame = self.psyche.comprehend(
                heard,
                semantic=self.semantic,
                dialogue_respond=self._dialogue_for_psyche,
                episodic_recall=self._episodic_recall_merged,
                wm_context=self.working_memory.context_words(),
                visual_themes=vis_themes,
            )
            self._last_comp_frame = comp_frame
            if comp_frame.taught_say and comp_frame.intent in ("taught", "causal", "word_meaning"):
                if not dialogue_text or comp_frame.depth >= 0.65:
                    dialogue_text = comp_frame.taught_say
            if comp_frame.inhibit_lexicon_dump and comp_frame.intent == "social":
                dialogue_text = ""
                pathway_words = _tokens_from_heard((heard or "").lower())
            if comp_frame.inhibit_shallow and comp_frame.intent.startswith("narrative"):
                long_form = True

        vis_cue = self._last_vision_hash if self._vision_fresh else ""
        if not code_out and heard:
            hl = heard.lower()
            if any(k in hl for k in ("programma", "codice", "scrivi", "python", "funzione")):
                code_out = self.code.produce(heard) or None

        # --- MIND come knowledge layer E response candidate ---
        # MIND attiva frammenti semantici → arricchisce memory_themes E offre
        # risposta candidata. La variazione viene dall'articulability check:
        # frammenti che Baby non sa ancora articolare vengono usati solo come contesto.
        mind_fragments: list[str] = []
        mind_candidate: str = ""
        if heard and not code_out:
            from mind.types import Cue, CueKind
            import re as _re_mind
            mind_result = org.mind_bridge.mind.think(
                Cue(kind=CueKind.TEXT, value=heard.strip(), meta={"human": True})
            )
            if mind_result.fragments:
                mind_fragments = [f.title for f in mind_result.fragments[:4]]
                best = mind_result.fragments[0]
                mind_candidate = best.title
                for ft in mind_fragments[:2]:
                    self._append_consciousness([f"memoria: {ft[:60]}"])
            # Dialogo insegnato lungo prende priorità su MIND
            # Dialogo brevissimo (< 4 parole) → MIND offre risposta più ricca
            if dialogue_text and mind_candidate:
                _dt_w = [w for w in _re_mind.findall(r"[a-zàèéìòù']+", dialogue_text.lower()) if len(w) > 2]
                _mf_w = [w for w in _re_mind.findall(r"[a-zàèéìòù']+", mind_candidate.lower()) if len(w) > 2]
                if len(_dt_w) < 4 and len(_mf_w) >= 6:
                    dialogue_text = ""  # MIND ha risposta più ricca
            if comp_frame and comp_frame.inhibit_lexicon_dump and comp_frame.intent == "social":
                taught = (comp_frame.taught_say or "").strip().lower().rstrip(".")
                if mind_candidate.strip().lower().rstrip(".") == taught:
                    mind_candidate = ""

        unknown_words: list[str] = []
        if heard and not has_path and not code_out and not mind_candidate:
            unknown_words = self.self_learner.detect_unknown(
                heard,
                has_pathway=has_path,
            )

        vis_cue = self._last_vision_hash if self._vision_fresh else ""
        pre_thought = self.thought_engine.think(
            org.brain,
            org.memory,
            self.curiosity.state,
            heard_text=heard or "",
            synapses_grown=grown,
            visual_cue=vis_cue,
            visual_themes=vis_themes,
        )
        ws = self.workspace.cycle(
            org.brain,
            pre_thought,
            sensory_symbols=sensory_symbols,
            novelty=self.curiosity.state.novelty,
            wave_phase=wave_phase,
            has_learned_path=has_path,
        )
        if self.impulse:
            if heard:
                self.impulse.perceive_text(heard)
            self.impulse.pulse(steps=2)
            ws = merge_workspace(ws, self.impulse)
            for line in impulse_consciousness_lines(self.impulse):
                self._append_consciousness([line])
        layers = org.brain.layer_activation_summary()
        neural_act = sum(layers.values()) / max(1, len(layers))
        qstate = self.quantum_layer.tick(
            neural_activity=neural_act,
            workspace_ignition=ws.ignition,
            thought_seed=(heard or ws.focus or "")[:80],
        )
        if qstate.last_moment and self.quantum_layer.collapse_boost() > 0.05:
            self._append_consciousness([f"microtubuli · {qstate.last_moment[:56]}"])
        thought = self.thought_engine.think(
            org.brain,
            org.memory,
            self.curiosity.state,
            heard_text=heard or "",
            synapses_grown=grown,
            workspace_broadcast=ws.broadcast,
            visual_cue=vis_cue,
            visual_themes=vis_themes,
        )
        if heard:
            self.composer.absorb(heard, boost=0.5)

        # Topic threading: inietta le parole chiave dei frammenti MIND attivati
        # nella working memory e nei temi del pensiero per coerenza multi-turno.
        # I frammenti sono KNOWLEDGE — colorano il pensiero, non generano risposte fisse.
        if mind_fragments:
            import re as _re_topic
            topic_words: list[str] = []
            for ft in mind_fragments[:3]:
                for w in _re_topic.findall(r"[a-zàèéìòù]+", ft.lower()):
                    if len(w) > 3 and w not in topic_words:
                        topic_words.append(w)
            if topic_words:
                self.working_memory.activate(topic_words[:8], weight=0.7)
                for w in topic_words[:4]:
                    if w not in thought.themes:
                        thought.themes.append(w)

        _code_noise = ("print", "def", "return", "range", "for", "else")
        thought.themes = [
            t for t in thought.themes
            if not any(c in t for c in _code_noise) and not t.startswith("VIS:sig=")
        ]

        if self._vision_fresh and self._last_vision_hash:
            conf = self.visual_binder.confidence(self._last_visual_features)
            min_sim = 0.62 if conf > 0.7 else 0.55
            rec = self.visual_binder.recognize_object(
                self._last_visual_features, min_sim=min_sim
            )
            color = str(self._last_visual_features.get("color", ""))
            if color:
                thought.symbols.append(f"COL:{color}")
            for sym in self._last_visual_features.get("symbols", []):
                if sym.startswith("COL:") and sym not in thought.symbols:
                    thought.symbols.append(sym)
            if rec and rec not in thought.themes:
                thought.themes.insert(0, rec)
            if color and color not in thought.themes:
                thought.themes.insert(0, color)
            if rec or color:
                vis_speech = self.visual_binder.speech_themes(
                    self._last_visual_features,
                    question="vision",
                    articulable=lambda w: self.composer.lexicon.is_articulable(w, min_exposure=0.25),
                )
                for w in vis_speech:
                    if w not in thought.themes:
                        thought.themes.insert(0, w)
            face_data = self._last_visual_features.get("face") or {}
            if face_data.get("detected"):
                for sym in face_data.get("symbols", []):
                    if sym not in thought.symbols:
                        thought.symbols.append(sym)
                for w in self.face_binder.speech_themes(face_data):
                    if w not in thought.themes:
                        thought.themes.insert(0, w)
                emo = self.face_binder.recognize_emotion(face_data) or self.face_binder.infer_emotion_hint(face_data)
                if emo:
                    self.affect.note_visual_emotion(emo)
                    thought.symbols.append(f"EMO:vis:{emo}")
        elif heard and not self._vision_fresh:
            known_objs = set(self.visual_binder._object_names.values())
            thought.themes = [
                t for t in thought.themes
                if t not in known_objs and not t.startswith(("OBJ:", "COL:", "VIS:"))
            ]

        heard_l = (heard or "").strip().lower()
        _vision_q = (
            "cosa vedi", "che vedi", "che colore", "cosa è",
            "di che colore", "quale colore",
        )
        _knowledge_q = is_question(heard or "") and not has_path
        _yesno_q = ("è un ", "è una ", "è il ", "è la ", "è questo", "è questa", "vero che", "giusto che")
        if heard_l and any(k in heard_l for k in ("che colore", "di che colore", "quale colore")):
            thought.symbols.append("QUESTION:color")
            if self._vision_fresh:
                color = str(self._last_visual_features.get("color", ""))
                if color:
                    thought.symbols.append(f"COL:{color}")
                vis_themes = self.visual_binder.speech_themes(
                    self._last_visual_features,
                    question="color",
                    articulable=lambda w: self.composer.lexicon.is_articulable(w, min_exposure=0.3),
                )
                prime_visual_percept(
                    org.brain, self.speech, self.composer.lexicon, themes=vis_themes, boost=0.5
                )
                thought.themes = vis_themes or self.visual_binder.percept_words(self._last_visual_features)
        elif heard_l and any(heard_l.startswith(p) for p in _vision_q):
            thought.symbols.append("QUESTION:vision")
            if self._vision_fresh:
                vis_themes = self.visual_binder.speech_themes(
                    self._last_visual_features,
                    question="vision",
                    articulable=lambda w: self.composer.lexicon.is_articulable(w, min_exposure=0.3),
                )
                prime_visual_percept(
                    org.brain, self.speech, self.composer.lexicon, themes=vis_themes, boost=0.5
                )
                thought.themes = vis_themes or self.visual_binder.percept_words(self._last_visual_features)
        elif heard_l in ("sì", "si", "no", "è vero", "è sbagliato", "è giusto", "non è vero"):
            thought.symbols.append("QUESTION:yesno")
            answer = "no" if any(w in heard_l for w in ("no", "sbagliato", "falso", "non")) else "sì"
            thought.themes = [answer]
        elif heard_l and (heard_l.rstrip("?") in ("sì", "si", "no") or any(heard_l.startswith(p) for p in _yesno_q)):
            thought.symbols.append("QUESTION:yesno")
            neg = any(w in heard_l for w in ("non", "sbagliato", "falso"))
            rec = self.visual_binder.recognize_object(self._last_visual_features, min_sim=0.52) if self._vision_fresh else None
            asked_obj = ""
            for token in heard_l.replace("?", "").split():
                if len(token) > 3 and token not in ("questo", "questa", "quello", "quella", "vero", "giusto"):
                    asked_obj = token
                    break
            if asked_obj and rec:
                match = asked_obj in rec or rec in asked_obj
                answer = "sì" if match != neg else "no"
                thought.themes = [answer]
            elif neg:
                thought.themes = ["no"]
            elif heard_l.startswith("è un") or heard_l.startswith("è una"):
                thought.themes = ["sì" if rec else "no"]

        if heard and has_path:
            for n in self.speech.phoneme_neurons[:12]:
                n.activation = min(1.0, n.activation + 0.08)

        if heard:
            for w in _tokens_from_heard(heard_l):
                if w not in thought.themes:
                    thought.themes.append(w)
            wm_ctx = self.working_memory.context_words()
            thought.themes = coherent_themes(
                thought.themes,
                heard=heard,
                pathway_words=pathway_words,
                context_words=wm_ctx or self._context_words,
                max_themes=6,
            )

        if unknown_words:
            self.goal_stack.emerge_from_state(unknown_word=unknown_words[0])
        if (unknown_words or _knowledge_q) and not has_path:
            impulse = "ask"
            self.curiosity.state.uncertainty = min(1.0, 0.85)
            self.curiosity.state.level = min(1.0, self.curiosity.state.level + 0.2)
            self.goal_stack.emerge_from_state(uncertainty=self.curiosity.state.uncertainty)
            thought.symbols.append("IMPULSE:ask")
            heard_tokens = _tokens_from_heard(heard_l)
            focus_word = unknown_words[0] if unknown_words else (heard_tokens[-1] if heard_tokens else "")
            if focus_word:
                thought.symbols.append(f"UNKNOWN:{focus_word}")
                self.self_learner.note_asked(focus_word)
            prime_curiosity_pathway(
                org.brain,
                self.speech,
                self.composer.lexicon,
                focus=focus_word or "",
                heard=heard_l,
                boost=0.55,
            )
            ask_pool = ([focus_word] if focus_word else []) + _tokens_from_heard(heard_l)
            ask_themes = self.composer.lexicon.active_words(8, min_act=0.08)
            for w in self.composer.lexicon.ranked(ask_pool):
                if w not in ask_themes:
                    ask_themes.append(w)
            thought.themes = ask_themes[:6]
            if focus_word and focus_word not in thought.themes:
                thought.themes.insert(0, focus_word)

        tone = social_tone or SocialTone(0.0, 0.1, False, False, False, False)
        visual_energy = float(self._last_visual_features.get("contrast", 0.0))
        has_learned = has_path
        self._pulse_amygdala(org)
        ask_mode_pre = (unknown_words or _knowledge_q) and not has_path
        bg = self.basal_ganglia.select(
            default=impulse,
            heard=heard,
            curiosity=self.curiosity.state.level,
            novelty=self.curiosity.state.novelty,
            boredom=self.curiosity.state.boredom,
            amygdala_inhibition=self.amygdala.speech_inhibition(),
            has_association=has_path,
            wants_ask=bool(ask_mode_pre),
        )
        if bg.urgency > 0.42 and bg.impulse in ("speak", "ask", "vocalize"):
            impulse = bg.impulse
        em = self.speech.readout(extra_inhibition=self.amygdala.speech_inhibition())
        pres = self.presence.evaluate(
            curiosity=min(1.0, self.curiosity.state.level * 0.6 + self.affect.state.curiosity * 0.4),
            novelty=self.curiosity.state.novelty,
            boredom=self.curiosity.state.boredom,
            stimulus_key=skey,
            visual_energy=visual_energy,
            has_learned_phrase=has_learned,
            impulse=impulse,
            wants_voice=em.will_speak or thought.pressure > 0.1 or has_path,
        )
        caregiver_spoke = bool((heard or "").strip())
        wants_voice = caregiver_spoke and (
            pres.speaks
            or has_path
            or is_question(heard or "")
            or impulse in ("vocalize", "ask")
        )
        if wants_voice and not pres.speaks:
            pres = self.presence.evaluate(
                curiosity=self.curiosity.state.level,
                novelty=self.curiosity.state.novelty,
                boredom=self.curiosity.state.boredom,
                stimulus_key=skey,
                visual_energy=visual_energy,
                has_learned_phrase=has_learned,
                impulse=impulse,
                wants_voice=True,
            )

        from organism.motor.compose_speech import ComposedSpeech

        memory_themes = (
            self.episodic_memory.recent_themes(limit=10)
            + self.photo_memory.label_hints(self._last_visual_features)
        )
        memory_themes = list(dict.fromkeys(memory_themes))[:12]
        if memory_themes:
            thought.themes = list(dict.fromkeys(memory_themes + thought.themes))[:14]
        for stream in self.working_memory.parallel_streams():
            for w in stream:
                if w not in thought.themes:
                    thought.themes.append(w)
        goal_words = self.goal_stack.attention_bias_words()
        for w in goal_words:
            if w not in thought.themes:
                thought.themes.insert(0, w)
        for w in self.hypothesis_engine.theme_hints():
            if w not in thought.themes:
                thought.themes.append(w)
        if self._vision_fresh:
            flash = self.visual_imagination.flash(
                features=self._last_visual_features,
                neuron_activation=thought.pressure,
            )
            if flash:
                for lb in flash.labels:
                    if lb not in thought.themes:
                        thought.themes.insert(0, lb)

        affect_valence = (
            self.affect.state.joy - self.affect.state.fear - self.affect.state.sadness * 0.5
        )

        composed: ComposedSpeech
        if code_out:
            composed = ComposedSpeech(code_out, "code", False, thought.themes[:4])
        elif mind_candidate and wants_voice and not dialogue_text:
            # MIND ha trovato un frammento rilevante — usalo come risposta se articolabile.
            import re as _re_mc
            _mc_words = _re_mc.findall(r"[a-zàèéìòù']+", mind_candidate.lower())
            _mc_artic = sum(1 for w in _mc_words if len(w) > 2 and
                            self.composer._articulable(w, min_exposure=0.25))
            if _mc_artic >= max(3, len(_mc_words) // 2):
                _mc = mind_candidate.strip()
                if _mc and _mc[-1] not in ".?!":
                    _mc += "."
                _mc = _mc[0].upper() + _mc[1:]
                _plan = self.speech.plan_from_text(_mc) if self.speech else None
                composed = ComposedSpeech(
                    text=_mc, kind="speech", from_thought=True,
                    thought_used=thought.themes[:6] + mind_fragments[:2],
                    motor_plan=_plan,
                )
            elif long_form:
                composed = self.composer.long_form(
                    thought=thought, motor=self.speech,
                    heard=heard, memory_themes=mind_fragments[:4] + (memory_themes or []),
                )
            else:
                # Fallback a produce() con contesto MIND
                speech_mode = "speak" if heard else ws.mode
                if speech_mode not in ("speak", "reflect", "flow"):
                    speech_mode = "speak"
                recent_flat = [w for s in self._recent_spokes[-6:]
                               for w in s.replace(".", "").replace("?", "").split() if len(w) > 2]
                conv_ctx = self.working_memory.context_words(limit=8)
                enriched_memory = list(dict.fromkeys(mind_fragments[:4] + (memory_themes or []) + conv_ctx))[:16]
                composed = self.composer.produce(
                    thought=thought, motor=self.speech, mode=speech_mode,
                    reflective=False, pathway_primed=bool(pathway_words),
                    pathway_words=pathway_words, amygdala=self.amygdala,
                    recent_words=recent_flat, heard=heard,
                    memory_themes=enriched_memory, valence=affect_valence,
                    dialogue_text="",
                )
        elif dialogue_text and wants_voice:
            # Risposta dialogica appresa — usata DIRETTAMENTE per risposte lunghe (>= 5 parole).
            # Risposte brevi passano al percorso emergente (produce/long_form) con pathway_words.
            import re as _re
            _dt_words = _re.findall(r"[a-zàèéìòù']+", dialogue_text.lower())
            _dt_words = [w for w in _dt_words if len(w) > 2]
            _artic = sum(1 for w in _dt_words if self.composer._articulable(w, min_exposure=0.25))
            if len(_dt_words) >= 5 and _artic >= max(3, len(_dt_words) // 2):
                _text = dialogue_text.strip()
                if _text and _text[-1] not in ".?!":
                    _text += "."
                _text = _text[0].upper() + _text[1:]
                _plan = self.speech.plan_from_text(_text) if self.speech else None
                composed = ComposedSpeech(
                    text=_text,
                    kind="speech",
                    from_thought=True,
                    thought_used=thought.themes[:10],
                    motor_plan=_plan,
                )
            elif long_form:
                composed = self.composer.long_form(
                    thought=thought,
                    motor=self.speech,
                    heard=heard,
                    memory_themes=memory_themes,
                )
            else:
                # Risposta breve → produce() con pathway_words come guida
                speech_mode = "speak" if heard else ws.mode
                if speech_mode not in ("speak", "reflect", "flow"):
                    speech_mode = "speak"
                recent_flat = [
                    w for s in self._recent_spokes[-6:]
                    for w in s.replace(".", "").replace("?", "").split()
                    if len(w) > 2
                ]
                conv_ctx = self.working_memory.context_words(limit=8)
                enriched_memory = list(dict.fromkeys((memory_themes or []) + conv_ctx))[:16]
                composed = self.composer.produce(
                    thought=thought,
                    motor=self.speech,
                    mode=speech_mode,
                    reflective=False,
                    pathway_primed=bool(pathway_words),
                    pathway_words=pathway_words,
                    amygdala=self.amygdala,
                    recent_words=recent_flat,
                    heard=heard,
                    memory_themes=enriched_memory,
                    valence=affect_valence,
                    dialogue_text=dialogue_text,
                )
        elif long_form and wants_voice:
            composed = self.composer.long_form(
                thought=thought,
                motor=self.speech,
                heard=heard,
                memory_themes=memory_themes,
            )
        elif wants_voice:
            # Domande e stimoli esterni → parla sempre; coscienza osserva in parallelo
            speech_mode = "speak" if heard else ws.mode
            if speech_mode not in ("speak", "reflect", "flow"):
                speech_mode = "speak"
            ask_mode = any(s.startswith("IMPULSE:ask") for s in thought.symbols)
            recent_flat = [
                w
                for s in self._recent_spokes[-6:]
                for w in s.replace(".", "").replace("?", "").split()
                if len(w) > 2
            ]
            # Arricchisci i memory_themes con il contesto conversazionale corrente
            conv_ctx = self.working_memory.context_words(limit=8)
            enriched_memory = list(dict.fromkeys((memory_themes or []) + conv_ctx))[:16]
            composed = self.composer.produce(
                thought=thought,
                motor=self.speech,
                mode=speech_mode,
                reflective=False,
                pathway_primed=bool(pathway_words),
                pathway_words=pathway_words,
                amygdala=self.amygdala,
                recent_words=recent_flat,
                heard=heard,
                memory_themes=enriched_memory,
                valence=affect_valence,
                dialogue_text=dialogue_text,
            )
        else:
            composed = ComposedSpeech("", "speech", False, [])

        mood = read_brain_mood(
            org.brain,
            synapses_at_birth=self._synapses_at_birth,
            motor_pressure=em.motor_pressure,
            inhibition=em.inhibition,
            wants_voice=wants_voice,
            arousal=arousal,
        )

        spoke = composed.text if composed.kind == "speech" else ""
        code = composed.text if composed.kind == "code" else ""
        # Rimuovi eventuali artefatti "→" da vecchi frammenti MIND
        if spoke and "→" in spoke:
            spoke = spoke.split("→")[-1].strip()
            if spoke and spoke[0].islower():
                spoke = spoke[0].upper() + spoke[1:]
        spoke = self._apply_superego(spoke, heard or "", comp_frame)
        spoke = self._anti_repeat_speech(
            spoke,
            org=org,
            heard=heard,
            ask_mode=any(s.startswith("IMPULSE:ask") for s in thought.symbols),
        )
        if spoke:
            self._recent_spokes.append(spoke.lower().strip())
            self._recent_spokes = self._recent_spokes[-8:]
        if heard or thought.themes:
            self.working_memory.push(thought.themes, heard=heard or "")
            self._context_words = self.working_memory.context_words()
        understood = (
            has_path
            or (bool(spoke) and thought.memory_hits > 0)
            or composed.from_thought
        )
        plan = composed.motor_plan
        if spoke and not plan:
            plan = self.speech.plan_from_text(spoke)
        motor_fb: dict[str, Any] = {}
        motion_fb: dict[str, Any] = {}
        if self.motion:
            pose_d = self.body_schema.pose.to_dict()
            self._last_motion = self.motion.express(
                spatial_gesture=self.body_schema.gesture_from_state(
                    joy=self.affect.state.joy,
                    fear=self.affect.state.fear,
                ),
                heading_deg=float(pose_d.get("heading_deg", 0.0)),
                velocity=float(pose_d.get("velocity", 0.0)),
                body_mode=self.body_schema.navigate_mode,
            )
            motion_fb = {
                "gesture": self._last_motion.frames[0].gesture if self._last_motion.frames else "",
                "mode": self.body_schema.navigate_mode,
                "pose": pose_d,
            }
        if spoke:
            self._last_baby_spoke = spoke
            self._last_spoke_wall_t = time.time()
            self.presence.note_spoke()
            self.corrections.note_baby_spoke(spoke)
            motor_fb = self._close_motor_loop(org, spoke, plan)
        fc = self._face_context()
        if fc.get("detected") and spoke:
            face_name = fc.get("face")
            if face_name:
                self.composer.lexicon.prime_word(str(face_name), boost=0.25)
            emo = fc.get("emotion")
            if emo:
                self.composer.lexicon.prime_word(str(emo), boost=0.15)
        if spoke or heard:
            self.episodic_memory.record(
                heard=heard or "",
                spoke=spoke or "",
                themes=thought.themes[:8],
                objects=[w for w in thought.themes if self.visual_binder.recognize_object(self._last_visual_features) == w][:3],
                emotion=str(fc.get("emotion") or ""),
            )
            if self.disk_vault:
                self.disk_vault.append_episode(
                    heard=heard or "",
                    spoke=spoke or "",
                    themes=thought.themes[:8],
                    emotion=str(fc.get("emotion") or ""),
                    meta={"pulse": self.affect._pulse_count},
                )
        self._last_presence = pres.to_dict()
        self._last_tone = tone.to_dict()
        thought_d = thought.to_dict()
        ws_d = ws.to_dict()
        ep_ctx = self._episodic_recall_merged(heard or "", limit=1)
        mem_line = ""
        if ep_ctx:
            mem_line = str(ep_ctx[0].get("spoke") or ep_ctx[0].get("heard") or "")[:60]
        self._last_thought_coherence = thought_coherence(thought.themes, heard=heard or "")
        self._last_inner_voice = compose_inner_voice(
            themes=thought.themes,
            heard=heard or "",
            emotion=str(fc.get("emotion") or self.affect.state.label or ""),
            objects=[
                w
                for w in thought.themes
                if self.visual_binder.recognize_object(self._last_visual_features) == w
            ][:2],
            memory_line=mem_line,
        )
        stream = build_consciousness_stream(
            heard=heard,
            thought=thought_d,
            workspace=ws_d,
            self_state=self.self_model.state.to_dict(),
            spoke=spoke,
            wants_voice=wants_voice,
            presence=pres.to_dict(),
            motor_will=em.will_speak,
            wave=self.waves.last.to_dict(),
            emotion=self.affect.state.to_dict(),
            phase=impulse,
            taught_anchor=heard or "",
        )
        stream = enrich_consciousness_lines(
            stream,
            inner_voice=self._last_inner_voice,
            coherence=self._last_thought_coherence,
        )
        self._last_ws = ws_d
        self._append_consciousness(stream)
        if motor_fb:
            thought_d.setdefault("motor_loop", motor_fb)
        if motion_fb:
            thought_d.setdefault("body_motion", motion_fb)
        return spoke, code, thought_d, ws_d, mood, understood, composed.from_thought, plan, stream

    def _close_motor_loop(
        self,
        org: OrganismRuntime,
        spoke: str,
        plan: Any,
    ) -> dict[str, Any]:
        """Chiusura DIVA — auto-ascolto + rinforzo sillabe prodotte."""
        if not spoke.strip() or self.speech is None:
            return {}
        from organism.cognition.motor_plan import MotorPlan

        mp: MotorPlan | None = plan if plan else None
        if mp and mp.syllables:
            self.speech.phonemes.reinforce_sequence(mp.syllables, boost=0.05)
        elif spoke:
            from organism.teaching.phonemes import split_italian_syllables

            self.speech.phonemes.reinforce_sequence(
                split_italian_syllables(spoke), boost=0.05
            )
        if spoke.strip():
            self.narrator.train_from_text(spoke, boost=0.05)
            self.hypothesis_engine.observe(supports=spoke[:60])
        return {
            "deferred_self_hear": True,
            "text": spoke,
            "syllables_reinforced": bool(mp and mp.syllables),
        }

    def brain_pulse_tick(self, *, persist: bool = False) -> dict[str, Any]:
        """Cervello ad onde — percepisce, pensa, sogna, riflette senza corpo."""
        if not self._born or self.org is None:
            return {"alive": False}
        org = self.org
        idle = time.time() - self._last_sense_t
        if self.affect._pulse_count % 45 == 0:
            inject_circadian(org.brain)

        if self.impulse:
            reading = self.impulse.pulse(steps=1)
            if reading.conscious and reading.thoughts:
                self._append_consciousness([f"pulse · {reading.thoughts[0][:64]}"])

        wave = self.waves.advance(idle=idle > 10, arousal=self.affect.state.curiosity)
        inject_wave(org.brain, wave.phase, tick=wave.tick, amplitude=wave.amplitude)
        self._tick_subcortex(org, idle_s=idle, wave_phase=wave.phase)
        self.affect.pulse_tick(org.brain, idle=idle > 25)
        self._pulse_amygdala(org)

        dream_st = self.dream_engine.state
        if wave.phase == "dream":
            dream_st = self.dream_engine.cycle(
                org.brain, org.memory, lexicon=self.composer.lexicon, idle_s=idle
            )
            if dream_st.content:
                self._last_dream_content = dream_st.content
                self.composer.absorb(dream_st.content, boost=0.15)

        if not self._dormant and self.affect._pulse_count % 20 == 0:
            wire_self_recurrence(org.brain)

        for n in list(org.brain.neurons.values())[::3]:
            n.activation = max(0.01, n.activation * 0.992)
        if org.brain.plasticity and self.affect._pulse_count % 15 == 0:
            org.brain.plasticity.apply_hebbian(org.brain, org.brain.tick)

        growth = (
            self.brain_growth.maybe_grow(org.brain, pulses=self.affect._pulse_count)
            if not self._dormant
            else {"skipped": "dormant"}
        )

        self.self_model.update(
            org.brain,
            org.memory,
            wave=wave,
            affect=self.affect.state,
            dream_fragment=dream_st.content if dream_st.active else "",
            saw=self._last_vision_hash,
            body_mode=self.body_schema.navigate_mode,
            interoception=self.interoception.state.label,
            place_context=self.body_schema.hippocampus_context(),
        )

        if persist:
            self._persist()
        orch = self.brain_orchestrator.stats() if self.brain_orchestrator else {}
        return {
            "alive": True,
            "brain_tick": org.brain.tick,
            "wave": wave.to_dict(),
            "self": self.self_model.state.to_dict(),
            "dream": dream_st.to_dict(),
            "emotion": self.affect.state.to_dict(),
            "neurochemistry": self.neurochemistry.stats(),
            "endocrine": self.endocrine.stats(),
            "interoception": self.interoception.stats(),
            "body_schema": self.body_schema.stats(),
            "quantum": self.quantum_layer.stats(),
            "pulses": self.affect._pulse_count,
            "idle_s": round(idle, 1),
            "growth": growth,
            "neurons": org.brain.neuron_count,
            "architecture": orch,
            "impulse": impulse_state_dict(self.impulse),
        }

    def sense(
        self,
        *,
        image_gray: list[int] | None = None,
        image_b64: str | None = None,
        image_w: int = 64,
        image_h: int = 64,
        audio_b64: str | None = None,
        text: str | None = None,
        color_rgb: dict[str, float] | None = None,
        image_rgba: list[int] | None = None,
    ) -> dict[str, Any]:
        org = self._ensure()
        self._last_sense_t = time.time()
        self._vision_fresh = bool(image_gray or image_b64 or image_rgba)
        input_data: dict[str, Any] = {}
        v_hash = a_hash = ""

        if self._vision_fresh:
            grid, sig = self._decode_vision(
                image_gray=image_gray,
                image_b64=image_b64,
                image_w=image_w,
                image_h=image_h,
                color_rgb=color_rgb,
                image_rgba=image_rgba,
            )
            v_hash = sig or vision_hash(grid)
            self._last_vision_hash = v_hash
            input_data["image"] = grid
            input_data["width"] = image_w
            input_data["height"] = image_h
            input_data["visual_features"] = self._last_visual_features
        if audio_b64:
            audio = decode_audio_b64(audio_b64)
            a_hash = audio_hash(audio)
            self._last_audio_hash = a_hash
            input_data["audio"] = audio
        social_fb: dict[str, Any] = {}
        if text and self._is_own_echo(text):
            fb = self.self_hear(text=text)
            return {
                "moment": fb.get("moment"),
                "mode": "self_feedback",
                "stats": org.stats,
            }
        tone = analyze_social_tone(text or "", last_spoke=self._last_baby_spoke)
        correction_fb: dict[str, Any] = {}
        if text:
            input_data["text"] = text
            self.speech.hear(text, boost=0.7)
            if tone.is_correction:
                correction_fb = self.corrections.try_learn(
                    text,
                    is_correction=True,
                    dialogue_teach=self.dialogue.teach,
                    phonemes=self.speech.phonemes,
                    lexicon=self.composer.lexicon,
                )
                if correction_fb.get("applied"):
                    wrong = str(correction_fb.get("wrong", ""))
                    right = str(correction_fb.get("right", ""))
                    if wrong and right:
                        self.self_model.note_correction(wrong, right)
            self.affect.update_from_social(
                tone,
                correction_applied=correction_fb.get("applied", False),
                curiosity_level=self.curiosity.state.level,
            )
            self.affect.inject_into_brain(org.brain)
            if not tone.is_angry:
                social_fb = self.speech_loop.social_feedback(
                    org.brain, self.speech, caregiver_text=text, within_seconds=8.0
                )
                self.presence.note_caregiver()
            else:
                self.presence.note_caregiver(duration=4.0)

        sensory = org.perceive(input_data) if input_data else None
        if sensory and input_data:
            org.brain.propagate(steps=2)
            org.brain.reinforce_active_pathway(boost=0.03)
            if org.brain.plasticity:
                org.brain.plasticity.apply_hebbian(org.brain, org.brain.tick)
            self._wire_from_input(
                org,
                had_text=bool(text),
                had_vision=bool(image_gray or image_b64 or image_rgba),
            )

        skey = stimulus_key_from_sensory(vision_hash=v_hash, audio_hash=a_hash, text=text or "")
        self.curiosity.observe(skey, pattern_gap=False, learned=False)
        impulse = self._choose_impulse(org)
        long_form = bool(text and self._long_form_trigger(text))
        sens_syms = sensory.all_symbols() if sensory else []
        spoke, code_out, thought_d, ws_d, mood, understood, from_thought, plan, stream = self._act(
            org,
            heard=text,
            impulse=impulse,
            long_form=long_form,
            skey=skey,
            sensory_symbols=sens_syms,
            social_tone=tone if text else None,
        )
        motor_fb = thought_d.get("motor_loop") or {}

        task_fb: dict[str, Any] = {}
        if spoke:
            task_fb = self.tasks.evaluate_attempt(
                spoke,
                corrected=bool(correction_fb.get("applied")),
            )

        vis_syms = list(self._last_visual_features.get("symbols", []))[:6]
        symbols = sens_syms + vis_syms + thought_d.get("symbols", [])[:6]
        if social_fb.get("social"):
            symbols.append("SOCIAL:caregiver_response")
        if correction_fb.get("applied"):
            symbols.append(f"LEARN:correction:{correction_fb.get('right', '')[:20]}")
        if tone.is_angry:
            symbols.append("TONE:angry")
        elif tone.is_correction:
            symbols.append("TONE:correction")
        elif tone.is_praise:
            symbols.append("TONE:praise")
        if ws_d.get("conscious"):
            symbols.append(f"CONSCIOUS:{ws_d.get('focus', '')[:20]}")
        symbols.append(f"EMO:{self.affect.state.dominant}")
        wave_st = self.waves.last
        inject_wave(org.brain, wave_st.phase, tick=wave_st.tick, amplitude=wave_st.amplitude)
        self_st = self.self_model.update(
            org.brain,
            org.memory,
            wave=wave_st,
            affect=self.affect.state,
            heard=text or "",
            spoke=spoke,
            saw=self._last_vision_hash,
        )
        moment = BabyMoment(
            impulse=impulse,
            spoke=spoke,
            code=code_out or "",
            understood=understood or from_thought or correction_fb.get("applied", False),
            from_thought=from_thought,
            self_heard=bool(motor_fb.get("self_heard")),
            speech_error=motor_fb.get("speech_error") or {},
            consciousness=ws_d,
            self_state=self_st.to_dict(),
            wave=wave_st.to_dict(),
            dream=self.dream_engine.state.to_dict(),
            emotion=self.affect.state.to_dict(),
            social_tone=tone.to_dict() if text else {},
            presence=self._last_presence,
            task=task_fb,
            consciousness_stream=stream,
            thought=thought_d,
            stimulus_key=skey,
            curiosity=self.curiosity.state.to_dict(),
            learned=understood or from_thought or task_fb.get("done", False),
            brain=mood.to_dict(),
            wanted_to_speak=mood.wants_voice and not spoke and not code_out,
            symbols=symbols,
        )
        self._last_moment = moment
        if text:
            self._log_dialogue("tu", text, channel="text")
        if spoke:
            self._log_dialogue("organism", spoke, channel="text")
        self.journal.record(
            cycle=org.learner.total_cycles,
            phase="sense",
            input_data=input_data or {"idle": True},
            thought_symbols=moment.symbols + [f"INNER:{ln}" for ln in stream[:12]],
            mind_action=impulse,
            mind_fragments=[],
            expression_text=spoke,
            learning=None,
            brain_snapshot={
                "synapses": org.brain.synapse_count,
                "mean_weight": org.brain.mean_synapse_weight(),
            },
        )
        self._maybe_persist()
        if text or understood or spoke or code_out:
            self._persist()
        return {"moment": moment.to_dict(), "stats": org.stats}

    def conscious_flow(
        self,
        *,
        image_gray: list[int] | None = None,
        image_b64: str | None = None,
        image_w: int = 64,
        image_h: int = 64,
        color_rgb: dict[str, float] | None = None,
        image_rgba: list[int] | None = None,
    ) -> dict[str, Any]:
        """Flusso continuo — percepisce e pensa in silenzio; parla solo se il caregiver parla."""
        return self.perceive_vision(
            image_gray=image_gray,
            image_b64=image_b64,
            image_w=image_w,
            image_h=image_h,
            color_rgb=color_rgb,
            image_rgba=image_rgba,
        )

    def flow(
        self,
        *,
        image_gray: list[int] | None = None,
        image_b64: str | None = None,
        image_w: int = 64,
        image_h: int = 64,
        color_rgb: dict[str, float] | None = None,
        image_rgba: list[int] | None = None,
    ) -> dict[str, Any]:
        """Un passo di coscienza — percezione silenziosa, niente speech spontaneo."""
        return self.conscious_flow(
            image_gray=image_gray,
            image_b64=image_b64,
            image_w=image_w,
            image_h=image_h,
            color_rgb=color_rgb,
            image_rgba=image_rgba,
        )

    def autonomous_tick(self) -> dict[str, Any]:
        """Alias del flusso silenzioso — niente balbettio autonomo."""
        return self.conscious_flow()


def normalize_dialogue_key(when: str) -> str:
    import hashlib

    return hashlib.sha256(when.strip().lower().encode()).hexdigest()[:12]
