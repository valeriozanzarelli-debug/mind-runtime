"""Baby agent — voce e crescita emergono dal grafo neurale."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from organism.autonomous.baby_store import BabyStore, baby_state_path
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
    wire_dialogue_pathway,
)
from organism.cognition.visual_pathway import prime_visual_percept, wire_visual_association
from organism.cognition.self_model import SelfModel
from organism.cognition.consciousness_stream import (
    build_consciousness_stream,
    consciousness_events_from_log,
    is_question,
)
from organism.cognition.theme_coherence import coherent_themes
from organism.cognition.self_learn import SelfLearner
from organism.cognition.growth import BrainGrowth
from organism.cognition.tasks import TaskKind, TaskRunner
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
from organism.autonomous.thought_generator import ThoughtGenerator
from organism.sensory.vision_sense import VisionSense
from organism.cognition.human_thought import (
    compose_inner_voice,
    enrich_consciousness_lines,
    thought_coherence,
)
from organism.dna.evolution import DNAEvolver
from organism.cognition.motor_plan import MotorPlan
from organism.cognition.speech_loop import SpeechSensorimotorLoop
from organism.cognition.thought import ThoughtEngine
from organism.sensory.social_tone import SocialTone, analyze_social_tone
from organism.teaching.correction import CorrectionLearner
from organism.motor.code_emergent import EmergentCodeMotor
from organism.motor.compose_speech import SpeechComposer
from organism.motor.emergent_speech import EmergentSpeechMotor
from organism.nursery.journal import ThoughtJournal
from organism.research.senses import research_human_senses
from organism.runtime import OrganismRuntime
from organism.sensory.visual_scene import scene_features, scene_signature
from organism.sensory.web_media import (
    audio_hash,
    decode_audio_b64,
    decode_image_gray,
    decode_image_pixels,
    vision_hash,
)
from organism.teaching.dialogue import DialogueTeacher
from organism.teaching.repetition import RepetitionTeacher

TZ = ZoneInfo("Europe/Madrid")
_COLOR_WORDS = frozenset(
    {"rosso", "verde", "blu", "giallo", "nero", "bianco", "grigio", "rosa", "arancione", "marrone"}
)


def _tokens_from_heard(heard: str) -> list[str]:
    import re

    stop = frozenset({"che", "chi", "come", "cosa", "il", "la", "un", "una", "di", "è", "era", "sono", "sei"})
    return [w for w in re.findall(r"[a-zàèéìòù']+", heard.lower()) if len(w) > 2 and w not in stop]


@dataclass
class BabyMoment:
    impulse: str
    spoke: str
    stimulus_key: str
    curiosity: dict[str, Any]
    learned: bool
    brain: dict[str, Any]
    thought: dict[str, Any]
    wanted_to_speak: bool = False
    code: str = ""
    understood: bool = False
    from_thought: bool = False
    self_heard: bool = False
    speech_error: dict[str, Any] = field(default_factory=dict)
    consciousness: dict[str, Any] = field(default_factory=dict)
    self_state: dict[str, Any] = field(default_factory=dict)
    wave: dict[str, Any] = field(default_factory=dict)
    dream: dict[str, Any] = field(default_factory=dict)
    emotion: dict[str, Any] = field(default_factory=dict)
    social_tone: dict[str, Any] = field(default_factory=dict)
    presence: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    consciousness_stream: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "impulse": self.impulse,
            "spoke": self.spoke,
            "code": self.code,
            "understood": self.understood,
            "from_thought": self.from_thought,
            "self_heard": self.self_heard,
            "speech_error": self.speech_error,
            "consciousness": self.consciousness,
            "self": self.self_state,
            "wave": self.wave,
            "dream": self.dream,
            "emotion": self.emotion,
            "social_tone": self.social_tone,
            "presence": self.presence,
            "task": self.task,
            "consciousness_stream": self.consciousness_stream,
            "thought": self.thought,
            "stimulus_key": self.stimulus_key,
            "curiosity": self.curiosity,
            "learned": self.learned,
            "brain": self.brain,
            "wanted_to_speak": self.wanted_to_speak,
            "symbols": self.symbols,
        }


class BabyAgent:
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
        self.working_memory = WorkingMemory(capacity=7)
        self.photo_memory = PhotographicMemory(capacity=2000)
        self.episodic_memory = EpisodicMemory(short_slots=32, long_capacity=1200)
        self.visual_imagination = VisualImagination(self.photo_memory)
        self.goal_stack = GoalStack()
        self.hypothesis_engine = HypothesisEngine()
        self.thought_generator = ThoughtGenerator(seed=seed)
        self.vision_sense = VisionSense()
        self.dna_evolver = DNAEvolver.load()
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
        loaded = self.store.load(self)
        self._resume_meta = loaded if loaded.get("loaded") else None

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
        if not themes:
            return
        line = f"pensiero spontaneo: {' '.join(themes[:6])}"
        if self._consciousness_log and self._consciousness_log[-1] == line:
            return
        self._append_consciousness([line])
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
        stats = self.narrator.bootstrap_curriculum(repeats=3)
        return {"syntax": stats, "narrator": self.narrator.stats()}

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
    ) -> tuple[bool, str | None, list[str]]:
        """Attiva percorsi sinaptici appresi — hint lessicali, non copia scriptata."""
        has_path = False
        code_out: str | None = None
        pathway_words: list[str] = []
        if text and self.speech:
            say, kind = self.dialogue.respond(text)
            if say and kind == "code":
                return True, say, []
            if say:
                for w in _tokens_from_heard(text):
                    self.composer.absorb(w, boost=0.35)
                self.speech.hear(text, boost=0.45)
                pathway_words = _tokens_from_heard(say)
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
        return has_path, code_out, pathway_words

    def _visual_themes(self) -> list[str]:
        if not self._vision_fresh or not self._last_vision_hash:
            return []
        return self.visual_binder.themes(
            self._last_vision_hash, self._last_visual_features, limit=10
        )

    def _match_scene_phrase(self, text: str) -> str | None:
        from organism.teaching.dialogue import _content_words, normalize_text

        words = _content_words(normalize_text(text))
        if not words:
            return None
        best: str | None = None
        best_score = 0
        need = max(1, len(words)) if len(words) <= 2 else 2
        for phrase in self.teacher.all_learned().values():
            pw = _content_words(normalize_text(phrase))
            overlap = len(words & pw)
            if overlap >= need and overlap > best_score:
                best_score = overlap
                best = phrase
        return best

    def _decode_vision(
        self,
        *,
        image_gray: list[int] | None,
        image_b64: str | None,
        image_w: int,
        image_h: int,
        color_rgb: dict[str, float] | None = None,
        image_rgba: list[int] | None = None,
    ) -> tuple[list[list[int]], str]:
        from organism.sensory.web_media import decode_rgba_flat
        from organism.sensory.ventral_vision import process_ventral_stream

        grid: list[list[int]] = []
        rgb_grid: list[list[tuple[int, int, int]]] | None = None
        if image_rgba and len(image_rgba) >= image_w * image_h * 4:
            grid, rgb_grid = decode_rgba_flat(image_rgba, image_w, image_h)
        elif image_gray and len(image_gray) >= image_w * image_h * 4:
            grid, rgb_grid = decode_rgba_flat(image_gray, image_w, image_h)
        elif image_gray:
            grid = decode_image_gray(image_gray, image_w, image_h)
        elif image_b64:
            grid = decode_image_pixels(image_b64, image_w, image_h)
        else:
            return [], ""
        vs = self.vision_sense.process(grid, rgb_grid=rgb_grid)
        if vs.get("skipped") and self._last_visual_features:
            return grid, self._last_vision_hash or str(vs.get("sig", ""))
        sig = str(vs.get("sig") or scene_signature(grid))
        base = dict(vs.get("features") or scene_features(grid))
        ventral = process_ventral_stream(grid, rgb_grid=rgb_grid, color_rgb=color_rgb)
        feat = dict(ventral.get("features", {}))
        gist = feat.get("gist", {})
        if gist:
            gist["luminance"] = base.get("luminance", gist.get("luminance", 0))
            gist["contrast"] = base.get("contrast", gist.get("contrast", 0))
            feat["luminance"] = gist["luminance"]
            feat["contrast"] = gist["contrast"]
        symbols = list(ventral.get("symbols", []))
        face_data = self._analyze_face_on_grid(grid, rgb_grid=rgb_grid)
        if face_data.get("detected"):
            symbols.extend(face_data.get("symbols", []))
        self._last_visual_features = {
            **base,
            **feat,
            "symbols": symbols,
            "object_sig": ventral.get("object_sig", feat.get("object_sig", "")),
            "face": face_data,
        }
        thumb = [grid[y][x] for y in range(min(len(grid), 16)) for x in range(min(len(grid[0]) if grid else 0, 16))]
        labels = [
            w
            for w in self.visual_binder.themes(sig, self._last_visual_features, limit=4)
            if not str(w).startswith(("VIS:", "OBJ:", "COL:"))
        ]
        self.photo_memory.capture(
            object_sig=str(self._last_visual_features.get("object_sig", sig)),
            features=self._last_visual_features,
            labels=labels,
            thumb_gray=thumb,
        )
        return grid, sig

    def _analyze_face_on_grid(
        self,
        grid: list[list[int]],
        *,
        rgb_grid: list[list[tuple[int, int, int]]] | None = None,
    ) -> dict[str, Any]:
        from organism.sensory.face_vision import analyze_face

        return analyze_face(grid, rgb_grid=rgb_grid)

    def _face_context(self) -> dict[str, Any]:
        feat = self._last_visual_features
        face = feat.get("face") or {}
        if not face.get("detected"):
            return {"detected": False}
        return {
            "detected": True,
            "face": self.face_binder.recognize_face(face),
            "emotion": self.face_binder.recognize_emotion(face)
            or self.face_binder.infer_emotion_hint(face),
            "score": face.get("face_score", 0),
        }

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
        arousal = inject_circadian(org.brain)
        if impulse == "vocalize":
            for n in self.speech.phoneme_neurons[:6]:
                n.activation = min(1.0, n.activation + 0.08 * arousal)
        org.brain.propagate(steps=1)

        grown = org.brain.synapse_count - self._synapses_at_birth
        wave_phase = self.waves.last.phase
        vis_themes = self._visual_themes()
        has_path, code_out, pathway_words = self._prime_learned_pathways(
            org, skey, heard, vision_fresh=self._vision_fresh
        )
        if not code_out and heard:
            hl = heard.lower()
            if any(k in hl for k in ("programma", "codice", "scrivi", "python", "funzione")):
                code_out = self.code.produce(heard) or None

        unknown_words: list[str] = []
        if heard and not has_path and not code_out:
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
        wants_voice = (
            pres.speaks
            or em.will_speak
            or thought.pressure > 0.1
            or has_path
            or impulse in ("vocalize", "ask")
        )
        if heard and heard.strip():
            wants_voice = True
        if is_question(heard or ""):
            wants_voice = True
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
                memory_themes=memory_themes,
                valence=affect_valence,
            )
            if not composed.text and em.will_speak and not ask_mode:
                plan = self.speech.utter_with_plan()
                if plan.text and len(plan.text) > 4:
                    composed = ComposedSpeech(
                        plan.text,
                        "speech",
                        True,
                        thought.themes[:4],
                        plan,
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
        self._last_presence = pres.to_dict()
        self._last_tone = tone.to_dict()
        thought_d = thought.to_dict()
        ws_d = ws.to_dict()
        ep_ctx = self.episodic_memory.recall_context(heard or "", limit=1)
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
        return self.speech_loop.self_hear(
            org.brain,
            self.speech,
            heard_text=spoke,
            plan=mp,
            source="self",
        )

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

    def correct_speech(self, *, heard: str, correct: str) -> dict[str, Any]:
        """Correzione caregiver — penalizza mismatch fonetico."""
        org = self._ensure()
        if not self.speech:
            return {"ok": False}
        from organism.teaching.phonemes import split_italian_syllables

        wrong_syl = split_italian_syllables(heard)
        right_syl = split_italian_syllables(correct)
        penalized = self.speech.phonemes.penalize_mismatch(wrong_syl, right_syl, penalty=0.03)
        reinforced = self.speech.phonemes.reinforce_sequence(right_syl, boost=0.4)
        self.speech.hear(correct, boost=1.0)
        self.composer.absorb(correct, boost=0.8)
        r = self.dialogue.teach(heard.strip()[:80], correct.strip())
        org.perceive({"text": correct})
        org.brain.propagate(steps=1)
        self._persist()
        return {
            "ok": True,
            "penalized": penalized,
            "reinforced": reinforced,
            "dialogue": r,
        }

    def brain_pulse_tick(self, *, persist: bool = False) -> dict[str, Any]:
        """Cervello ad onde — percepisce, pensa, sogna, riflette senza corpo."""
        if not self._born or self.org is None:
            return {"alive": False}
        org = self.org
        idle = time.time() - self._last_sense_t
        if self.affect._pulse_count % 45 == 0:
            inject_circadian(org.brain)

        wave = self.waves.advance(idle=idle > 10, arousal=self.affect.state.curiosity)
        inject_wave(org.brain, wave.phase, tick=wave.tick, amplitude=wave.amplitude)
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

        if self.affect._pulse_count % 20 == 0:
            wire_self_recurrence(org.brain)

        for n in list(org.brain.neurons.values())[::3]:
            n.activation = max(0.01, n.activation * 0.992)
        if org.brain.plasticity and self.affect._pulse_count % 15 == 0:
            org.brain.plasticity.apply_hebbian(org.brain, org.brain.tick)

        growth = self.brain_growth.maybe_grow(org.brain, pulses=self.affect._pulse_count)

        self.self_model.update(
            org.brain,
            org.memory,
            wave=wave,
            affect=self.affect.state,
            dream_fragment=dream_st.content if dream_st.active else "",
            saw=self._last_vision_hash,
        )

        if persist:
            self._persist()
        return {
            "alive": True,
            "brain_tick": org.brain.tick,
            "wave": wave.to_dict(),
            "self": self.self_model.state.to_dict(),
            "dream": dream_st.to_dict(),
            "emotion": self.affect.state.to_dict(),
            "pulses": self.affect._pulse_count,
            "idle_s": round(idle, 1),
            "growth": growth,
            "neurons": org.brain.neuron_count,
        }

    def _self_listen(
        self,
        org: OrganismRuntime,
        spoke: str,
        plan: MotorPlan | None,
        *,
        source: str = "self",
    ) -> dict[str, Any]:
        if not spoke.strip():
            return {}
        org.perceive({"text": spoke})
        org.brain.propagate(steps=1)
        return self.speech_loop.self_hear(
            org.brain,
            self.speech,
            heard_text=spoke,
            plan=plan,
            source=source,  # type: ignore[arg-type]
        )

    def _normalize_phrase(self, text: str) -> str:
        import re

        t = re.sub(r"[.?!,]", " ", text.lower())
        return re.sub(r"\s+", " ", t).strip()

    def _is_own_echo(self, phrase: str, *, within_s: float = 14.0) -> bool:
        """Riconosce la propria voce — evita dialogo con sé stesso."""
        p = self._normalize_phrase(phrase)
        if not p or time.time() - self._last_spoke_wall_t > within_s:
            return False
        own = self._normalize_phrase(self._last_baby_spoke)
        if not own:
            return False
        pw = {w for w in p.split() if len(w) > 2}
        ow = {w for w in own.split() if len(w) > 2}
        # Single-word inputs are rarely genuine echoes; only trigger for exact-match
        # within a very short window (TTS latency), not for substring matches.
        if len(pw) < 2:
            return time.time() - self._last_spoke_wall_t < 3.0 and p == own
        if p == own or p in own or own in p:
            return True
        if pw and ow:
            overlap = len(pw & ow) / max(len(pw), len(ow))
            if overlap >= 0.55:
                return True
        for recent in self._recent_spokes[-4:]:
            rw = self._normalize_phrase(recent)
            if not rw:
                continue
            if p == rw or p in rw or rw in p:
                return True
            rset = {w for w in rw.split() if len(w) > 2}
            if pw and rset and len(pw & rset) / max(len(pw), len(rset)) >= 0.6:
                return True
        return False

    def self_hear(self, *, text: str) -> dict[str, Any]:
        """Chiusura loop uditivo — browser TTS o eco interna."""
        org = self._ensure()
        text = text.strip()
        if not text:
            return {"feedback": {}, "moment": None}
        plan = self.speech.plan_from_text(text) if self.speech else None
        fb = self._self_listen(org, text, plan, source="self")
        self.composer.absorb(text, boost=0.25)
        err = fb.get("speech_error") or {}
        self._append_consciousness([f"sé: ho detto «{text[:48]}» (sim {err.get('similarity', 0):.0%})"])
        if self._last_moment:
            self._last_moment.self_heard = True
            self._last_moment.speech_error = err
        elif self._last_baby_spoke:
            self._last_moment = BabyMoment(
                impulse="reflect",
                spoke="",
                stimulus_key="",
                curiosity=self.curiosity.state.to_dict(),
                learned=False,
                brain={},
                thought={"themes": [], "pressure": 0.2},
                self_heard=True,
                speech_error=err,
            )
        self._persist()
        return {"feedback": fb, "moment": self._last_moment.to_dict() if self._last_moment else None}

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

    def flow(self) -> dict[str, Any]:
        """Un passo di coscienza — senza loop forzato, solo quando invocato."""
        return self.autonomous_tick()

    def autonomous_tick(self) -> dict[str, Any]:
        org = self._ensure()
        idle = time.time() - self._last_sense_t
        self.curiosity.tick_idle(idle)
        impulse = self._choose_impulse(org)
        skey = self.curiosity.state.last_stimulus_key or "idle"
        spoke, code_out, thought_d, ws_d, mood, understood, from_thought, plan, stream = self._act(
            org, heard=None, impulse=impulse, long_form=False, skey=skey
        )

        moment = BabyMoment(
            impulse=impulse,
            spoke=spoke,
            code=code_out or "",
            understood=from_thought,
            from_thought=from_thought,
            self_heard=False,
            speech_error={},
            consciousness=ws_d,
            consciousness_stream=stream,
            self_state=self.self_model.state.to_dict(),
            wave=self.waves.last.to_dict(),
            dream=self.dream_engine.state.to_dict(),
            emotion=self.affect.state.to_dict(),
            social_tone={},
            presence=self._last_presence,
            thought=thought_d,
            stimulus_key=skey,
            curiosity=self.curiosity.state.to_dict(),
            learned=from_thought,
            brain=mood.to_dict(),
            wanted_to_speak=mood.wants_voice and not spoke,
            symbols=[f"AUTO:{impulse}", f"EMO:{self.affect.state.dominant}"]
            + thought_d.get("symbols", [])[:4],
        )
        self._last_moment = moment
        return {"moment": moment.to_dict(), "stats": org.stats}

    def reflect(self, *, prompt: str = "") -> dict[str, Any]:
        """Riflessione autonoma — coscienza decide se pensare o parlare."""
        org = self._ensure()
        self._last_sense_t = time.time()
        heard = prompt.strip() or None
        if heard:
            org.perceive({"text": heard})
            org.brain.propagate(steps=2)
            self.speech.hear(heard, boost=0.7)
            self.composer.absorb(heard, boost=0.5)
        else:
            inject_wave(org.brain, "reflect", tick=int(org.brain.tick), amplitude=0.8)
        impulse = "vocalize"
        spoke, code_out, thought_d, ws_d, mood, _, from_thought, plan, stream = self._act(
            org,
            heard=heard,
            impulse=impulse,
            long_form=bool(heard),
            skey="reflect",
        )
        moment = BabyMoment(
            impulse=impulse,
            spoke=spoke,
            code=code_out or "",
            understood=from_thought or bool(spoke),
            from_thought=from_thought,
            self_heard=False,
            speech_error={},
            consciousness=ws_d,
            consciousness_stream=stream,
            self_state=self.self_model.state.to_dict(),
            wave=self.waves.last.to_dict(),
            dream=self.dream_engine.state.to_dict(),
            emotion=self.affect.state.to_dict(),
            social_tone={},
            presence=self._last_presence,
            thought=thought_d,
            stimulus_key="reflect",
            curiosity=self.curiosity.state.to_dict(),
            learned=True,
            brain=mood.to_dict(),
            wanted_to_speak=False,
            symbols=thought_d.get("symbols", []),
        )
        self._last_moment = moment
        self._persist()
        return {"moment": moment.to_dict(), "stats": org.stats}

    def _is_stuck_repeat(self, candidate: str) -> bool:
        c = candidate.lower().strip()
        if not c or len(c) < 4:
            return False
        hits = sum(1 for s in self._recent_spokes if s and (s == c or c in s or s in c))
        return hits >= 2

    def _anti_repeat_speech(
        self,
        spoke: str,
        *,
        org: OrganismRuntime,
        heard: str | None,
        ask_mode: bool,
    ) -> str:
        if not spoke or ask_mode:
            return spoke
        if not self._is_stuck_repeat(spoke):
            return spoke
        if self._vision_fresh and self.speech:
            alt = self._visual_utterance(org)
            if alt and not self._is_stuck_repeat(alt):
                return alt
        if heard and not ask_mode:
            uw = self.self_learner.detect_unknown(heard, has_pathway=False)
            if uw and self.speech:
                from organism.cognition.thought import Thought

                t = Thought(
                    symbols=["IMPULSE:ask", f"UNKNOWN:{uw[0]}"],
                    themes=["non", "so", uw[0]],
                    pressure=0.55,
                )
                q = self.composer._from_curiosity_question(t, self.speech)
                if q:
                    return q
        return spoke

    def _visual_utterance(self, org: OrganismRuntime) -> str:
        if not self.speech or not self._vision_fresh:
            return ""
        feat = self._last_visual_features
        rec = self.visual_binder.recognize_object(feat, min_sim=0.46)
        if not rec:
            return ""
        themes = self.visual_binder.speech_themes(
            feat,
            question="vision",
            articulable=lambda w: self.composer.lexicon.is_articulable(w, min_exposure=0.25),
        )
        if not themes:
            themes = ["vedo", rec]
        prime_visual_percept(org.brain, self.speech, self.composer.lexicon, themes=themes, boost=0.55)
        plan = self.speech.lexical_readout(themes[:5], punct=".")
        return plan.text

    def _pulse_amygdala(self, org: OrganismRuntime) -> None:
        self.affect.state.curiosity = min(
            1.0,
            0.65 * self.affect.state.curiosity + 0.35 * self.curiosity.state.level,
        )
        if self.amygdala.state.play > 0.5:
            self.affect.state.joy = min(1.0, self.affect.state.joy + 0.02)
        self.amygdala.pulse(
            org.brain,
            joy=self.affect.state.joy,
            fear=self.affect.state.fear,
            shame=self.affect.state.shame,
            anger=self.affect.state.anger,
            trust=self.affect.state.trust,
            curiosity=self.affect.state.curiosity,
            curiosity_drive=self.curiosity.state,
        )

    def _choose_impulse(self, org: OrganismRuntime) -> str:
        self._pulse_amygdala(org)
        return self.curiosity.choose_impulse(amygdala=self.amygdala)

    def _ensure_ask_speech(
        self,
        org: OrganismRuntime,
        *,
        themes: list[str],
        symbols: list[str],
        focus: str,
    ) -> str:
        """Garantisce output motorio su IMPULSE:ask — solo lessico appreso + balbettio."""
        if not self.speech:
            return ""
        from organism.cognition.thought import Thought

        self._pulse_amygdala(org)
        t = Thought(themes=themes, symbols=symbols, pressure=0.55, memory_hits=0)
        spoke = self.composer._from_curiosity_question(t, self.speech, amygdala=self.amygdala)
        if spoke:
            return spoke
        active = self.composer.lexicon.active_words(8, min_act=0.05)
        if active:
            plan = self.speech.lexical_readout(active, punct="?")
            if plan.text:
                return plan.text
        babble = self.speech.utter_with_plan()
        if babble.text:
            txt = babble.text.strip().rstrip(".!")
            return f"{txt}?" if not txt.endswith("?") else txt
        if focus and self.composer.lexicon.is_articulable(focus, min_exposure=0.15):
            plan = self.speech.lexical_readout([focus], punct="?")
            return plan.text
        return ""

    def _vision_attend(
        self,
        org: OrganismRuntime,
        *,
        image_gray: list[int] | None = None,
        image_b64: str | None = None,
        image_w: int = 64,
        image_h: int = 64,
        color_rgb: dict[str, float] | None = None,
        image_rgba: list[int] | None = None,
    ) -> dict[str, Any] | None:
        self._last_sense_t = time.time()
        self._vision_fresh = bool(image_gray or image_b64 or image_rgba)
        grid, sig = self._decode_vision(
            image_gray=image_gray,
            image_b64=image_b64,
            image_w=image_w,
            image_h=image_h,
            color_rgb=color_rgb,
            image_rgba=image_rgba,
        )
        if not grid:
            return None
        self._last_vision_hash = sig
        org.perceive({"image": grid, "width": image_w, "height": image_h})
        org.brain.propagate(steps=2)
        self._wire_from_input(org, had_text=False, had_vision=True)
        feat = self._last_visual_features
        rec = self.visual_binder.recognize_object(feat, min_sim=0.42)
        conf = self.visual_binder.confidence(feat)
        if rec and conf < 0.52:
            rec = None
        impulse = "vocalize"
        symbols: list[str] = []
        themes: list[str] = []
        focus = ""
        if rec:
            symbols.append(f"OBJ:name={rec}")
            themes = self.visual_binder.speech_themes(
                feat,
                question="vision",
                articulable=lambda w: self.composer.lexicon.is_articulable(w, min_exposure=0.25),
            )
            prime_visual_percept(org.brain, self.speech, self.composer.lexicon, themes=themes, boost=0.6)
        else:
            impulse = "ask"
            symbols.append("IMPULSE:ask")
            symbols.append("UNKNOWN:scene")
            percept = self.visual_binder.percept_words(feat)
            for pw in percept:
                if self.composer.lexicon.is_articulable(pw, min_exposure=0.2):
                    symbols.append(f"UNKNOWN:{pw}")
                    focus = pw
                    break
            prime_curiosity_pathway(
                org.brain,
                self.speech,
                self.composer.lexicon,
                focus=focus,
                heard="",
                boost=0.5,
            )
            theme_pool = self.composer.lexicon.active_words(8, min_act=0.05)
            for w in percept:
                if w not in theme_pool:
                    theme_pool.append(w)
            themes = self.composer.lexicon.ranked(theme_pool)[:6]
            self.curiosity.observe(
                stimulus_key_visual_context(vision_hash=sig),
                pattern_gap=True,
                learned=False,
            )
            self.affect.state.curiosity = min(1.0, self.affect.state.curiosity + 0.12)
            self._pulse_amygdala(org)
        return {
            "sig": sig,
            "feat": feat,
            "rec": rec,
            "conf": conf,
            "impulse": impulse,
            "symbols": symbols,
            "themes": themes,
            "focus": focus,
        }

    def _vision_moment(
        self,
        org: OrganismRuntime,
        attended: dict[str, Any],
        *,
        spoke: str,
    ) -> BabyMoment:
        impulse = str(attended["impulse"])
        themes = list(attended["themes"])
        symbols = list(attended["symbols"])
        feat = attended["feat"]
        rec = attended["rec"]
        sig = attended["sig"]
        if spoke:
            self.speech.hear(spoke, boost=0.4)
            org.perceive({"text": spoke})
            self.composer.absorb(spoke, boost=0.35)
            self._last_baby_spoke = spoke
            self._recent_spokes.append(spoke.lower().strip())
            self._recent_spokes = self._recent_spokes[-8:]
        em = self.speech.readout()
        arousal = inject_circadian(org.brain)
        moment = BabyMoment(
            impulse=impulse,
            spoke=spoke,
            stimulus_key=stimulus_key_visual_context(vision_hash=sig),
            curiosity=self.curiosity.state.to_dict(),
            learned=bool(rec),
            brain=read_brain_mood(
                org.brain,
                synapses_at_birth=self._synapses_at_birth,
                motor_pressure=em.motor_pressure,
                inhibition=em.inhibition,
                wants_voice=bool(spoke),
                arousal=arousal,
            ).to_dict(),
            thought={"themes": themes, "symbols": symbols, "pressure": 0.5},
            wanted_to_speak=bool(spoke),
            understood=bool(rec),
            from_thought=True,
            symbols=symbols + list(feat.get("symbols", []))[:6],
        )
        self._last_moment = moment
        self._maybe_persist()
        return moment

    def hear_spoken(
        self,
        phrase: str,
        *,
        teach_focus: bool = False,
        source: str = "caregiver",
        image_gray: list[int] | None = None,
        image_b64: str | None = None,
        image_w: int = 64,
        image_h: int = 64,
        color_rgb: dict[str, float] | None = None,
        image_rgba: list[int] | None = None,
    ) -> dict[str, Any]:
        """Udito manuale — caregiver parla; eco propria → auto-ascolto e miglioramento."""
        from organism.teaching.vision_phrase import parse_vision_teaching

        phrase = phrase.strip()
        if not phrase:
            return {"moment": None, "reason": "empty"}
        if source == "self" or self._is_own_echo(phrase):
            fb = self.self_hear(text=phrase)
            return {
                "heard": phrase,
                "mode": "self_feedback",
                "feedback": fb.get("feedback", {}),
                "moment": fb.get("moment"),
            }
        parsed = parse_vision_teaching(phrase)
        vision_kw = dict(
            image_gray=image_gray,
            image_b64=image_b64,
            image_w=image_w,
            image_h=image_h,
            color_rgb=color_rgb,
            image_rgba=image_rgba,
        )
        if teach_focus or (parsed.has_object and (image_gray or image_b64 or image_rgba)):
            result = self.teach_attention(phrase, **vision_kw)
            moment = self._last_moment.to_dict() if self._last_moment else None
            return {**result, "moment": moment, "heard": phrase}
        return self.sense(text=phrase, **vision_kw)

    def teach_attention(
        self,
        phrase: str,
        *,
        image_gray: list[int] | None = None,
        image_b64: str | None = None,
        image_w: int = 64,
        image_h: int = 64,
        color_rgb: dict[str, float] | None = None,
        image_rgba: list[int] | None = None,
    ) -> dict[str, Any]:
        """Manina + occhi — frase naturale mentre mostri l'oggetto."""
        from organism.teaching.vision_phrase import parse_vision_teaching

        parsed = parse_vision_teaching(phrase)
        if not parsed.has_object:
            return {
                **self.teach_repetition(
                    phrase,
                    image_gray=image_gray,
                    image_b64=image_b64,
                    image_w=image_w,
                    image_h=image_h,
                ),
                "mode": "phrase",
            }
        if parsed.color:
            self.composer.absorb(parsed.color, boost=1.0)
        say = parsed.phrase
        if not say:
            say = f"vedo {parsed.object_name}"
            if parsed.color:
                say += f" {parsed.color}"
        result = self.teach_object(
            parsed.object_name,
            image_gray=image_gray,
            image_b64=image_b64,
            image_w=image_w,
            image_h=image_h,
            color_rgb=color_rgb,
            image_rgba=image_rgba,
            phrase=say,
        )
        if parsed.color:
            cp = f"è {parsed.color}"
            self.dialogue.teach(f"che colore è il {parsed.object_name}", cp)
            wire_dialogue_pathway(
                self._ensure().brain,
                self.speech,
                when=f"che colore è il {parsed.object_name}",
                say=cp,
            )
            self.composer.absorb(cp, boost=0.8)
        result["mode"] = "vision_object"
        result["parsed"] = {
            "object": parsed.object_name,
            "color": parsed.color,
            "phrase": say,
        }
        return result

    def look(
        self,
        *,
        image_gray: list[int] | None = None,
        image_b64: str | None = None,
        image_w: int = 64,
        image_h: int = 64,
        color_rgb: dict[str, float] | None = None,
        image_rgba: list[int] | None = None,
    ) -> dict[str, Any]:
        """Solo occhi — richiami attenzione, riconosce o chiede cos'è."""
        org = self._ensure()
        attended = self._vision_attend(
            org,
            image_gray=image_gray,
            image_b64=image_b64,
            image_w=image_w,
            image_h=image_h,
            color_rgb=color_rgb,
            image_rgba=image_rgba,
        )
        if not attended:
            return {"moment": None, "reason": "no_image"}
        rec = attended["rec"]
        conf = attended["conf"]
        impulse = attended["impulse"]
        themes = attended["themes"]
        symbols = attended["symbols"]
        focus = attended["focus"]
        feat = attended["feat"]

        spoke = self._visual_utterance(org) if rec else ""
        if not spoke and impulse == "ask":
            spoke = self._ensure_ask_speech(org, themes=themes, symbols=symbols, focus=str(focus))
        elif not spoke and self.speech:
            plan = self.speech.lexical_readout(themes[:5], punct=".")
            spoke = plan.text

        moment = self._vision_moment(org, attended, spoke=spoke)
        return {
            "moment": moment.to_dict(),
            "recognized": rec,
            "confidence": round(conf, 2),
            "features": feat,
        }

    def glance(
        self,
        *,
        image_gray: list[int] | None = None,
        image_b64: str | None = None,
        image_w: int = 64,
        image_h: int = 64,
        color_rgb: dict[str, float] | None = None,
        image_rgba: list[int] | None = None,
        min_interval_s: float = 10.0,
    ) -> dict[str, Any]:
        """Sguardo passivo — scena nuova: riconosce o chiede da solo (telefono in tasca/mano)."""
        now = time.time()
        if now - self._last_glance_t < min_interval_s:
            return {"moment": None, "skipped": "rate_limit"}

        org = self._ensure()
        self._vision_fresh = bool(image_gray or image_b64 or image_rgba)
        grid, sig = self._decode_vision(
            image_gray=image_gray,
            image_b64=image_b64,
            image_w=image_w,
            image_h=image_h,
            color_rgb=color_rgb,
            image_rgba=image_rgba,
        )
        if not grid:
            return {"moment": None, "skipped": "no_image"}
        if sig == self._last_glance_sig:
            return {"moment": None, "skipped": "same_scene", "scene_sig": sig}

        self._last_glance_t = now
        self._last_glance_sig = sig

        attended = self._vision_attend(
            org,
            image_gray=image_gray,
            image_b64=image_b64,
            image_w=image_w,
            image_h=image_h,
            color_rgb=color_rgb,
            image_rgba=image_rgba,
        )
        if not attended:
            return {"moment": None, "skipped": "no_image"}

        rec = attended["rec"]
        conf = attended["conf"]
        impulse = attended["impulse"]
        themes = attended["themes"]
        symbols = attended["symbols"]
        focus = attended["focus"]
        feat = attended["feat"]

        spoke = ""
        if rec:
            spoke = self._visual_utterance(org)
            if spoke and self._is_stuck_repeat(spoke):
                spoke = ""
        if not spoke and impulse == "ask":
            spoke = self._ensure_ask_speech(org, themes=themes, symbols=symbols, focus=str(focus))

        if not spoke:
            return {
                "moment": None,
                "skipped": "silent",
                "recognized": rec,
                "confidence": round(conf, 2),
                "scene_sig": sig,
            }

        moment = self._vision_moment(org, attended, spoke=spoke)
        fc = self._face_context()
        return {
            "moment": moment.to_dict(),
            "recognized": rec,
            "confidence": round(conf, 2),
            "features": feat,
            "scene_sig": sig,
            "novel": True,
            "face": fc,
        }

    def teach_from_web_image(
        self,
        url: str,
        *,
        name: str,
        phrase: str | None = None,
        kind: str = "object",
        image_prefetch: dict[str, Any] | None = None,
        hd_size: int = 256,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Insegna da immagine internet — oggetto, volto o emozione (HD)."""
        from organism.teaching.web_fetch import fetch_image_rgba

        name = name.strip().lower()
        if not url or not name:
            return {"ok": False, "reason": "missing_url_or_name"}
        try:
            img = image_prefetch or fetch_image_rgba(url, size=hd_size)
        except Exception as e:
            return {"ok": False, "reason": str(e)[:120], "url": url[:80]}

        face_data = {}
        grid, sig = self._decode_vision(
            image_gray=img.get("image_gray"),
            image_b64=None,
            image_rgba=img.get("image_rgba"),
            image_w=int(img.get("image_w", hd_size)),
            image_h=int(img.get("image_h", hd_size)),
            color_rgb=img.get("color_rgb"),
        )
        face_data = self._last_visual_features.get("face") or {}
        self.photo_memory.capture(
            object_sig=str(self._last_visual_features.get("object_sig", "")),
            features=self._last_visual_features,
            labels=[name],
            thumb_gray=(img.get("image_gray") or [])[:256],
        )

        say = phrase or f"vedo {name}"
        result: dict[str, Any] = {"ok": True, "name": name, "kind": kind, "url": url[:120]}

        if kind == "emotion":
            emo_r = self.face_binder.teach_emotion(name, face_data)
            say = phrase or f"questa espressione è {name}"
            self.composer.absorb(f"{name} emozione volto {say}", boost=1.1)
            self.dialogue.teach(f"che emozione è", f"è {name}")
            result.update(emo_r)
        elif kind == "face":
            face_r = self.face_binder.teach_face(name, face_data)
            say = phrase or f"questo è un {name}"
            self.composer.absorb(f"volto {name} {say}", boost=1.0)
            result.update(face_r)
        else:
            obj_r = self.teach_object(
                name,
                image_gray=img.get("image_gray"),
                image_rgba=img.get("image_rgba"),
                image_w=int(img.get("image_w", hd_size)),
                image_h=int(img.get("image_h", hd_size)),
                color_rgb=img.get("color_rgb"),
                phrase=say,
            )
            result.update(obj_r)

        if kind in ("face", "emotion") and self.speech:
            org = self._ensure()
            self.speech.hear(say, boost=0.9)
            org.perceive({"text": say})
            org.brain.propagate(steps=1)
            self._wire_from_input(org, had_text=True, had_vision=True, teach=True)

        if persist:
            self._persist()
        result["phrase"] = say
        result["face_detected"] = bool(face_data.get("detected"))
        return result

    def run_web_curriculum(
        self,
        *,
        objects: bool = True,
        emotions: bool = True,
        faces: bool = True,
    ) -> dict[str, Any]:
        """Curriculum automatico da Wikimedia — oggetti, volti, emozioni."""
        from organism.teaching.web_curriculum import run_web_curriculum

        return run_web_curriculum(
            self.teach_from_web_image,
            objects=objects,
            emotions=emotions,
            faces=faces,
        )

    def run_mega_curriculum(self, *, limit: int = 1000, pause_s: float = 0.35) -> dict[str, Any]:
        """Curriculum 1000 oggetti — visione HD."""
        from organism.teaching.mega_curriculum import run_mega_curriculum

        def _teach_batch(**kwargs: Any) -> dict[str, Any]:
            return self.teach_from_web_image(persist=False, **kwargs)

        result = run_mega_curriculum(_teach_batch, limit=limit, pause_s=pause_s)
        self._persist()
        return result

    def run_code_curriculum(self) -> dict[str, Any]:
        """Insegna programmazione — snippet e programmi interi."""
        from organism.teaching.code_curriculum import run_code_curriculum

        return run_code_curriculum(self.teach_dialogue)

    def train_syntax_curriculum(self, *, repeats: int = 3, absorb_limit: int = 250) -> dict[str, Any]:
        """Fase 2 — rinforzo transizioni sintattiche Layer 2."""
        from organism.teaching.syntax_curriculum import curriculum_sentences

        org = self._ensure()
        stats = self.narrator.bootstrap_curriculum(repeats=repeats)
        absorbed = 0
        for sent in curriculum_sentences()[:absorb_limit]:
            self.composer.absorb(sent, boost=0.35)
            if self.speech:
                self.speech.hear(sent, boost=0.25)
            absorbed += 1
        org.brain.propagate(steps=1)
        if org.brain.plasticity:
            org.brain.plasticity.apply_hebbian(org.brain, org.brain.tick)
        self._persist()
        return {
            "ok": True,
            "syntax": stats,
            "absorbed": absorbed,
            "narrator": self.narrator.stats(),
        }

    def run_fluency_benchmark(self, *, limit: int = 100, pause_s: float = 0.12) -> dict[str, Any]:
        """Benchmark 100 prompt — fluenza, coerenza, SVO."""
        from organism.teaching.fluency_benchmark import run_benchmark

        def _sense(prompt: str) -> dict[str, Any]:
            return self.sense(text=prompt)

        result = run_benchmark(_sense, limit=limit, pause_s=pause_s)
        self._persist()
        return result

    def run_dog_curriculum(self, *, limit: int = 50, pause_s: float = 0.3) -> dict[str, Any]:
        """Training visivo cane — pose e contesti diversi."""
        from organism.teaching.dog_curriculum import run_dog_curriculum

        def _teach(**kwargs: Any) -> dict[str, Any]:
            return self.teach_from_web_image(persist=False, **kwargs)

        result = run_dog_curriculum(_teach, limit=limit, pause_s=pause_s)
        self._persist()
        return result

    def train_integrated_cycle(self, cycle: int) -> dict[str, Any]:
        """Un passo training Fase 3 — storie, dialoghi, motor loop, oggetti."""
        import random

        from organism.teaching.corpus import REASONING, PHILOSOPHY, STORIES_EXTENDED
        from organism.teaching.story_curriculum import dialogue_chains, long_story_chunks

        org = self._ensure()
        rng = random.Random(self.seed + cycle)
        log: dict[str, Any] = {"cycle": cycle, "actions": []}

        if cycle == 1 or cycle % 500 == 0:
            r = self.train_syntax_curriculum(repeats=3, absorb_limit=200)
            log["actions"].append({"syntax": r.get("syntax", {})})

        if cycle % 10 == 0:
            chunks = long_story_chunks()
            chunk = chunks[cycle % len(chunks)]
            self.read(chunk)
            self.self_hear(text=chunk[:120])
            log["actions"].append({"story_chunk": chunk[:60]})

        if cycle % 25 == 0:
            pairs = STORIES_EXTENDED + REASONING + PHILOSOPHY
            when, say = pairs[cycle % len(pairs)]
            self.teach_dialogue(when, say)
            log["actions"].append({"dialogue": when[:40]})

        if cycle % 40 == 0:
            when, say = dialogue_chains()[cycle % len(dialogue_chains())]
            self.teach_dialogue(when, say)
            for _ in range(2):
                m = self.sense(text=when).get("moment") or {}
                spoke = str(m.get("spoke", ""))
                if spoke:
                    self.self_hear(text=spoke)
            log["actions"].append({"dialogue_chain": when})

        if cycle % 100 == 0:
            dog = self.run_dog_curriculum(limit=20, pause_s=0.2)
            log["actions"].append({"dog": dog.get("taught", 0)})

        if cycle % 200 == 0:
            mega = self.run_mega_curriculum(limit=30, pause_s=0.2)
            log["actions"].append({"mega": mega.get("taught", 0)})

        if cycle % 300 == 0:
            self.sleep_cycle()
            log["actions"].append({"sleep": True})

        # Ogni ciclo: stimolo casuale + motor loop chiuso
        probes = ["cosa pensi", "cosa vedi", "chi sei", "perché impari", "cosa provi"]
        q = rng.choice(probes)
        m = self.sense(text=q).get("moment") or {}
        spoke = str(m.get("spoke", ""))
        if spoke and len(spoke) > 8:
            self.self_hear(text=spoke)
        log["probe"] = q
        log["spoke_words"] = len(spoke.split())

        org.brain.propagate(steps=1)
        if cycle % 50 == 0:
            self._persist()
        return log

    def browse_web(self, url: str) -> dict[str, Any]:
        """Legge una pagina web e assorbe conoscenza."""
        from organism.cognition.browser_agent import fetch_page_text

        org = self._ensure()
        page = fetch_page_text(url)
        if not page.get("ok"):
            return page
        text = str(page.get("text", ""))[:4000]
        self.read(text)
        for kw in page.get("keywords", [])[:20]:
            self.composer.absorb(kw, boost=0.35)
        self.episodic_memory.record(
            heard=f"browse {url[:80]}",
            spoke="",
            themes=page.get("keywords", [])[:8],
            objects=[],
            emotion="",
        )
        org.brain.propagate(steps=1)
        self._persist()
        return {
            **page,
            "absorbed_words": len(page.get("keywords", [])),
            "title": page.get("title", ""),
        }

    def teach_object(
        self,
        name: str,
        *,
        image_gray: list[int] | None = None,
        image_b64: str | None = None,
        image_w: int = 64,
        image_h: int = 64,
        color_rgb: dict[str, float] | None = None,
        image_rgba: list[int] | None = None,
        phrase: str | None = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Insegna nome oggetto con vista — forma+colore associati all'immagine."""
        org = self._ensure()
        name = name.strip().lower()
        if not name:
            return {"learned": False}
        grid, sig = self._decode_vision(
            image_gray=image_gray,
            image_b64=image_b64,
            image_w=image_w,
            image_h=image_h,
            color_rgb=color_rgb,
            image_rgba=image_rgba,
        )
        if not grid:
            return {"learned": False, "reason": "no_image"}
        self._last_vision_hash = sig
        self._vision_fresh = True
        org.perceive({"image": grid, "width": image_w, "height": image_h})
        org.brain.propagate(steps=2)
        color = str(self._last_visual_features.get("color", ""))
        if phrase:
            say = phrase
        elif color and color not in ("colore", "grigio"):
            say = f"vedo un {name} {color}"
        else:
            say = f"vedo un {name}"
        obj_sig = str(self._last_visual_features.get("object_sig", sig))
        self.visual_binder.bind(
            sig,
            say,
            boost=1.4,
            features=self._last_visual_features,
            object_sig=obj_sig,
            object_name=name,
        )
        self.photo_memory.capture(
            object_sig=obj_sig,
            features=self._last_visual_features,
            labels=[name, color] if color else [name],
            thumb_gray=(image_gray or [])[:256] if image_gray else None,
        )
        skey = stimulus_key_visual_context(vision_hash=sig)
        self.speech.hear(say, boost=1.2)
        org.perceive({"text": say})
        org.brain.propagate(steps=2)
        grown = self._wire_from_input(org, had_text=True, had_vision=True, teach=True)
        grown += wire_visual_association(org.brain, self.speech, name=name, color=color, boost=0.65)
        if org.brain.plasticity:
            org.brain.plasticity.apply_hebbian(org.brain, org.brain.tick)
        result = self.teacher.teach(skey, say)
        self.composer.absorb(f"{name} {color} {say}".strip(), boost=1.2)
        consolidated = self.visual_binder._object_names.get(obj_sig) == name
        if consolidated and color and color not in ("colore", "grigio"):
            color_phrase = f"è {color}"
            for _ in range(2):
                self.dialogue.teach(f"che colore è il {name}", color_phrase)
                wire_dialogue_pathway(org.brain, self.speech, when=f"che colore è il {name}", say=color_phrase)
                self.composer.absorb(color_phrase, boost=0.9)
        trials = self.visual_binder._object_trials.get(obj_sig, 0)
        if persist:
            self._persist()
        return {
            **result,
            "name": name,
            "object_sig": obj_sig,
            "consolidated": consolidated,
            "trials": trials,
            "confidence": round(self.visual_binder.confidence(self._last_visual_features, object_sig=obj_sig), 2),
            "symbols": self._last_visual_features.get("symbols", []),
            "stimulus_key": skey,
            "new_wires": grown,
        }

    def teach_repetition(
        self,
        phrase: str,
        *,
        stimulus_key: str | None = None,
        image_gray: list[int] | None = None,
        image_b64: str | None = None,
        image_w: int = 64,
        image_h: int = 64,
    ) -> dict[str, Any]:
        org = self._ensure()
        if image_gray or image_b64:
            grid, sig = self._decode_vision(
                image_gray=image_gray,
                image_b64=image_b64,
                image_w=image_w,
                image_h=image_h,
            )
            self._last_vision_hash = sig or vision_hash(grid)
            org.perceive({"image": grid, "width": image_w, "height": image_h})
            org.brain.propagate(steps=1)
            self.visual_binder.bind(
                self._last_vision_hash,
                phrase,
                features=self._last_visual_features,
                object_sig=str(self._last_visual_features.get("object_sig", "")),
            )
        skey = stimulus_key or self._context_stimulus_key()
        if not skey:
            skey = stimulus_key_from_sensory(text=phrase)
        self.speech.hear(phrase, boost=1.3)
        org.perceive({"text": phrase})
        org.brain.propagate(steps=2)
        grown = self._wire_from_input(
            org,
            had_text=True,
            had_vision=bool(image_gray or image_b64),
            teach=True,
        )
        if org.brain.plasticity:
            org.brain.plasticity.apply_hebbian(org.brain, org.brain.tick)
        result = self.teacher.teach(skey, phrase)
        if result.get("learned"):
            from mind.types import Fragment

            org.memory.add(
                Fragment(
                    id=f"taught_{skey}",
                    title=phrase,
                    weight=0.7,
                    sensation_id="taught",
                    hooks=phrase.lower().split()[:6],
                    functions=["learned"],
                )
            )
        self._persist()
        return {
            **result,
            "stimulus_key": skey,
            "syllables_heard": self.speech.phonemes.count,
            "synapses": org.brain.synapse_count,
            "synapses_grown": org.brain.synapse_count - self._synapses_at_birth,
            "new_wires": grown,
        }

    def teach_dialogue(
        self,
        when: str,
        say: str,
        *,
        kind: str = "speech",
    ) -> dict[str, Any]:
        """Insegna: quando senti «when» → rispondi «say» (dialogo o codice)."""
        org = self._ensure()
        when = when.strip()
        say = say.strip()
        self.speech.hear(when, boost=0.6)
        self.speech.hear(say, boost=1.0)
        self.composer.absorb(when, boost=0.8)
        self.composer.absorb(say, boost=1.2)
        org.perceive({"text": f"{when} → {say}"})
        org.brain.propagate(steps=2)
        grown = self._wire_from_input(org, had_text=True, had_vision=False, teach=True)
        for w in list(self.self_learner.gaps()):
            if w in when.lower() or w in say.lower():
                self.self_learner.note_learned(w)
        if kind == "code":
            self.code.learn_snippet(when, say)
            post = [n.id for n in org.brain.get_neurons("motor", "text_generator")][:8]
            if not post:
                post = [n.id for n in org.brain.get_neurons("motor", "speech_phoneme_generator")][:8]
            pre = active_ids(org.brain, "sensory", "text_semantic_encoder")
            wire_coactive(org.brain, pre[:10], post, max_new=4)
        if org.brain.plasticity:
            org.brain.plasticity.apply_hebbian(org.brain, org.brain.tick)
        wire_dialogue_pathway(org.brain, self.speech, when=when, say=say)
        if self._last_vision_hash:
            self.visual_binder.bind(self._last_vision_hash, f"{when} {say}")
        result = self.dialogue.teach(when, say, kind=kind)  # type: ignore[arg-type]
        if result.get("learned"):
            from mind.types import Fragment

            org.memory.add(
                Fragment(
                    id=f"dialogue_{normalize_dialogue_key(when)}",
                    title=f"{when} → {say}",
                    weight=0.75,
                    sensation_id="dialogue",
                    hooks=when.lower().split()[:6],
                    functions=["dialogue", kind],
                )
            )
        self._persist()
        return {
            **result,
            "synapses": org.brain.synapse_count,
            "synapses_grown": org.brain.synapse_count - self._synapses_at_birth,
            "new_wires": grown,
            "dialogue_pairs": len(self.dialogue.all_pairs()),
            "words_known": self.composer.lexicon.count,
        }

    def read(self, text: str) -> dict[str, Any]:
        """Lettura autonoma — testo come flusso sensoriale, impara da solo."""
        org = self._ensure()
        self._last_sense_t = time.time()
        if self._reading is None:
            self._reading = ReadingChannel(org.brain)
        result = self._reading.perceive(text)
        self.composer.absorb(text, boost=0.45)
        if self.speech:
            self.speech.hear(text, boost=0.5)
        org.perceive({"text": text[:400]})
        org.brain.propagate(steps=2)
        self._wire_from_input(org, had_text=True, had_vision=False)
        active = self.tasks.active()
        if active and active.kind == "read_aloud":
            self.tasks.evaluate_attempt(text[:120])
        self._persist()
        return {
            "reading": result.to_dict(),
            "words_known": self.composer.lexicon.count,
            "task": self.tasks.active().to_dict() if self.tasks.active() else None,
        }

    def assign_task(self, kind: TaskKind, prompt: str, target: str) -> dict[str, Any]:
        """Assegna un compito — responsabilità e portare a termine."""
        self._ensure()
        t = self.tasks.assign(kind, prompt, target)
        self._persist()
        return {"task": t.to_dict(), "tasks": self.tasks.to_dict()}

    def absorb_vocabulary(self, texts: list[str], *, boost: float = 1.0) -> dict[str, Any]:
        """Assorbe testi nel lessico — migliaia di parole senza coppie dialogo."""
        org = self._ensure()
        before = self.composer.lexicon.count
        n = 0
        for raw in texts:
            t = raw.strip()
            if not t:
                continue
            self.composer.absorb(t, boost=boost)
            if self.speech:
                self.speech.hear(t, boost=0.35)
            n += 1
        if n:
            sample = " ".join(texts[:3])[:240]
            org.perceive({"text": sample})
            org.brain.propagate(steps=1)
            if org.brain.plasticity:
                org.brain.plasticity.apply_hebbian(org.brain, org.brain.tick)
        self._persist()
        return {
            "absorbed": n,
            "words_known": self.composer.lexicon.count,
            "words_new": self.composer.lexicon.count - before,
        }

    def research_once(self) -> dict[str, Any]:
        org = self._ensure()
        if self._researched:
            return {"already": True, "topics": []}
        topics_cfg = org.dna.genome.get("baby", {}).get("research_topics", [])
        topics = research_human_senses(org.memory, topics=topics_cfg, lang="it")
        self._researched = True
        self._persist()
        return {"researched": True, "topics": topics}

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


def normalize_dialogue_key(when: str) -> str:
    import hashlib

    return hashlib.sha256(when.strip().lower().encode()).hexdigest()[:12]
