"""ORGANISM runtime — perceive → think → express → learn → sleep."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from organism.dna.interpreter import DNAInterpreter
from organism.learning.live import LearningReport, LiveLearner
from organism.learning.store import OrganismStore
from organism.mind_bridge import MindBridge, OrganismThought, SensoryBundle
from organism.motor.motion import MotionModule, MotionResult
from organism.motor.song import SongModule, SongResult
from organism.motor.speech import SpeechModule, SpeechResult
from organism.motor.text_out import TextMotor, TextOutput
from organism.sensory.audio import AudioModule
from organism.sensory.text import TextModule
from organism.sensory.vision import VisionModule

OutputModality = Literal["speech", "song", "text", "motion", "full"]


@dataclass
class OrganismCycleLog:
    sensory_symbols: list[str] = field(default_factory=list)
    thought_symbols: list[str] = field(default_factory=list)
    expression_symbols: list[str] = field(default_factory=list)
    learning_symbols: list[str] = field(default_factory=list)
    brain_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrganismExpression:
    modality: OutputModality
    speech: SpeechResult | None = None
    song: SongResult | None = None
    text: TextOutput | None = None
    motion: MotionResult | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"modality": self.modality}
        if self.speech:
            d["speech"] = {
                "text": self.speech.text,
                "prosody": self.speech.prosody,
                "ssml": self.speech.ssml,
            }
        if self.song:
            d["song"] = {
                "scale": self.song.scale,
                "melody": [{"pitch": n.pitch, "beats": n.duration_beats} for n in self.song.melody],
                "style": self.song.style,
            }
        if self.text:
            d["text"] = {"body": self.text.text, "emotion": self.text.emotion}
        if self.motion:
            d["motion"] = {
                "frames": [{"gesture": f.gesture, "i": f.intensity} for f in self.motion.frames],
            }
        return d


class OrganismRuntime:
    def __init__(
        self,
        dna_path: str | Path | None = None,
        variant_path: str | Path | None = None,
        seed: int = 42,
        *,
        store_path: str | Path | None = None,
        auto_load: bool = False,
    ) -> None:
        self.dna = DNAInterpreter(genome_path=dna_path, variant_path=variant_path)
        self.brain = self.dna.grow_brain(seed=seed)
        self.vision = VisionModule(self.brain)
        self.audio = AudioModule(self.brain)
        self.text = TextModule(self.brain, lexicon=self.dna.pattern_lexicon())
        self.mind_bridge = MindBridge()
        self.speech = SpeechModule(self.brain)
        self.song = SongModule(self.brain, default_style=self.dna.motor_defaults().get("song_style", "lo_fi"))
        self.text_motor = TextMotor()
        self.motion = MotionModule(self.brain, style=self.dna.motor_defaults().get("motion_style", "neutral"))
        learn_cfg = self.dna.genome.get("live_learning", {})
        self.learner = LiveLearner(learn_cfg)
        self.store = OrganismStore(store_path)
        self._learn_enabled = bool(learn_cfg.get("enabled", True))
        self._auto_save_every = int(learn_cfg.get("auto_save_every", 0))
        self._last_log = OrganismCycleLog()
        self._last_learning: LearningReport | None = None
        if auto_load:
            self.load_state()

    @classmethod
    def studio_assistant(cls, seed: int = 42, **kwargs) -> OrganismRuntime:
        variant = Path(__file__).parent / "dna" / "variants" / "studio_assistant.yaml"
        return cls(variant_path=variant, seed=seed, **kwargs)

    @classmethod
    def baby(cls, seed: int = 42, **kwargs) -> OrganismRuntime:
        from mind.runtime import MindRuntime
        from mind.seed_it import build_italian_memory, build_italian_circuits

        variant_name = os.environ.get("ORGANISM_DNA_VARIANT", "baby")
        if variant_name == "genesis":
            return cls.genesis(seed=seed, **kwargs)
        variant = Path(__file__).parent / "dna" / "variants" / f"{variant_name}.yaml"
        if not variant.exists():
            variant = Path(__file__).parent / "dna" / "variants" / "baby.yaml"
        org = cls(variant_path=variant, seed=seed, **kwargs)
        # MIND italiano — rete di frammenti semantici, catene causali, identità
        org.mind_bridge = MindBridge(MindRuntime(
            memory=build_italian_memory(),
            patterns=__import__("mind.pattern", fromlist=["PatternEngine"]).PatternEngine(),
            sensations=build_italian_circuits(),
        ))
        return org

    @classmethod
    def genesis(cls, seed: int = 42, **kwargs) -> OrganismRuntime:
        """Genesis — DNA evoluto se presente, MIND italiano."""
        from mind.runtime import MindRuntime
        from mind.seed_it import build_italian_memory, build_italian_circuits
        from organism.dna.evolution import genesis_variant_path

        org = cls(variant_path=genesis_variant_path(), seed=seed, **kwargs)
        org.mind_bridge = MindBridge(MindRuntime(
            memory=build_italian_memory(),
            patterns=__import__("mind.pattern", fromlist=["PatternEngine"]).PatternEngine(),
            sensations=build_italian_circuits(),
        ))
        return org

    @classmethod
    def mega(cls, seed: int = 42, **kwargs) -> OrganismRuntime:
        return cls._baby_variant("mega", seed=seed, **kwargs)

    @classmethod
    def giga(cls, seed: int = 42, **kwargs) -> OrganismRuntime:
        return cls._baby_variant("giga", seed=seed, **kwargs)

    @classmethod
    def ultra(cls, seed: int = 42, **kwargs) -> OrganismRuntime:
        return cls._baby_variant("ultra", seed=seed, **kwargs)

    @classmethod
    def _baby_variant(cls, name: str, seed: int = 42, **kwargs) -> OrganismRuntime:
        from mind.runtime import MindRuntime

        variant = Path(__file__).parent / "dna" / "variants" / f"{name}.yaml"
        org = cls(variant_path=variant, seed=seed, **kwargs)
        org.mind_bridge = MindBridge(MindRuntime.blank())
        return org

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "neurons": self.brain.neuron_count,
            "synapses": self.brain.synapse_count,
            "mean_synapse_weight": round(self.brain.mean_synapse_weight(), 5),
            "species": self.dna.genome.get("species"),
            "learning_cycles": self.learner.total_cycles,
            "efficiency": self.brain.efficiency_stats(),
        }

    @property
    def memory(self):
        return self.mind_bridge.mind.memory

    def perceive(self, input_data: dict[str, Any]) -> SensoryBundle:
        bundle = SensoryBundle()
        if "text" in input_data:
            bundle._raw_text = str(input_data["text"])
        if "shapes" in input_data:
            bundle._raw_shapes = str(input_data["shapes"])
        if "image" in input_data:
            img = input_data["image"]
            w = int(input_data.get("width", 16))
            h = int(input_data.get("height", 16))
            bundle.vision = self.vision.perceive(img, width=w, height=h)
        if "shapes" in input_data:
            bundle.vision = self.vision.perceive_shapes_ascii(input_data["shapes"])
        if "audio" in input_data:
            bundle.audio = self.audio.perceive(input_data["audio"])
        if "tone_hz" in input_data:
            bundle.audio = self.audio.perceive_tone(float(input_data["tone_hz"]))
        if "text" in input_data:
            bundle.text = self.text.perceive(str(input_data["text"]))
        return bundle

    def think(self, sensory: SensoryBundle, **kwargs) -> OrganismThought:
        return self.mind_bridge.think(sensory, **kwargs)

    def express(
        self,
        thought: OrganismThought,
        modality: OutputModality = "speech",
        emotion: str | None = None,
    ) -> OrganismExpression:
        emo = emotion or self.dna.motor_defaults().get("speech_emotion", "calm")
        if thought.mind_result.human_emotion_active and emo == "calm":
            emo = "empathetic"
        mr = thought.mind_result

        if modality == "speech":
            return OrganismExpression(modality="speech", speech=self.speech.express(mr, emotion=emo))
        if modality == "song":
            return OrganismExpression(modality="song", song=self.song.express(mr, emotion=emo))
        if modality == "text":
            return OrganismExpression(modality="text", text=self.text_motor.express(mr, emotion=emo))
        if modality == "motion":
            return OrganismExpression(modality="motion", motion=self.motion.express(mr))
        return OrganismExpression(
            modality="full",
            speech=self.speech.express(mr, emotion=emo),
            song=self.song.express(mr, emotion=emo),
            text=self.text_motor.express(mr, emotion=emo),
            motion=self.motion.express(mr),
        )

    def live(
        self,
        input_data: dict[str, Any],
        output_modality: OutputModality = "speech",
        *,
        learn: bool | None = None,
        outcome_success: bool = True,
        **think_kwargs,
    ) -> tuple[OrganismThought, OrganismExpression, LearningReport | None]:
        sensory = self.perceive(input_data)
        thought = self.think(sensory, **think_kwargs)
        expression = self.express(thought, modality=output_modality)

        learn_report: LearningReport | None = None
        do_learn = self._learn_enabled if learn is None else learn
        if do_learn:
            learn_report = self.learner.consolidate(
                self.brain,
                self.memory,
                sensory,
                thought,
                expression,
                outcome_success=outcome_success,
            )
            self._last_learning = learn_report
            if self._auto_save_every and self.learner.total_cycles % self._auto_save_every == 0:
                self.save_state()

        self._log_cycle(sensory, thought, expression, learn_report)
        return thought, expression, learn_report

    def replay(self, episodes: list[dict[str, Any]], *, learn: bool = True) -> list[dict[str, Any]]:
        """Batch live cycles for training — each episode: input_data + optional kwargs."""
        results = []
        for ep in episodes:
            data = ep.get("input", ep)
            kwargs = {k: v for k, v in ep.items() if k != "input"}
            thought, expr, lr = self.live(data, learn=learn, **kwargs)
            results.append(
                {
                    "action": thought.mind_result.action.id if thought.mind_result.action else None,
                    "learning": lr.__dict__ if lr else None,
                    "text": expr.speech.text if expr.speech else (expr.text.text if expr.text else ""),
                }
            )
        return results

    def sleep(self) -> dict[str, Any]:
        cfg = self.dna.pruning_config()
        removed = self.brain.prune_weak_synapses(
            threshold=float(cfg.get("weak_synapse_threshold", 0.05)),
            keep_percentage=float(cfg.get("keep_percentage", 0.85)),
        )
        consolidated = self.brain.consolidate_memory()
        return {"pruned_synapses": removed, **consolidated, "synapses_after": self.brain.synapse_count}

    def save_state(self, path: str | Path | None = None) -> Path:
        if path:
            self.store = OrganismStore(path)
        return self.store.save(self.brain, self.memory, self.learner, meta=self.stats)

    def load_state(self, path: str | Path | None = None) -> dict[str, Any]:
        if path:
            self.store = OrganismStore(path)
        return self.store.load(self.brain, self.memory, self.learner)

    def brain_visualize(self, **kwargs) -> str:
        return self.brain.visualize_json(**kwargs)

    def live_json(self, input_data: dict[str, Any], **kwargs) -> str:
        thought, expression, learn_report = self.live(input_data, **kwargs)
        return json.dumps(
            {
                "stats": self.stats,
                "thought": thought.mind_result.to_dict(),
                "symbols": thought.symbols,
                "expression": expression.to_dict(),
                "learning": learn_report.__dict__ if learn_report else None,
                "log": {
                    "sensory": self._last_log.sensory_symbols,
                    "thought": self._last_log.thought_symbols,
                    "expression": self._last_log.expression_symbols,
                    "learning": self._last_log.learning_symbols,
                },
            },
            ensure_ascii=False,
            indent=2,
        )

    def _log_cycle(
        self,
        sensory: SensoryBundle,
        thought: OrganismThought,
        expression: OrganismExpression,
        learn_report: LearningReport | None,
    ) -> None:
        self._last_log = OrganismCycleLog(
            sensory_symbols=sensory.all_symbols(),
            thought_symbols=thought.symbols,
            expression_symbols=self._expr_symbols(expression),
            learning_symbols=learn_report.symbols if learn_report else [],
            brain_stats={"tick": self.brain.tick, **self.stats},
        )

    def _expr_symbols(self, expr: OrganismExpression) -> list[str]:
        out: list[str] = []
        for part in (expr.speech, expr.song, expr.text, expr.motion):
            if part is not None and hasattr(part, "symbols"):
                out.extend(part.symbols)
        return out
