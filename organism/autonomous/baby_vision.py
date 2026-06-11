"""Visione e percezione visiva — metodi del baby agent."""
from __future__ import annotations

import time
from typing import Any

from organism.autonomous.baby_types import BabyMoment, _COLOR_WORDS
from organism.runtime import OrganismRuntime
from organism.sensory.visual_scene import scene_features, scene_signature
from organism.sensory.web_media import decode_image_gray, decode_image_pixels, vision_hash
from organism.cognition.pathway import prime_curiosity_pathway
from organism.cognition.visual_pathway import prime_visual_percept, wire_visual_association
from organism.drives.curiosity import stimulus_key_visual_context
from organism.drives.brain_readout import inject_circadian, read_brain_mood
from organism.sensory.face_vision import analyze_face


class BabyVisionMixin:
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
