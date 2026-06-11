"""Udito e attenzione uditiva — metodi del baby agent."""
from __future__ import annotations

import re
import time
from typing import Any

from organism.autonomous.baby_types import BabyMoment
from organism.runtime import OrganismRuntime
from organism.cognition.motor_plan import MotorPlan
from organism.cognition.pathway import wire_dialogue_pathway
from organism.sensory.web_media import decode_image_gray, decode_image_pixels, vision_hash, decode_audio_b64, audio_hash
from organism.sensory.social_tone import analyze_social_tone
from organism.drives.curiosity import stimulus_key_from_sensory


class BabyHearingMixin:
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
