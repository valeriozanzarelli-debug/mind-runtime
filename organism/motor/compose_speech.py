"""Compone parlato da pensiero + motore — zero lookup di risposte fisse."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

from organism.cognition.amygdala import AmygdalaEngine
from organism.cognition.discourse import compose_discourse, is_babble
from organism.cognition.linguistic_narrator import LinguisticNarrator
from organism.cognition.motor_plan import MotorPlan
from organism.cognition.neural_lexicon import NeuralLexicon
from organism.cognition.thought import Thought
from organism.cognition.workspace import OutputMode
from organism.motor.emergent_speech import EmergentSpeechMotor

Kind = Literal["speech", "code"]


@dataclass
class ComposedSpeech:
    text: str
    kind: Kind
    from_thought: bool
    thought_used: list[str]
    motor_plan: MotorPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind,
            "from_thought": self.from_thought,
            "thought_used": self.thought_used,
            "motor_plan": self.motor_plan.to_dict() if self.motor_plan else None,
        }


class SpeechComposer:
    def __init__(self, seed: int = 0, *, narrator: LinguisticNarrator | None = None) -> None:
        self.rng = random.Random(seed)
        self.lexicon = NeuralLexicon()
        self.narrator = narrator or LinguisticNarrator(seed=seed)

    def bind(self, brain: Any, motor: EmergentSpeechMotor | None = None) -> None:
        self.lexicon.bind(brain, motor)

    def absorb(self, text: str, *, boost: float = 1.0) -> None:
        self.lexicon.absorb(text, boost=boost)
        if text.strip():
            self.narrator.train_from_text(text, boost=0.03 * boost)

    def produce(
        self,
        *,
        thought: Thought,
        motor: EmergentSpeechMotor,
        mode: OutputMode = "speak",
        reflective: bool = False,
        pathway_primed: bool = False,
        pathway_words: list[str] | None = None,
        amygdala: AmygdalaEngine | None = None,
        recent_words: list[str] | None = None,
        heard: str = "",
        memory_themes: list[str] | None = None,
        valence: float = 0.0,
    ) -> ComposedSpeech:
        """Layer 2 narratore → motore; fallback emergente se narratore insufficiente."""
        if mode == "silent":
            return ComposedSpeech(text="", kind="speech", from_thought=False, thought_used=[])

        if mode == "flow":
            mode = "speak"

        body = ""
        avoid = [w.lower() for w in (recent_words or []) if w]

        # Layer 2 — traduzione grammaticale dei temi non-verbali
        self.narrator.observe_themes(thought.themes, valence=valence)
        narration = self.narrator.narrate(
            thought, heard=heard, valence=valence, min_words=6, max_words=15
        )
        if narration.from_layer2 and narration.text and not is_babble(narration.text, min_words=5):
            articulable = sum(
                1 for w in narration.words if self.lexicon.is_articulable(w, min_exposure=0.2)
            )
            if articulable >= max(2, len(narration.words) // 3) or self.lexicon.count < 12:
                body = narration.text
        if heard or pathway_words or memory_themes:
            body = self._mature_discourse(
                thought,
                heard=heard,
                pathway_words=pathway_words,
                memory_themes=memory_themes,
                avoid=avoid,
            )
        if any(s.startswith("IMPULSE:ask") for s in thought.symbols) and not body:
            body = self._from_curiosity_question(thought, motor, amygdala=amygdala)
        if not body and pathway_primed and pathway_words:
            body = self._from_pathway_emergent(
                pathway_words, thought, motor, avoid=avoid
            )
        if not body:
            body = self._from_thought_emergent(
                thought, motor=motor, min_words=2, avoid=avoid
            )
        if body and (self._looks_garbage(body, thought) or is_babble(body, min_words=4)):
            body = self._honest_unknown(thought, motor, amygdala=amygdala) or ""
        if not body or is_babble(body, min_words=5):
            body = self._mature_discourse(
                thought,
                heard=heard,
                pathway_words=pathway_words,
                memory_themes=memory_themes,
            )
        if not body:
            body = self._from_thought(thought, motor, min_words=2, avoid=avoid)
        if body and is_babble(body, min_words=5):
            body = self._mature_discourse(
                thought,
                heard=heard,
                pathway_words=pathway_words,
                memory_themes=memory_themes,
            )
        if body and len(body.strip()) > 2:
            plan = motor.plan_from_text(body) if motor else None
            return ComposedSpeech(
                text=body,
                kind="speech",
                from_thought=True,
                thought_used=thought.themes[:10],
                motor_plan=plan,
            )

        inh = amygdala.speech_inhibition() if amygdala else 0.0
        em = motor.readout(extra_inhibition=inh)
        if em.will_speak:
            plan = motor.utter_with_plan()
            if plan.text and len(plan.text) > 4:
                return ComposedSpeech(
                    text=plan.text,
                    kind="speech",
                    from_thought=True,
                    thought_used=thought.themes[:4],
                    motor_plan=plan,
                )

        return ComposedSpeech(text="", kind="speech", from_thought=False, thought_used=[])

    def _question_word_order(
        self,
        thought: Thought,
        focus: str,
        *,
        amygdala: AmygdalaEngine | None = None,
    ) -> list[str]:
        """Ordine parole per domanda — solo lessico neurale attivo, zero liste fisse."""
        pool: list[str] = []
        for t in thought.themes:
            tl = t.lower().strip()
            if not tl or tl.startswith(("vis:", "obj:", "col:", "unknown:", "impulse:")):
                continue
            if tl not in pool:
                pool.append(tl)
        for w in self.lexicon.active_words(12, min_act=0.06):
            if w not in pool:
                pool.append(w)
        for w in self.lexicon.known_words(16):
            if w not in pool:
                pool.append(w)
        focus_l = focus.lower().strip()
        if focus_l and not focus_l.startswith("scene") and focus_l not in pool:
            pool.append(focus_l)

        scored: list[tuple[float, str]] = []
        for w in pool:
            act = self.lexicon.activation(w)
            if amygdala:
                act = amygdala.word_boost(w, activation=act)
            if self._articulable(w, min_exposure=0.22):
                scored.append((act, w))
        scored.sort(key=lambda x: (-x[0], x[1]))

        ordered = [w for _, w in scored[:4]]
        if focus_l and self._articulable(focus_l, min_exposure=0.15) and focus_l not in ordered:
            ordered.append(focus_l)
        return ordered[:5]

    def _from_curiosity_question(
        self,
        thought: Thought,
        motor: EmergentSpeechMotor,
        *,
        amygdala: AmygdalaEngine | None = None,
    ) -> str:
        """Domanda emergente — assemblaggio motorio lessicale, non template."""
        unknown = [
            s.replace("UNKNOWN:", "").lower().strip()
            for s in thought.symbols
            if s.startswith("UNKNOWN:") and s != "UNKNOWN:scene"
        ]
        focus = unknown[0] if unknown else ""
        ordered = self._question_word_order(thought, focus, amygdala=amygdala)
        if ordered:
            plan = motor.lexical_readout(ordered, punct="?")
            if plan.text:
                return plan.text
        if focus and self._articulable(focus, min_exposure=0.15):
            plan = motor.lexical_readout([focus], punct="?")
            return plan.text
        active = self.lexicon.active_words(6, min_act=0.05)
        if active:
            plan = motor.lexical_readout(active, punct="?")
            return plan.text
        return ""

    def _honest_unknown(
        self,
        thought: Thought,
        motor: EmergentSpeechMotor,
        *,
        amygdala: AmygdalaEngine | None = None,
    ) -> str:
        unknown = [
            s.replace("UNKNOWN:", "").lower().strip()
            for s in thought.symbols
            if s.startswith("UNKNOWN:") and s != "UNKNOWN:scene"
        ]
        focus = unknown[0] if unknown else ""
        ordered = self._question_word_order(thought, focus, amygdala=amygdala)[:5]
        if not ordered:
            return ""
        plan = motor.lexical_readout(ordered, punct=".")
        return plan.text

    def _articulable(self, w: str, *, min_exposure: float = 0.7) -> bool:
        """Filtra parole inventate — solo lessico con percorso neurale."""
        return self.lexicon.is_articulable(w, min_exposure=min_exposure)

    def _looks_garbage(self, text: str, thought: Thought) -> bool:
        words = text.lower().replace("?", ".").split()
        if len(words) > 8:
            return True
        theme_set = {t.lower() for t in thought.themes}
        if not theme_set:
            return len(words) > 4
        overlap = sum(
            1 for w in words if w in theme_set or self.lexicon.is_articulable(w, min_exposure=0.4)
        )
        return overlap < max(1, len(words) // 2)

    def _from_pathway_emergent(
        self,
        pathway_words: list[str],
        thought: Thought,
        motor: EmergentSpeechMotor,
        *,
        avoid: list[str] | None = None,
    ) -> str:
        """Risposta dal percorso neurale attivato — ordine da attivazione, non copia verbatim."""
        em = motor.readout()
        if em.motor_pressure < 0.08 and thought.pressure < 0.15:
            return ""
        theme_set = {t.lower() for t in thought.themes}
        candidates = [w.lower().strip() for w in pathway_words if w.isalpha()]
        for t in thought.themes:
            if self._articulable(t):
                candidates.append(t.lower())
        chosen = self.lexicon.ranked(candidates, avoid=avoid)
        chosen = [w for w in chosen if w in theme_set or w in {x.lower() for x in pathway_words}][:7]
        if len(chosen) < 2:
            for t in thought.themes:
                if self._articulable(t) and t.lower() not in chosen:
                    chosen.append(t.lower())
                if len(chosen) >= 5:
                    break
        if not chosen:
            return ""
        n = min(7, max(2, len(chosen)))
        chunk = chosen[:n]
        sentence = " ".join(chunk)
        return sentence[0].upper() + sentence[1:] + (
            "." if not sentence.endswith((".", "?", "!")) else ""
        )

    def _from_visual_motor(
        self,
        thought: Thought,
        motor: EmergentSpeechMotor,
        q_tag: str,
    ) -> str:
        """Risposta visiva/colore — assemblaggio motorio lessicale."""
        ordered: list[str] = []
        col_syms = [s.replace("COL:", "") for s in thought.symbols if s.startswith("COL:")]
        if q_tag == "color":
            pool = col_syms + [w for w in thought.themes if not w.startswith(("VIS:", "OBJ:", "COL:"))]
            ordered = self.lexicon.ranked(pool)
        elif q_tag == "vision":
            pool = [w for w in thought.themes if not w.startswith(("VIS:", "OBJ:", "COL:", "IMPULSE:", "UNKNOWN:"))]
            ordered = self.lexicon.ranked(pool)[:5]
        elif q_tag == "yesno":
            pool = [w for w in thought.themes if self._articulable(w, min_exposure=0.25)]
            ordered = self.lexicon.ranked(pool)[:1]
        ordered = [w for w in ordered if w and self._articulable(w, min_exposure=0.22)]
        if not ordered:
            return ""
        plan = motor.lexical_readout(ordered[:6], punct=".")
        return plan.text

    def _from_thought_emergent(
        self,
        thought: Thought,
        *,
        motor: EmergentSpeechMotor | None = None,
        min_words: int = 2,
        avoid: list[str] | None = None,
    ) -> str:
        """Frase dai temi attivi nell'ordine del pensiero — non copia insegnato."""
        ordered: list[str] = []
        skip = ("VIS:", "OBJ:", "MEM:", "BRAIN:", "QUESTION:", "COL:")
        for t in thought.themes:
            if t.startswith(skip):
                continue
            if self._articulable(t) and t not in ordered:
                ordered.append(t)
        q_tags = {s.replace("QUESTION:", "") for s in thought.symbols if s.startswith("QUESTION:")}
        yn = [w for w in ordered if self._articulable(w, min_exposure=0.2) and len(w) <= 3]
        if yn and "yesno" in q_tags:
            w = yn[0]
            return w[0].upper() + w[1:] + "."
        if motor and "color" in q_tags:
            body = self._from_visual_motor(thought, motor, "color")
            if body:
                return body
        if motor and "vision" in q_tags:
            body = self._from_visual_motor(thought, motor, "vision")
            if body:
                return body
        allow_fill = not q_tags and len(ordered) < 2 and thought.pressure > 0.35
        if allow_fill and self.lexicon.count:
            for w in self.lexicon.sample(3, self.rng, prefer=ordered, avoid=avoid):
                if self._articulable(w) and w not in ordered:
                    ordered.append(w)
        if avoid and len(ordered) >= 2:
            ordered = self.lexicon.ranked(ordered, avoid=avoid)[:6]
        n = min(6, max(min_words, 2 + int(thought.pressure * 2)))
        chunk = ordered[:n]
        if not chunk:
            return ""
        sentence = " ".join(chunk)
        return sentence[0].upper() + sentence[1:] + (
            "." if not sentence.endswith((".", "?", "!")) else ""
        )

    def _from_thought(
        self,
        thought: Thought,
        motor: EmergentSpeechMotor | None,
        *,
        min_words: int = 2,
        avoid: list[str] | None = None,
    ) -> str:
        prefer = [t for t in thought.themes if len(t) > 2 and not t.startswith("VIS:")]
        prefer = [w for w in prefer if self._articulable(w) or self.lexicon.count < 8]
        if not prefer and self.lexicon.count:
            prefer = [
                w
                for w in self.lexicon.sample(6, self.rng, prefer=[], avoid=avoid)
                if self._articulable(w)
            ]
        if not prefer:
            return ""

        n_words = min(8, max(min_words, 3 + int(thought.pressure * 4)))
        chunk = self.lexicon.sample(n_words, self.rng, prefer=prefer)
        chunk = [w for w in chunk if self._articulable(w)]
        if not chunk:
            return ""
        clean: list[str] = []
        for w in chunk:
            if not clean or clean[-1] != w:
                clean.append(w)
        sentence = " ".join(clean[:8]).strip()
        if not sentence:
            return ""
        return sentence[0].upper() + sentence[1:] + (
            "." if not sentence.endswith((".", "?", "!")) else ""
        )

    def _mature_discourse(
        self,
        thought: Thought,
        *,
        heard: str = "",
        pathway_words: list[str] | None = None,
        memory_themes: list[str] | None = None,
        avoid: list[str] | None = None,
    ) -> str:
        """Periodo umano da temi attivi — evita balbettio."""
        themes: list[str] = []
        skip = frozenset({"ciao", "grazie", "bene", "sono", "stai", "cosa", "come", "chi", "penso", "pensi"})
        for src in (pathway_words or [], memory_themes or [], thought.themes):
            for w in src:
                wl = w.lower().strip()
                if (
                    wl
                    and wl.isalpha()
                    and len(wl) > 2
                    and wl not in skip
                    and not wl.startswith(("vis:", "obj:"))
                    and self._articulable(wl, min_exposure=0.15)
                ):
                    if wl not in themes:
                        themes.append(wl)
        for w in self.lexicon.active_words(14, min_act=0.08):
            if w not in themes and w not in skip:
                themes.append(w)
        themes = self.lexicon.ranked(themes, avoid=avoid)[:12]
        if heard:
            hw = [w for w in heard.lower().split() if len(w) > 2 and w not in skip][:2]
            themes = list(dict.fromkeys(hw + themes))[:12]
        body = compose_discourse(themes[:12], rng=self.rng, heard=heard, min_words=8, max_words=18)
        if body and not is_babble(body, min_words=5):
            return body
        return ""

    def long_form(
        self,
        *,
        thought: Thought,
        motor: EmergentSpeechMotor,
        heard: str = "",
        memory_themes: list[str] | None = None,
    ) -> ComposedSpeech:
        """Monologo — solo temi e percorsi neurali, nessun template dialogo."""
        body = self._mature_discourse(
            thought,
            heard=heard,
            memory_themes=memory_themes,
        )
        if not body:
            prefer = [t for t in thought.themes if self._articulable(t)]
            if prefer:
                extra = self.lexicon.sample(4, self.rng, prefer=prefer)
                extra = [w for w in extra if self._articulable(w)]
                chunk = prefer[:3] + [w for w in extra if w not in prefer[:3]]
                chunk = chunk[:10]
                if chunk:
                    body = " ".join(chunk).capitalize() + "."
        if not body:
            body = self._from_thought(thought, motor, min_words=3)
        if body and is_babble(body, min_words=5):
            body = self._mature_discourse(thought, heard=heard, memory_themes=memory_themes)
        return ComposedSpeech(
            text=body.strip(),
            kind="speech",
            from_thought=bool(body),
            thought_used=thought.themes[:12],
            motor_plan=motor.plan_from_text(body) if body else None,
        )
