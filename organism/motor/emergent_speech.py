"""Voce emergente — solo attivazione neurale motoria + sillabe udite."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from organism.cognition.motor_plan import MotorPlan
from organism.teaching.phonemes import PhonemeLearner, split_italian_syllables


@dataclass
class SpeechEmergence:
    motor_pressure: float
    inhibition: float
    will_speak: bool
    syllables_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "motor_pressure": round(self.motor_pressure, 3),
            "inhibition": round(self.inhibition, 3),
            "will_speak": self.will_speak,
            "syllables_ready": self.syllables_ready,
        }


class EmergentSpeechMotor:
    def __init__(self, brain, seed: int = 0) -> None:
        self.brain = brain
        self.rng = random.Random(seed)
        self.phonemes = PhonemeLearner()
        self.phoneme_neurons = brain.get_neurons("motor", "speech_phoneme_generator")
        self._neuron_syllable: dict[int, str] = {}

    def hear(self, text: str, *, boost: float = 1.0) -> list[str]:
        heard = self.phonemes.hear(text, boost=boost)
        for i, syl in enumerate(heard):
            if i < len(self.phoneme_neurons):
                n = self.phoneme_neurons[i]
                self._neuron_syllable[n.id] = syl
                n.activation = min(1.0, n.activation + 0.2 * boost)
        return heard

    def readout(self, *, extra_inhibition: float = 0.0) -> SpeechEmergence:
        layers = self.brain.layer_activation_summary()
        motor = float(layers.get("motor", 0.0))
        sensory = float(layers.get("sensory", 0.0))
        assoc = float(layers.get("associative", 0.0))
        phoneme_act = 0.0
        if self.phoneme_neurons:
            phoneme_act = sum(n.activation for n in self.phoneme_neurons) / len(self.phoneme_neurons)
        motor_pressure = max(motor, phoneme_act)
        inhibition = max(0.0, assoc - sensory * 0.35) + max(0.0, extra_inhibition)
        ready = self.phonemes.count > 0
        will = ready and motor_pressure > 0.06 + inhibition * 0.12
        return SpeechEmergence(
            motor_pressure=motor_pressure,
            inhibition=inhibition,
            will_speak=will,
            syllables_ready=ready,
        )

    def utter(self) -> str:
        return self.utter_with_plan().text

    def utter_with_plan(self) -> MotorPlan:
        """Balbettio motorio — esplorazione con piano tracciabile per auto-ascolto."""
        em = self.readout()
        if not em.will_speak:
            return MotorPlan(text="", syllables=[], from_motor=True)

        pairs: list[tuple[str, float, int]] = []
        for n in self.phoneme_neurons:
            syl = self._neuron_syllable.get(n.id)
            if not syl:
                continue
            w = n.activation
            if w > 0.02:
                pairs.append((syl, w, n.id))

        if not pairs:
            active = sorted(
                (
                    (self._neuron_syllable[n.id], n.activation, n.id)
                    for n in self.phoneme_neurons
                    if n.id in self._neuron_syllable and n.activation > 0.03
                ),
                key=lambda x: -x[1],
            )
            parts = [s for s, _, _ in active[: 1 + int(em.motor_pressure * 3)]]
            text = "-".join(parts) if parts else ""
            return MotorPlan(text=text, syllables=parts, from_motor=True)

        pairs.sort(key=lambda x: -x[1])
        n_out = 1 + int(em.motor_pressure * 3)
        chosen: list[str] = []
        neuron_ids: list[int] = []
        for _ in range(n_out):
            total = sum(w for _, w, _ in pairs)
            r = self.rng.random() * total
            acc = 0.0
            for syl, w, nid in pairs:
                acc += w
                if r <= acc:
                    chosen.append(syl)
                    neuron_ids.append(nid)
                    break

        text = "-".join(chosen) if chosen else ""
        return MotorPlan(text=text, syllables=chosen, neuron_ids=neuron_ids, from_motor=True)

    def plan_from_text(self, text: str) -> MotorPlan:
        """Piano per parlato composto (dialogo/pensiero) — intenzione lessicale."""
        syl = split_italian_syllables(text)
        ids = [n.id for n in self.phoneme_neurons[: len(syl)]]
        return MotorPlan(text=text, syllables=syl, neuron_ids=ids, from_motor=False)

    def lexical_readout(
        self,
        words: list[str],
        *,
        punct: str = "",
        min_activation: float = 0.04,
    ) -> MotorPlan:
        """Assembla parole in sequenza dal motore fonetico — zero template di frase."""
        spoken: list[str] = []
        neuron_ids: list[int] = []
        all_syl: list[str] = []
        slot = 0
        seen: set[str] = set()
        for raw in words:
            w = raw.lower().strip()
            if not w or w in seen:
                continue
            seen.add(w)
            syls = split_italian_syllables(w)
            if not syls:
                continue
            word_ids: list[int] = []
            peak = 0.0
            for syl in syls:
                if slot >= len(self.phoneme_neurons):
                    break
                n = self.phoneme_neurons[slot]
                slot += 1
                self._neuron_syllable[n.id] = syl
                n.activation = min(1.0, n.activation + 0.28)
                peak = max(peak, n.activation)
                word_ids.append(n.id)
                all_syl.append(syl)
            if peak >= min_activation:
                spoken.append(w)
                neuron_ids.extend(word_ids)
        if self.brain:
            self.brain.propagate(steps=1)
        text = " ".join(spoken)
        if punct and text and not text.endswith(punct):
            text += punct
        if text:
            text = text[0].upper() + text[1:]
        return MotorPlan(text=text, syllables=all_syl, neuron_ids=neuron_ids, from_motor=True)
