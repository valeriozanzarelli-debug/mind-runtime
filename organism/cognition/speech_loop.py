"""Loop sensorimotorio del parlato — modello DIVA / Hebb.

Fasi (come nello sviluppo infantile):
1. Balbettio — motore esplora, ascolta la propria voce, corregge
2. Associazione — A1+M1 co-attivi si legano (Hebb)
3. Feedback sociale — risposta del caregiver rinforza la vocalizzazione

Riferimenti: Guenther DIVA, Pulvermüller Hebbian assemblies, Goldstein social vocal learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from organism.brain.growth import active_ids, depress_synapses, wire_coactive, wire_motor_to_sensory
from organism.brain.topology import NeuralTopology
from organism.cognition.motor_plan import MotorPlan
from organism.teaching.phonemes import PhonemeLearner, split_italian_syllables

if TYPE_CHECKING:
    from organism.motor.emergent_speech import EmergentSpeechMotor

Source = Literal["self", "caregiver", "social"]


@dataclass
class SpeechError:
    """Discrepanza tra intenzione motoria e feedback uditivo."""

    intended: list[str]
    heard: list[str]
    similarity: float
    mismatch: float
    corrected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "intended": self.intended,
            "heard": self.heard,
            "similarity": round(self.similarity, 3),
            "mismatch": round(self.mismatch, 3),
            "corrected": self.corrected,
        }


def syllable_similarity(intended: list[str], heard: list[str]) -> float:
    """LCS normalizzata — quanto la produzione assomiglia al piano."""
    if not intended and not heard:
        return 1.0
    if not intended or not heard:
        return 0.0
    m, n = len(intended), len(heard)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if intended[i - 1] == heard[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    return lcs / max(m, n)


class SpeechSensorimotorLoop:
    """Chiusura del loop produzione → percezione → errore → plasticità."""

    def __init__(self) -> None:
        self._last_spoke_t: float = 0.0
        self._last_spoke_text: str = ""
        self._babble_cycles: int = 0
        self._corrections: int = 0

    def compare(self, intended: list[str], heard: list[str]) -> SpeechError:
        sim = syllable_similarity(intended, heard)
        mismatch = 1.0 - sim
        corrected = mismatch > 0.15
        return SpeechError(
            intended=intended,
            heard=heard,
            similarity=sim,
            mismatch=mismatch,
            corrected=corrected,
        )

    def self_hear(
        self,
        brain: NeuralTopology,
        speech: EmergentSpeechMotor,  # noqa: F821
        *,
        heard_text: str,
        plan: MotorPlan | None = None,
        source: Source = "self",
    ) -> dict[str, Any]:
        """Ascolta la propria voce — attiva ventrale (comprensione) + dorsale (correzione)."""
        heard_text = heard_text.strip()
        if not heard_text:
            return {}

        heard_syl = split_italian_syllables(heard_text)
        intended_syl = plan.syllables if plan else heard_syl
        err = self.compare(intended_syl, heard_syl)

        boost = 0.35 if source == "self" else 0.85
        speech.hear(heard_text, boost=boost)

        pre_motor = plan.neuron_ids if plan and plan.neuron_ids else [
            n.id for n in speech.phoneme_neurons if n.activation > 0.15
        ]
        sensory_pre = active_ids(brain, "sensory", "text_semantic_encoder", min_act=0.12)
        if pre_motor:
            wire_coactive(brain, pre_motor[:10], sensory_pre[:10], max_new=3, weight=0.14)
        wires = wire_motor_to_sensory(brain)

        brain.propagate(steps=1)

        phoneme_delta = self._apply_phoneme_feedback(speech.phonemes, err, source)
        synapse_delta = self._apply_brain_feedback(brain, err, pre_motor)

        if err.corrected:
            self._corrections += 1
        if plan and plan.from_motor:
            self._babble_cycles += 1

        import time

        self._last_spoke_t = time.time()
        self._last_spoke_text = heard_text

        return {
            "self_heard": True,
            "source": source,
            "speech_error": err.to_dict(),
            "wires_grown": wires,
            "phoneme_adjusted": phoneme_delta,
            "synapses_adjusted": synapse_delta,
            "babble_cycles": self._babble_cycles,
            "corrections": self._corrections,
        }

    def social_feedback(
        self,
        brain: NeuralTopology,
        speech: EmergentSpeechMotor,  # noqa: F821
        *,
        caregiver_text: str,
        within_seconds: float,
    ) -> dict[str, Any]:
        """Risposta contingente del caregiver — rinforza la vocalizzazione precedente."""
        import time

        elapsed = time.time() - self._last_spoke_t
        if not self._last_spoke_text or elapsed > within_seconds:
            return {"social": False, "reason": "no_recent_vocalization"}

        speech.hear(caregiver_text, boost=1.1)
        pre = active_ids(brain, "motor", "speech_phoneme_generator", min_act=0.1)
        post = active_ids(brain, "sensory", "text_semantic_encoder", min_act=0.1)
        grown = wire_coactive(brain, pre[:8], post[:8], max_new=4, weight=0.22)
        brain.reinforce_active_pathway(boost=0.04)
        if brain.plasticity:
            brain.plasticity.apply_hebbian(brain, brain.tick)

        return {
            "social": True,
            "latency_s": round(elapsed, 2),
            "after": self._last_spoke_text[:60],
            "wires_grown": grown,
        }

    def _apply_phoneme_feedback(
        self,
        phonemes: PhonemeLearner,
        err: SpeechError,
        source: Source,
    ) -> int:
        adjusted = 0
        if err.similarity >= 0.85:
            adjusted += phonemes.reinforce_sequence(err.heard, boost=0.25 if source == "self" else 0.4)
        elif err.corrected:
            adjusted += phonemes.reinforce_sequence(err.intended, boost=0.12)
            adjusted += phonemes.penalize_mismatch(err.intended, err.heard, penalty=0.18)
        return adjusted

    def _apply_brain_feedback(
        self,
        brain: NeuralTopology,
        err: SpeechError,
        motor_ids: list[int],
    ) -> int:
        if err.similarity >= 0.85:
            return brain.reinforce_active_pathway(boost=0.035)
        if err.corrected and motor_ids:
            return depress_synapses(brain, motor_ids, amount=0.025)
        return 0

    def stats(self) -> dict[str, Any]:
        return {
            "babble_cycles": self._babble_cycles,
            "corrections": self._corrections,
            "last_spoke": self._last_spoke_text[:80],
        }
