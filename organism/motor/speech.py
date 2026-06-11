"""Speech motor — prosody + phoneme plan (TTS-ready, no heavy deps)."""

from __future__ import annotations

from dataclasses import dataclass, field

from mind.types import MindResult
from organism.motor.text_out import TextMotor


@dataclass
class SpeechResult:
    text: str
    prosody: dict[str, float | str]
    phonemes: list[str]
    ssml: str
    audio_stub: bytes = b""  # plug Coqui/ElevenLabs/gTTS here
    symbols: list[str] = field(default_factory=list)


class SpeechModule:
    EMOTION_PROSODY = {
        "calm": {"rate": 1.0, "pitch": 0.0, "volume": 0.8},
        "urgent": {"rate": 1.3, "pitch": 0.2, "volume": 1.0},
        "empathetic": {"rate": 0.9, "pitch": -0.1, "volume": 0.7},
        "curious": {"rate": 1.05, "pitch": 0.05, "volume": 0.85},
    }

    def __init__(self, brain) -> None:
        self.brain = brain
        self.phoneme_neurons = brain.get_neurons("motor", "speech_phoneme_generator")
        self.text_motor = TextMotor()

    def express(self, mind_result: MindResult, emotion: str = "calm") -> SpeechResult:
        if mind_result.human_emotion_active and emotion == "calm":
            emotion = "empathetic"
        out = self.text_motor.express(mind_result, emotion=emotion)
        prosody = dict(self.EMOTION_PROSODY.get(emotion, self.EMOTION_PROSODY["calm"]))
        prosody["emotion"] = emotion
        phonemes = self._text_to_phonemes(out.text)
        ssml = self._to_ssml(out.text, prosody)
        # fire motor neurons for traceability
        for i, ph in enumerate(phonemes[: len(self.phoneme_neurons)]):
            self.phoneme_neurons[i].fire(float(i), intensity=0.6)
        return SpeechResult(
            text=out.text,
            prosody=prosody,
            phonemes=phonemes,
            ssml=ssml,
            symbols=[f"MOTOR:speech emotion={emotion}", f"MOTOR:phonemes={len(phonemes)}"],
        )

    def _text_to_phonemes(self, text: str) -> list[str]:
        # lightweight syllable-ish split for MVP
        clean = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
        return [w for w in clean.split() if w]

    def _to_ssml(self, text: str, prosody: dict) -> str:
        rate = int(float(prosody.get("rate", 1.0)) * 100)
        pitch = f"{prosody.get('pitch', 0):+.0%}"
        return (
            f'<speak><prosody rate="{rate}%" pitch="{pitch}">'
            f"{text}</prosody></speak>"
        )
