"""Song motor — emotion → scale → melody (MIDI-like, synth-ready)."""

from __future__ import annotations

from dataclasses import dataclass, field

from mind.types import MindResult


@dataclass
class Note:
    pitch: str
    duration_beats: float


@dataclass
class SongResult:
    scale: str
    melody: list[Note]
    style: str
    symbols: list[str] = field(default_factory=list)


class SongModule:
    EMOTION_SCALE = {
        "happy": "C_major",
        "calm": "G_major",
        "sad": "A_minor",
        "tense": "E_phrygian",
        "empathetic": "D_mixolydian",
    }

    SCALE_NOTES = {
        "C_major": ["C4", "D4", "E4", "G4", "A4"],
        "G_major": ["G3", "A3", "B3", "D4", "E4"],
        "A_minor": ["A3", "C4", "D4", "E4", "G4"],
        "E_phrygian": ["E4", "F4", "G4", "A4", "B4"],
        "D_mixolydian": ["D4", "E4", "F#4", "A4", "B4"],
    }

    def __init__(self, brain, default_style: str = "lo_fi") -> None:
        self.brain = brain
        self.composers = brain.get_neurons("motor", "song_melody_composer")
        self.default_style = default_style

    def express(self, mind_result: MindResult, style: str | None = None, emotion: str = "calm") -> SongResult:
        style = style or self.default_style
        scale = self.EMOTION_SCALE.get(emotion, "C_major")
        notes = self.SCALE_NOTES.get(scale, self.SCALE_NOTES["C_major"])
        fragments = [f.title for f in mind_result.fragments[:4]]
        melody: list[Note] = []
        for i, title in enumerate(fragments or ["rest"]):
            pitch = notes[i % len(notes)]
            dur = 0.5 + (len(title) % 3) * 0.25
            melody.append(Note(pitch=pitch, duration_beats=dur))
        if not melody:
            melody = [Note(pitch=notes[0], duration_beats=1.0)]
        for i, _ in enumerate(melody[: len(self.composers)]):
            self.composers[i].fire(float(i), intensity=0.5)
        return SongResult(
            scale=scale,
            melody=melody,
            style=style,
            symbols=[f"MOTOR:song scale={scale} style={style}"],
        )
