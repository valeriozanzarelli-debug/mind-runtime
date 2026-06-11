"""Audio sensory module — frequency bands → spikes."""

from __future__ import annotations

import math
import time
import wave
from dataclasses import dataclass, field
from io import BytesIO

from organism.brain.topology import ActivePattern, NeuralTopology, Spike
from organism.sensory._array import fft_band_energy


@dataclass
class AudioResult:
    spikes: list[Spike]
    patterns: list[ActivePattern]
    dominant_band: str = ""
    symbols: list[str] = field(default_factory=list)


class AudioModule:
    def __init__(self, brain: NeuralTopology) -> None:
        self.brain = brain
        self.analyzers = brain.get_neurons("sensory", "audio_frequency_analyzer")

    def perceive(self, audio_bytes: bytes) -> AudioResult:
        samples, rate = self._decode(audio_bytes)
        t = time.time()
        spikes: list[Spike] = []
        band_power: dict[str, float] = {}

        for an in self.analyzers:
            band = an.frequency_band()
            if band is None:
                continue
            low, high = band
            power = fft_band_energy(samples, rate, low, high)
            band_power[f"{int(low)}-{int(high)}"] = power
            threshold = float(an.meta.get("threshold", 0.1))
            norm = power / max(1.0, len(samples))
            if norm > threshold:
                spikes.append(Spike(neuron_id=an.id, timestamp=t, intensity=min(1.0, norm * 10)))

        self.brain.inject_spikes(spikes)
        self.brain.propagate(steps=2)
        patterns = self.brain.get_active_patterns(threshold=0.3, modality="audio")

        dominant = max(band_power, key=band_power.get) if band_power else ""
        return AudioResult(
            spikes=spikes,
            patterns=patterns,
            dominant_band=dominant,
            symbols=[f"AUD:spikes={len(spikes)}", f"AUD:band={dominant}"],
        )

    def perceive_tone(self, frequency_hz: float, duration_s: float = 0.25, sample_rate: int = 8000) -> AudioResult:
        """Synthetic tone for tests without WAV files."""
        n = int(sample_rate * duration_s)
        samples = [math.sin(2 * math.pi * frequency_hz * i / sample_rate) for i in range(n)]
        raw = self._encode_pcm(samples, sample_rate)
        return self.perceive(raw)

    def _decode(self, data: bytes) -> tuple[list[float], int]:
        try:
            with wave.open(BytesIO(data), "rb") as wf:
                rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                width = wf.getsampwidth()
                if width == 1:
                    samples = [float(b) - 128 for b in frames]
                else:
                    samples = []
                    for i in range(0, len(frames), width):
                        val = int.from_bytes(frames[i : i + width], "little", signed=True)
                        samples.append(float(val))
                return samples, rate
        except Exception:
            # raw PCM fallback
            samples = [float(b) - 128 for b in data[:8000]]
            return samples, 8000

    def _encode_pcm(self, samples: list[float], rate: int) -> bytes:
        buf = BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            frames = b"".join(
                int(max(-32768, min(32767, s * 3000))).to_bytes(2, "little", signed=True) for s in samples
            )
            wf.writeframes(frames)
        return buf.getvalue()
