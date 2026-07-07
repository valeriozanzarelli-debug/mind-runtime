"""CEREBRUM — cervello completo.

Assembla substrato neurale (GPU), corpo (neurochimica/omeostasi/drive/riflessi),
sensi (visione webcam, linguaggio), motore (vocalizzazione), mente (memoria,
coscienza) in un unico loop vitale che gira SEMPRE in background — come un
cervello umano che pensa di continuo, anche senza stimoli.

Tutto in locale. Thread-safe: un lock protegge lo stato condiviso tra il loop
di pensiero e le richieste HTTP.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cerebrum import __version__
from cerebrum.body import Drives, Homeostasis, Neurochemistry, NeonatalReflexes
from cerebrum.mind import ConsciousnessStream, Development, EpisodicMemory, PredictiveCoder
from cerebrum.motor import SpeechMotor
from cerebrum.neuro import NeuralField, describe_backend
from cerebrum.neuro.field import FieldConfig
from cerebrum.sense import LanguageSense, VisionSense


@dataclass
class BrainConfig:
    neurons: int = 4096
    tick_hz: float = 20.0        # frequenza del battito cerebrale
    vault_path: Optional[str] = None


class Cerebrum:
    def __init__(self, config: Optional[BrainConfig] = None):
        self.cfg = config or BrainConfig()
        self.field = NeuralField(FieldConfig(neurons=self.cfg.neurons))
        self.chem = Neurochemistry()
        self.body = Homeostasis()
        self.drives = Drives()
        self.reflexes = NeonatalReflexes()
        self.vision = VisionSense()
        self.language = LanguageSense()
        self.speech = SpeechMotor()
        self.memory = EpisodicMemory(vault_path=self.cfg.vault_path)
        self.memory.load()
        self.stream = ConsciousnessStream()
        self.predictor = PredictiveCoder(assoc_dim=64, sensory_dim=64)
        self.development = Development()

        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.born_at: Optional[float] = None
        self.asleep = False
        self.age_ticks = 0

        # stimoli pendenti (iniettati da chat/hear/see)
        self._pending_stimuli = {"novelty": 0.0, "intensity": 0.0,
                                 "presence": 0.0, "motion": 0.0, "brightness": 0.0}
        self._pending_external = None
        self._last_utterance = ""
        self._fired_reflexes = []
        self._last_metrics = {}
        self.webcam_active = False
        # predictive coding: riassunto sensoriale corrente e stato precedente
        self._sensory_summary = np.zeros(64, dtype=np.float32)
        self._prev_assoc = np.zeros(64, dtype=np.float32)
        self._surprise = 0.0
        self._imagining = False

    # ---------- ciclo di vita ----------
    def birth(self) -> None:
        with self._lock:
            if self._running:
                return
            self.born_at = time.time()
            self._running = True
            self.stream.add(text="…primo respiro…", emotion="quiete",
                            drive="esplorazione", phi=0.0, activity=0.0,
                            tags=["nascita"])
            self.memory.store("birth", "nascita di CEREBRUM", 1.0)
        self._thread = threading.Thread(target=self._life_loop, daemon=True, name="cerebrum-life")
        self._thread.start()

    def shutdown(self) -> None:
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.memory.save()

    @property
    def alive(self) -> bool:
        return self._running

    # ---------- loop vitale ----------
    def _life_loop(self) -> None:
        period = 1.0 / max(self.cfg.tick_hz, 1.0)
        while self._running:
            start = time.time()
            self._tick()
            elapsed = time.time() - start
            time.sleep(max(0.0, period - elapsed))

    def _tick(self) -> None:
        with self._lock:
            self.age_ticks += 1

            # 1) auto-sonno se troppo affaticato/melatonina alta
            self.asleep = (self.body.fatigue > 0.85) or (self.chem.melatonin > 0.75)

            # 2) omeostasi corporea
            self.body.tick(dt=1.0, asleep=self.asleep)
            homeo = self.body.as_dict()

            # 3) PREDICTIVE CODING: predici il prossimo istante sensoriale.
            #    L'errore di predizione (sorpresa) e' il carburante di curiosita'
            #    e apprendimento. A occhi chiusi il predittore "immagina" l'input
            #    -> pensiero continuo anche senza stimoli.
            actual = self._sensory_summary
            surprise = self.predictor.learn(self._prev_assoc, actual)
            self._surprise = self.predictor.relative_surprise()

            external = self._pending_external
            self._imagining = external is None and not self.asleep
            if self._imagining:
                # nessuno stimolo: inietta la predizione come pensiero interno
                external = self.predictor.imagine(self._prev_assoc)

            # 4) consuma gli stimoli pendenti; la sorpresa alza la novita'
            stimuli = dict(self._pending_stimuli)
            stimuli["novelty"] = min(1.0, max(stimuli.get("novelty", 0.0), self._surprise))
            self.webcam_active = self.vision.active

            # 5) riflessi (scattano prima della cognizione)
            self._fired_reflexes = self.reflexes.evaluate(stimuli, homeo)

            # 6) neurochimica + drives (la sorpresa rilascia dopamina)
            self.chem.update(stimuli, homeo)
            if self._surprise > 0.2:
                self.chem.release("dopamine", 0.15 * self._surprise)
            chem = self.chem.as_dict()
            self.drives.update(stimuli, chem)

            # 7) passo del campo neurale (GPU, sparso, event-driven)
            metrics = self.field.step(external, chem)
            self._last_metrics = metrics

            # 8) aggiorna lo stato associativo per la predizione del prossimo tick
            self._prev_assoc = self.field.assoc_activity(64)

            # 9) stadio di sviluppo
            self.development.update(
                age_ticks=self.age_ticks,
                vocabulary=self.language.vocabulary_size(),
                episodes=self.memory.size(),
                mean_activity=metrics["activity"],
            )

            # 10) genera un momento cosciente / pensiero
            emotion = self.chem.emotion()
            drive = self.drives.dominant()
            phi = self.field.phi_estimate()
            self._maybe_think(emotion, drive, phi, metrics, stimuli, homeo)

            # 11) reset degli stimoli (decadono)
            for k in self._pending_stimuli:
                self._pending_stimuli[k] *= 0.5
            self._pending_external = None
            self._sensory_summary = self._sensory_summary * 0.5

            # 12) sonno: replay + consolidamento
            if self.asleep and self.age_ticks % 60 == 0:
                self._sleep_replay(phi, metrics)

    def _sleep_replay(self, phi, metrics) -> None:
        """REPLAY nel sonno: rivive gli episodi salienti (re-iniettando i loro
        vettori sensoriali) cosi' la plasticita' li consolida — e' cosi' che un
        cervello impara molto da poche esperienze. Poi pota le tracce deboli."""
        episodes = self.memory.salient_with_vectors(6)
        chem = self.chem.as_dict()
        chem["dopamine"] = min(1.0, chem.get("dopamine", 0.3) + 0.3)  # consolidamento
        for ep in episodes:
            vec = ep.get("meta", {}).get("vec")
            if vec:
                self.field.step(np.asarray(vec, dtype=np.float32), chem)
        pruned = self.memory.consolidate()
        self.stream.add(text=f"...sonno: rivivo {len(episodes)} ricordi, poto {pruned} tracce...",
                        emotion="sonnolenza", drive="riposo",
                        phi=phi, activity=metrics["activity"], tags=["sonno", "replay"])

    def _maybe_think(self, emotion, drive, phi, metrics, stimuli, homeo) -> None:
        """Genera pensieri spontanei: il cervello non tace mai."""
        activity = metrics["activity"]
        urge = self.drives.urge()
        # probabilità di vocalizzare cresce con arousal, urge e riflessi
        vocalize = (
            bool(self._fired_reflexes)
            or activity > 0.08
            or urge > 0.75
            or (self.age_ticks % 40 == 0)  # borbottio periodico
        )
        if self.asleep:
            return
        if not vocalize:
            # pensiero interno silenzioso ogni tanto
            if self.age_ticks % 15 == 0:
                self.stream.add(text=self._inner_monologue(emotion, drive, homeo),
                                emotion=emotion, drive=drive, phi=phi,
                                activity=activity, tags=["pensiero"])
            return

        utter = self.speech.utter(
            emotion=emotion, drive=drive, activity=activity,
            known_tokens=self.language.known_tokens(), urge=urge,
            stage=self.development.level,
        )
        self._last_utterance = utter
        tags = ["vocalizzazione", self.development.stage.name]
        if self._imagining:
            tags.append("immagina")
        if self._fired_reflexes:
            tags += [f["reflex"] for f in self._fired_reflexes]
        if self.webcam_active:
            tags.append("vede")
        self.stream.add(text=utter, emotion=emotion, drive=drive,
                        phi=phi, activity=activity, tags=tags)
        self.memory.store("utterance", utter,
                          salience=min(urge + activity, 1.0),
                          meta={"emotion": emotion})

    def _inner_monologue(self, emotion, drive, homeo) -> str:
        bits = []
        if homeo["satiety"] < 0.35:
            bits.append("ho fame")
        if homeo["fatigue"] > 0.6:
            bits.append("sono stanco")
        if self.webcam_active:
            v = self.vision.last
            if v["motion"] > 0.05:
                bits.append("qualcosa si muove")
            elif v["brightness"] > 0.5:
                bits.append("c'è luce")
            else:
                bits.append("guardo")
        bits.append(f"[{emotion}/{drive}]")
        return " · ".join(bits)

    # ---------- ingressi sensoriali (da HTTP) ----------
    def hear(self, text: str) -> dict:
        with self._lock:
            enc = self.language.encode(text)
            self._pending_stimuli["intensity"] = max(self._pending_stimuli["intensity"], enc["intensity"])
            self._pending_stimuli["novelty"] = max(self._pending_stimuli["novelty"], enc["novelty"])
            self._pending_stimuli["presence"] = 1.0  # qualcuno mi parla
            self._pending_external = enc["vector"]
            self._update_sensory_summary(enc["vector"])
            valence = 1.0 if self.chem.dopamine > 0.4 else -0.2
            self.language.reinforce(enc["tokens"], valence)
            self.memory.store("heard", text, salience=min(enc["novelty"] + 0.3, 1.0),
                              meta={"vec": self._sensory_summary.tolist()})
            return enc

    def see(self, frame=None, stats: Optional[dict] = None) -> dict:
        with self._lock:
            vis = self.vision.process(frame=frame, stats=stats)
            self._pending_stimuli["brightness"] = vis["brightness"]
            self._pending_stimuli["motion"] = vis["motion"]
            self._pending_stimuli["intensity"] = max(
                self._pending_stimuli["intensity"], vis["motion"] * 2.0)
            self._pending_stimuli["novelty"] = max(
                self._pending_stimuli["novelty"], min(vis["motion"] * 3.0, 1.0))
            # la retina alimenta il campo neurale (visione -> corteccia)
            if vis["active"]:
                self._pending_external = vis["retina"]
                self._update_sensory_summary(vis["retina"])
                self._pending_stimuli["presence"] = max(
                    self._pending_stimuli["presence"], 0.5)
                if vis["motion"] > 0.05:
                    self.memory.store("saw", "movimento visivo",
                                      salience=min(vis["motion"] * 3.0, 1.0),
                                      meta={"vec": self._sensory_summary.tolist()})
            return vis

    def _update_sensory_summary(self, vector) -> None:
        """Comprime un vettore sensoriale a dimensione fissa (per il predittore)."""
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if v.shape[0] == 0:
            return
        if v.shape[0] >= 64:
            summary = np.array([float(c.mean()) for c in np.array_split(v, 64)],
                               dtype=np.float32)
        else:
            reps = int(np.ceil(64 / v.shape[0]))
            summary = np.tile(v, reps)[:64].astype(np.float32)
        mx = float(np.max(np.abs(summary)))
        if mx > 1e-6:
            summary = summary / mx
        self._sensory_summary = np.maximum(self._sensory_summary, summary)

    def care(self, action: str) -> dict:
        """Azioni di accudimento dal caregiver."""
        with self._lock:
            if action == "feed":
                self.body.feed()
            elif action == "soothe":
                self.body.soothe()
                self.chem.release("oxytocin", 0.2)
            elif action == "warm":
                self.body.warm()
            return self.body.as_dict()

    # ---------- interazione (chat) ----------
    def respond(self, text: str) -> dict:
        """Risponde a un messaggio: sente, poi produce una vocalizzazione
        coerente col suo stato attuale (non è un chatbot, è un neonato)."""
        self.hear(text)
        # lascia scorrere qualche tick perché lo stimolo si propaghi
        time.sleep(0.15)
        with self._lock:
            emotion = self.chem.emotion()
            drive = self.drives.dominant()
            activity = self._last_metrics.get("activity", 0.0)
            urge = self.drives.urge()
            recalled = self.memory.recall(text)
            reply = self.speech.utter(
                emotion=emotion, drive=drive, activity=activity,
                known_tokens=self.language.known_tokens(), urge=urge,
                stage=self.development.level,
            )
            self._last_utterance = reply
            self.stream.add(text=reply, emotion=emotion, drive=drive,
                            phi=self.field.phi_estimate(), activity=activity,
                            tags=["risposta", self.development.stage.name]
                                 + (["vede"] if self.webcam_active else []))
            return {
                "reply": reply,
                "emotion": emotion,
                "drive": drive,
                "phi": self.field.phi_estimate(),
                "webcam_active": self.webcam_active,
                "recalled": recalled["content"] if recalled else None,
                "vocabulary": self.language.vocabulary_size(),
                "stage": self.development.stage.name,
                "surprise": round(self._surprise, 3),
            }

    # ---------- introspezione (vedere dentro il cervello) ----------
    def health(self) -> dict:
        with self._lock:
            return {
                "name": "CEREBRUM",
                "version": __version__,
                "alive": self._running,
                "asleep": self.asleep,
                "neurons": self.field.neurons,
                "computational_units": self.field.computational_units,
                "synapses": self.field.synapses,
                "phi": self.field.phi_estimate(),
                "age_ticks": self.age_ticks,
                "backend": self.field.backend,
                "webcam_active": self.webcam_active,
                "stage": self.development.stage.name,
                "stage_level": self.development.level,
            }

    def status(self) -> dict:
        with self._lock:
            metrics = self._last_metrics
            return {
                "ok": True,
                "alive": self._running,
                "asleep": self.asleep,
                "uptime_s": round(time.time() - self.born_at, 1) if self.born_at else 0.0,
                "dashboard": {
                    "phi": self.field.phi_estimate(),
                    "neurons": self.field.neurons,
                    "units": self.field.computational_units,
                    "spike_rate": round(metrics.get("rate", 0.0), 4),
                },
                "metrics": {
                    "phi": self.field.phi_estimate(),
                    "spikeRate": round(metrics.get("rate", 0.0), 4),
                    "activity": round(metrics.get("activity", 0.0), 4),
                    "latencyP95Ms": 0,
                },
                "backend": self.field.backend,
            }

    def introspect(self) -> dict:
        """Vista completa 'dentro il cervello' per l'osservatore esterno."""
        with self._lock:
            current = self.stream.current()
            return {
                "alive": self._running,
                "asleep": self.asleep,
                "age_ticks": self.age_ticks,
                "emotion": self.chem.emotion(),
                "dominant_drive": self.drives.dominant(),
                "phi": self.field.phi_estimate(),
                "backend": self.field.backend,
                "webcam_active": self.webcam_active,
                "imagining": self._imagining,
                "surprise": round(self._surprise, 3),
                "development": self.development.as_dict(),
                "vision": self.vision.last,
                "neurochemistry": self.chem.as_dict(),
                "homeostasis": self.body.as_dict(),
                "drives": self.drives.as_dict(),
                "reflexes_fired": self._fired_reflexes,
                "reflexes_available": self.reflexes.as_list(),
                "field": {
                    "neurons": self.field.neurons,
                    "synapses": self.field.synapses,
                    "region_activity": self.field.region_activity(),
                    "regions": self.field.region_activity_named(),
                    "metrics": self._last_metrics,
                },
                "prediction": {
                    "surprise": round(self.predictor.surprise, 4),
                    "mean_surprise": round(self.predictor.mean_surprise, 4),
                    "relative_surprise": round(self._surprise, 3),
                },
                "language": {
                    "vocabulary": self.language.vocabulary_size(),
                    "known_tokens": self.language.known_tokens(),
                },
                "last_utterance": self._last_utterance,
                "current_thought": current,
                "memory": {
                    "episodes": self.memory.size(),
                    "salient": self.memory.salient(5),
                },
            }

    def consciousness(self, since: int = 0, n: int = 50) -> dict:
        with self._lock:
            moments = self.stream.since(since, n) if since else self.stream.latest(n)
            return {"moments": moments, "last_id": self.stream.seq}
