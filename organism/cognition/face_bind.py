"""Volti ed emozioni — apprendimento graduale da esempi visivi."""

from __future__ import annotations

from typing import Any

from organism.sensory.face_vision import face_gist_similarity

_EMOTIONS = frozenset(
    {
        "felice", "triste", "arrabbiato", "sorpreso", "spaventato",
        "calmo", "neutro", "gioia", "rabbia", "paura", "disgusto",
    }
)


class FaceEmotionBinder:
    """Associa volti noti e espressioni emotive a gist facciali."""

    CONSOLIDATE_AT = 3

    def __init__(self) -> None:
        self._face_prototypes: dict[str, list[dict[str, Any]]] = {}
        self._face_trials: dict[str, int] = {}
        self._face_names: dict[str, str] = {}  # face_sig -> name
        self._emotion_prototypes: dict[str, list[dict[str, Any]]] = {}
        self._emotion_trials: dict[str, int] = {}
        self._emotion_labels: dict[str, str] = {}  # face_sig -> emotion

    def teach_face(
        self,
        name: str,
        face_data: dict[str, Any],
        *,
        boost: float = 1.0,
    ) -> dict[str, Any]:
        name = name.strip().lower()
        gist = face_data.get("gist") or {}
        if not name or not gist:
            return {"learned": False}
        sig = str(face_data.get("face_sig", ""))
        protos = self._face_prototypes.setdefault(name, [])
        protos.append(dict(gist))
        if len(protos) > 10:
            self._face_prototypes[name] = protos[-10:]
        if sig:
            self._face_trials[sig] = self._face_trials.get(sig, 0) + 1
            if self._face_trials[sig] >= self.CONSOLIDATE_AT:
                self._face_names[sig] = name
        consolidated = len(protos) >= self.CONSOLIDATE_AT
        return {
            "learned": consolidated,
            "name": name,
            "trials": len(protos),
            "consolidated": consolidated,
        }

    def teach_emotion(
        self,
        emotion: str,
        face_data: dict[str, Any],
        *,
        boost: float = 1.0,
    ) -> dict[str, Any]:
        emotion = emotion.strip().lower()
        gist = face_data.get("gist") or {}
        if not emotion or not gist:
            return {"learned": False}
        sig = str(face_data.get("face_sig", ""))
        protos = self._emotion_prototypes.setdefault(emotion, [])
        protos.append(dict(gist))
        if len(protos) > 12:
            self._emotion_prototypes[emotion] = protos[-12:]
        if sig:
            self._emotion_trials[sig] = self._emotion_trials.get(sig, 0) + 1
            if self._emotion_trials[sig] >= self.CONSOLIDATE_AT:
                self._emotion_labels[sig] = emotion
        consolidated = len(protos) >= self.CONSOLIDATE_AT
        return {
            "learned": consolidated,
            "emotion": emotion,
            "trials": len(protos),
            "consolidated": consolidated,
        }

    def recognize_face(self, face_data: dict[str, Any], *, min_sim: float = 0.52) -> str | None:
        gist = face_data.get("gist") or {}
        if not gist:
            return None
        sig = str(face_data.get("face_sig", ""))
        if sig and sig in self._face_names:
            return self._face_names[sig]
        best: str | None = None
        best_sim = 0.0
        for name, protos in self._face_prototypes.items():
            if not protos:
                continue
            sim = max(face_gist_similarity(gist, p) for p in protos[-6:])
            if sim >= min_sim and sim > best_sim:
                best_sim = sim
                best_name = name
        return best

    def recognize_emotion(self, face_data: dict[str, Any], *, min_sim: float = 0.48) -> str | None:
        if not face_data.get("detected"):
            return None
        gist = face_data.get("gist") or {}
        sig = str(face_data.get("face_sig", ""))
        if sig and sig in self._emotion_labels:
            return self._emotion_labels[sig]
        best: str | None = None
        best_sim = 0.0
        for emo, protos in self._emotion_prototypes.items():
            if not protos:
                continue
            sim = max(face_gist_similarity(gist, p) for p in protos[-6:])
            if sim >= min_sim and sim > best_sim:
                best_sim = sim
                best_name = emo
        return best

    def infer_emotion_hint(self, face_data: dict[str, Any]) -> str | None:
        """Indizio debole prima del consolidamento — solo se volto rilevato."""
        if not face_data.get("detected"):
            return None
        g = face_data.get("gist") or {}
        smile = float(g.get("smile_hint", 0))
        frown = float(g.get("frown_hint", 0))
        surprise = float(g.get("surprise_hint", 0))
        if smile > 0.14 and smile > frown:
            return "felice"
        if frown > 0.12 and frown > smile:
            return "arrabbiato"
        if surprise > 0.14:
            return "sorpreso"
        if float(g.get("mouth_curve", 0)) < -0.05:
            return "triste"
        return None

    def speech_themes(self, face_data: dict[str, Any]) -> list[str]:
        out: list[str] = []
        if not face_data.get("detected"):
            return out
        face = self.recognize_face(face_data)
        if face:
            out.append(face)
        emo = self.recognize_emotion(face_data) or self.infer_emotion_hint(face_data)
        if emo and emo not in out:
            out.append(emo)
        if face_data.get("detected") and "volto" not in out:
            out.insert(0, "volto")
        return out[:4]

    def stats(self) -> dict[str, Any]:
        return {
            "faces_known": len(self._face_prototypes),
            "emotions_known": len(self._emotion_prototypes),
            "face_names": list(self._face_prototypes.keys()),
            "emotion_labels": list(self._emotion_prototypes.keys()),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_prototypes": {k: list(v) for k, v in self._face_prototypes.items()},
            "face_trials": dict(self._face_trials),
            "face_names": dict(self._face_names),
            "emotion_prototypes": {k: list(v) for k, v in self._emotion_prototypes.items()},
            "emotion_trials": dict(self._emotion_trials),
            "emotion_labels": dict(self._emotion_labels),
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self._face_prototypes = {
            str(k): [dict(p) for p in v] for k, v in data.get("face_prototypes", {}).items()
        }
        self._face_trials = {str(k): int(v) for k, v in data.get("face_trials", {}).items()}
        self._face_names = {str(k): str(v) for k, v in data.get("face_names", {}).items()}
        self._emotion_prototypes = {
            str(k): [dict(p) for p in v] for k, v in data.get("emotion_prototypes", {}).items()
        }
        self._emotion_trials = {str(k): int(v) for k, v in data.get("emotion_trials", {}).items()}
        self._emotion_labels = {str(k): str(v) for k, v in data.get("emotion_labels", {}).items()}
