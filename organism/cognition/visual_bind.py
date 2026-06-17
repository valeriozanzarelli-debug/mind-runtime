"""Associazioni visive — apprendimento graduale, colore+forma."""

from __future__ import annotations

import re
from typing import Any

from organism.sensory.object_scene import feature_similarity

_COLOR_NAMES = frozenset(
    {
        "rosso", "verde", "blu", "giallo", "nero", "bianco", "grigio", "rosa",
        "arancione", "marrone", "azzurro", "viola",
    }
)
_GEOM_SHAPES = frozenset(
    {"quadrato", "rotondo", "allungato", "ovale", "rettangolo", "triangolo", "cerchio"}
)


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zàèéìòù']+", text.lower()) if len(w) > 2]


class VisualBinder:
    """Firma scena → parole co-insegnate; consolidamento dopo N esposizioni."""

    CONSOLIDATE_AT = 3

    def __init__(self) -> None:
        self._scene_words: dict[str, dict[str, float]] = {}
        self._word_scenes: dict[str, set[str]] = {}
        self._object_features: dict[str, dict[str, Any]] = {}
        self._object_names: dict[str, str] = {}
        self._object_trials: dict[str, int] = {}
        self._object_candidates: dict[str, dict[str, int]] = {}
        self._prototypes: dict[str, list[dict[str, Any]]] = {}
        self._user_taught: set[str] = set()

    def bind(
        self,
        scene_sig: str,
        phrase: str,
        *,
        boost: float = 1.0,
        features: dict[str, Any] | None = None,
        object_sig: str = "",
        object_name: str = "",
        user_taught: bool = False,
    ) -> int:
        if not scene_sig or scene_sig == "empty" or not phrase.strip():
            return 0
        bucket = self._scene_words.setdefault(scene_sig, {})
        n = 0
        for w in _words(phrase):
            bucket[w] = min(5.0, bucket.get(w, 0.0) + boost)
            self._word_scenes.setdefault(w, set()).add(scene_sig)
            n += 1
        if features:
            self._object_features[scene_sig] = dict(features)
        osig = object_sig or scene_sig
        if object_name and osig:
            name = object_name.strip().lower()
            if user_taught:
                self._user_taught.add(name)
                self._object_names[osig] = name
                self._object_names[scene_sig] = name
            candidates = self._object_candidates.setdefault(osig, {})
            candidates[name] = candidates.get(name, 0) + 1
            self._object_trials[osig] = self._object_trials.get(osig, 0) + 1
            if candidates[name] >= self.CONSOLIDATE_AT:
                self._object_names[osig] = name
                self._object_names[scene_sig] = name
            if features:
                protos = self._prototypes.setdefault(name, [])
                protos.append(dict(features))
                if len(protos) > 8:
                    self._prototypes[name] = protos[-8:]
        color = (features or {}).get("color", "")
        if color:
            bucket[f"colore_{color}"] = min(5.0, bucket.get(f"colore_{color}", 0.0) + boost)
            bucket[color] = min(5.0, bucket.get(color, 0.0) + boost * 1.1)
        if object_name:
            bucket[object_name.strip().lower()] = min(
                5.0, bucket.get(object_name.strip().lower(), 0.0) + boost * 1.3
            )
        return n

    def _min_sim_for(self, features: dict[str, Any], base: float) -> float:
        scene = str(features.get("scene_type", "oggetto"))
        if scene in ("distesa", "orizzonte"):
            return min(base, 0.36)
        return base

    def confidence(self, features: dict[str, Any], *, object_sig: str = "") -> float:
        """Quanto è sicuro del riconoscimento 0..1."""
        rec = self.recognize_object(features, object_sig=object_sig, min_sim=0.38)
        if not rec:
            return 0.0
        osig = object_sig or str(features.get("object_sig", ""))
        trials = self._object_trials.get(osig, 0)
        base = min(1.0, trials / self.CONSOLIDATE_AT) if trials else 0.5
        for sig, feat in self._object_features.items():
            if self._object_names.get(sig) == rec:
                return min(1.0, base * feature_similarity(features, feat))
        protos = self._prototypes.get(rec, [])
        if protos:
            sim = max(feature_similarity(features, p) for p in protos)
            return min(1.0, base * sim)
        return base * 0.5

    def recognize_object(
        self,
        features: dict[str, Any],
        *,
        object_sig: str = "",
        min_sim: float = 0.55,
    ) -> str | None:
        thresh = self._min_sim_for(features, min_sim)
        if object_sig and object_sig in self._object_names:
            return self._object_names[object_sig]

        best_name: str | None = None
        best_sim = 0.0
        for name, protos in self._prototypes.items():
            if not protos:
                continue
            sim = max(feature_similarity(features, p) for p in protos[-6:])
            if sim >= thresh and sim > best_sim:
                best_sim = sim
                best_name = name

        for sig, feat in self._object_features.items():
            name = self._object_names.get(sig)
            if not name:
                continue
            sim = feature_similarity(features, feat)
            if sim >= thresh and sim > best_sim:
                best_sim = sim
                best_name = name
        return best_name

    def percept_words(self, features: dict[str, Any]) -> list[str]:
        """Nomi appresi — mai forme geometriche al posto del significato."""
        rec = self.recognize_object(
            features, object_sig=str(features.get("object_sig", "")), min_sim=0.38
        )
        if rec:
            return [rec]
        scene = str(features.get("scene_type", "oggetto"))
        if scene in ("distesa", "orizzonte"):
            return []
        return []

    def speech_themes(
        self,
        features: dict[str, Any],
        *,
        question: str = "vision",
        articulable: Any | None = None,
    ) -> list[str]:
        """Temi per parlato visivo — solo nomi consolidati, non quadrato+blu."""
        words = self.percept_words(features)
        color = str(features.get("color", ""))

        def ok(w: str) -> bool:
            if articulable is None:
                return True
            return articulable(w)

        ordered: list[str] = []
        for w in words:
            if ok(w) and w not in ordered and w not in _GEOM_SHAPES:
                ordered.append(w)
        if question == "color" and color and ok(color) and color not in ordered:
            ordered.insert(0, color)
        return [w for w in ordered if w][:5]

    def themes(
        self,
        scene_sig: str,
        features: dict[str, Any] | None = None,
        *,
        limit: int = 8,
    ) -> list[str]:
        out: list[str] = []
        feat = features or {}
        lum = float(feat.get("luminance", 0))
        con = float(feat.get("contrast", 0))
        color = str(feat.get("color", ""))
        if color:
            out.append(f"COL:{color}")
            out.append(color)
        if lum > 0.65:
            out.append("VIS:bright")
        elif lum < 0.25:
            out.append("VIS:dark")
        if con > 0.35:
            out.append("VIS:edges")
        rec = self.recognize_object(feat, object_sig=str(feat.get("object_sig", "")), min_sim=0.38)
        if rec:
            out.insert(0, rec)
            out.insert(0, f"OBJ:name={rec}")
        if scene_sig and scene_sig in self._scene_words:
            ranked = sorted(self._scene_words[scene_sig].items(), key=lambda x: -x[1])
            for w, _ in ranked[:limit]:
                if w not in out and not w.startswith("colore_") and w not in _GEOM_SHAPES:
                    out.append(w)
        for sym in feat.get("symbols", [])[:4]:
            if sym not in out:
                out.append(sym)
        return out[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenes": len(self._scene_words),
            "words": len(self._word_scenes),
            "bindings": {k: dict(v) for k, v in self._scene_words.items()},
            "object_features": dict(self._object_features),
            "object_names": dict(self._object_names),
            "object_trials": dict(self._object_trials),
            "object_candidates": {k: dict(v) for k, v in self._object_candidates.items()},
            "prototypes": {k: list(v) for k, v in self._prototypes.items()},
        }

    def load_dict(self, data: dict[str, Any]) -> None:
        self._scene_words = {
            str(k): {str(w): float(p) for w, p in v.items()} for k, v in data.get("bindings", {}).items()
        }
        self._object_features = {str(k): dict(v) for k, v in data.get("object_features", {}).items()}
        self._object_names = {str(k): str(v) for k, v in data.get("object_names", {}).items()}
        self._object_trials = {str(k): int(v) for k, v in data.get("object_trials", {}).items()}
        self._object_candidates = {
            str(k): {str(n): int(c) for n, c in v.items()}
            for k, v in data.get("object_candidates", {}).items()
        }
        self._prototypes = {
            str(k): [dict(p) for p in v] for k, v in data.get("prototypes", {}).items()
        }
        self._word_scenes = {}
        for sig, words in self._scene_words.items():
            for w in words:
                self._word_scenes.setdefault(w, set()).add(sig)
