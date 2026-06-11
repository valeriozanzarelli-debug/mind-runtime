"""Benchmark fluenza — 100 prompt standard, scoring coerenza e sintassi."""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from organism.cognition.discourse import is_babble
from organism.cognition.human_thought import thought_coherence

# --- 100 prompt standard (categorie bilanciate) ---
_CORE: list[str] = [
    "ciao",
    "come stai",
    "chi sei",
    "cosa pensi",
    "cosa vedi",
    "cosa senti",
    "cosa impari",
    "perché esisti",
    "cos'è la coscienza",
    "cosa provi adesso",
    "non sono sicuro",
    "sto imparando",
    "descrivi i tuoi pensieri",
    "raccontami qualcosa",
    "raccontami una storia",
    "raccontami pinocchio",
    "perché piove",
    "se piove cosa succede",
    "come funziona il cervello",
    "cos'è la gravità",
    "che colore vedi",
    "c'è qualcosa sul tavolo",
    "il cane sta correndo",
    "vedi un oggetto",
    "riconosci questo",
    "è vero che impari",
    "mi spieghi",
    "non ho capito",
    "ripeti per favore",
    "grazie",
]

_REASONING: list[str] = [
    "perché il cielo è blu",
    "come fanno le piante a crescere",
    "cosa succede quando dormi",
    "perché le sinapsi si rinforzano",
    "se imparo cosa cambia",
    "quando capisci cosa senti",
    "mentre parli cosa pensi",
    "anche se sbagli cosa fai",
    "quindi cosa concludi",
    "perché fai domande",
]

_META: list[str] = [
    "sei sicuro di quello che dici",
    "cosa ricordi di ieri",
    "cosa vuoi imparare",
    "cosa ti incuriosisce",
    "ti senti tranquillo",
    "hai paura di qualcosa",
    "cosa ti rende felice",
    "stai riflettendo",
    "pensi a più cose insieme",
    "cosa significa capire",
]

_VISION: list[str] = [
    "cosa c'è davanti a te",
    "descrivi la scena",
    "quale forma vedi",
    "c'è movimento",
    "è grande o piccolo",
    "di che colore è",
    "è un animale",
    "è un oggetto",
    "riconosci la penna",
    "riconosci il cane",
]

_DIALOGUE: list[str] = [
    "dimmi di te",
    "parlami del mare",
    "cosa ne pensi del mondo",
    "hai mai visto il sole",
    "conosci il fuoco",
    "sai cos'è l'acqua",
    "cosa fai quando non capisci",
    "come rispondi alle domande",
    "perché parli così",
    "vuoi imparare di più",
]

_IT_VERBS = frozenset(
    {
        "sono", "sei", "è", "siamo", "siete", "ho", "hai", "ha", "vedo", "vedi", "vede",
        "penso", "pensi", "pensa", "sento", "sente", "capisco", "capisci", "imparo",
        "impari", "parlo", "parli", "corro", "corre", "sto", "sta", "voglio", "vuoi",
        "credo", "noto", "ricordo", "osservo", "rispondo", "chiedo", "spiego",
    }
)
_IT_SUBJECTS = frozenset({"io", "tu", "lui", "lei", "noi", "voi", "loro", "c", "il", "la", "un", "una"})


def standard_prompts(*, limit: int = 100) -> list[str]:
    """100 prompt — core + ragionamento + meta + visione + dialogo + oggetti."""
    pool = list(dict.fromkeys(_CORE + _REASONING + _META + _VISION + _DIALOGUE))
    obj_path = Path(__file__).resolve().parents[2] / "data" / "objects_it_1000.txt"
    if obj_path.exists():
        objects = [ln.strip().lower() for ln in obj_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rng = random.Random(42)
        sample = rng.sample(objects, min(40, len(objects)))
        for obj in sample:
            pool.append(f"vedi un {obj}")
            pool.append(f"cos'è un {obj}")
    # Pad con varianti
    extras = [
        "io penso che sia interessante",
        "quando guardo capisco meglio",
        "se fallisco riprovo",
        "mi chiedo come funziona",
        "noto qualcosa di nuovo",
    ]
    pool.extend(extras)
    return pool[:limit]


def has_subject_verb(text: str) -> bool:
    """Euristica SVO italiano — soggetto + verbo."""
    words = re.findall(r"[a-zàèéìòù']+", text.lower())
    if len(words) < 3:
        return False
    has_verb = any(w in _IT_VERBS or w.endswith(("are", "ere", "ire", "ando", "endo")) for w in words)
    has_subj = any(w in _IT_SUBJECTS for w in words[:4]) or words[0] == "io"
    return has_verb and (has_subj or words[0] in _IT_VERBS)


@dataclass
class PromptScore:
    prompt: str
    spoke: str
    words: int = 0
    coherent: bool = False
    score: float = 0.0
    from_thought: bool = False
    babble: bool = True
    svo: bool = False
    themes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "spoke": self.spoke[:160],
            "words": self.words,
            "coherent": self.coherent,
            "score": round(self.score, 3),
            "from_thought": self.from_thought,
            "babble": self.babble,
            "svo": self.svo,
        }


def score_response(
    prompt: str,
    spoke: str,
    *,
    themes: list[str] | None = None,
    from_thought: bool = False,
) -> PromptScore:
    words_n = len(spoke.split())
    babble = is_babble(spoke, min_words=5) if spoke.strip() else True
    svo = has_subject_verb(spoke) if spoke.strip() else False
    coh = thought_coherence(themes or [], heard=prompt) if themes else 0.0

    score = 0.0
    if spoke.strip():
        score += 0.15
    if not babble:
        score += 0.25
    if 10 <= words_n <= 15:
        score += 0.3
    elif 6 <= words_n <= 20:
        score += 0.15
    if svo:
        score += 0.2
    if from_thought:
        score += 0.1
    if coh > 0.35:
        score += 0.1

    coherent = score >= 0.55 and not babble and words_n >= 5
    return PromptScore(
        prompt=prompt,
        spoke=spoke,
        words=words_n,
        coherent=coherent,
        score=min(1.0, score),
        from_thought=from_thought,
        babble=babble,
        svo=svo,
        themes=list(themes or [])[:8],
    )


def aggregate_scores(results: list[PromptScore]) -> dict[str, Any]:
    if not results:
        return {"coherent_ratio": 0.0, "avg_words": 0.0, "avg_score": 0.0, "n": 0}
    coherent = sum(1 for r in results if r.coherent)
    words = [r.words for r in results if r.spoke.strip()]
    scores = [r.score for r in results]
    mature = sum(1 for r in results if 10 <= r.words <= 15 and not r.babble)
    return {
        "n": len(results),
        "coherent_ratio": round(coherent / len(results), 3),
        "mature_ratio": round(mature / len(results), 3),
        "avg_words": round(sum(words) / max(1, len(words)), 2),
        "avg_score": round(sum(scores) / len(scores), 3),
        "svo_ratio": round(sum(1 for r in results if r.svo) / len(results), 3),
        "babble_ratio": round(sum(1 for r in results if r.babble) / len(results), 3),
        "from_thought_ratio": round(sum(1 for r in results if r.from_thought) / len(results), 3),
        "pass_90": coherent / len(results) >= 0.9,
    }


def run_benchmark(
    sense_fn: Callable[[str], dict[str, Any]],
    *,
    limit: int = 100,
    pause_s: float = 0.15,
) -> dict[str, Any]:
    """Esegue benchmark — sense_fn(prompt) → {moment: {...}}."""
    prompts = standard_prompts(limit=limit)
    results: list[PromptScore] = []
    t0 = time.time()
    for p in prompts:
        try:
            out = sense_fn(p)
            m = out.get("moment") or out
            spoke = str(m.get("spoke", ""))
            thought = m.get("thought") or {}
            themes = list(thought.get("themes", []))
            results.append(
                score_response(
                    p,
                    spoke,
                    themes=themes,
                    from_thought=bool(m.get("from_thought")),
                )
            )
        except Exception as e:
            results.append(
                PromptScore(prompt=p, spoke="", coherent=False, score=0.0, babble=True)
            )
            results[-1].spoke = f"[err:{e!s:.40}]"
        time.sleep(pause_s)
    summary = aggregate_scores(results)
    return {
        "ok": True,
        "elapsed_s": round(time.time() - t0, 1),
        "summary": summary,
        "samples": [r.to_dict() for r in results[:12]],
        "worst": [r.to_dict() for r in sorted(results, key=lambda x: x.score)[:5]],
        "best": [r.to_dict() for r in sorted(results, key=lambda x: -x.score)[:5]],
        "results": [r.to_dict() for r in results],
    }
