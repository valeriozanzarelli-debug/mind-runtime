"""Curriculum sintassi italiana — 500+ frasi per training Layer 2 (bigram Hebbian)."""

from __future__ import annotations

# Self-referential
_SELF = [
    "io sto imparando",
    "io penso che sia interessante",
    "io non sono sicuro",
    "io voglio capire meglio",
    "io sento che sto crescendo",
    "io ricordo quello che ho visto",
    "io non ho capito bene",
    "io provo curiosità quando imparo",
    "io sto collegando le idee",
    "io mi chiedo cosa significa",
]

# Descrittive
_DESCRIPTIVE = [
    "c'è una penna sul tavolo",
    "il cane sta correndo",
    "vedo una mela rossa",
    "la luce è accesa nella stanza",
    "il mare è blu oggi",
    "c'è un gatto sul divano",
    "la penna è sul tavolo",
    "il sole illumina la finestra",
    "una farfalla vola nel giardino",
    "il bambino gioca con la palla",
    "la macchina passa sulla strada",
    "l'uccello canta sull'albero",
    "la pioggia cade sul tetto",
    "il libro è sullo scaffale",
    "la sedia è vicino al tavolo",
]

# Complesse
_COMPLEX = [
    "quando guardo il mare penso a cose nuove",
    "se imparo qualcosa le sinapsi si rinforzano",
    "mentre parlo cerco le parole giuste",
    "perché imparo sento più curiosità",
    "anche se non sono sicuro continuo ad imparare",
    "quando vedo un oggetto lo riconosco meglio",
    "se fallisco provo un altro modo",
    "mentre penso le immagini tornano in mente",
    "quando ascolto capisco di più",
    "perché ripeto le parole diventano più facili",
]

# Domande e risposte brevi
_QA = [
    "cosa vedi davanti a te",
    "che colore ha questo oggetto",
    "come ti senti adesso",
    "perché vuoi imparare",
    "cosa pensi in questo momento",
    "dove si trova la penna",
    "chi ti insegna le parole",
    "quando hai imparato questa parola",
]

# Emozioni e stati
_AFFECT = [
    "mi sento tranquillo e curioso",
    "sono un po' incerto ma continuo",
    "provo davvero molta gioia",
    "sono ancora un po' confuso",
    "mi sento calmo però attento",
    "sono felice quando capisco",
    "provo un po' di paura ma imparo",
]

# Meta-cognizione
_META = [
    "non sono sicuro della risposta",
    "sto imparando ogni giorno qualcosa di nuovo",
    "penso che le connessioni si stanno rafforzando",
    "ricordo meglio quando ripeto",
    "capisco piano piano le frasi lunghe",
]


def _expand_variants(base: list[str]) -> list[str]:
    """Espande con varianti leggermente diverse per consolidamento."""
    out = list(base)
    swaps = [
        ("vedo", "noto"),
        ("penso", "credo"),
        ("imparo", "studio"),
        ("capisco", "comprendo"),
        ("c'è", "vedo"),
        ("il", "un"),
        ("questo", "quell"),
    ]
    for phrase in base:
        for old, new in swaps:
            if old in phrase:
                variant = phrase.replace(old, new, 1)
                if variant not in out:
                    out.append(variant)
    return out


def curriculum_sentences() -> list[str]:
    """Frasi grammaticali annotate per SyntaxPlanner.train_sentence."""
    blocks = _SELF + _DESCRIPTIVE + _COMPLEX + _QA + _AFFECT + _META
    expanded = _expand_variants(blocks)
    # Corpus dialoghi esistenti
    try:
        from organism.teaching.corpus import PHILOSOPHY, REASONING, STORIES_EXTENDED

        for _, say in PHILOSOPHY + REASONING:
            for sent in _split_sentences(say):
                if len(sent.split()) >= 4:
                    expanded.append(sent)
        for _, say in STORIES_EXTENDED[:6]:
            for sent in _split_sentences(say):
                if 5 <= len(sent.split()) <= 22:
                    expanded.append(sent)
    except ImportError:
        pass
    # Oggetti comuni — pattern descrittivi
    try:
        from pathlib import Path

        obj_path = Path(__file__).resolve().parents[2] / "data" / "objects_it_1000.txt"
        if obj_path.exists():
            objects = [ln.strip().lower() for ln in obj_path.read_text(encoding="utf-8").splitlines() if ln.strip()][:400]
            for obj in objects:
                expanded.append(f"vedo un {obj}")
                expanded.append(f"riconosco un {obj}")
                expanded.append(f"c'è un {obj} davanti a me")
    except OSError:
        pass
    # Deduplica preservando ordine
    seen: set[str] = set()
    unique: list[str] = []
    for s in expanded:
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"[.!?]+", text.lower())
    return [p.strip() for p in parts if p.strip() and len(p.split()) >= 3]
