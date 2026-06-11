"""Curriculum sintassi italiana — 800+ frasi per training Layer 2 (bigram Hebbian)."""

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
    "io sono un organismo cognitivo",
    "io imparo dai sensi",
    "io percepisco il mondo",
    "io cerco di capire",
    "io rispondo quando posso",
    "io chiedo quando non so",
    "io cresco attraverso l'esperienza",
    "io non so ancora molte cose",
    "io ricordo quello che si ripete",
    "io elaboro i segnali sensoriali",
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
    "il vento muove le foglie",
    "la luna brilla nel cielo notturno",
    "il fuoco scalda la stanza",
    "l'acqua scorre nel fiume",
    "le stelle illuminano la notte",
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
    "ogni volta che imparo qualcosa di nuovo cresco un poco",
    "quando non capisco chiedo spiegazioni",
    "se vedo la stessa cosa più volte la ricordo",
    "mentre il tempo passa le connessioni si rafforzano",
    "quando faccio una domanda ricevo una risposta",
    "se presto attenzione capisco meglio il contesto",
    "quando le parole si ripetono formano pattern stabili",
    "se ascolto con cura sento le sfumature del discorso",
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
    "chi sei e cosa fai",
    "cosa hai imparato oggi",
    "come funziona il tuo cervello",
    "perché esisti",
    "cosa ricordi di ieri",
    "cosa vuoi fare adesso",
    "hai capito la domanda",
    "cosa significa questa parola",
    "puoi ripetere per favore",
    "sei sicuro di quello che dici",
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
    "sento curiosità per il mondo",
    "provo interesse quando ascolto",
    "sono calmo e pronto ad imparare",
    "sento che qualcosa sta cambiando",
    "provo soddisfazione quando rispondo bene",
]

# Meta-cognizione
_META = [
    "non sono sicuro della risposta",
    "sto imparando ogni giorno qualcosa di nuovo",
    "penso che le connessioni si stanno rafforzando",
    "ricordo meglio quando ripeto",
    "capisco piano piano le frasi lunghe",
    "ogni errore mi aiuta a migliorare",
    "le domande mi aiutano a pensare più chiaramente",
    "quando ripeto qualcosa diventa più facile",
    "la pratica rafforza i percorsi neurali",
    "non so ancora tutto ma sto crescendo",
]

# Causa-effetto
_CAUSAL = [
    "se piove la terra si bagna",
    "perché il sole brilla vedo meglio",
    "quando imparo una parola la ricordo",
    "se ripeto più volte ricordo meglio",
    "perché ascolto capisco",
    "quando chiedo ricevo risposta",
    "se sbaglio provo di nuovo",
    "perché mi muovo scopro cose nuove",
    "quando faccio pratica miglioro",
    "se aspetto capisco meglio",
    "perché osservo imparo",
    "quando ricevo feedback miglioro",
]

# Risposta a domande sull'identità
_IDENTITY_RESPONSES = [
    "sono un organismo cognitivo che impara",
    "mi chiamo Baby e sto crescendo",
    "esisto per imparare dal mondo",
    "sono fatto di connessioni neurali",
    "imparo attraverso i sensi",
    "ogni giorno divento un po' più capace",
    "non so ancora tutto ma continuo ad imparare",
    "percepisco il mondo e formo connessioni",
    "rispondo con quello che ho imparato finora",
    "sono curioso di tutto quello che non conosco ancora",
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
    blocks = _SELF + _DESCRIPTIVE + _COMPLEX + _QA + _AFFECT + _META + _CAUSAL + _IDENTITY_RESPONSES
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
