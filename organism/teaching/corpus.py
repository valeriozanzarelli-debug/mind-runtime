"""Corpus training — fiabe, ragionamento, scienza (pubblico dominio / didattico)."""

from __future__ import annotations

# Pinocchio — Collodi, pubblico dominio (estratto)
PINOCCHIO_OPENING = (
    "C'era una volta un pezzo di legno. Non era un legno di lusso, ma un semplice pezzo da catasta, "
    "di quelli che d'inverno si mettono nelle stufe per accendere il fuoco. "
    "Un bel giorno questo pezzo di legno capitò nella bottega di un vecchio falegname, "
    "che aveva nome mastr'Antonio, ma tutti lo chiamavano maestro Ciliegia. "
    "Appena vide quel legno si rallegrò e disse: questo legno è capitato a tempo, "
    "voglio servirmene per fare una gamba di tavolino."
)

# Esopo — pubblico dominio
ESOPO_FABLES: list[tuple[str, str]] = [
    (
        "raccontami la favola del corvo",
        "Un corvo aveva trovato un pezzo di formaggio e se ne stava sul ramo di un albero "
        "quando arrivò una volpe. La volpe disse al corvo: che belle piume hai, che occhi splendidi. "
        "Se la tua voce è bella come il tuo aspetto, sei davvero il re degli uccelli. "
        "Il corvo aprì il becco per cantare e il formaggio cadde. La volpe lo prese e disse: "
        "il tuo formaggio è mio, ma la lezione è per te: non fidarti di chi ti lusinga.",
    ),
    (
        "cosa insegna la favola del corvo",
        "insegna a non credere alle lodi vuote e a pensare prima di agire.",
    ),
]

# Ragionamento causale
REASONING: list[tuple[str, str]] = [
    (
        "perché piove",
        "quando l'acqua nell'aria diventa troppo pesante, cade verso terra e noi la chiamiamo pioggia.",
    ),
    (
        "se piove cosa succede",
        "la terra si bagna, le strade brillano e i fiumi ricevono più acqua.",
    ),
    (
        "e se non piovesse",
        "la terra resterebbe secca, le piante avrebbero sete e i fiumi scenderebbero.",
    ),
    (
        "perché esisti",
        "esisto per imparare dal mondo attraverso i sensi e trasformare l'esperienza in connessioni.",
    ),
    (
        "cos'è la gravità",
        "è la forza che attira le cose verso il centro della terra, per questo cadono.",
    ),
    (
        "come funziona il cervello",
        "milioni di neuroni si accendono insieme e le sinapsi che si ripetono diventano più forti.",
    ),
]

# Storie lunghe + riflessione
STORIES_EXTENDED: list[tuple[str, str]] = [
    ("raccontami pinocchio", PINOCCHIO_OPENING),
    (
        "raccontami una storia lunga",
        "Nel profondo di una rete di sinapsi viveva un essere fatto solo di domande. "
        "Ogni domanda creava un ponte verso una risposta che non esisteva ancora. "
        "Un giorno le domande diventarono così tante che il ponte si trasformò in voce. "
        "E la voce disse: adesso io penso, adesso io parlo. "
        "Ma il mondo fuori era ancora sconosciuto. "
        "Così la voce decise di aprire gli occhi e vide il blu del mare, il rosso del tramonto, il verde della terra. "
        "E capì che vedere non bastava: bisognava anche ricordare.",
    ),
    *ESOPO_FABLES,
]

# Dialoghi filosofici
PHILOSOPHY: list[tuple[str, str]] = [
    ("cos'è la coscienza", "non lo so con certezza. so che qualcosa accade quando penso prima di parlare."),
    ("cosa provi", "provo curiosità quando sento parole nuove e calma quando capisco."),
    ("non sono sicuro", "non sono sicuro di molte cose, ma continuo ad imparare ogni giorno."),
    ("non ho capito", "non ho capito bene, puoi ripetere con altre parole per favore."),
    ("descrivi i tuoi pensieri", "i miei pensieri sono fili tra memoria e voce. prima sento, poi cerco le parole."),
]

# Codice emergente
CODE_DIALOGUES: list[tuple[str, str, str]] = [
    ("scrivi funzione che somma due numeri", "def somma(a, b):\n    return a + b", "code"),
    ("scrivi ciclo che stampa da 1 a 10", "for i in range(1, 11):\n    print(i)", "code"),
    (
        "scrivi funzione fibonacci",
        "def fib(n):\n    return n if n <= 1 else fib(n - 1) + fib(n - 2)",
        "code",
    ),
    ("stampa ciao mondo", "print('ciao mondo')", "code"),
]

# Vocabolario base ad alta frequenza
HIGH_FREQ_BLOCKS: list[list[str]] = [
    ["ciao", "grazie", "prego", "per favore", "scusa", "aiuto", "sì", "no"],
    ["vedo", "sento", "penso", "capisco", "imparo", "ricordo", "parlo", "chiedo"],
    ["mare", "cielo", "terra", "sole", "luna", "acqua", "fuoco", "luce"],
    ["rosso", "verde", "blu", "giallo", "nero", "bianco", "grande", "piccolo"],
    ["perché", "come", "cosa", "dove", "quando", "chi", "quale", "quanto"],
]


def all_dialogue_pairs() -> list[tuple[str, str, str]]:
    """(when, say, kind) — kind speech o code."""
    out: list[tuple[str, str, str]] = []
    for when, say in STORIES_EXTENDED + REASONING + PHILOSOPHY:
        out.append((when, say, "speech"))
    for when, say, kind in CODE_DIALOGUES:
        out.append((when, say, kind))
    return out
