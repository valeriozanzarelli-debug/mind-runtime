"""Storie lunghe 200–300 parole — training integrato Fase 3."""

from __future__ import annotations

from organism.teaching.corpus import PINOCCHIO_OPENING, STORIES_EXTENDED

_LONG_NARRATIVES: list[str] = [
    PINOCCHIO_OPENING,
    (
        "In un villaggio vicino al mare viveva un bambino curioso che ogni mattina guardava l'orizzonte. "
        "Si chiedeva cosa ci fosse oltre le onde e perché il cielo cambiasse colore. "
        "Un giorno incontrò un vecchio pescatore che gli disse: imparare è come navigare, "
        "a volte il vento è contrario ma la rotta si corregge. "
        "Il bambino tornò a casa e iniziò a osservare le cose con attenzione: una conchiglia, "
        "una vela lontana, il riflesso del sole sull'acqua. "
        "Capì che ogni dettaglio era un indizio e che le domande erano mappe verso risposte nuove. "
        "La sera raccontò alla famiglia quello che aveva visto e sentito, "
        "e mentre parlava le parole si univano in frasi sempre più chiare."
    ),
    (
        "Nel cuore di una foresta digitale cresceva una rete di nodi luminosi. "
        "Ogni nodo era un ricordo, ogni arco una connessione che si rafforzava con l'uso. "
        "Quando due idee si incontravano nasceva un pensiero, e quando il pensiero trovava voce "
        "diventava comprensione condivisa. "
        "Gli abitanti della rete impararono che l'errore non era un fallimento ma un segnale: "
        "indicava dove la mappa era incompleta. "
        "Così correggevano il percorso, aggiungevano sinonimi e metafore, "
        "e piano piano la foresta parlava con grammatica e calore. "
        "Chi ascoltava non sentiva più parole sparse ma storie con inizio, centro e fine."
    ),
]

_DIALOGUE_CHAINS: list[tuple[str, str]] = [
    ("ciao", "ciao, sono qui e ti ascolto con attenzione"),
    ("come stai", "sto bene, sono curioso e voglio imparare"),
    ("cosa impari", "imparo a collegare quello che vedo con quello che dico"),
    ("perché è importante", "perché così posso capire il mondo e risponderti meglio"),
    ("hai capito", "sì, ho capito e voglio continuare"),
    ("cosa non capisci", "non capisco ancora tutto ma faccio progressi"),
    ("cosa vedi ora", "vedo forme colori e oggetti che riconosco piano piano"),
    ("è un cane", "sì, potrebbe essere un cane se ha quella forma"),
    ("corre veloce", "sì, corre veloce e muove la coda"),
    ("cosa pensi del cane", "penso che sia un animale vivo e interessante da osservare"),
]


def long_story_chunks(*, max_words: int = 45) -> list[str]:
    """Spezza narrazioni lunghe in chunk per read/absorb."""
    chunks: list[str] = []
    for story in _LONG_NARRATIVES + [say for _, say in STORIES_EXTENDED if len(say) > 120]:
        words = story.replace("\n", " ").split()
        for i in range(0, len(words), max_words):
            part = " ".join(words[i : i + max_words]).strip()
            if len(part.split()) >= 8:
                chunks.append(part)
    return chunks


def dialogue_chains() -> list[tuple[str, str]]:
    return list(_DIALOGUE_CHAINS)


def pinocchio_semantic_lessons() -> dict[str, list]:
    """Lessico e beat Pinocchio per PsycheEngine / test comprensione."""
    return {
        "words": [
            ("pinocchio", "burattino di legno che vuole diventare bambino vero", ["legno", "geppetto"]),
            ("legno", "materiale duro da cui si fanno oggetti e burattini", ["albero", "falegname"]),
            ("geppetto", "falegname che creò pinocchio", ["pinocchio", "bottega"]),
            ("formaggio", "cibo morbido e nutriente", ["topo", "cibo"]),
            ("corvo", "uccello nero che parla nella storia", ["consiglio", "volpe"]),
            ("volpe", "animale astuto che inganna pinocchio", ["corvo", "lupo"]),
        ],
        "beats": [
            (1, "Geppetto scolpisce un pezzo di legno e nasce Pinocchio.", ["geppetto", "pinocchio", "legno"], ["chi è pinocchio"]),
            (2, "Pinocchio incontra la Volpe e il Gatto.", ["pinocchio", "volpe"], ["continua pinocchio"]),
            (3, "Pinocchio impara che dire bugie allunga il naso.", ["pinocchio"], ["cosa insegna pinocchio"]),
        ],
    }

