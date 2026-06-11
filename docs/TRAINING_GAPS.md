# Gap architetturali — Organism vs umano

Cosa rende l'organismo "stupido" rispetto a un bambino umano. Ogni punto è un target di polish.

## 1. Lessico senza grounding (critico)

| Umano | Organism oggi |
|-------|-----------------|
| Vede mela → sente "mela" → assaggia/tocca | `absorb()` alza exposure senza significato |
| Non usa parola finché non sa cosa significa | Può articolare filler sovraesposti senza concetto |
| **Fix:** `teach-word` + immagine oggetto + beat semantico prima dell'uso in narrazione |

## 2. Retrieval vs comprensione (critico)

| Umano | Organism oggi |
|-------|-----------------|
| Racconta con parole sue riordinando episodi | Coppie `when→say` verbatim per risposte lunghe |
| Generalizza ("perché nevica?" da "perché piove") | Match fuzzy su trigger, zero inferenza |
| **Fix:** narrative mode + beat + `verbatim=false`; Q&A brevi causali ok verbatim |

## 3. Visione non semantica (alto)

| Umano | Organism oggi |
|-------|-----------------|
| Riconosce oggetti in contesti diversi | Picsum seed ≠ foto reale; prototipo fragile |
| Collega volto ↔ emozione ↔ nome | Face binder ok ma pochi esempi |
| **Fix:** mega-curriculum 1000 + web curriculum + teach_attention live |

## 4. Nessun modello causale esplicito (alto)

| Umano | Organism oggi |
|-------|-----------------|
| "piove → terra bagnata" come regola | Fragment hooks solo su trigger words |
| Controfattuali ("se non piovesse") | 6 coppie reasoning, non consolidate in grafo |
| **Fix:** micro-pairs causali + episodic + ripetizione 5× |

## 5. Working memory corta (medio)

| Umano | Organism oggi |
|-------|-----------------|
| Tiene filo del discorso 30s+ | Working memory 7 slot, non thread narrativo |
| **Fix:** episodic `recall_story_thread` (TODO) |

## 6. Motor loop senza errore (medio)

| Umano | Organism oggi |
|-------|-----------------|
| Si corregge se caregiver dice "no" | `self_hear` similarity=1.0, solo rinforzo |
| **Fix:** mismatch penalty + `correct_speech` API in training loop |

## 7. Metriche che mentono (medio)

| Umano | Organism oggi |
|-------|-----------------|
| Sa se ha capito | evolution_score 1.0 con word salad |
| **Fix:** semantic_overlap + babble + coverage gate |

## 8. Stato monolitico 120MB (operativo)

| Umano | Organism oggi |
|-------|-----------------|
| — | Ogni persist serializza tutto il cervello |
| **Fix:** persist debounced; training locale batch; pruning sleep |

## 9. Nessun pragmatics (basso)

| Umano | Organism oggi |
|-------|-----------------|
| Turn-taking, non interrompe | Anti echo ok; no dialogo bidirezionale fluido |

## 10. Scala lessico (obiettivo training)

| Umano ~5 anni | Organism oggi |
|---------------|-----------------|
| ~5.000–10.000 parole attive | ~3.000 exposure entries |
| **Training serio:** absorb batch + definizioni + 1000 oggetti visivi + lettura web |

---

## Ordine polish (priorità)

1. Grounding parola+immagine prima di narrazione
2. Mega-curriculum completo + semantic stories
3. Vocab massivo con definizioni (non solo absorb)
4. Causal pairs ripetuti + probe semantici
5. Sleep/prune regolare
6. Correzioni caregiver nel loop motor
