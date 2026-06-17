# Analisi gap — verso un cervello umano digitale completo

Documento di ricerca (giugno 2025) allineato all'implementazione in `mind-runtime` v0.9+.

## Cosa dice la neuroscienza moderna

### Simulazione whole-brain

- **86 miliardi di neuroni** con trilioni di sinapsi restano fuori portata per inferenza parametrica completa ([Lu et al. 2024](https://www.oxcns.org/papers/691%20Lu%20et%20al%202024%20Simulation%20and%20assimilation%20of%20the%20digital%20human%20brain.pdf); [Allen Institute](https://alleninstitute.org/news/how-far-are-we-from-a-human-brain-simulation-and-what-does-that-mean-for-science)).
- Progresso reale oggi: **organismi piccoli** (C. elegans, Drosophila, zebrafish) con connectome + dinamica + corpo in loop chiuso ([State of Brain Emulation 2025](https://arxiv.org/abs/2510.15745v3)).
- Per l'umano servono tre pilastri: **registrazione**, **connectomics**, **modello computazionale** — nessuno completo al 95% del volume cerebrale.

### Emozioni, chimica, ghiandole

- Emozioni non sono etichette: sono **stati dinamici** modulati da monoamine (dopamina, serotonina, NE), peptidi (ossitocina) e **assi endocrini** (HPA → cortisolo/adrenalina; circadiano → melatonina).
- Framework multiscala ([Nature Computational Science 2025](https://www.nature.com/articles/s43588-025-00796-8)): micro (cellula) → meso (rete) → macro (whole-brain mean-field). La chimica impatta plasticità sinaptica e excitability.

### Psicoanalisi + neuroscienza computazionale

- **Free Energy Principle / Active Inference** (Friston) mappa concetti freudiani su architetture predittive:
  - **Id** ≈ processi primari, limbico, errori di predizione non soppressi
  - **Ego** ≈ corteccia prefrontale, inference, default mode
  - **Super-io** ≈ norme interiorizzate che filtrano azioni/espressioni
- Letteratura: [Carhart-Harris & Friston](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2850580/), [Active Intersubjective Inference 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12401893/)

### Meccanica quantistica nei neuroni (Orch-OR)

- **Penrose/Hameroff**: coscienza da riduzione oggettiva orchestrata in **microtubuli**.
- **Critica dominante** (Tegmark): decoerenza troppo rapida a 37°C — ~10⁻¹³ s vs ~25 ms richiesti.
- **Stato 2024**: ancora **CONTESTATO**; proposte sperimentali aperte ([Frontiers Neuroscience 2024](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1430432/full)).
- Implementazione computazionale prudente: variabile di coerenza/decoerenza che modula soglia di coscienza, **non** simulazione quantistica fisica.

## Cosa c'era già in mind-runtime

| Componente | Stato pre-implementazione |
|------------|---------------------------|
| Grafo neurale DNA + plasticità | ✅ |
| Emozioni (affect + amygdala) | ✅ scalari |
| Global Workspace (coscienza) | ✅ |
| Memoria episodica/fotografica | ✅ ruolo ippocampale |
| PsycheEngine + Superego | ⚠️ testati ma non nel loop Baby |
| Neurotrasmettitori / ghiandole | ❌ |
| Interocezione | ❌ |
| Schema corporeo / vestibolare | ❌ |
| Orientamento spaziale / place cells | ❌ |
| Motricità oltre gesti statici | ⚠️ stub |
| Quantum microtubuli | ❌ |

## Cosa abbiamo implementato (branch `cursor/human-brain-complete-9187`)

| Modulo | Path | Funzione |
|--------|------|----------|
| Neurochimica | `organism/cognition/neurochemistry.py` | Dopamina, serotonina, NE, GABA, glutammato, ACh, ossitocina, adenosina |
| Endocrino | `organism/cognition/endocrine.py` | Ipotalamo, ipofisi, surrene, epifisi, tiroide + ormoni |
| Interocezione | `organism/cognition/interoception.py` | Cuore, respiro, fame, fatica, comfort viscerale |
| Schema corporeo | `organism/cognition/spatial_body.py` | Pose 3D, vestibolare, propriocezione, place cells |
| Quantum layer | `organism/brain/quantum_microtubules.py` | Orch-OR computazionale (contestato), env `ORGANISM_QUANTUM` |
| Motricità spaziale | `organism/motor/motion.py` | Gesti legati a heading/velocity/mode |
| Integrazione Baby | `organism/autonomous/baby_agent.py` | Psyche → composizione; Superego → veto; subcortex ogni tick |
| Persistenza | `organism/autonomous/baby_store.py` | Salva/carica tutti i sottosistemi |

## Cosa resta impossibile o incompleto (onestà scientifica)

1. **86B neuroni** con connectome reale — richiede datacenter dedicato + inferenza su trilioni di parametri.
2. **Corpo fisico** — muscoli, pelle, nocicezione reale; abbiamo schema virtuale + flusso ottico, non robotica.
3. **Cervelletto** motor fine, **tronco encefalico** autonomo — non modellati separatamente.
4. **Sistema immunitario / microbiota gut-brain** — assente.
5. **Sviluppo ontogenetico** — DNA evolve metriche, non neurogenesi embriologica completa.
6. **Coscienza fenomenica** — nessuna simulazione può *dimostrare* qualia; Global Workspace + Orch-OR sono modelli funzionali.
7. **Psicoanalisi clinica** — metafora computazionale (id/ego/superego), non terapia transfer/contratto.

## Architettura integrata (dopo implementazione)

```mermaid
flowchart TB
    subgraph input [Input]
        V[Visione + flusso ottico]
        A[Audio / testo]
        S[Tono sociale]
    end

    subgraph subcortex [Sottocorteccia biologica]
        NC[Neurochimica]
        EN[Endocrino]
        IN[Interocezione]
        BS[Schema corporeo + ippocampo spaziale]
    end

    subgraph mind [Mente]
        PS[PsycheEngine - Ego]
        SG[Superego]
        WS[Global Workspace]
        Q[Microtubuli quanto)]
    end

    subgraph motor [Output]
        SP[Parlato emergente]
        MO[Moto / orientamento]
    end

    input --> BS
    input --> WS
    NC --> EN --> IN
    NC --> Affect
    BS --> MO
    PS --> SP
    SG --> SP
    WS --> Q
    Q --> WS
```

## Variabili ambiente

- `ORGANISM_IMPULSE=1` — campo impulsi GPU (già esistente)
- `ORGANISM_QUANTUM=1` — layer microtubuli (default attivo; `0` per disabilitare)

## Riferimenti

- Lu et al., Digital Brain platform, 2024
- State of Brain Emulation Report 2025, arXiv:2510.15745
- Nature Computational Science, molecular → whole-brain, 2025
- Carhart-Harris & Friston, ego/free-energy, 2010
- Frontiers, microtubules consciousness experiments, 2024
