# mind-runtime + ORGANISM

**Runtime cognitivo completo** — non un LLM.

## 🍼 Lancia il Baby (interfaccia umana)

```bash
cd mind-runtime
pip install -e ".[full]"
python3 -m organism.cli nursery --browser
```

### Produzione (inkconscius.eu + Caddy)

**https://inkconscius.eu/organism/**

Vedi [deploy/DEPLOY.md](deploy/DEPLOY.md) — snippet Caddy + `systemctl organism-nursery`.

### Locale

```bash
python3 -m organism.cli nursery --browser
```

Apri **http://127.0.0.1:8765** — orb viola, webcam, microfono, altoparlante.

| Cosa fa | Come |
|---------|------|
| **Nasce da solo** | DNA dispiega il cervello all'avvio |
| **Parla da solo** | Balbettio quando curioso (ogni ~4s) |
| **Vede** | Webcam → segnali visivi |
| **Senta** | Microfono → testo/udio |
| **Parla** | Web Speech API (come voce umana) |
| **Impara** | Tieni ✋ e ripeti 3 volte la stessa frase |
| **Ricerca** | 🔬 legge come funzionano occhio/coclea umani |

**Niente risposte hardcoded** — solo istinto di curiosità + ciò che insegni tu per ripetizione.

Lab tecnico (grafo, curriculum): http://127.0.0.1:8765/lab

| Layer | Cosa fa |
|-------|---------|
| **DNA** | Genoma YAML (~3KB) → migliaia di neuroni, 10k+ sinapsi a runtime |
| **Brain** | Spike, plasticità Hebbian/STDP, pruning |
| **Sensory** | Vista, udito, testo → pattern neurali |
| **MIND** | Memoria frammentata, risonanza, policy esperienza |
| **Motor** | Parola, canto, testo, gesto |

## Install

```bash
git clone https://github.com/valeriozanzarelli-debug/mind-runtime.git
cd mind-runtime
pip install -e ".[full]"
```

## Test

```bash
python3 -m pytest -v          # 34 test (MIND + ORGANISM + nursery)
python3 -m mind.cli demo      # MIND puro
python3 -m organism.cli demo  # ciclo completo
python3 -m organism.cli nursery --browser  # 🧬 dashboard visiva
```

## Nursery — vedi nascere l'organismo

Dashboard web per osservare in tempo reale:

- **Nascita DNA** → topologia neurale che si dispiega
- **Flusso pensieri** → simboli MEM, SEN, ACT, LEARN ad ogni ciclo
- **Grafo connessioni** → neuroni attivi e sinapsi (vis-network)
- **Curriculum** → vista → udito → linguaggio → sociale → mondo
- **Verifica auto-sviluppo** → peso sinapsi, frammenti, cicli learning

```bash
python3 -m organism.cli nursery
# oppure: python3 -m organism.nursery.server --port 8765 --browser
# apri http://127.0.0.1:8765
```

## ORGANISM quick start

```bash
# Stats cervello generato dal DNA
python3 -m organism.cli stats

# Demo studio assistant
python3 -m organism.cli demo --variant studio

# Ciclo singolo
python3 -m organism.cli live --text "lampadina non si accende" --modality speech
python3 -m organism.cli live --shapes "quadrato+cerchio,triangolo+cerchio,rettangolo+"
python3 -m organism.cli live --text "cliente whatsapp diffidente preventivo" \
  --resonate-with "cliente marzo diffidente whatsapp"

# Auto-learning batch
python3 -m organism.cli replay --file examples/replay_episodes.json

# Persistenza stato (cervello + memoria)
python3 -m organism.cli save --path ~/.organism/state.json
python3 -m organism.cli load --path ~/.organism/state.json

# Mock WhatsApp → ink-api bridge
python3 -m organism.cli wa --text "preventivo braccio realistico"

# Consolidamento notturno (pruning sinapsi deboli)
python3 -m organism.cli sleep

# Visualizza grafo neurale
python3 -m organism.cli brain
```

## Python API

```python
from organism import OrganismRuntime

org = OrganismRuntime.studio_assistant()
print(org.stats)  # neurons, synapses, species

thought, expr = org.live(
    {"text": "Ciao, vorrei prenotare per giovedì", "tone_hz": 220},
    output_modality="full",
)
print(expr.speech.text, expr.song.scale, expr.motion.frames)
```

## Struttura

```
mind-runtime/
├── mind/           # MIND decisionale (core)
├── organism/
│   ├── dna/        # genoma + varianti
│   ├── brain/      # topologia neurale
│   ├── sensory/    # vision, audio, text
│   ├── motor/      # speech, song, motion
│   └── runtime.py  # loop perceive → think → express
└── tests/
```

Vedi [ARCHITECTURE.md](ARCHITECTURE.md).

## Licenza

MIT
