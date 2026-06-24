# mind-runtime

Cervello neurobiologico operativo — **23.800 neuroni**, 18 regioni, 7 sistemi.

## Architettura

Prima del training massivo, il cervello deve essere **vero**: grafo neurale sparso con plasticità STDP, dopamina da prediction error, e calcolo Φ dinamico per la coscienza.

Vedi [docs/BIOLOGICAL_ARCHITECTURE.md](docs/BIOLOGICAL_ARCHITECTURE.md) per la mappa completa.

## Quick Start

```bash
pip install -e ".[dev]"

# Costruisci e verifica
organism build

# Test cognitivo
organism tick --text "ciao mondo" --chat

# Benchmark capacità GPU
organism capacity

# Server per Ink Admin
organism nursery
```

## Sistemi

| Sistema | Neuroni | Regioni |
|---------|---------|---------|
| Sensory Input | 4.300 | Auditory, Visual, Proprioceptive, Interoceptive |
| Integration Core | 9.000 | Association, Temporal, Insula |
| Prefrontal + Limbic | 4.300 | PFC, ACC, Amygdala, Dopamine |
| Language | 3.000 | Wernicke, Broca, Motor |
| Consciousness Loop | 3.200 | Thalamus, PCC, TPJ, Φ Integrator |

**Totale: 23.800 neuroni**

## API Nursery

Server HTTP su `:8765` — compatibile con Ink Admin organism client.

- `GET /api/baby/health` — metriche cervello + capacità GPU
- `GET /api/baby/state` — stato live (Φ, dopamina, emozione)
- `POST /api/baby/birth` — attiva il cervello
- `POST /api/baby/chat` — ciclo sensory → language → output

## Licenza

MIT
