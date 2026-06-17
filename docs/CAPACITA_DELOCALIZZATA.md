# Capacità cervello delocalizzato — 16 GB RAM server + 8 GB GPU locale

## Architettura

```
┌──────────────────────── SERVER (16 GB RAM) ────────────────────────┐
│  Grafo DNA sparse — neuroni funzionali (pensiero + linguaggio)    │
│  MIND · Baby · memoria · chimica · endocrino · Psyche              │
│  ORGANISM_DNA_VARIANT=mind_giga                                    │
│  ORGANISM_GPU_REMOTE=http://PC-LOCALE:8770                         │
└────────────────────────────┬───────────────────────────────────────┘
                             │ HTTP /pulse
┌────────────────────────────▼───────────────────────────────────────┐
│  PC LOCALE (NVIDIA 8 GB VRAM)                                      │
│  gpu-worker — campo impulsi 4096×3072 = 12,5M neuroni-pixel      │
│  Sinapsi virtuali (kernel conv), CUDA                              │
└────────────────────────────────────────────────────────────────────┘
```

## Quanti neuroni per pensare?

### Cervello umano (cosa NON ci serve simulare)

| Area | Neuroni | Ruolo |
|------|---------|--------|
| **Cerebellum** | ~69 miliardi | Coordinazione motoria — **assente nel nostro modello** |
| Corteccia motoria/somatica | ~2,5 miliardi | Corpo, movimento muscolare — **rimosso** |
| **Pensiero** (associativa, PFC, limbico cognitivo) | ~10 miliardi | **Questo è il target** |

### Il nostro grafo DNA (profilo `mind` / `mind_giga`)

| Profilo | Neuroni totali | **Pensiero** | Corpo motorio | Linguaggio |
|---------|----------------|--------------|---------------|------------|
| `baby` | 1.482 | 51% | 19% | speech |
| `giga` (con corpo) | 2,9M | 43% | 8% motorio |
| **`mind`** | 2.107 | **63,5%** | **0,6%** | sì |
| **`mind_giga`** | **~4,3M** | **~59%** | **0,4%** | sì |

**Neuroni che “pensano” con `mind_giga`:** ~**2,5–3,0 milioni** (corteccia associativa + memoria + emozione).

### GPU pixel (PC locale 8 GB)

| Risoluzione | Neuroni-pixel | VRAM |
|-------------|---------------|------|
| 512×384 (default vecchio) | 196k | ~16 MB |
| 2048×1536 | 3,1M | ~264 MB |
| **4096×3072 (consigliato)** | **12,6M** | **~450 MB** |

### Totale efficace (mind_giga + GPU)

**~16–17 milioni** unità computazionali (4,3M grafo + 12,6M pixel).

Sinapsi virtuali GPU: convoluzione 3×3 su tutto il campo — equivalente a **ordini di grandezza in più** connessioni senza RAM per edge singoli.

## Perché siamo più efficienti del biologico

1. **Propagazione sparsa** — per tick si attiva ~0,001% del grafo; 4M neuroni costano ~0,07 ms/pulse (benchmark mega).
2. **Sinapsi virtuali GPU** — kernel, non lista di 15 trilioni di edge.
3. **Un neurone digitale ≠ una cellula** — un `pattern_matcher` condensa un microcircuito.
4. **Niente cerebellum** — risparmio dell’80% del “budget” neuronale umano.

## Setup

### PC locale (GPU)

```bash
pip install -e ".[gpu]"
ORGANISM_IMPULSE_W=4096 ORGANISM_IMPULSE_H=3072 \
ORGANISM_GPU_WORKER_DEVICE=cuda \
python3 -m organism.distributed.gpu_worker_server --port 8770
```

### Server (16 GB RAM)

```bash
ORGANISM_DNA_VARIANT=mind_giga
ORGANISM_GPU_REMOTE=http://IP-TUO-PC:8770
ORGANISM_IMPULSE=1
# nursery / systemd
```

### Verifica piano

```bash
python3 scripts/capacity_plan.py --variant mind_giga --grow
```

## Limiti onesti

- **RAM server:** `mind_giga` usa ~11–13 GB al boot — lascia 3 GB a OS + nursery.
- **GPU:** 4096×3072 è conservativo; si può salire a 6144×4608 se resta VRAM libera.
- **86 miliardi:** irrilevante come target; conta l’**equivalenza funzionale** del pensiero.
