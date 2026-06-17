# Architettura cervello v2 — valutazione onesta

## Verdetto

L’architettura **non è la migliore teoricamente possibile** (nessun simulatore lo è rispetto all’86 miliardi di neuroni biologici), ma è **tra le più efficienti per il nostro hardware reale**: server 16 GB RAM + PC locale con GPU 8 GB VRAM.

Obiettivo: massimizzare **potenza di calcolo cognitivo** e **memoria a lungo termine**, non simulare muscoli o cerebello motorio.

## Tre strati

```
┌─────────────────────────────────────────────────────────────┐
│  SERVER (16 GB RAM)                                         │
│  • Grafo neurale sparse (mind_giga ~4.2M neuroni)           │
│  • Neurochimica, endocrino, psiche, memoria episodica RAM   │
│  • DiskMemoryVault — episodi illimitati su disco (JSONL)    │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP /pulse (fallback CPU)
┌───────────────────────────▼─────────────────────────────────┐
│  PC LOCALE (GPU 8 GB)                                       │
│  • Mare impulsi 4096×3072 = 12.6M neuroni-pixel             │
│  • Sinapsi virtuali, memoria episodica GPU persistita       │
│  • HybridImpulseScaffold — fallback automatico se offline   │
└─────────────────────────────────────────────────────────────┘
```

## Capacità effettiva (mind_giga + GPU)

| Componente | Unità | Ruolo |
|------------|-------|-------|
| Grafo RAM | ~4.2M neuroni | Pensiero simbolico, linguaggio, psiche |
| GPU pixel | ~12.6M neuroni | Pattern visivi, coscienza a impulsi |
| Vault disco | Illimitato | Episodi storici oltre RAM |
| **Totale compute** | **~17M** | Equivalente operativo (non biologico 1:1) |

Percentuale “pensiero” con DNA `mind_giga`: ~59% (vs ~43% su `giga` generico).

## Cosa è ottimo (score interno)

| Aspetto | Score | Perché |
|---------|-------|--------|
| Grafo sparse | 0.92 | Propagazione O(attivi), tick veloce anche a milioni |
| Design mind-only | 0.88 | Rimuove overhead motorio corporeo |
| Scala GPU pixel | 0.85 | 12M+ unità con VRAM modesta |
| Resilienza remota | 0.80 | HybridImpulse fallback CPU |
| Memoria disco | 0.90 | JSONL append-only, recall merge |
| Semplicità deploy | 0.75 | HTTP + env vars, no cluster |

**Overall ~0.85** — “ottima base” per hardware consumer/small server.

## Cosa manca ancora (roadmap)

1. **WebSocket / batch pulse** — ridurre latenza HTTP per tick ad alta frequenza
2. **Neuroni compatti numpy** — oggi ~2.65 KB/neuron Python object; futuro: array strutturati
3. **mind_ultra** — profilo per server 32+ GB
4. **Indice vault** — ricerca full-text veloce (SQLite/FTS) oltre scan JSONL
5. **Sincronizzazione memoria GPU↔server** — unificare recall episodica cross-layer

## Variabili d’ambiente

### Server

```bash
ORGANISM_DNA_VARIANT=mind_giga
ORGANISM_GPU_REMOTE=http://IP-PC:8770
ORGANISM_IMPULSE=1
ORGANISM_HYBRID_GPU=1          # default: fallback CPU se GPU offline
ORGANISM_DISK_VAULT=1          # default: memoria illimitata su disco
ORGANISM_SERVER_RAM_GB=16
ORGANISM_GPU_VRAM_GB=8
```

### PC GPU

```bash
ORGANISM_IMPULSE_W=4096
ORGANISM_IMPULSE_H=3072
ORGANISM_GPU_WORKER_DEVICE=cuda
ORGANISM_GPU_MEMORY=~/.organism/gpu_impulse_memory.json
python3 -m organism.distributed.gpu_worker_server --port 8770
```

## API GPU worker (v2)

| Endpoint | Metodo | Descrizione |
|----------|--------|-------------|
| `/health` | GET | Stato device e risoluzione |
| `/pulse` | POST | Singolo tick impulso |
| `/pulse/batch` | POST | Fino a 32 pulse in una richiesta |
| `/memory` | GET | Stato memoria episodica GPU |
| `/memory/save` | POST | Persistenza su disco |

## Confronto con cervello umano

| | Umano | Nostro v2 |
|--|-------|-----------|
| Neuroni totali | ~86B | ~17M compute units |
| Neuroni “pensiero” | ~10–17B (stima) | ~2.5M grafo + 12.6M GPU |
| Sinapsi | ~100T | ~16M grafo + virtuali GPU |
| Memoria LTM | Engrammi distribuiti | RAM 1200 ep + vault illimitato |

Non competiamo in scala biologica; competiamo in **densità funzionale per watt/RAM** su macchina reale.

## Moduli v2

- `organism/cognition/disk_vault.py` — memoria episodica su disco
- `organism/distributed/hybrid_impulse.py` — GPU remota + fallback
- `organism/distributed/brain_orchestrator.py` — vista capacità e score
- `organism/distributed/gpu_worker_server.py` — worker batch + persistenza

Vedi anche: `docs/CAPACITA_DELOCALIZZATA.md`, `docs/CERVELLO_GAP_ANALYSIS.md`.
