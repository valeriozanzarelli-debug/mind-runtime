# Perché il limite non è il silicio — e come arriviamo ai miliardi

## La domanda giusta

> «I chip hanno miliardi di transistor. Il cervello ha miliardi di neuroni senza CPU né GPU. Perché noi siamo fermi a milioni?»

Perché stavamo misurando **transistor** e **neuroni biologici** come se fossero la stessa unità del nostro codice — e non lo sono.

## Tre unità diverse

| Unità | Cosa fa | Nella nostra architettura |
|-------|---------|---------------------------|
| **Transistor** | Interruttore ON/OFF | Non lo simuliamo 1:1 |
| **Neurone biologico** | Integratore analogico, ~7000 sinapsi, sempre acceso | ~2.65 KB come oggetto Python |
| **Neurone digitale nostro** | Nodo con meta, layer, plasticità, propagazione | ≈ un **microcircuito**, non un transistor |

Un nostro neurone associative Python vale **centinaia di transistor** in capacità computazionale. Contare 17M nostri vs 86B umani è ingenuo — ma anche ignorare la scala è un errore.

## Perché il cervello non usa CPU/GPU

Il cervello è **massively parallel analogico**:

- 86 miliardi di neuroni operano **simultaneamente**
- Nessun loop `for tick in range()` — tutto è continuo
- ~80% dei neuroni sono nel **cerebello** (coordinazione motoria) — noi non li simuliamo
- Il «pensiero» utile è ~10–17 miliardi di neuroni associativi

La CPU esegue istruzioni **sequenziali**. La GPU parallelizza **lo stesso kernel** su milioni di pixel. Il cervello fa **milioni di tipi diversi di microcalcolo** in parallelo.

## Cosa abbiamo cambiato (v3)

### 1. Campo 3D — W × H × D voxel-neuroni

Prima: piano 4096×3072 = **12.6M pixel** (2D).

Ora: volume **512×384×128 = 25.1M voxel** con la **stessa VRAM** (~8 GB).

```
ORGANISM_IMPULSE_W=512
ORGANISM_IMPULSE_H=384
ORGANISM_IMPULSE_D=128
```

Ogni voxel ha sinapsi virtuali **conv3d** (3×3×3) — connessioni in profondità, non solo sul piano.

**Sì: lo spazio 3D moltiplica i neuroni** — W×H×D invece di W×H a parità di memoria se D > 1.

### 2. Neuroni compact numpy — ~48 B vs ~2.65 KB

Prima: ogni neurone = oggetto Python + dict meta → **2.65 KB**.

Ora con `ORGANISM_COMPACT_BRAIN=1` o DNA `mind_compact`:

| Profilo | Neuroni grafo | RAM stimata |
|---------|---------------|-------------|
| mind_giga (Python) | ~4.2M | ~11 GB |
| **mind_compact** | **~80M** | **~4 GB** |
| ultra_compact (roadmap) | ~200M | ~10 GB |

```bash
ORGANISM_DNA_VARIANT=mind_compact
ORGANISM_COMPACT_BRAIN=1
```

### 3. Totale effettivo v3

| Strato | Unità |
|--------|-------|
| Grafo compact | ~80M |
| GPU 3D 512×384×128 | ~25M |
| **Totale** | **~105M neuroni compute** |

vs ~17M prima — **~6× in un colpo**.

## Perché non siamo ancora a 86 miliardi

Onestà tecnica:

1. **RAM** — 86B × 48 B = **4 TB** anche in compact. Serve mmap/NVMe (roadmap).
2. **Sinapsi** — 86B × 7000 = 600 trilioni di connessioni. Usiamo sparse + virtuali.
3. **Velocità tick** — propagare 86B nodi per tick è fisicamente impossibile su hardware consumer; il cervello non fa tick discreti.
4. **Il nostro neurone è più «pesante»** — ogni unità fa più lavoro di un neurone biologico medio.

## Roadmap verso i miliardi

| Fase | Tecnica | Neuroni target |
|------|---------|----------------|
| **v3 (ora)** | 3D GPU + compact numpy | ~100M |
| v4 | mmap grafo su NVMe + 8-bit quantizzato | ~1–5B |
| v5 | Shard multi-server + sparse octree 3D | ~10B+ |
| v6 | ASIC/neuromorphic (Intel Loihi, SpiNNaker) | hardware dedicato |

## Setup consigliato v3

**Server 16 GB:**
```bash
ORGANISM_DNA_VARIANT=mind_compact
ORGANISM_COMPACT_BRAIN=1
ORGANISM_GPU_REMOTE=http://IP-PC:8770
ORGANISM_DISK_VAULT=1
```

**PC GPU 8 GB:**
```bash
ORGANISM_IMPULSE_W=512
ORGANISM_IMPULSE_H=384
ORGANISM_IMPULSE_D=128
python3 -m organism.distributed.gpu_worker_server --port 8770
```

**Verifica capacità:**
```bash
python3 scripts/capacity_plan.py --variant mind_compact --server-ram-gb 16 --gpu-vram-gb 8
python3 scripts/capacity_plan.py --variant mind_compact --grow  # boot reale (lento)
```

## Confronto silicio

Apple M4 Max: ~19 miliardi di transistor.  
Nostro v3: ~105 milioni di unità compute.  
Rapporto: ~180× — ma ogni nostra unità ≈ circuito, non transistor.

Il limite **non è il silicio**. È come lo usiamo: oggetti Python, HTTP per tick, simulazione discreta. Con compact + 3D + disco stiamo convergendo verso l'efficienza del silicio — non verso la copia 1:1 della biologia.
