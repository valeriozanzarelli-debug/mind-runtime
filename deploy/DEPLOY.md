# Deploy ORGANISM su inkconscius.eu

URL pubblico: **https://inkconscius.eu/organism/**

> **Stato (giu 2026):** servizio **spento** su produzione — sviluppo offline in `mind-runtime/`.  
> Riaccendere: `bash scripts/start-organism.sh` · Spegnere: `bash scripts/stop-organism.sh`

(Microfono e webcam richiedono **HTTPS** — Caddy già lo gestisce.)

## 1. Sul server (ssh ink)

```bash
cd /opt/mind-runtime   # oppure clone/pull del repo
sudo bash deploy/install-on-ink-server.sh
```

Verifica:

```bash
curl -s http://127.0.0.1:8765/api/baby/state | head
systemctl status organism-nursery
```

## 2. Caddy — aggiungi al Caddyfile esistente

Copia il blocco da `deploy/caddy/organism.caddy` **dentro** il sito `inkconscius.eu { ... }`:

```caddy
@organism path /organism /organism/*
handle @organism {
	uri strip_prefix /organism
	header Permissions-Policy "microphone=*, camera=*"
	reverse_proxy 127.0.0.1:8765
}
```

Poi:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

## 3. Apri nel browser

**https://inkconscius.eu/organism/**

- Consenti **microfono** e **camera** (Chrome consigliato)
- Lab tecnico: **https://inkconscius.eu/organism/lab**

## Variabili (systemd)

| Variabile | Valore | Ruolo |
|-----------|--------|--------|
| `ORGANISM_BASE_PATH` | `/organism` | prefisso URL |
| `ORGANISM_PUBLIC_URL` | `https://inkconscius.eu/organism` | log / link |
| `ORGANISM_PORT` | `8765` | porta locale |
| `ORGANISM_DNA_VARIANT` | `baby` / `mega` / `giga` / `ultra` | profilo DNA (vedi tabella sotto) |
| `ORGANISM_NEURON_MULTIPLIER` | (opzionale) | override moltiplicatore neuroni |
| `ORGANISM_PULSE_INTERVAL` | `1.2` | secondi tra impulsi cerebrali |

### Profili scala (scegli in base alla RAM libera)

| Variante | Neuroni | Sinapsi | RAM | Quando usarlo |
|----------|---------|---------|-----|---------------|
| **`baby`** (default) | ~1.5k | ~63k | **~50 MB** | server con poca RAM — **usa questo** |
| `compact` | ~15k | ~500k | **~200 MB** | un po' più grande, ancora leggero |
| `mega` | ~1.6M | ~14M | **~4 GB** | solo se hai ≥6 GB liberi |
| `giga` | ~3M | ~15M | **~9 GB** | solo se hai ≥12 GB liberi |
| `ultra` | ~5M | ~18M | **~14 GB** | macchina dedicata |

> **Attenzione:** `giga`/`ultra` su un VPS piccolo saturano la RAM e il servizio va giù (OOM). Torna a `baby` e riavvia.

Propagazione **sparsa** — tick veloce anche con molti neuroni, ma la **nascita** alloca tutta la RAM del profilo.

```bash
# default sicuro
ORGANISM_DNA_VARIANT=baby
sudo systemctl restart organism-nursery
```

### Recupero dopo OOM (server bloccato)

```bash
ssh ink
# forza profilo leggero
sed -i 's/ORGANISM_DNA_VARIANT=.*/ORGANISM_DNA_VARIANT=baby/' /etc/systemd/system/organism-nursery.service
systemctl daemon-reload
systemctl restart organism-nursery
free -h
curl -s http://127.0.0.1:8765/organism/api/baby/state | head -c 200
```

Dopo cambio variante, rigenera il cervello conservando i dialoghi:

```bash
curl -s -X POST https://inkconscius.eu/organism/api/baby/rebirth \
  -H 'Content-Type: application/json' \
  -d '{"keep_dialogue": true}'
```

Benchmark locale:

```bash
python3 scripts/benchmark_brain_scale.py --tier giga --pulses 5
python3 scripts/benchmark_brain_scale.py --all --allow-mega --allow-giga
```

## Dominio inkconscious.eu

Se il DNS punta a `inkconscious.eu` (con la «o»), sostituisci `inkconscius.eu` nel Caddyfile e in `ORGANISM_PUBLIC_URL`.

## Sottodominio (alternativa)

```caddy
organism.inkconscius.eu {
	header Permissions-Policy "microphone=*, camera=*"
	reverse_proxy 127.0.0.1:8765
}
```

In quel caso `ORGANISM_BASE_PATH` resta vuoto e `ORGANISM_PUBLIC_URL=https://organism.inkconscius.eu`.
