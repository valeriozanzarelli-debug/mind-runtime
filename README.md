# CEREBRUM

Runtime cerebrale **locale**. Tutto il cervello gira sul PC dell'utente
(GPU se disponibile via CUDA + torch). Il server remoto è solo un canale di
distribuzione/consultazione: **non** esegue mai il cervello.

CEREBRUM non è un chatbot. È un organismo cognitivo che nasce come un neonato:
ha un loop vitale continuo (pensa sempre, anche senza stimoli), sistemi di
sopravvivenza di base (omeostasi, neurochimica, drives, riflessi neonatali),
sensi (visione da webcam, udito/linguaggio) e un apparato motorio vocale che
lallazione dopo lallazione impara a imitare le parole.

## Architettura

```
Sensi (webcam, testo/voce)
      │  corrente sensoriale
      ▼
Campo neurale  ──►  spiking LIF ricorrente + plasticità Hebbian   [GPU: torch/CUDA, fallback numpy]
      ▲                                    │ attività, Φ
      │ modulazione                        ▼
Corpo:  neurochimica · omeostasi · drives · riflessi neonatali
      │                                    │
      ▼                                    ▼
Mente:  memoria episodica · flusso di coscienza  ──►  Motore vocale (lallazione → parole)
```

- `cerebrum/neuro/field.py` — substrato neurale (GPU): neuroni LIF, sinapsi
  plastiche, attività spontanea di fondo, stima di Φ.
- `cerebrum/body/` — neurochimica (dopamina, serotonina, cortisolo, ossitocina…),
  omeostasi (fame, energia, stanchezza, temperatura, dolore), drives (curiosità,
  attaccamento, esplorazione), riflessi neonatali (Moro, rooting, orienting, pianto…).
- `cerebrum/sense/` — visione (retina dai frame webcam: luminosità, movimento,
  contrasto), linguaggio (token → stimolo, lessico esperienziale).
- `cerebrum/motor/speech.py` — vocalizzazione emergente.
- `cerebrum/mind/` — memoria episodica con consolidamento nel sonno, flusso di coscienza.
- `cerebrum/brain.py` — assembla tutto in un loop vitale continuo, thread-safe.
- `cerebrum/server.py` — server HTTP locale (`127.0.0.1:8788`).

## Avvio

```bash
pip install -e ".[full]"     # torch+CUDA per sfruttare la GPU
python -m cerebrum serve     # avvia il cervello + server su 127.0.0.1:8788
python -m cerebrum info      # mostra il backend (cuda/cpu)
python -m cerebrum selftest  # test rapido
```

Su Windows con il pacchetto: doppio click su `START_WINDOWS.bat`, oppure lascia
che Ink Admin lo avvii (`scripts/serve_local.mjs`).

## API locale (usata da Ink Admin)

| Endpoint | Metodo | Descrizione |
|----------|--------|-------------|
| `/health` | GET | vivo, neuroni, unità, Φ, backend, webcam |
| `/ready` | GET | pronto |
| `/status` | GET | telemetria (Φ, spike rate, attività) |
| `/introspect` | GET | **vedere dentro il cervello**: emozione, chimica, corpo, drives, riflessi, pensiero corrente |
| `/consciousness?since=&n=` | GET | flusso di coscienza (pensieri/vocalizzazioni) |
| `/chat` | POST `{text}` | parla con lui → `{reply, emotion, drive, phi, webcam_active}` |
| `/hear` | POST `{text}` | stimolo uditivo |
| `/see` | POST `{frame|stats}` | frame webcam o statistiche visive |
| `/care` | POST `{action}` | accudimento: `feed` / `soothe` / `warm` |
| `/train` | POST `{steps}` | esposizione guidata a esperienza |
| `/power` | POST `{action}` | `on` / `off` |

## GPU

Con `torch` e CUDA installati, il campo neurale gira interamente sulla GPU.
Senza torch, fallback trasparente a numpy su CPU. Controlla con
`python -m cerebrum info`.

## Distribuzione

- Manifest: `releases/manifest.json`
- Build Windows: GitHub Actions (`.github/workflows/build-windows.yml`) →
  release `cerebrum-latest`, asset `CEREBRUM-Windows.zip`.
