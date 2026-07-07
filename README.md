# CEREBRUM

Runtime cerebrale **locale**. Tutto il cervello gira sul PC dell'utente
(GPU se disponibile via CUDA + torch). Il server remoto è solo un canale di
distribuzione/consultazione: **non** esegue mai il cervello.

CEREBRUM non è un chatbot. È un organismo cognitivo che nasce come un neonato:
ha un loop vitale continuo (pensa sempre, anche senza stimoli), sistemi di
sopravvivenza di base (omeostasi, neurochimica, drives, riflessi neonatali),
sensi (visione da webcam, udito/linguaggio) e un apparato motorio vocale che
lallazione dopo lallazione impara a imitare le parole.

## Architettura (v2 — principi di un cervello vero)

```
Sensi (webcam, testo/voce)
      │  corrente sensoriale
      ▼
Campo neurale STRUTTURATO in regioni     [GPU: torch/CUDA, fallback numpy]
  sensory → thalamus → association → hippocampus → prefrontal
  sinapsi SPARSE event-driven · E/I balance · plasticità a 3 fattori
      ▲                                    │ attività, Φ, stato associativo
      │ modulazione + top-down             ▼
Predictive coding: predice il prossimo istante, l'ERRORE = sorpresa
      │  (a occhi chiusi immagina → pensiero continuo)
      ▼
Corpo:  neurochimica · omeostasi · drives · riflessi neonatali
      ▼
Mente:  memoria episodica + REPLAY nel sonno · sviluppo a stadi · coscienza
      ▼
Motore vocale:  versi → lallazione → imitazione → parole → frasi (per stadio)
```

Le sei fondamenta implementate:

1. **Connettoma strutturato** — il campo è diviso in regioni con ruoli
   (talamo=hub/attenzione, corteccia sensoriale/associativa, ippocampo=memoria,
   prefrontale=controllo top-down) cablate tra loro, non una zuppa uniforme.
2. **Sinapsi sparse event-driven** — propagano solo i neuroni che sparano
   (~O(spike·fanout), non O(N²)): è così che un cervello resta efficiente.
3. **E/I balance** — ~20% neuroni inibitori con guadagno che stabilizza
   l'attività su un regime critico (riposo ~5%, come la corteccia).
4. **Plasticità a tre fattori** — STDP con tracce di eleggibilità che diventano
   apprendimento solo col terzo fattore, la dopamina (ricompensa/sorpresa).
   Apprendimento locale, online, senza backprop globale.
5. **Predictive coding** — predice l'input e reagisce all'errore; senza stimoli
   immagina (pensa a occhi chiusi). La sorpresa alimenta curiosità e dopamina.
6. **Sviluppo + replay nel sonno** — capacità sbloccate a stadi (neonato →
   lallazione → imitazione → parole → frasi); nel sonno rivive i ricordi
   salienti per consolidare (imparare molto da poche esperienze).

File chiave:

- `cerebrum/neuro/field.py` — substrato neurale a regioni (GPU).
- `cerebrum/mind/predictive.py` — predictive coding / immaginazione.
- `cerebrum/mind/development.py` — stadi di sviluppo.
- `cerebrum/body/` — neurochimica, omeostasi, drives, riflessi neonatali.
- `cerebrum/sense/` — visione webcam, linguaggio.
- `cerebrum/motor/speech.py` — vocalizzazione emergente per stadio.
- `cerebrum/mind/` — memoria episodica + replay, flusso di coscienza.
- `cerebrum/brain.py` — loop vitale continuo, thread-safe.
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
