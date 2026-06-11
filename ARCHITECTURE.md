# MIND + ORGANISM — Architettura

## Panoramica

```
SENSORY (vision, audio, text)
        ↓ spikes
   DNA-grown BRAIN (topology + plasticity)
        ↓ patterns
   MIND (pattern, memory, sensation, policy)
        ↓ action
   MOTOR (speech, song, text, motion)
```

## MIND (decisionale)

Vedi `mind/` — runtime simbolico-esperienziale:
- Pattern visivo, memoria frammentata, circuiti sensazione, policy costo/esperienza.

## ORGANISM (organismo completo)

### DNA Layer (`organism/dna/`)

- `organism_dna.yaml` — genoma ~3KB che definisce tipi neuroni, regole crescita, plasticità, lessico pattern.
- `variants/studio_assistant.yaml` — specializzazione IC Studio (merge).
- `interpreter.py` — genera topologia a runtime (10k–100k+ sinapsi su MVP).

### Brain (`organism/brain/`)

- `Neuron`, `Synapse`, `NeuralTopology` — grafo sparso, spike injection, propagazione.
- `PlasticityEngine` — Hebbian, STDP, homeostatic.
- `sleep()` — pruning + consolidamento.

### Sensory (`organism/sensory/`)

- **Vision** — Sobel edges → spike su edge detectors.
- **Audio** — FFT bande → frequency analyzers (WAV o tono sintetico).
- **Text** — embedding hash 128-dim + lexicon DNA → semantic encoders.

### Motor (`organism/motor/`)

- **Speech** — testo + prosody + SSML (TTS-pluggable).
- **Song** — emozione → scala → melodia (note struct).
- **Text** — template da azione MIND.
- **Motion** — gesture primitives per avatar.

### Runtime (`organism/runtime.py`)

```python
org = OrganismRuntime.studio_assistant()
thought, expression = org.live({"text": "...", "audio": b"..."}, output_modality="full")
org.sleep()  # pruning notturno
```

## Auto-learning (v0.3)

Ogni `live()` con `learn=True` (default):

1. **Hebbian + STDP** sulle sinapsi attive
2. **Pathway reinforce** sensory → associative → motor
3. **Fragment weight boost** sui frammenti MIND recuperati
4. Dopo N episodi uguali → **nuovo frammento compresso** `learned_*`

```bash
organism replay --file examples/replay_episodes.json
organism save / organism load
organism wa --text "..."   # ink-api bridge (mock)
```

## Integrazione futura ink-api

```
WhatsApp → organism.perceive → organism.think → organism.express
                                              ↘ ink-api tools
```

LLM opzionale solo per parafrasi testo; decisione resta MIND.
