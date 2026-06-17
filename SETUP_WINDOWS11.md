# Setup Mindruntime GPU V2 — Windows 11 + RTX 1060

Guida completa per il cervello emergente locale. Per una versione breve vedi anche [SETUP.md](SETUP.md).

## Prerequisiti hardware

- HP Pavilion Gaming 16 (o equivalente)
- NVIDIA RTX 1060 6 GB
- Windows 11
- Webcam (opzionale)

---

## STEP 1: CUDA Toolkit 12.4

1. Scarica [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) per Windows x86_64.
2. Installa con driver NVIDIA aggiornato (≥ 535).
3. Verifica:

```powershell
nvcc --version
nvidia-smi
```

---

## STEP 2: cuDNN 9.x (opzionale)

Numba CUDA **non richiede** cuDNN per `mindruntime`. Serve solo per training PyTorch (`ai_trainer`).

1. Registrati su [NVIDIA Developer](https://developer.nvidia.com/cudnn).
2. Scarica cuDNN 9.x per CUDA 12.
3. Copia DLL/header/lib nelle cartelle `CUDA\v12.4\`.

---

## STEP 3: Python 3.11 + venv

```powershell
cd C:\path\to\mind-runtime
python -m venv venv_brain
.\venv_brain\Scripts\activate
python -m pip install -U pip
pip install -e ".[mindruntime]"
```

Dipendenze installate: `numpy`, `numba`, `scipy`, `opencv-python`, `matplotlib`.

### Cupy (opzionale)

```powershell
pip install cupy-cuda12x
```

---

## STEP 4: Verifica CUDA + Numba

```powershell
python scripts\test_cuda_numba.py
```

Output atteso:

```
✅ CUDA disponibile
✅ Device: NVIDIA GeForce GTX 1060
✅ Test kernel PASSED
✅ BrainEngineV2 step OK
```

---

## STEP 5: Primo avvio

```powershell
python -m mindruntime.visualizer
# oppure
python -m mindruntime.visualizer_v2
# oppure doppio clic
ORGANISM-Windows.exe
```

**Controlli tastiera:**

| Tasto | Azione |
|-------|--------|
| ESC / Q | Esci |
| M | Cambia modalità rendering (`phase_coherence`, `voltage`, `impulse`) |
| S | Salva snapshot in `%USERPROFILE%\.organism\mindruntime\` |

**Cosa aspettarsi:**

- Finestra OpenCV nativa (nessun browser)
- Mappa fase/coerenza con inset webcam
- FPS 30+ su RTX 1060 a 256×256
- Zone ad alta coherence evidenziate con cerchi verdi

---

## STEP 6: Variabili ambiente

| Variabile | Default | Effetto |
|-----------|---------|---------|
| `ORGANISM_ENGINE` | `v2` | `legacy` = motore dendritico V1 |
| `CUDA_VISIBLE_DEVICES` | `0` | Seleziona GPU |
| `ORGANISM_DATA_DIR` | `%USERPROFILE%\.organism` | Cartella salvataggio |

---

## STEP 7: Troubleshooting

| Problema | Soluzione |
|----------|-----------|
| `cuda: false` | Aggiorna driver, reinstalla `numba`, riavvia |
| FPS < 10 | `--width 128 --height 128` o `--steps-per-frame 1` |
| Webcam nera | `--camera 1` o `--image foto.jpg` |
| Out of memory | Chiudi giochi/browser, `nvidia-smi` |
| ModuleNotFoundError cv2 | `pip install opencv-python` |

---

## STEP 8: Validazione fisica

```powershell
python -m pytest tests/test_validation_v2.py -v
```

Vedi [docs/VALIDATION_TESTS.md](docs/VALIDATION_TESTS.md) per i criteri di successo.

---

## Architettura V2

```
mindruntime/
  field_v2.py         # 12 canali + spike_time
  gpu_physics_v2.py   # kernel HH, Turing, SOC, gamma, predictive (CUDA+CPU)
  gpu_engine_v2.py    # BrainEngineV2 / GPUBrainEngineV2
  resonators.py       # template FFT geometrici + lettere
  visualizer.py       # loop OpenCV principale
  visualizer_v2.py    # alias V2
```

Salvataggio locale: `%USERPROFILE%\.organism\mindruntime\state_latest.json`
