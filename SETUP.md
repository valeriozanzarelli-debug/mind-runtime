# Mindruntime — setup Windows 11 + RTX 1060 (6 GB)

Motore GPU **locale**: nessun server, nessun deploy cloud. Tutto gira sul tuo PC.

## Requisiti hardware

- Windows 11
- NVIDIA RTX 1060 6 GB (o GPU CUDA compatibile, compute capability ≥ 6.1)
- Webcam (opzionale, per input live)

## 1. CUDA Toolkit 12.x

1. Scarica [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) per Windows.
2. Installa con driver NVIDIA aggiornato (Game Ready o Studio, ≥ 535).
3. Verifica in PowerShell:

```powershell
nvcc --version
nvidia-smi
```

## 2. cuDNN 9.x (opzionale per questo stack)

Numba CUDA non richiede cuDNN per i kernel `mindruntime`. Serve solo se usi PyTorch training (`ai_trainer`).

1. Registrati su [NVIDIA Developer](https://developer.nvidia.com/cudnn).
2. Scarica cuDNN 9.x per CUDA 12.
3. Copia le DLL nella cartella CUDA o aggiungi al PATH.

## 3. Ambiente Python 3.11

```powershell
cd C:\path\to\mind-runtime
python -m venv venv_mindruntime
.\venv_mindruntime\Scripts\activate
python -m pip install -U pip
```

## 4. Installazione dipendenze

### Solo motore GPU locale (consigliato)

```powershell
pip install -e ".[mindruntime]"
```

Equivale a: `numpy`, `numba`, `scipy`, `opencv-python`, `matplotlib`

### Con training PyTorch opzionale

```powershell
pip install -e ".[mindruntime,gpu]"
```

### Cupy (opzionale, accelerazione array)

```powershell
pip install cupy-cuda12x
```

## 5. Test CUDA Numba

```powershell
python -c "from mindruntime.cuda_util import cuda_info; print(cuda_info())"
```

Output atteso con RTX 1060:

```json
{"numba": true, "cuda": true, "device": "NVIDIA GeForce GTX 1060", "compute_capability": (6, 1)}
```

Se `cuda: false`, reinstalla driver NVIDIA e verifica che `numba` veda la GPU.

## 6. Avvio visualizer (webcam)

```powershell
python -m mindruntime.visualizer
```

Opzioni utili:

```powershell
# Griglia 256×256 (default, ~65k neuroni, <300 MB VRAM)
python -m mindruntime.visualizer --width 256 --height 256

# Immagine statica
python -m mindruntime.visualizer --image foto.jpg

# Benchmark senza finestra
python -m mindruntime.visualizer --image foto.jpg --no-display --max-frames 120
```

**Target performance:** ≥ 30 FPS su 256×256 con RTX 1060 (2 step/tick).

## 7. ORGANISM Baby (browser locale, opzionale)

L'EXE / launcher Baby usa PyTorch impulse field + server HTTP su `127.0.0.1`:

```powershell
pip install -e ".[full,gpu]"
python -m organism.launcher
```

Per il motore **Numba puro** senza browser, usa sempre `python -m mindruntime.visualizer`.

## Variabili ambiente

| Variabile | Default | Effetto |
|-----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | `0` | Seleziona GPU |
| `NUMBA_ENABLE_CUDASIM` | — | `1` = simulazione CPU (debug) |

## Risoluzione VRAM (6 GB)

| Griglia | Neuroni | VRAM stimata |
|---------|---------|--------------|
| 256×256 | 65 536 | ~80–150 MB |
| 384×384 | 147 456 | ~200–280 MB |
| 512×512 | 262 144 | ~350–450 MB |

Resta su **256×256** per margine OS + browser.

## Troubleshooting

| Problema | Soluzione |
|----------|-----------|
| `cuda: false` | Aggiorna driver, reinstalla `numba`, riavvia |
| Webcam nera | `--camera 1` o usa `--image` |
| FPS bassi | Riduci `--steps-per-frame` a 1 |
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |

## Architettura file

```
mindruntime/
  gpu_core.py      # kernel CUDA Numba
  gpu_engine.py    # GPUBrainEngine (triple-buffer)
  resonators.py    # template FFT 2D
  ai_trainer.py    # stub training PyTorch
  visualizer.py    # loop OpenCV locale
```
