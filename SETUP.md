# Mindruntime GPU — setup Windows 11 + RTX 1060 (6 GB)

Motore GPU **locale**: Hodgkin-Huxley + Turing + SOC + biforcazione. Nessun server richiesto.

## Requisiti hardware

- Windows 11
- NVIDIA RTX 1060 6 GB (compute capability ≥ 6.1)
- Webcam (opzionale)

## 1. CUDA Toolkit 12.x

1. Scarica [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive) per Windows.
2. Installa con driver NVIDIA aggiornato (≥ 535).
3. Verifica in PowerShell:

```powershell
nvcc --version
nvidia-smi
```

## 2. cuDNN 9.x (opzionale)

Numba CUDA **non** richiede cuDNN per `gpu_physics.py`. Serve solo per training PyTorch opzionale.

1. Registrati su [NVIDIA Developer](https://developer.nvidia.com/cudnn).
2. Scarica cuDNN 9.x per CUDA 12.
3. Copia DLL in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin`.

## 3. Ambiente Python 3.11

```powershell
cd C:\path\to\mind-runtime
python -m venv venv_mindruntime
.\venv_mindruntime\Scripts\activate
python -m pip install -U pip
```

## 4. Installazione dipendenze

```powershell
pip install -e ".[mindruntime]"
```

Include: `numpy`, `numba`, `scipy`, `opencv-python`, `matplotlib`

### Cupy (opzionale)

```powershell
pip install cupy-cuda12x
```

## 5. Test CUDA Numba

Crea `test_cuda.py`:

```python
from numba import cuda
from mindruntime.cuda_util import cuda_info

print(cuda_info())
print(f"CUDA available: {cuda.is_available()}")
```

```powershell
python test_cuda.py
```

Output atteso:

```json
{"numba": true, "cuda": true, "device": "NVIDIA GeForce GTX 1060", "compute_capability": [6, 1]}
```

## 6. Avvio visualizer

```powershell
python -m mindruntime.visualizer
```

Finestra OpenCV **Mindruntime GPU — Cervello emergente**:
- mappa coerenza (fase → hue, impulso → saturation)
- inset webcam
- simboli riconosciuti per risonanza FFT

Motore dendritico legacy:

```powershell
python -m mindruntime.visualizer --engine dendritic
```

**Q** o **ESC** per uscire.

## Architettura file

```
mindruntime/
  gpu_physics.py   # kernel CUDA A-F (HH, Turing, SOC, biforcazione, memoria)
  gpu_engine.py    # GPUBrainEngine (triple-buffer 8 canali)
  gpu_core.py      # kernel legacy onde/Hebb (deprecato)
  resonators.py    # template FFT A-Z + forme
  visualizer.py    # loop OpenCV webcam
  dendritic_*.py   # motore dendritico alternativo
```

## VRAM stimata (6 GB)

| Griglia | Pixel | VRAM |
|---------|-------|------|
| 256×256 | 65 536 | ~200–300 MB |
| 128×128 | 16 384 | ~50–80 MB |

Riduci risoluzione se FPS < 10:

```powershell
python -m mindruntime.visualizer --width 128 --height 128
```

## Troubleshooting

| Problema | Soluzione |
|----------|-----------|
| `cuda: false` | Aggiorna driver, reinstalla `numba`, riavvia |
| `ModuleNotFoundError: numba` | Attiva venv: `.\venv_mindruntime\Scripts\activate` |
| Webcam nera | `--camera 1` o `--image test.jpg` |
| FPS bassi | `--width 128 --height 128` |
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
