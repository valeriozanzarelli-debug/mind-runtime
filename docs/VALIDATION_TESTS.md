# Validation Tests — Verifica cervello emergente V2

Test automatici in `tests/test_validation_v2.py`. Esegui:

```powershell
python -m pytest tests/test_validation_v2.py -v
```

---

## Test 1: Emergenza pattern (coherence sale)

**Criterio:** dopo 80+ step su rumore, `coherence_mean` > 0.05.

Il sistema deve auto-organizzarsi verso criticità (SOC + gamma binding).

---

## Test 2: Forme diverse → zone diverse

**Criterio:** cerchio vs quadrato producono mappe coherence con media diversa (|Δ| > 0.01).

Pattern geometrici diversi → attrattori diversi.

---

## Test 3: Order parameter (gamma binding)

**Criterio:** dopo 40 step, order parameter globale R > 0.05.

Verifica sincronizzazione di fase parziale (kernel gamma).

---

## Test 4: Free energy (predictive coding)

**Criterio:** free energy definita e finita dopo step multipli.

Proxy: mean |impulse − predizione locale|.

---

## Test 5: API Part 2

**Criterio:** `export_state`, `import_state`, `get_statistics`, `get_recognition_zones`, `render(mode=...)` funzionano.

---

## Test manuali (opzionali)

### Avalanche power-law

Raccogli `active_neurons` per 1000+ step e plotta istogramma log-log. Distribuzione dovrebbe avere coda lunga (SOC).

### Webcam live

```powershell
python -m mindruntime.visualizer --width 256 --height 256
```

FPS target: **≥ 25** su RTX 1060.

---

## Se un test fallisce

| Test | Kernel da verificare |
|------|---------------------|
| Coherence bassa | SOC (`soc_avalanche`), gamma (`gamma_phase_lock`) |
| R basso | Turing RD + gamma |
| Free energy NaN | Predictive coding |
| FPS bassi | CUDA non attivo — esegui `scripts/test_cuda_numba.py` |
