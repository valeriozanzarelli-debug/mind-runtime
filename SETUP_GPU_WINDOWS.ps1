# Setup GPU per CEREBRUM su Windows.
#
# Installa il runtime cerebrum in Python con torch CUDA, cosi' il cervello
# gira sulla GPU. Dopo questo setup, "Avvia" da Ink Admin usa automaticamente
# la GPU (il launcher preferisce python+CUDA all'exe CPU).
#
# Uso:  .\SETUP_GPU_WINDOWS.ps1
#
# Requisiti: Python 3.10+ nel PATH e driver NVIDIA aggiornati.
# NOTA: file in puro ASCII (Windows PowerShell 5.1 corrompe i caratteri speciali).

$ErrorActionPreference = "Stop"

Write-Host "== CEREBRUM - setup GPU (Windows) ==" -ForegroundColor Cyan

# 1) Python presente?
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python non trovato nel PATH. Installa Python 3.10+ da python.org e riprova."
}
python --version

# 2) Installa il package cerebrum (da GitHub, sempre l'ultima versione su main)
Write-Host "[1/3] Installo il runtime cerebrum..." -ForegroundColor Cyan
python -m pip install --upgrade pip | Out-Null
python -m pip install --upgrade "git+https://github.com/valeriozanzarelli-debug/mind-runtime.git"
if ($LASTEXITCODE -ne 0) { throw "Installazione cerebrum fallita" }

# 3) Installa torch con CUDA (prova gli indici dal piu' recente)
Write-Host "[2/3] Installo torch CUDA (puo' scaricare 2+ GB, pazienta)..." -ForegroundColor Cyan
$cudaIndexes = @(
    "https://download.pytorch.org/whl/cu128",
    "https://download.pytorch.org/whl/cu126",
    "https://download.pytorch.org/whl/cu121"
)
$torchOk = $false
foreach ($idx in $cudaIndexes) {
    Write-Host "  Provo $idx ..."
    python -m pip install torch --index-url $idx
    if ($LASTEXITCODE -eq 0) { $torchOk = $true; break }
}
if (-not $torchOk) { throw "Installazione torch CUDA fallita da tutti gli indici" }

# 4) Verifica: il cervello deve vedere la GPU
Write-Host "[3/3] Verifica GPU..." -ForegroundColor Cyan
python -m cerebrum info
python -c "import torch; print('CUDA disponibile:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nessuna')"
if ($LASTEXITCODE -ne 0) { throw "Verifica fallita" }

Write-Host ""
Write-Host "Setup GPU completato." -ForegroundColor Green
Write-Host "Ora da Ink Admin: CEREBRUM -> Ferma -> Avvia. Il launcher scegliera' la GPU."
Write-Host "(Se avevi gia' scaricato il pacchetto, rifai 'Scarica e installa' per avere il launcher aggiornato.)"
