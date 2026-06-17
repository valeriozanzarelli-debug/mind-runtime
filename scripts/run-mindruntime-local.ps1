# Avvio rapido mindruntime su Windows (RTX 1060, solo locale)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

if (-not (Test-Path "venv_mindruntime")) {
    python -m venv venv_mindruntime
}
& .\venv_mindruntime\Scripts\Activate.ps1
pip install -U pip
pip install -e ".[mindruntime]"

Write-Host "`nTest CUDA..."
python -c "from mindruntime.cuda_util import cuda_info; print(cuda_info())"

Write-Host "`nAvvio visualizer (webcam)..."
python -m mindruntime.visualizer --width 256 --height 256
