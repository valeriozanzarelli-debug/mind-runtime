# Installazione GPU locale su Windows (NVIDIA CUDA)
# Esegui in PowerShell dalla root del repo:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\scripts\install-gpu-windows.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== mind-runtime · Retina GPU Windows ===" -ForegroundColor Cyan

# PyTorch con CUDA 12.4 (compatibile con driver recenti NVIDIA)
Write-Host "Installo PyTorch CUDA..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

Write-Host "Installo mind-runtime con dipendenze GPU..." -ForegroundColor Yellow
python -m pip install -e ".[full,gpu]"

Write-Host ""
Write-Host "Diagnostica GPU:" -ForegroundColor Green
python scripts/retina_gpu_local.py --info

Write-Host ""
Write-Host "Benchmark retina 1024x768 (milioni di neuroni-pixel):" -ForegroundColor Green
python scripts/retina_gpu_local.py --preset hd --pulses 20

Write-Host ""
Write-Host "Fatto. Per Baby locale:" -ForegroundColor Cyan
Write-Host "  python -m organism.cli retina --preset hd"
