# Avvio Organism su Windows — nursery + GPU worker (mind_compact + 3D)

Param(
    [switch]$GpuOnly,
    [switch]$NurseryOnly,
    [string]$DnaVariant = "mind_compact",
    [int]$GpuPort = 8770,
    [int]$NurseryPort = 8765
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Test-Python {
    try {
        $v = python --version 2>&1
        Write-Host "Python: $v"
        return $true
    } catch {
        Write-Host "ERRORE: Python non trovato. Installa Python 3.11+ da python.org" -ForegroundColor Red
        return $false
    }
}

function Start-GpuWorker {
    Write-Host "`n=== GPU Worker (3D impulse) :$GpuPort ===" -ForegroundColor Cyan
    $env:ORGANISM_IMPULSE_W = "512"
    $env:ORGANISM_IMPULSE_H = "384"
    $env:ORGANISM_IMPULSE_D = "128"
    $env:ORGANISM_GPU_WORKER_DEVICE = "cuda"
    $env:ORGANISM_GPU_WORKER_PORT = "$GpuPort"
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "cd '$Root'; `$env:ORGANISM_IMPULSE_W='512'; `$env:ORGANISM_IMPULSE_H='384'; `$env:ORGANISM_IMPULSE_D='128'; `$env:ORGANISM_GPU_WORKER_DEVICE='cuda'; python -m organism.distributed.gpu_worker_server --port $GpuPort"
    )
    Write-Host "GPU worker avviato in finestra separata. Health: http://127.0.0.1:$GpuPort/health"
}

function Start-Nursery {
    Write-Host "`n=== Nursery (Baby) :$NurseryPort ===" -ForegroundColor Cyan
    $env:ORGANISM_DNA_VARIANT = $DnaVariant
    $env:ORGANISM_COMPACT_BRAIN = "1"
    $env:ORGANISM_GPU_REMOTE = "http://127.0.0.1:$GpuPort"
    $env:ORGANISM_IMPULSE = "1"
    $env:ORGANISM_HYBRID_GPU = "1"
    $env:ORGANISM_DISK_VAULT = "1"
    $env:ORGANISM_PORT = "$NurseryPort"
    if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) {
        Write-Host "Creazione venv..."
        python -m venv venv
        .\venv\Scripts\Activate.ps1
        python -m pip install -U pip
        pip install -e ".[full,gpu]"
    } else {
        .\venv\Scripts\Activate.ps1
    }
    Write-Host "DNA: $DnaVariant | GPU remote: http://127.0.0.1:$GpuPort"
    Write-Host "UI: http://127.0.0.1:$NurseryPort/baby"
    Write-Host "Health: http://127.0.0.1:$NurseryPort/api/baby/health"
    python -m organism.cli nursery --host 127.0.0.1 --port $NurseryPort --browser
}

if (-not (Test-Python)) { exit 1 }

if (-not (Test-Path "pyproject.toml")) {
    Write-Host "Esegui dalla cartella mind-runtime" -ForegroundColor Red
    exit 1
}

if ($GpuOnly) {
    Start-GpuWorker
    exit 0
}
if ($NurseryOnly) {
    Start-Nursery
    exit 0
}

Start-GpuWorker
Start-Sleep -Seconds 3
Start-Nursery
