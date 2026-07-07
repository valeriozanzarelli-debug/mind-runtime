# Avvio manuale di CEREBRUM su Windows (PowerShell).
# Sfrutta la GPU se CUDA + torch sono installati.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not $env:CEREBRUM_PORT) { $env:CEREBRUM_PORT = "8788" }

if (Test-Path ".\cerebrum.exe") {
    Write-Host "Avvio CEREBRUM (exe) sulla porta $env:CEREBRUM_PORT ..."
    & .\cerebrum.exe serve --port $env:CEREBRUM_PORT
    return
}

$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    Write-Host "Avvio CEREBRUM (python) sulla porta $env:CEREBRUM_PORT ..."
    & python -m cerebrum serve --port $env:CEREBRUM_PORT
    return
}

Write-Error "Ne' cerebrum.exe ne' python trovati. Installa Python 3.10+ e poi: pip install -e ."
