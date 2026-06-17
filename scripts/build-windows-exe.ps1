# Build ORGANISM-Windows.exe — esegui su Windows (PowerShell)
#   .\scripts\build-windows-exe.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== Build ORGANISM Windows .exe ===" -ForegroundColor Cyan

python -m pip install --upgrade pip
python -m pip install -e ".[full]" pyinstaller tzdata

Write-Host "Compilo con PyInstaller..." -ForegroundColor Yellow
python -m PyInstaller packaging/organism_windows.spec --noconfirm --clean

$out = "dist/ORGANISM-Windows.exe"
if (-not (Test-Path $out)) {
    Write-Error "Build fallita — $out non trovato"
}

$destDir = "organism/nursery/static/releases"
New-Item -ItemType Directory -Force -Path $destDir | Out-Null
Copy-Item $out "$destDir/ORGANISM-Windows.exe" -Force

$sizeMb = [math]::Round((Get-Item $out).Length / 1MB, 1)
Write-Host ""
Write-Host "OK: $out ($sizeMb MB)" -ForegroundColor Green
Write-Host "Copiato in $destDir per il sito"
Write-Host ""
Write-Host "Prova locale:" -ForegroundColor Cyan
Write-Host "  .\dist\ORGANISM-Windows.exe"
Write-Host ""
Write-Host "Sul sito:" -ForegroundColor Cyan
Write-Host "  https://inkconscius.eu/organism/download"
