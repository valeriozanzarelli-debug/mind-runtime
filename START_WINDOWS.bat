@echo off
REM Avvio manuale di CEREBRUM su Windows (senza Ink Admin).
REM Usa la GPU se sono installati CUDA + torch.
setlocal
cd /d "%~dp0"

if not defined CEREBRUM_PORT set CEREBRUM_PORT=8788

REM 1) EXE impacchettato
if exist "cerebrum.exe" (
  echo Avvio CEREBRUM (exe) sulla porta %CEREBRUM_PORT% ...
  cerebrum.exe serve --port %CEREBRUM_PORT%
  goto :eof
)

REM 2) Python installato
where python >nul 2>nul
if %errorlevel%==0 (
  echo Avvio CEREBRUM (python) sulla porta %CEREBRUM_PORT% ...
  python -m cerebrum serve --port %CEREBRUM_PORT%
  goto :eof
)

echo [ERRORE] Ne' cerebrum.exe ne' python trovati.
echo Installa Python 3.10+ e poi:  pip install -e .
pause
