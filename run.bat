@echo off
rem Start ThaiASR without a console window.
rem Double-click this file, or put a shortcut to it in the Start menu / Startup.
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PY%" set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo ThaiASR is not installed yet. Run install.bat first.
  pause
  exit /b 1
)
set "PYTHONPATH=%~dp0src"
start "" "%PY%" -m thaiasr %*
