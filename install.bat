@echo off
rem One-time setup: virtual environment, dependencies, model download.
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] creating .venv ...
  where uv >nul 2>nul
  if not errorlevel 1 (
    uv venv .venv --python 3.12 || goto :fail
  ) else (
    python -m venv .venv || goto :fail
  )
) else (
  echo [1/3] .venv already exists
)

echo [2/3] installing dependencies ...
where uv >nul 2>nul
if not errorlevel 1 (
  uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt || goto :fail
) else (
  ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
)

echo [3/3] downloading the model (about 2.6 GB for fp32, one time) ...
".venv\Scripts\python.exe" -m thaiasr --download || goto :fail

echo.
echo Done. Start the dictation app with run.bat
pause
exit /b 0

:fail
echo.
echo Setup failed. See the messages above.
pause
exit /b 1
