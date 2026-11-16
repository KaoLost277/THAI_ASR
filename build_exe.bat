@echo off
rem Build the standalone .exe bundle: dist\ThaiASR\ThaiASR.exe (no console)
rem and dist\ThaiASR\ThaiASR-cli.exe (console, for --download / --transcribe).
rem The ONNX model is NOT included: it is downloaded on first run.
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo ThaiASR is not installed yet. Run install.bat first.
  pause
  exit /b 1
)

"%PY%" -c "import PyInstaller" 2>nul
if errorlevel 1 (
  echo [1/2] installing PyInstaller + tray dependencies ...
  where uv >nul 2>nul
  if not errorlevel 1 (
    uv pip install --python "%PY%" pyinstaller pystray pillow || goto :fail
  ) else (
    "%PY%" -m pip install pyinstaller pystray pillow || goto :fail
  )
) else (
  echo [1/2] PyInstaller already installed
)

"%PY%" -c "from PIL import Image" 2>nul
if errorlevel 1 (
  where uv >nul 2>nul
  if not errorlevel 1 (
    uv pip install --python "%PY%" pillow || goto :fail
  ) else (
    "%PY%" -m pip install pillow || goto :fail
  )
)

if not exist "packaging\thaiasr.ico" (
  echo generating the icon ...
  "%PY%" packaging\make_icon.py || goto :fail
)

echo [2/2] building (takes a few minutes) ...
"%PY%" -m PyInstaller packaging\thaiasr.spec --noconfirm --clean --distpath dist --workpath build || goto :fail

echo.
echo Done: dist\ThaiASR\ThaiASR.exe
echo Next: install_shortcut.bat  (adds Start Menu + Desktop shortcuts)
echo Note: the first run downloads the model (about 2.6 GB) into %%LOCALAPPDATA%%\ThaiASR
pause
exit /b 0

:fail
echo.
echo Build failed. See the messages above.
pause
exit /b 1
