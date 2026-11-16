@echo off
rem Add Start Menu + Desktop shortcuts for the built app (run build_exe.bat first).
rem   install_shortcut.bat            -> Start Menu + Desktop
rem   install_shortcut.bat startmenu  -> Start Menu only
rem   install_shortcut.bat desktop    -> Desktop only
rem   install_shortcut.bat remove     -> delete both shortcuts
setlocal
cd /d "%~dp0"
set "EXE=%~dp0dist\ThaiASR\ThaiASR.exe"
set "WHAT=%~1"
if "%WHAT%"=="" set "WHAT=both"

if /i not "%WHAT%"=="remove" if not exist "%EXE%" (
  echo Not built yet: %EXE%
  echo Run build_exe.bat first.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\make_shortcut.ps1" -Exe "%EXE%" -What "%WHAT%"
if errorlevel 1 (
  echo.
  echo Creating the shortcut failed.
  pause
  exit /b 1
)
pause
exit /b 0
