@echo off
rem Close ThaiASR if it is stuck and does not respond to the close button
rem (or to Ctrl+Alt+Q).  Only processes started from this folder are stopped.
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$me = Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' or Name='python.exe'\" | Where-Object { $_.CommandLine -like '*thaiasr*' };" ^
  "if (-not $me) { Write-Host 'ThaiASR is not running.'; exit 0 }" ^
  "foreach ($p in $me) { Write-Host ('Stopping PID ' + $p.ProcessId); Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }" ^
  "Start-Sleep -Milliseconds 400;" ^
  "if (Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\" | Where-Object { $_.CommandLine -like '*thaiasr*' }) { Write-Host 'Still running - try Task Manager.' } else { Write-Host 'Closed.' }"
