@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\schedule.ps1" %*
set "status=%errorlevel%"
pause
exit /b %status%
