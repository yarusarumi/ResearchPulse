@echo off
setlocal
set "PYTHONHOME="
set "PYTHONPATH=%~dp0app"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
"%~dp0runtime\python.exe" -P -X utf8 -m neuroei.desktop %*
set "status=%errorlevel%"
if not "%status%"=="0" pause
exit /b %status%
