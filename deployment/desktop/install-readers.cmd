@echo off
setlocal
set "PYTHONHOME="
"%~dp0runtime\python.exe" -IB -X utf8 "%~dp0tools\install_readers.py" %*
set "status=%errorlevel%"
pause
exit /b %status%
