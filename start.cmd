@echo off
cd /d "%~dp0"
set "NEUROSEG_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%NEUROSEG_PYTHON%" (
  "%NEUROSEG_PYTHON%" start.py
) else (
  py -3.12 start.py
)
pause
