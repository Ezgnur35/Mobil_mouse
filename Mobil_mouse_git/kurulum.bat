@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" kurulum_setup.py
) else (
    python kurulum_setup.py
)
pause
