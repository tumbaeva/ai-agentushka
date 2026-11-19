@echo off
REM ============================================================
REM  {{AGENT_NAME}} - portable launcher (Windows)
REM  Creates a virtual environment, installs dependencies once,
REM  then starts the agent and opens the browser.
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [setup] Creating virtual environment...
  python -m venv .venv || goto :error
  echo [setup] Installing dependencies (first run only)...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)

echo [run] Starting {{AGENT_NAME}}...
".venv\Scripts\python.exe" run.py
goto :eof

:error
echo [error] Setup failed. Ensure Python 3.10+ is installed and on PATH.
exit /b 1
