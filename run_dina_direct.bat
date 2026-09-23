@echo off
title Dina Direct - 100%% Offline

rem --- LLM Model Configuration ---
rem Single model for both mentor and extraction.
set MENTOR_MODEL=qwen2.5:3b

rem --- Voice Transcription (STT) Model Configuration ---
rem Options: tiny.en, base.en (default), small.en (recommended for better accuracy), medium.en
rem Note: larger models take more RAM and CPU but have much better accuracy.
set WHISPER_MODEL=tiny.en
set "PYTHON="
set "BASE_PY="
set "PYTHONPATH="
if exist "%~dp0.venv\pyvenv.cfg" (
    for /f "tokens=1,* delims==" %%a in ('findstr /b /c:"base-executable =" "%~dp0.venv\pyvenv.cfg"') do set "BASE_PY=%%b"
)
if defined BASE_PY set "BASE_PY=%BASE_PY:~1%"
if defined BASE_PY if exist "%BASE_PY%" (
    set "PYTHON=%BASE_PY%"
    set "PYTHONPATH=%~dp0.venv\Lib\site-packages"
)
if not defined PYTHON if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
)
if not defined PYTHON if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON=%~dp0venv\Scripts\python.exe"
)
if not defined PYTHON (
    set "PYTHON=python"
)

echo Running Dynamic Hardware Auto-Tweaker...
"%PYTHON%" "%~dp0autotweak.py"

echo ===========================================
echo   DINA DIRECT - SIMPLIFIED OFFLINE VOICE
echo ===========================================
echo 1. Ensure Ollama is running in background.
echo 2. loading models (Whisper + Silero)...
echo.
"%PYTHON%" "%~dp0dina_direct.py"
pause
