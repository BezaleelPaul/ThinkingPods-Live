@echo off
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
if not defined PYTHON if exist "C:\Users\bezal\AppData\Local\Programs\Python\Python312\python.exe" (
    set "PYTHON=C:\Users\bezal\AppData\Local\Programs\Python\Python312\python.exe"
)
if not defined PYTHON (
    set "PYTHON=python"
)

rem --- LLM Model Configuration ---
rem autotweak builds optimized-pods from OLLAMA_BASE_MODEL.
rem MENTOR_MODEL targets the optimized profile with automatic fallback.
set OLLAMA_BASE_MODEL=qwen2.5:3b
set MENTOR_MODEL=optimized-pods
set LLM_NUM_CTX=1024
set REPLY_CACHE_ANCHOR=false
set FAST_SHORT_REPLIES=true
set DYNAMIC_SHORT_PHRASING=true
set FAST_SHORT_REPLIES_MAX_WORDS=30
set COMPLEXITY_GATE=false
set MENTOR_USE_LLM_EXTRACTION=false

rem --- Voice Transcription (STT) Model Configuration ---
rem Options: tiny.en, base.en (default), small.en (recommended for better accuracy), medium.en
rem Note: larger models take more RAM and CPU but have much better accuracy.
set WHISPER_MODEL=tiny.en

rem --- Hardware Acceleration ---
rem Enable Vulkan execution for Intel HD / iGPU offloading if supported
set OLLAMA_VULKAN=1

rem --- Observation-only text classification (HuggingFace transformers) ---
rem Default OFF; enabled here so the Developer Console shows Sentiment /
rem Message Classification during demos. Tests run without this file.
set SENTIMENT_ENABLED=true
set CLASSIFIER_ENABLED=true

echo Checking for existing processes on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo Killing process %%a using port 8000...
    taskkill /F /PID %%a
)

echo Running Dynamic Hardware Auto-Tweaker...
"%PYTHON%" "%~dp0autotweak.py"

start "ReqGPT Backend" /D "%~dp0" "%PYTHON%" server.py
start "ReqGPT Frontend" /D "%~dp0" "%PYTHON%" -m streamlit run app.py
echo Both servers launched. Close this window or press Ctrl+C to stop.
pause
