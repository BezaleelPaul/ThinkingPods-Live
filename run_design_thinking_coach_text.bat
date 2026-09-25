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
if not defined PYTHON (
    set "PYTHON=python"
)

rem --- LLM Model Configuration ---
rem Single model for both mentor and extraction.
set MENTOR_MODEL=qwen2.5:3b

echo Starting Text-Based Design Thinking Coach...
"%PYTHON%" "%~dp0cli\design_thinking_coach_text.py"
pause
