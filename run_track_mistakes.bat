@echo off
title ReqGPT - Mistake & Failure Tracker

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

echo ====================================================================
echo   REQGPT MISTAKE ^& CONVERSATION FAILURE TRACKER
echo ====================================================================
echo Running comprehensive audit on golden fixtures and saved sessions...
echo.
"%PYTHON%" "%~dp0track_mistakes.py" --verbose --export
echo.
echo Audit complete. Full report written to docs\reports\Mistakes_Audit_Report.md
pause
