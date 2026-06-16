@echo off
set PYTHON=C:\Users\bezal\AppData\Local\Programs\Python\Python312\python.exe

echo Checking for existing processes on port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo Killing process %%a using port 8000...
    taskkill /F /PID %%a
)

start "ReqGPT Backend" "%PYTHON%" server.py
start "ReqGPT Frontend" "%PYTHON%" -m streamlit run app.py
echo Both servers launched. Close this window or press Ctrl+C to stop.
pause
