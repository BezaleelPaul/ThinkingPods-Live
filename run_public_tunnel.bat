@echo off
setlocal
title ThinkingPods Public Tunnel (Free)

echo ====================================================================
echo   ThinkingPods - Instant Free Public Link (Cloudflare Tunnel)
echo ====================================================================
echo.
echo Checking for Cloudflare Tunnel tool (cloudflared.exe)...

if not exist "%~dp0cloudflared.exe" (
    echo Downloading portable cloudflared tool...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile '%~dp0cloudflared.exe'"
    if not exist "%~dp0cloudflared.exe" (
        echo [ERROR] Failed to download cloudflared. Please check your internet connection.
        pause
        exit /b 1
    )
    echo [OK] cloudflared downloaded successfully.
)

echo.
echo ====================================================================
echo   Make sure run.bat is already running in another window!
echo   Starting free public tunnel to Streamlit (Port 8501)...
echo ====================================================================
echo.
echo Look for the "https://....trycloudflare.com" link below:
echo.

"%~dp0cloudflared.exe" tunnel --url http://127.0.0.1:8501
pause
