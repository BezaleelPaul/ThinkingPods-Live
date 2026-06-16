import os
import sys
import subprocess

def check_package(package):
    try:
        __import__(package)
        print(f"[OK] {package} is installed.")
    except ImportError:
        print(f"[ERROR] {package} is NOT installed.")

def check_file(file):
    if os.path.exists(file):
        print(f"[OK] {file} found.")
    else:
        print(f"[ERROR] {file} NOT found.")

print("--- ReqGPT Diagnostics ---")
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}")

expected_python = r"C:\Users\bezal\AppData\Local\Programs\Python\Python312\python.exe"
if sys.executable.lower() != expected_python.lower():
    print(f"[WARNING] You are NOT running the Python from run.bat!")
    print(f"Expected: {expected_python}")
else:
    print(f"[OK] Python executable matches run.bat.")

packages = ["streamlit", "fastapi", "uvicorn", "faster_whisper", "ollama", "torch", "torchaudio", "scipy"]
for pkg in packages:
    check_package(pkg)

files = ["server.py", "app.py"]
for f in files:
    check_file(f)

print("\nChecking for FFmpeg...")
try:
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[OK] FFmpeg is installed and in path.")
except FileNotFoundError:
    print("[WARNING] FFmpeg is NOT found. (Fixed by using scipy in server.py)")

print("\nChecking for port 8000...")
try:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('localhost', 8000)) == 0:
            print("[WARNING] Port 8000 is already in use. Please kill the process using it.")
        else:
            print("[OK] Port 8000 is available.")
except Exception as e:
    print(f"[ERROR] Could not check port 8000: {e}")

print("\n--- Diagnostics Complete ---")
