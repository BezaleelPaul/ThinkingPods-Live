import subprocess
import sys
import platform
import os

scripts_dir = os.path.dirname(os.path.abspath(__file__))
python = sys.executable

common = dict(cwd=scripts_dir)
if platform.system() == "Windows":
    common["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
else:
    common["start_new_session"] = True

# Run Dynamic Hardware Profiler & Auto-Tweaker
try:
    print("Initializing hardware optimization settings...")
    subprocess.run([python, "autotweak.py"], cwd=scripts_dir)
except Exception as e:
    print(f"Auto-tweak script skipped or encountered error: {e}")

backend = subprocess.Popen([python, "server.py"], **common)
frontend = subprocess.Popen([python, "-m", "streamlit", "run", "app.py"], **common)

print(f"Backend → http://localhost:8000")
print(f"Frontend → http://localhost:8501")
print(f"Python: {python}")
print("Press Ctrl+C to stop both.")

import signal

def cleanup():
    if backend.poll() is None:
        backend.terminate()
    if frontend.poll() is None:
        frontend.terminate()

try:
    backend.wait()
    if backend.returncode != 0:
        print(f"Backend exited with code {backend.returncode}. Stopping frontend.")
        frontend.terminate()
    frontend.wait()
except KeyboardInterrupt:
    cleanup()
