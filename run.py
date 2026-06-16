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

backend = subprocess.Popen([python, "server.py"], **common)
frontend = subprocess.Popen([python, "-m", "streamlit", "run", "app.py"], **common)

print(f"Backend → http://localhost:8000")
print(f"Frontend → http://localhost:8501")
print(f"Python: {python}")
print("Press Ctrl+C to stop both.")

try:
    backend.wait()
    frontend.wait()
except KeyboardInterrupt:
    backend.terminate()
    frontend.terminate()
