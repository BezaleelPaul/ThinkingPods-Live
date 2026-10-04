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

# Set Optimized Hardware & Pipeline Defaults
os.environ.setdefault("OLLAMA_BASE_MODEL", "qwen2.5:3b")
os.environ.setdefault("MENTOR_MODEL", "optimized-pods")
os.environ.setdefault("LLM_NUM_CTX", "1024")
os.environ.setdefault("REPLY_CACHE_ANCHOR", "false")
os.environ.setdefault("FAST_SHORT_REPLIES", "true")
os.environ.setdefault("DYNAMIC_SHORT_PHRASING", "true")
os.environ.setdefault("COMPLEXITY_GATE", "false")
os.environ.setdefault("WHISPER_MODEL", "tiny.en")
os.environ.setdefault("OLLAMA_VULKAN", "1")
os.environ.setdefault("SENTIMENT_ENABLED", "true")
os.environ.setdefault("CLASSIFIER_ENABLED", "true")
os.environ.setdefault("MENTOR_USE_LLM_EXTRACTION", "false")
os.environ.setdefault("FAST_SHORT_REPLIES_MAX_WORDS", "30")

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
