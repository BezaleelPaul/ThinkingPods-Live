"""
measure_startup.py — boot + first-response + memory benchmark for server.py.

Launches the backend as a child process and measures, until the app answers
GET /health:

  * import_ms   — wall-clock to `import server` complete (synchronous, blocks
                  the uvicorn bind in the eager-import build)
  * boot_ms     — wall-clock from process launch until /health returns 200
  * rss_mb      — child RSS at the moment /health first responds

Prints one line of JSON. Deterministic-enough for before/after comparison.
No production behavior is touched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

HARNESS = r"""
import json, sys, time, os
t0 = time.perf_counter()
import server
t_import = (time.perf_counter() - t0) * 1000
with open(os.environ["STARTUP_REPORT"], "w") as f:
    json.dump({"import_ms": t_import, "pid": None}, f)
os.dup2(os.open(os.devnull, os.O_WRONLY), 1)  # silence model-load prints
os.dup2(os.open(os.devnull, os.O_WRONLY), 2)  # silence uvicorn logs
import uvicorn
uvicorn.run(server.app, host="127.0.0.1", port=8765, log_level="error")
"""

PORT = 8765


def _health_ok() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1.0) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> None:
    import tempfile
    import psutil
    report_path = os.path.join(tempfile.gettempdir(), "reqgpt_startup_report.json")
    if os.path.exists(report_path):
        os.remove(report_path)
    env = dict(os.environ, STARTUP_REPORT=report_path)
    t_launch = time.perf_counter()
    child = subprocess.Popen([sys.executable, "-c", HARNESS], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    import_ms = None
    deadline_file = time.monotonic() + 60
    while time.monotonic() < deadline_file:
        if os.path.exists(report_path):
            try:
                with open(report_path) as f:
                    import_ms = json.load(f)["import_ms"]
                break
            except Exception:
                pass
        time.sleep(0.02)

    # poll until the child serves /health
    deadline = time.monotonic() + 240
    rss_mb = None
    while time.monotonic() < deadline:
        if _health_ok():
            boot_ms = (time.perf_counter() - t_launch) * 1000
            try:
                rss_mb = psutil.Process(child.pid).memory_info().rss / (1024 * 1024)
            except Exception:
                rss_mb = None
            print(json.dumps({"import_ms": import_ms, "boot_ms": round(boot_ms),
                              "rss_mb": round(rss_mb, 1) if rss_mb else None}))
            child.terminate()
            return
        time.sleep(0.1)
    print(json.dumps({"error": "server never became ready"}))
    child.kill()


if __name__ == "__main__":
    main()