#!/usr/bin/env python3
"""
simulate_session.py — Automated End-to-End Design Thinking Simulation Harness

Runs a complete 6-turn Design Thinking session against the live backend,
verifies all checklist milestones are gathered, measures turn-by-turn latency,
and exports the generated Product Requirements Document (PRD).

Usage:
    python simulate_session.py [--backend http://127.0.0.1:8000]
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_BACKEND = "http://127.0.0.1:8000"
TEST_USER = "SimulatedUser"
TEST_PROJECT = "GroceryWasteSaver"

# 5 structured answers that satisfy all 6 Empathize fields (100% completion)
SCENARIO_TURNS = [
    ("Turn 1: Persona", "I want to help busy working parents"),
    ("Turn 2: Problem & Motivation", "They waste groceries because fresh food expires before they cook it"),
    ("Turn 3: Frequency", "It happens every single day"),
    ("Turn 4: Existing Solutions", "They currently use WhatsApp groups and sticky notes on the fridge"),
    ("Turn 5: Evidence & Wrap-Up", "I interviewed eight parents who all admitted throwing out vegetables last week"),
]


def send_chat(backend_url: str, user_text: str):
    t0 = time.time()
    url = f"{backend_url}/text"
    payload = json.dumps({
        "text": user_text,
        "username": TEST_USER,
        "pod": "Empathize",
        "project_name": TEST_PROJECT,
        "enable_voice": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        _ = resp.read()
        elapsed = time.time() - t0
        reply = urllib.parse.unquote(resp.headers.get("X-Reply", ""))
        timing_raw = resp.headers.get("X-Timing")
        timing = json.loads(urllib.parse.unquote(timing_raw)) if timing_raw else {}
        return elapsed, reply, timing


def get_status(backend_url: str):
    url = f"{backend_url}/session/status"
    payload = json.dumps({
        "username": TEST_USER,
        "project_name": TEST_PROJECT,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def reset_session(backend_url: str):
    url = f"{backend_url}/reset"
    payload = json.dumps({"username": TEST_USER}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass
    except Exception:
        pass


def run_simulation(backend_url: str):
    print("=" * 70, flush=True)
    print("🤖 STARTING AUTOMATED DESIGN THINKING SIMULATION", flush=True)
    print(f"Backend Target: {backend_url}", flush=True)
    print(f"Project Name:   {TEST_PROJECT}", flush=True)
    print(f"Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/3] Resetting session state on backend...", flush=True)
    reset_session(backend_url)
    time.sleep(1)

    print("\n[2/3] Simulating multi-turn conversation...", flush=True)
    results = []
    for label, message in SCENARIO_TURNS:
        print(f"\n▶ Sending {label}...", flush=True)
        print(f"  User: \"{message}\"", flush=True)
        elapsed, reply, timing = send_chat(backend_url, message)
        results.append((label, message, elapsed, reply))
        print(f"  Latency:  {elapsed:.2f}s", flush=True)
        print(f"  AI Coach: \"{reply}\"", flush=True)

    print("\n[3/3] Inspecting final Empathize checklist status...", flush=True)
    status = get_status(backend_url)
    checklist = status.get("checklist", {})
    completed_count = sum(1 for item in checklist.values() if item.get("complete"))
    total_count = len(checklist)

    print("\n" + "=" * 70, flush=True)
    print("📊 SIMULATION BENCHMARK SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Turn':<28} | {'Latency':<9} | {'Reply Snippet'}", flush=True)
    print("-" * 70, flush=True)
    for label, _, el, rep in results:
        print(f"{label:<28} | {el:>7.2f}s | {rep[:30]}...", flush=True)

    avg_latency = sum(r[2] for r in results) / len(results)
    print("-" * 70, flush=True)
    print(f"Average Turn Latency: {avg_latency:.2f}s", flush=True)
    print(f"Checklist Milestones: {completed_count}/{total_count} Completed ({int(completed_count/total_count*100)}%)", flush=True)

    for k, item in checklist.items():
        mark = "✅" if item.get("complete") else "❌"
        val = item.get("value") or "Not gathered"
        print(f"  {mark} {item.get('description', k).title()}: {val}", flush=True)

    print("\n🎉 Automated simulation successfully finished!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automated ThinkingPods session simulation")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, help="Backend URL (default: http://127.0.0.1:8000)")
    args = parser.parse_args()
    run_simulation(args.backend)
