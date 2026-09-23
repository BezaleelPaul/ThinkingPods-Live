"""
benchmark_latency.py — Phase 1.5 mentor-turn latency benchmark.

Runs a fixed 5-turn conversation through the live
``mentor.process_mentor_turn`` pipeline (default model ``optimized-pods``)
and records the per-turn ``TurnProfiler`` profile attached by the
instrumentation as ``timing.profile``.

Usage:
    python benchmark_latency.py [--runs 3] [--model optimized-pods]
        [--out benchmark_latency_results.json] [--warmup 1]

Design notes (see Phase 1.5 audit):
* One warmup turn per process (cold Ollama ``load_duration`` would
  otherwise dominate run 1 and hide steady-state behavior).
* Fresh session per run (``reset_runtime_state``) so history length — and
  therefore persistence/prompt size — is comparable across runs.
* No mocks: every LLM call is real. Slow on CPU (~1 min/turn); 3 runs of
  5 turns take roughly 15-20 minutes.
* Output JSON preserves every run (no averaging away); the printed table
  shows medians.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

sys.path.insert(0, ".")

import mentor  # noqa: E402
from session_manager import get_session_manager  # noqa: E402

# Canonical 5-turn benchmark conversation (fixed wording so Phase 2 can
# rerun it verbatim and diff stage timings turn-by-turn).
TURNS = [
    "I want to help college students manage assignment deadlines.",
    "They use sticky notes and phone alarms, and it happens every day.",
    "That's not what I meant — it's high-school teachers, not students.",
    "I don't understand.",
    "What do you mean by target user?",
]


def hardware_info() -> dict:
    """Best-effort runtime conditions record. Never raises."""
    info: dict = {}
    try:
        import platform
        info["platform"] = platform.platform()
        info["python"] = platform.python_version()
        info["cpu"] = platform.processor() or platform.machine()
    except Exception:
        pass
    try:
        import psutil
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    except Exception:
        info["psutil"] = "unavailable"
    try:
        import subprocess
        out = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=15
        )
        info["ollama_version"] = (out.stdout or out.stderr).strip()
    except Exception as exc:
        info["ollama_version"] = f"unknown ({exc})"
    return info


def fresh_session(username: str, project: str) -> None:
    get_session_manager().reset_runtime_state(
        username=username, project_title=project)


def run_turn(user_message: str, username: str, project: str, model: str) -> dict:
    """One instrumented mentor turn; returns the profile + reply metadata."""
    wall_start = time.perf_counter()
    reply, _session, timing, diag = mentor.process_mentor_turn(
        user_message, username=username, project_name=project,
        model_name=model,
    )
    wall_ms = round((time.perf_counter() - wall_start) * 1000.0, 1)
    profile = getattr(timing, "profile", None) or {}
    prompt = (diag or {}).get("Prompt", "") or ""
    brief = (diag or {}).get("ConversationBrief", {}) or {}
    return {
        "user": user_message,
        "turn_type": brief.get("turn_type"),
        "prompt_chars": len(prompt),
        "wall_ms": wall_ms,
        "turning_total_ms": timing.total_ms if timing else None,
        "stages": profile.get("stages", {}),
        "llm_calls": profile.get("llm_calls", []),
        "reply_chars": len(reply or ""),
        "reply": reply,
    }


def streaming_probe(model: str) -> dict:
    """One-off streaming call (NOT part of the turn path) to split TTFT
    from generation time. Documents what non-streaming leaves unmeasured."""
    import ollama

    prompt = "Explain in one short sentence why Design Thinking starts with empathy."
    start = time.perf_counter()
    first_token_ms = None
    chunks = 0
    try:
        for chunk in ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.4, "num_predict": 60},
            stream=True,
        ):
            chunks += 1
            content = ((chunk.get("message") or {}) if hasattr(chunk, "get") else {}).get("content")
            if content is None and hasattr(chunk, "message"):
                content = getattr(chunk.message, "content", None)
            if content and first_token_ms is None:
                first_token_ms = round((time.perf_counter() - start) * 1000.0, 1)
    except Exception as exc:
        return {"error": str(exc)}
    total_ms = round((time.perf_counter() - start) * 1000.0, 1)
    return {
        "model": model,
        "ttft_ms": first_token_ms,
        "total_ms": total_ms,
        "generation_after_first_ms": (
            round(total_ms - first_token_ms, 1)
            if first_token_ms is not None else None
        ),
        "chunks": chunks,
    }


def session_bytes(username: str) -> dict:
    """Current persisted session footprint (persistence magnitude)."""
    try:
        mgr = get_session_manager()
        session_id = mgr.get_active_session_id()
        if not session_id:
            return {}
        from pathlib import Path
        import session_manager as sm
        d = Path(sm.__file__).parent / "sessions" / session_id
        files = {}
        for name in ("session_data.json", "session.json"):
            p = d / name
            if p.exists():
                files[name] = p.stat().st_size
        return {"session_id": session_id, **files}
    except Exception as exc:
        return {"error": str(exc)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="optimized-pods")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--out", default="benchmark_latency_results.json")
    parser.add_argument("--no-stream-probe", action="store_true")
    args = parser.parse_args(argv)

    print(f"hardware: {json.dumps(hardware_info())}")
    print(f"model={args.model} runs={args.runs} warmup={args.warmup}")

    for w in range(args.warmup):
        fresh_session(f"bench_warm{w}", "BenchWarm")
        run_turn(TURNS[0], f"bench_warm{w}", "BenchWarm", args.model)
        print(f"warmup {w + 1} done", flush=True)

    all_runs = []
    for r in range(args.runs):
        username = f"bench_r{r}"
        fresh_session(username, "BenchLatency")
        run_record = {"run": r + 1, "turns": []}
        for i, msg in enumerate(TURNS):
            rec = run_turn(msg, username, "BenchLatency", args.model)
            run_record["turns"].append(rec)
            llm_sum = sum(
                c.get("duration_ms", 0) for c in rec["llm_calls"])
            print(
                f"run {r + 1} turn {i + 1}: wall={rec['wall_ms']}ms "
                f"llm_sum={round(llm_sum, 1)}ms "
                f"calls={len(rec['llm_calls'])} "
                f"prompt={rec['prompt_chars']}ch "
                f"type={rec['turn_type']}",
                flush=True,
            )
        run_record["session_bytes"] = session_bytes(username)
        all_runs.append(run_record)

    result = {
        "hardware": hardware_info(),
        "model": args.model,
        "turns": TURNS,
        "runs": all_runs,
    }
    if not args.no_stream_probe:
        result["streaming_probe"] = streaming_probe(args.model)
        print(f"streaming probe: {json.dumps(result['streaming_probe'])}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    print(f"saved {args.out}")

    # Median summary table (informational; raw runs preserved above).
    for i, msg in enumerate(TURNS):
        walls = [run["turns"][i]["wall_ms"] for run in all_runs]
        llms = [sum(c.get("duration_ms", 0)
                     for c in run["turns"][i]["llm_calls"])
                for run in all_runs]
        print(f"turn {i + 1} ({msg[:40]}...): "
              f"wall median={statistics.median(walls):.0f}ms "
              f"llm median={statistics.median(llms):.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
