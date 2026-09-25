import time
import sys
import statistics
import os
import traceback

import ollama as ollama_module

import mentor
from mentor import process_mentor_turn

original_ollama_chat = ollama_module.chat

_timestamps = {}
_ollama_call_durations = []

def patched_chat(*args, **kwargs):
    call_start = time.perf_counter()
    stream = kwargs.pop("stream", False)
    if stream:
        class StreamWrapper:
            def __init__(self, inner_iter):
                self.inner_iter = inner_iter
                self.first_token = True
                self.first_token_time = None
                self.complete_time = None
            def __iter__(self):
                for chunk in self.inner_iter:
                    if self.first_token and "message" in chunk and "content" in chunk["message"]:
                        if chunk["message"]["content"]:
                            self.first_token_time = time.perf_counter()
                            self.first_token = False
                    yield chunk
                self.complete_time = time.perf_counter()
                _ollama_call_durations.append({
                    "start": call_start,
                    "first_token": self.first_token_time,
                    "complete": self.complete_time,
                })
        stream_iter = original_ollama_chat(*args, **kwargs)
        return StreamWrapper(stream_iter)
    else:
        result = original_ollama_chat(*args, **kwargs)
        call_end = time.perf_counter()
        _ollama_call_durations.append({
            "start": call_start,
            "first_token": call_end,
            "complete": call_end,
        })
        return result

ollama_module.chat = patched_chat

original_process_mentor_turn = process_mentor_turn

def instrumented_process_mentor_turn(user_message, username="User", project_name="MyProject", model_name="optimized-pods"):
    global _timestamps, _ollama_call_durations
    _timestamps = {}
    _ollama_call_durations = []
    _timestamps["T1_request_received"] = time.perf_counter()
    result = original_process_mentor_turn(user_message, username, project_name, model_name)
    _timestamps["T8_response_sent"] = time.perf_counter()
    return result, _timestamps, _ollama_call_durations

USER = "latency_user"
PROJECT = "latency_project"
TEST_MESSAGE = "We need a system to help elderly people remember their medication schedules"
MODEL = "optimized-pods"

def clear_sessions():
    from session_manager import get_session_manager
    try:
        mgr = get_session_manager()
        mgr.reset_runtime_state(username=USER, project_title=PROJECT)
    except Exception:
        pass

def run_measurement(num_runs=10, warmup_runs=3):
    print(f"Running {warmup_runs} warmup turns...")
    for i in range(warmup_runs):
        clear_sessions()
        try:
            result, ts, calls = instrumented_process_mentor_turn(TEST_MESSAGE, USER, PROJECT, MODEL)
            total_ms = (ts.get("T8_response_sent", 0) - ts.get("T1_request_received", 0)) * 1000
            print(f"  Warmup {i+1}: {total_ms:.0f} ms  ({result[0][:50]}...)")
        except Exception as e:
            print(f"  Warmup {i+1}: FAILED - {e}")
            traceback.print_exc()
        clear_sessions()

    print(f"\nRunning {num_runs} measured turns...")
    all_results = []
    for i in range(num_runs):
        clear_sessions()
        try:
            result, ts, calls = instrumented_process_mentor_turn(TEST_MESSAGE, USER, PROJECT, MODEL)
            all_results.append((ts, calls))
            total_ms = (ts.get("T8_response_sent", 0) - ts.get("T1_request_received", 0)) * 1000
            total_llm_ms = sum((c["complete"] - c["start"]) * 1000 for c in calls) if calls else 0
            py_ms = total_ms - total_llm_ms
            call_details = " + ".join(f"LLM{j+1}={(c['complete']-c['start'])*1000:.0f}ms" for j, c in enumerate(calls))
            print(f"  Run {i+1}: Total={total_ms:.0f}ms  Python={py_ms:.0f}ms  ({call_details})")
        except Exception as e:
            print(f"  Run {i+1}: FAILED - {e}")
            traceback.print_exc()
        clear_sessions()

    return all_results

def compute_statistics(all_results):
    if not all_results:
        return None

    runs = []
    for ts, calls in all_results:
        t1 = ts.get("T1_request_received", 0)
        t8 = ts.get("T8_response_sent", 0)

        if t1 > 0 and t8 > 0:
            run = {}
            run["total_ms"] = (t8 - t1) * 1000
            run["llm_total_ms"] = sum((c["complete"] - c["start"]) * 1000 for c in calls) if calls else 0
            run["num_llm_calls"] = len(calls) if calls else 0
            run["python_ms"] = run["total_ms"] - run["llm_total_ms"]
            run["python_pct"] = (run["python_ms"] / run["total_ms"] * 100) if run["total_ms"] > 0 else 0
            run["llm_pct"] = (run["llm_total_ms"] / run["total_ms"] * 100) if run["total_ms"] > 0 else 0

            # Individual LLM call stats
            for j, c in enumerate(calls):
                run[f"llm_call_{j+1}_ms"] = (c["complete"] - c["start"]) * 1000
                if c["first_token"]:
                    run[f"llm_call_{j+1}_ttft_ms"] = (c["first_token"] - c["start"]) * 1000

            runs.append(run)

    if not runs:
        return None

    keys = list(runs[0].keys())
    stats = {}
    for key in keys:
        values = [r[key] for r in runs if key in r]
        if values:
            stats[key] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }

    return stats

def save_markdown_report(stats):
    if not stats:
        return "No data to report"

    lines = []
    lines.append("# Latency Report - ReqGPT Mentor Pipeline")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Successful runs:** {stats.get('total_ms', {}).get('count', 0)}")
    lines.append(f"**Test message:** \"{TEST_MESSAGE}\"")
    lines.append(f"**Model:** {MODEL}")
    lines.append("")

    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Mean | Median | Min | Max | StdDev |")
    lines.append("|--------|------|--------|-----|-----|--------|")

    key_metrics = [
        ("total_ms", "Total Response Time"),
        ("llm_total_ms", "Total LLM (all calls)"),
        ("python_ms", "Python Pipeline (total - LLM)"),
        ("python_pct", "Python %"),
        ("llm_pct", "LLM %"),
    ]

    for key, label in key_metrics:
        if key in stats:
            s = stats[key]
            unit = "%" if "pct" in key else "ms"
            lines.append(f"| {label} | {s['mean']:.1f} {unit} | {s['median']:.1f} {unit} | {s['min']:.1f} {unit} | {s['max']:.1f} {unit} | {s['stdev']:.1f} {unit} |")

    # Per-LLM-call breakdown
    lines.append("")
    lines.append("## Per-Call LLM Breakdown")
    lines.append("")
    max_calls = 0
    for key in stats:
        if key.startswith("llm_call_") and key.endswith("_ms") and not "ttft" in key:
            call_num = int(key.split("_")[2])
            max_calls = max(max_calls, call_num)

    if max_calls > 0:
        lines.append("| LLM Call | Mean | Min | Max | Count |")
        lines.append("|----------|------|-----|-----|-------|")
        for j in range(1, max_calls + 1):
            key = f"llm_call_{j}_ms"
            if key in stats:
                s = stats[key]
                lines.append(f"| {key} | {s['mean']:.0f} ms | {s['min']:.0f} ms | {s['max']:.0f} ms | {s['count']} |")
    else:
        lines.append("(No per-call stats available)")

    lines.append("")
    lines.append("## Analysis")
    lines.append("")

    if "total_ms" in stats:
        avg_total = stats["total_ms"]["mean"]
        avg_python = stats.get("python_ms", {"mean": 0})["mean"]
        avg_llm = stats.get("llm_total_ms", {"mean": 0})["mean"]
        avg_pct_py = stats.get("python_pct", {"mean": 0})["mean"]
        avg_pct_ol = stats.get("llm_pct", {"mean": 0})["mean"]
        avg_calls = stats.get("num_llm_calls", {"mean": 0})["mean"]

        lines.append(f"- **Average total response time:** {avg_total:.0f} ms")
        lines.append(f"- **Average LLM calls per turn:** {avg_calls:.1f}")
        lines.append(f"- **Total LLM time:** {avg_llm:.0f} ms ({avg_pct_ol:.1f}%)")
        lines.append(f"- **Python pipeline:** {avg_python:.0f} ms ({avg_pct_py:.1f}%)")
        lines.append("")

        if avg_llm > avg_python and avg_python > 0:
            ratio = avg_llm / avg_python
            lines.append(f"> **LLM is the dominant latency source** ({ratio:.1f}x Python time).")
        elif avg_python > avg_llm and avg_llm > 0:
            ratio = avg_python / avg_llm
            lines.append(f"> **Python pipeline is the dominant latency source** ({ratio:.1f}x LLM time).")
        else:
            lines.append("> Latency is balanced between Python and LLM.")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Measurements taken with Ollama running locally (optimized-pods model, based on qwen2.5:3b)")
    lines.append("- All ollama.chat calls are timed individually and summed for total LLM time")
    lines.append("- Python pipeline includes session ops, memory extraction, state/objective/lifecycle/strategy engines, prompt builder, summary builder, and enforce_mentor_reply")
    lines.append("- Single-threaded CPU inference (llama.cpp via Ollama)")
    lines.append("- Session state cleared between each run (simulates first-turn latency)")
    lines.append(f"- Model: {MODEL}")

    report = "\n".join(lines)

    with open("Latency_Report.md", "w", encoding="utf-8") as f:
        f.write(report)

    return report

if __name__ == "__main__":
    print("=" * 70)
    print("ReqGPT End-to-End Latency Measurement")
    print("=" * 70)

    try:
        results = run_measurement(num_runs=5, warmup_runs=2)
        if results:
            stats = compute_statistics(results)
            report = save_markdown_report(stats)
            print("\n" + report)
            print(f"\n== Report saved to Latency_Report.md ==")
        else:
            print("\n== No successful runs ==")
            sys.exit(1)
    finally:
        ollama_module.chat = original_ollama_chat
        print("== Monkey patches removed ==")
