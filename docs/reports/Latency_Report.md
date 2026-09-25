# Latency Report - ReqGPT Mentor Pipeline

**Generated:** 2026-07-29 16:13:15
**Successful runs:** 5
**Test message:** "We need a system to help elderly people remember their medication schedules"
**Model:** optimized-pods

## Summary Metrics

| Metric | Mean | Median | Min | Max | StdDev |
|--------|------|--------|-----|-----|--------|
| Total Response Time | 122618.7 ms | 122601.0 ms | 121111.9 ms | 124268.6 ms | 1470.3 ms |
| Total LLM (all calls) | 122604.1 ms | 122588.6 ms | 121099.8 ms | 124246.3 ms | 1468.0 ms |
| Python Pipeline (total - LLM) | 14.6 ms | 12.3 ms | 12.0 ms | 22.3 ms | 4.4 ms |
| Python % | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.0 % |
| LLM % | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 0.0 % |

## Per-Call LLM Breakdown

| LLM Call | Mean | Min | Max | Count |
|----------|------|-----|-----|-------|
| llm_call_1_ms | 90870 ms | 90413 ms | 91778 ms | 5 |
| llm_call_2_ms | 31734 ms | 29322 ms | 33358 ms | 5 |

## Analysis

- **Average total response time:** 122619 ms
- **Average LLM calls per turn:** 2.0
- **Total LLM time:** 122604 ms (100.0%)
- **Python pipeline:** 15 ms (0.0%)

> **LLM is the dominant latency source** (8375.1x Python time).

## Notes

- Measurements taken with Ollama running locally (optimized-pods model, based on qwen2.5:3b)
- All ollama.chat calls are timed individually and summed for total LLM time
- Python pipeline includes session ops, memory extraction, state/objective/lifecycle/strategy engines, prompt builder, summary builder, and enforce_mentor_reply
- Single-threaded CPU inference (llama.cpp via Ollama)
- Session state cleared between each run (simulates first-turn latency)
- Model: optimized-pods