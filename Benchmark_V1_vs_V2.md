# Benchmark: MemoryExtractor V1 vs V2

## Methodology

- **Model**: `qwen2.5:3b` via Ollama (CPU-only, Ryzen 5 4500U)
- **Test input** (identical every trial): *"Students keep missing assignment deadlines because they forget due dates, and professors complain about inconsistent submission tracking."*
- **3 trials per version**, each preceded by `taskkill /F /IM ollama.exe` + fresh `ollama serve` + readiness check
- **Measurements**: Raw HTTP POST to `http://localhost:11434/api/chat` with streaming; `time.perf_counter()` for client-side timing; server-side durations from the final `"done": true` chunk
- **No production code modified** — both prompts imported as constants from `memory_extractor.py`

---

## Results

### Warm-state averages (trials 2–3, after model fully loaded)

Cold-start trial 1 (includes model load into memory) is excluded from the primary comparison.

| Metric | V1 | V2 | Δ |
|---|---|---|---|
| Prompt size (chars) | 21,551 | 6,171 | **−71%** |
| Prompt tokens (actual, server-reported) | 2,050 | 1,697 | **−17%** |
| **Prompt eval time** (server, s) | **0.09** | **0.08** | **−11%** |
| TTFT (client, s) | 2.45 | 2.45 | 0% |
| **Generation time** (client, s) | **20.05** | **25.30** | **+26%** |
| **Total time** (client, s) | **22.55** | **27.75** | **+23%** |
| Generated tokens | 239 | 300 | **+26%** |
| Generation rate (tok/s) | 11.9 | 11.9 | 0% |
| Prompt eval rate (tok/s) | 22,778 | 21,213 | −7% |

### All trials (including cold start)

| Metric | V1 | V2 | Δ |
|---|---|---|---|
| TTFT (client, s) | 14.33 | 15.47 | +8% |
| Prompt eval time (server, s) | 9.77 | 11.02 | +13% |
| Generation time (client, s) | 19.92 | 25.67 | +29% |
| Total time (client, s) | 34.34 | 41.14 | +20% |
| Total time stddev (s) | 20.40 | 23.24 | — |

Cold-start adds ~30–33 s of model-loading time to the first trial. This inflates means and standard deviations significantly; warm-state data (trials 2–3) is the reliable comparison.

---

## Answers

### 1. Did prompt evaluation decrease?

**No, not meaningfully in wall-clock time.**

The V2 prompt uses 1,697 tokens vs V1's 2,050 — a 17% reduction. However, the server-side prompt evaluation time is identical in warm state: **0.09 s vs 0.08 s**. Prompt evaluation is negligible (~0.4% of total latency), so token-count reductions at this scale are invisible to the user.

The 71% character reduction translated to only 17% token reduction because V1's verbose instruction text tokenizes very efficiently (~10.5 chars/token) while V2's denser examples with JSON blobs tokenize more compactly (~3.6 chars/token). The char-to-token estimation was misleading.

### 2. Did generation time change?

**Yes — it increased by 26%, exactly matching the increase in generated tokens.**

| | V1 | V2 |
|---|---|---|
| Generated tokens | 239 | 300 (+26%) |
| Generation time (warm, s) | 20.05 | 25.30 (+26%) |
| Generation rate (tok/s) | 11.9 | 11.9 |

The generation rate is unchanged at ~11.9 tok/s. V2 simply produces more output tokens (more extracted facts), so it takes proportionally longer.

### 3. Did total latency decrease?

**No — total latency increased by ~23% in warm state** (22.55 s → 27.75 s), driven entirely by the additional generation time for V2's richer output.

The 0.01 s saved in prompt evaluation is dwarfed by the 5.25 s extra generation time.

### 4. Were measurements statistically consistent?

**Generation time is highly consistent; TTFT has high variance due to cold start.**

- Generation time stddev: **0.55 s (V1) / 0.68 s (V2)** — very tight
- TTFT stddev: **20.6 s (V1) / 22.6 s (V2)** — dominated by the 30+ s cold-start trial
- In warm state, TTFT is consistent at 2.4–2.5 s per trial

The 3-trial design with one cold-start and two warm trials is sufficient to separate load effects from steady-state performance, but 3 warm trials per version would improve confidence.

### 5. Explain any unexpected observations.

1. **71% char reduction → only 17% token reduction.** The char-based token estimation assumed 4 chars/token for both prompts. In reality, V1's verbose prose tokenizes very efficiently while V2's structured JSON examples are denser. Token count, not char count, is the relevant metric for prompt eval time.

2. **Prompt evaluation is essentially free (~0.09 s).** On this platform, the 3B-parameter model evaluates prompts at >20,000 tok/s. The 353-token difference between V1 and V2 is processed in ~15 ms — indistinguishable from noise.

3. **Generation rate is cache-rate limited.** At ~12 tok/s, the CPU-bound qwen2.5:3b generates tokens at roughly the same speed regardless of prompt content. V2's extra 61 tokens cost ~5 s because the hardware has a fixed throughput ceiling.

4. **V2 produces 26% more output tokens** (300 vs 239). This is consistent with the regression finding that V2 extracts 75% more facts — more facts ≈ more tokens ≈ longer generation time.

---

## Conclusion

| Claim | Result |
|---|---|
| Smaller prompt → faster prompt eval | **False.** Prompt eval is already ~0.09 s; 17% token savings are invisible. |
| V2 is faster overall | **False.** V2 is ~23% slower in warm state, entirely due to more output tokens. |
| V2 extracts more facts per second | **True.** 75% more facts for 23% more time = ~60% better fact throughput. |
| Prompt size reduction was meaningful | **Partially.** 71% fewer chars but only 17% fewer tokens — the compact formatting oversold the savings. |

**Bottom line**: V2 achieves better extraction quality at the cost of proportionally more generation time. The V1→V2 swap is not a latency win; it's an accuracy-for-latency tradeoff. If latency is critical, the prompt could be shortened further by trimming output token budget rather than input token budget.
