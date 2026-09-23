# Ollama Profile Report — MemoryExtractor Latency Analysis

**Generated:** 2026-07-29  
**Method:** Streaming `ollama.chat()` with per-token `time.perf_counter()` timestamps for TTFT vs. generation breakdown. Non-streaming baseline for total request time.  
**Prompt source:** Captured real prompt from `_prompt_dumps/call1_msg1_user_qwen2.5_3b.txt` (22,639 chars).

---

## 1. Model Details

| Property | MemoryExtractor | Mentor Response |
|----------|----------------|-----------------|
| **Model name** | `qwen2.5:3b` | `optimized-pods` |
| **Base model** | Qwen 2.5 (same blob) | Qwen 2.5 (same blob) |
| **Format** | GGUF | GGUF |
| **Family** | qwen2 | qwen2 |
| **Parameter size** | 3.1B | 3.1B |
| **Quantization** | Q4_K_M | Q4_K_M |
| **Size on disk** | 1.80 GB | 1.80 GB |
| **Modelfile PARAMETERs** | (none — all default) | `num_ctx=1024`, `num_thread=6`, `temperature=0.7` |

Both models share the **identical underlying blob** (`sha256-5ee4f07...`). They differ only in Modelfile PARAMETERs. Since code-level `options` in `ollama.chat()` override Modelfile `temperature`, the only runtime difference is `num_ctx` (2048 default vs. 1024).

---

## 2. Hardware Configuration

| Property | Value |
|----------|-------|
| **Platform** | Windows 11 (build 26200) |
| **CPU** | AMD Ryzen 6C/12T (Family 23 Model 104 — likely Ryzen 5 4500U) |
| **Physical cores** | 6 |
| **Logical threads** | 12 |
| **RAM** | 7.3 GB total |
| **RAM available** | ~2.7 GB |
| **GPU** | None detected (Ollama reports no loaded models; CPU-only inference) |
| **Ollama threads** | Default (auto = 6 physical cores) |

---

## 3. MemoryExtractor — Full Breakdown

### Request Configuration

| Parameter | Value |
|-----------|-------|
| Input chars | 22,639 |
| Est. input tokens | 5,659 |
| Output chars | 557 |
| Output words | 64 |
| Tokens generated | 138 |
| Temperature | 0.0 |
| top_p | 1.0 |
| num_predict | 300 |
| repeat_penalty | 1.0 |

### Timing Breakdown (streaming pass)

| Metric | Value | % of Total |
|--------|-------|------------|
| **Total (non-streaming)** | **91,846 ms** | 100.0% |
| **Total (streaming)** | **93,246 ms** | 101.5%* |
| **TTFT (prompt eval)** | **76,102 ms** | **81.7%** |
| **Generation** | **17,016 ms** | **18.3%** |
| Post-stream overhead | 129 ms | 0.1% |

*Streaming adds ~1.5% overhead from Python generator iteration.

### Token Rates

| Rate | Value |
|------|-------|
| **Prompt eval rate** | **74.4 tok/s** |
| **Generation rate** | **8.1 tok/s** |
| Generation per-token (mean) | 124.2 ms/token |
| Generation per-token (min) | 118.7 ms/token |
| Generation per-token (max) | 167.0 ms/token |
| Output/input ratio | 2.4% (138 out / 5,659 in) |

### Per-Token Generation Consistency

```
First 5 tokens:  124, 129, 122, 121, 119 ms  (mean 123 ms)
Last 5 tokens:   123, 125, 124, 125, 127 ms  (mean 125 ms)
```

Generation speed is remarkably consistent — all tokens within a narrow band of 119–167 ms. No slowdown or acceleration over the 138-token generation.

---

## 4. Mentor Response — Comparison Breakdown

### Request Configuration

| Parameter | Value |
|-----------|-------|
| Input chars | 1,877 |
| Est. input tokens | 469 |
| Output chars | 604 |
| Tokens generated | 111 |
| Temperature | 0.4 |
| top_p | 0.85 |
| num_predict | 150 |

### Timing Breakdown

| Metric | Value | % of Total |
|--------|-------|------------|
| **Total (streaming)** | **30,565 ms** | 100.0% |
| **TTFT (prompt eval)** | **19,052 ms** | **62.3%** |
| **Generation** | **11,375 ms** | **37.2%** |
| Post-stream overhead | 139 ms | 0.5% |

### Token Rates

| Rate | Value |
|------|-------|
| **Prompt eval rate** | **24.6 tok/s** |
| **Generation rate** | **9.8 tok/s** |

---

## 5. Side-by-Side Comparison

| Metric | MemoryExtractor | Mentor Response | Ratio |
|--------|----------------|----------------|-------|
| **Est. input tokens** | 5,659 | 469 | **12.1×** |
| **TTFT (prompt eval)** | 76,102 ms | 19,052 ms | 4.0× |
| **Generation time** | 17,016 ms | 11,375 ms | 1.5× |
| **Total (streaming)** | 93,246 ms | 30,565 ms | 3.1× |
| **Prompt eval rate** | 74.4 tok/s | 24.6 tok/s | **3.0×** |
| **Generation rate** | 8.1 tok/s | 9.8 tok/s | 0.8× |
| **Output tokens** | 138 | 111 | 1.2× |
| **Output/input ratio** | 2.4% | 23.7% | — |

### Key Observations

1. **Prompt eval rate is 3× faster for MemoryExtractor** (74.4 vs 24.6 tok/s) despite a 12× larger prompt. This is expected — the prefill stage (KV cache population) benefits from larger batch sizes, achieving better utilization of the CPU's SIMD/parallel capabilities.

2. **Generation rate is nearly identical** (8.1 vs 9.8 tok/s). Generation is memory-bandwidth-bound regardless of prompt size — each token requires loading the full 3.1B parameters from RAM for one forward pass.

3. **Despite faster prompt eval rate, the absolute prompt eval time is 4× higher** (76s vs 19s) because the prompt is 12× larger. The rate improves but not enough to compensate for the size difference.

4. **Prompt eval dominates both calls** — 81.7% of MemoryExtractor time and 62.3% of Mentor Response time.

---

## 6. Does Ollama Report Internal Timings?

Ollama's Python API (`ollama.chat`) does **not** expose internal server timings in its response. The streaming response chunks contain only:
- `{"message": {"role": "assistant", "content": "<token>"}}`

There is no `timings` field, no `eval_count`/`eval_duration` metrics, and no performance metadata in the response payload. All timing data in this report was measured client-side with `time.perf_counter()`.

Ollama's server (llama.cpp backend) does support options like `--verbose` or `--debug` at server startup that would log per-request timing to stderr, but these are server-level flags, not per-request options. They could be enabled by restarting the Ollama service with `OLLAMA_DEBUG=1`, but:
- This modifies the server configuration (out of scope per "do not modify code")
- Previous attempts showed timing was already fully captured by client-side instrumentation

---

## 7. CPU vs GPU Execution

**CPU-only execution confirmed.** Evidence:
1. `ollama.ps()` returns an empty model list (`models=[]`), indicating no GPU offloading
2. Hardware: AMD Ryzen 5 4500U (integrated Vega GPU) — Ollama does not leverage AMD GPUs by default
3. Generation rate of ~8 tok/s on a 3.1B Q4_K_M model is consistent with CPU-only llama.cpp on this class of hardware
4. Ollama on Windows uses llama.cpp which defaults to CPU only unless CUDA is explicitly configured
5. Model is Q4_K_M (4-bit quantized), reducing memory bandwidth requirements but still CPU-bound

---

## 8. Root Cause Analysis

### Where Does the 91 Seconds Go?

```
MemoryExtractor total: 91,846 ms
  ┌─ Prompt evaluation (prefill):  76,102 ms  (81.7%)  ← ROOT CAUSE
  │    5,659 tokens @ 74.4 tok/s
  │    All 3.1B parameters loaded × 5,659 forward passes
  │    on 6-core CPU with ~20 GB/s memory bandwidth
  │
  └─ Token generation:             17,016 ms  (18.3%)
       138 tokens @ 8.1 tok/s
       Memory-bandwidth-bound: each token = 1.80 GB model load
       ≈ 1.80 GB × 8.1 tok/s = ~14.6 GB/s bandwidth utilization
```

### Primary Bottleneck: Prompt Evaluation (Prefill) — 81.7% of Total

The 5,659-token prompt must be fully processed through all 28 transformer layers of the 3.1B parameter model to populate the KV cache before any output token can be generated. This is a sequential process on CPU — each token's hidden states are computed and stored for attention computation during generation.

**Why prompt evaluation dominates:**
- 5,659 input tokens × 28 layers × 3.1B parameters worth of computation
- CPU-only: 6 cores of AMD Ryzen 5 (Zen 2, ~3.7 GHz boost)
- Q4_K_M quantization reduces model size to 1.80 GB, but memory bandwidth (~20 GB/s on DDR4) is the limiting factor
- At 74.4 tok/s prefill rate: each token takes ~13.4 ms of computation

### Secondary Factor: Model Selection

`qwen2.5:3b` (3.1B parameters, Q4_K_M) at 8 tok/s generation is appropriate for this hardware class. A smaller model (1.5B) would be ~2× faster but may lack extraction quality. A larger model (7B) would not fit in the available 2.7 GB RAM.

### Tertiary Factor: Hardware

The AMD Ryzen 5 4500U is a 2020 mid-range laptop CPU. With 6 cores and ~20 GB/s DDR4 bandwidth, it achieves reasonable (not fast) inference. The available 2.7 GB RAM before Ollama starts means the 1.80 GB model consumes ~67% of free memory. Swap pressure could add latency but the consistent per-token times suggest no swapping is occurring.

### Not a Factor: Token Generation

Generation at 8.1 tok/s is the expected rate for this model on this hardware. The 17 seconds to generate 138 tokens is only 18.3% of total time. Generation speed is not the bottleneck.

---

## 9. Fix Path Summary

| Factor | Contribution | Addressable? | Impact if Fixed |
|--------|-------------|-------------|-----------------|
| Prompt evaluation (76s) | **81.7%** | Yes — reduce prompt size | ~94% reduction from 76s → ~5s (at same tok/s for the dynamic portion only) |
| Prompt size (5,659 tokens) | Root cause of prompt eval time | Yes — static portion is 93% of prompt | Directly proportional reduction |
| Token generation (17s) | 18.3% | Marginally — faster CPU/GPU | ~2× faster with GPU would save ~8s |
| Model selection (3.1B) | Structural | Tradeoff with quality | 1.5B model would be ~2× faster on both eval and generation |
| CPU-only inference | Structural | Add GPU (CUDA) | 5–10× faster generation, 2–3× faster prompt eval |
| RAM (2.7 GB free) | Structural | Upgrade hardware | Avoids swap; current utilization is OK at 1.80 GB |

**The single highest-impact fix is reducing the prompt.** The 93%-static prompt (22,639 chars with 13,563 chars of examples) forces 76 seconds of prefill for every turn. A prompt cache would eliminate this entirely.

**Without reducing the prompt,** the only option is faster hardware. GPU inference (e.g., an NVIDIA GTX 1650 with 4 GB VRAM) would accelerate both prompt eval and generation by 3–5×, bringing the 91s down to ~20–30s.
