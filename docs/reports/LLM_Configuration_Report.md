# LLM Configuration Report — Mentor Pipeline

**Generated:** 2026-07-29

## Scope

All `ollama.chat()` call sites in the mentor pipeline (`process_mentor_turn`).
Non-mentor calls in `server.py` (`generate_llm_response`, `generate_mermaid_diagram`, `generate_requirements`) and
standalone scripts (`use_ollama.py`) are excluded — they are downstream/different code paths.

---

## Call Site 1: MemoryExtractor

| Field | Value |
|-------|-------|
| **File** | `memory_extractor.py:1284` |
| **Component** | `MemoryExtractor._call_model()` — Module 1 semantic extraction |
| **Trigger** | Always, on every `process_mentor_turn()` call (line 1126) |
| **Messages** | `[{"role": "user", "content": prompt}]` — single user message, no system prompt |

### Model Resolution

```
resolve_extractor_model()          # memory_extractor.py:1328
  ├── os.getenv("EXTRACTOR_MODEL")
  ├── os.getenv("MENTOR_MODEL")
  └── default: "qwen2.5:3b"
```

### Generation Parameters — Code Overrides

| Parameter | Value | Source |
|-----------|-------|--------|
| `temperature` | **0.0** | `_GENERATION_OPTIONS` (line 1144) |
| `top_p` | **1.0** | `_GENERATION_OPTIONS` (line 1145) |
| `num_predict` | **300** | `_GENERATION_OPTIONS` (line 1146) |
| `repeat_penalty` | **1.0** | `_GENERATION_OPTIONS` (line 1147) |

### Generation Parameters — Ollama Defaults (not overridden)

| Parameter | Value |
|-----------|-------|
| `top_k` | 40 |
| `num_ctx` | 2048 |
| `seed` | 0 (random) |
| `stream` | false |
| `keep_alive` | 5m |

### Defaults Assessment

All 4 code-level parameters are **explicitly overridden** — none rely on Ollama defaults.
The 5 remaining parameters use Ollama defaults.

---

## Call Site 2: Main Mentor Response (LLM)

| Field | Value |
|-------|-------|
| **File** | `mentor.py:1354` |
| **Component** | `process_mentor_turn()` — Module 6 LLM communicator |
| **Trigger** | Always, on every `process_mentor_turn()` call (after prompt building) |
| **Messages** | `[{"role": "user", "content": prompt}]` — single user message (the built 5-section mentor prompt), no system prompt |

### Model Resolution

```
process_mentor_turn(model_name=...)    # mentor.py:1065
  └── When called from server.py:
        mentor_model_name()            # server.py:113
          ├── os.getenv("MENTOR_MODEL")
          └── default: "qwen2.5:3b"
      When called from CLI/test:
        passed directly (e.g. "optimized-pods")
```

### Generation Parameters — Code Overrides

| Parameter | Value | Source |
|-----------|-------|--------|
| `temperature` | **0.4** | Inline dict (line 1357) |
| `top_p` | **0.85** | Inline dict (line 1357) |
| `num_predict` | **150** | Inline dict (line 1357) |

### Generation Parameters — Ollama Defaults (not overridden)

| Parameter | Value |
|-----------|-------|
| `repeat_penalty` | 1.1 |
| `top_k` | 40 |
| `num_ctx` | 2048 |
| `seed` | 0 (random) |
| `stream` | false |
| `keep_alive` | 5m |

### Defaults Assessment

3 parameters explicitly overridden, 6 using Ollama defaults.

---

## Call Site 3: Legacy LLM Extraction (only when `MENTOR_USE_LLM_EXTRACTION=true`)

| Field | Value |
|-------|-------|
| **File** | `mentor.py:594` |
| **Component** | `InputProcessor._llm_extract()` — legacy rule-based extraction fallback |
| **Trigger** | Only when `MENTOR_USE_LLM_EXTRACTION=true` AND (this is not a MEANINGFUL-update turn) — see line 1194 guard (P3 optimization) |
| **Messages** | `[{"role": "user", "content": prompt}]` — single user message with JSON extraction instruction |

### Model Resolution

```
model_name from process_mentor_turn parameter  →  MENTOR_MODEL or "qwen2.5:3b"
```

### Generation Parameters — Code Overrides

| Parameter | Value | Source |
|-----------|-------|--------|
| `temperature` | **0.1** | Inline dict (line 597) |
| `num_predict` | **256** | Inline dict (line 597) |

### Defaults Assessment

2 parameters overridden, 7 using Ollama defaults.
Defaults match Call Site 1 (both are extraction-oriented).

---

## Side-by-Side Comparison (Mentor Pipeline Only)

| Parameter | Call 1: MemoryExtractor | Call 2: Main Response | Call 3: Legacy Extraction |
|-----------|------------------------|----------------------|--------------------------|
| **File** | `memory_extractor.py:1284` | `mentor.py:1354` | `mentor.py:594` |
| **Active by default?** | Yes | Yes | No (`MENTOR_USE_LLM_EXTRACTION=false`) |
| **Typical model** | `qwen2.5:3b` (via `resolve_extractor_model()`) | `qwen2.5:3b` (via `MENTOR_MODEL`) or `optimized-pods` (explicit) | Same as Call 2 |
| **temperature** | **0.0** | **0.4** | **0.1** |
| **top_p** | **1.0** | **0.85** | (default 0.9) |
| **num_predict** | **300** | **150** | **256** |
| **repeat_penalty** | **1.0** | (default 1.1) | (default 1.1) |
| **top_k** | (default 40) | (default 40) | (default 40) |
| **num_ctx** | (default 2048) | (default 2048) | (default 2048) |
| **seed** | (default 0) | (default 0) | (default 0) |
| **stream** | false | false | false |
| **keep_alive** | 5m | 5m | 5m |
| **Purpose** | Deterministic JSON extraction | Natural-language mentor reply | Deterministic JSON extraction |
| **Prompt style** | Structured JSON instruction | 5-section mentor prompt | JSON extraction instruction |

## Model Identity

Both `qwen2.5:3b` and `optimized-pods` are based on the **same underlying model blob** (sha256 `5ee4f07c...`).
The difference is in the Modelfile PARAMETERs baked into the model:

| Parameter | `qwen2.5:3b` (no Modelfile overrides) | `optimized-pods` (via `autotweak.py`) |
|-----------|--------------------------------------|--------------------------------------|
| `num_ctx` | 2048 (Ollama default) | **1024** |
| `num_thread` | auto | **6** |
| `temperature` | 0.8 (Ollama default) | **0.7** |

**However**, code-level `options` in the `ollama.chat()` call **override** these Modelfile values at runtime.
This means the Modelfile difference between `qwen2.5:3b` and `optimized-pods` is **superseded** by the
explicit `temperature`, `top_p`, and `num_predict` in each call site. The `num_ctx=1024` from the
Modelfile *does* take effect (since no call site overrides it), but `num_thread` is also effectively
overridden because both models run on the same hardware with Ollama's runtime thread management.

## Key Finding: Different Models for Different Calls

Yes — **different components can and do use different models**.

| Call | Typical Model | Resolution Chain |
|------|-------------|-----------------|
| MemoryExtractor | `qwen2.5:3b` | `EXTRACTOR_MODEL` → `MENTOR_MODEL` → `"qwen2.5:3b"` |
| Main Response | `qwen2.5:3b` (from server) | `MENTOR_MODEL` → `"qwen2.5:3b"` (or explicit override) |

When server.py launches the mentor route, both calls **resolve to the same `MENTOR_MODEL`** by default.
When called directly with an explicit `model_name="optimized-pods"`, only the main response changes —
the MemoryExtractor still uses its own resolution chain (`resolve_extractor_model()`).

This means the system can be configured to use **two different models simultaneously**:
- Set `EXTRACTOR_MODEL` to a lightweight model for extraction
- Set `MENTOR_MODEL` to a heavier model for response generation
- Or vice versa

## Latency Impact (from 5-run measurement)

| Call | Mean Duration | % of Total |
|------|--------------|------------|
| MemoryExtractor (`qwen2.5:3b`) | **90,870 ms** | 74.1% |
| Main Response (`optimized-pods`) | **31,734 ms** | 25.9% |
| Python pipeline | **15 ms** | 0.01% |
| **Total** | **122,619 ms** | 100% |
