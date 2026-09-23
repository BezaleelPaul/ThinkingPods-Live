# Optimization Roadmap — ReqGPT / ThinkingPods

Generated: 2026-07-28
Benchmark baseline: `Performance_Report.md` (453 ms/turn, no Ollama)

---

## HIGH IMPACT

### H1 — Eliminate Dual Session Save (I/O Reduction)

| Field | Detail |
|-------|--------|
| **Current behavior** | `SessionManager.save()` in `mentor.py` writes BOTH to the new `SessionManager` (canonical `ProjectState`) AND the legacy `{user}_{project}_mentor.json` file on EVERY mentor turn. Two complete serialization + I/O rounds. |
| **Why it is inefficient** | I/O already consumes **79.7%** of pipeline time (Performance_Report.md, ~360ms of 453ms). Dual save means ~160ms extra for the legacy file that is only read by the `/session/status` dashboard endpoint and by legacy CLI tools. Every turn incurs this cost even though most clients use the new path. |
| **Expected improvement** | Switch legacy save to write-on-close or lazy flush (every N turns or on stage transition). The new `SessionManager` already persists canonical state; the legacy file is a backward-compat artifact. |
| **Implementation complexity** | Low (2 days). Change: condition legacy save on a dirty flag, flush only when `empathize_summary_presented` changes or stage transitions. |
| **Estimated speed improvement** | **~30–35%** reduction in per-turn wall time (≈140 ms saved). The dual `json.dump` + file write is the dominant I/O cost. |
| **Risk of regression** | Medium. If the process crashes between lazy-flush intervals, the legacy file lags behind canonical state. CLI tools reading `MentorSession` directly would see stale data. Mitigation: flush on stage transition + every 5 turns. |
| **Files affected** | `mentor.py` (`SessionManager.save`, `process_mentor_turn`) |

---

### H2 — Cache the Extraction System Prompt (Prompt Construction)

| Field | Detail |
|-------|--------|
| **Current behavior** | `memory_extractor.py`'s `_build_prompt()` method constructs the ~600-line system prompt template from scratch every turn using string concatenation. The template is static (same role description, field definitions, JSON schema, examples each turn). The 600-line prompt accounts for **96.8%** of the extraction prompt (Prompt_Size_Report.md: 22,143 chars, ~5,535 tokens). |
| **Why it is inefficient** | String concatenation of 600 lines on every turn wastes CPU and GC pressure. The prompt template never changes between turns — only the user message + current state differ. Every call re-builds the identical template header, field descriptions, and JSON schema. |
| **Expected improvement** | Cache the static template portion as a module-level compiled string. Only interpolate the dynamic parts (user message, current state, previous assistant message). |
| **Implementation complexity** | Low (1 day). Extract the static template to a module-level `_EXTRACTION_SYSTEM_PROMPT` constant, build once at import time. |
| **Estimated speed improvement** | **~5–8%** per-turn reduction (≈20–35 ms). Mostly GC and string-allocation savings. The LLM call itself dwarfs prompt construction, but this is free to implement. |
| **Risk of regression** | Very low. No behavior change — same template string, same interpolation logic. |
| **Files affected** | `memory_extractor.py` (`_build_prompt` or equivalent) |

---

### H3 — Skip RuleBasedExtractor When MemoryExtractor Is Live

| Field | Detail |
|-------|--------|
| **Current behavior** | `InputProcessor.extract_structured_data()` in `mentor.py` ALWAYS runs `RuleBasedExtractor.extract()` (regex matching over 30+ patterns) before the LLM-based extraction path, even when `MENTOR_USE_LLM_EXTRACTION=true` and the LLM path will produce strictly superior extractions. |
| **Why it is inefficient** | Regex scanning over the full user message (30+ patterns across 7 categories) is wasted CPU when the LLM extraction will run immediately after. The rule-based results are also mostly discarded when LLM extraction succeeds (merged via `_merge_extractions()` which prefers LLM values). |
| **Expected improvement** | Gate rule-based extraction to only run when `MENTOR_USE_LLM_EXTRACTION != true`. When LLM extraction is on, skip regex entirely. |
| **Implementation complexity** | Low (1 day). Add an early return in `extract_structured_data()` when `use_llm=True`. |
| **Estimated speed improvement** | **~2–5%** per-turn reduction (≈10–20 ms). Depends on message length and pattern complexity. |
| **Risk of regression** | Low. With `MENTOR_USE_LLM_EXTRACTION=true`, the LLM always produces richer extractions. The rule-based results were only used as fallback values that LLM overrides anyway. For the default `false` path, behavior is unchanged. |
| **Files affected** | `mentor.py` (`InputProcessor.extract_structured_data`) |

---

### H4 — Cache KnowledgeBase Instance (Singleton)

| Field | Detail |
|-------|--------|
| **Current behavior** | `get_knowledge_base()` in `knowledge_base.py` (line 653) constructs a new `KnowledgeBase` instance every time it is called. It is called separately by `inference_engine.py` (line 218: `self._kb = get_knowledge_base()`) and `hypothesis_manager.py` (line 211: `self._kb = get_knowledge_base()`). Two instances are created, each loading the same 6-domain concept definitions. |
| **Why it is inefficient** | The `KnowledgeBase` constructor iterates all 6 domains and registers their concept entries. Doing this twice doubles memory and initialization time. The knowledge base is read-only and thread-safe — a single shared instance suffices. |
| **Expected improvement** | Make `get_knowledge_base()` return a module-level singleton created at import time. Both consumers share the same instance. |
| **Implementation complexity** | Very low (0.5 day). Replace the function with a module-level `_KB = KnowledgeBase()` and `get_knowledge_base = lambda: _KB` pattern. |
| **Estimated speed improvement** | **~1–3%** per-startup, plus ~200 KB memory reduction. Negligible per-turn impact since both instances are created at process start, not per turn. |
| **Risk of regression** | Very low. The knowledge base is immutable and read-only — sharing is safe by design. |
| **Files affected** | `knowledge_base.py` (`get_knowledge_base`), `inference_engine.py`, `hypothesis_manager.py` |

---

## MEDIUM IMPACT

### M1 — Lazy JSON Serialization in Session Persistence

| Field | Detail |
|-------|--------|
| **Current behavior** | Each of the 14 scalar fields + 6 list fields in `MentorSession.to_dict()` is eagerly serialized into a flat dict, then `json.dump()`d to disk. The legacy file is re-written entirely every turn — all 20+ fields, even when only 1 field changed. |
| **Why it is inefficient** | Full serialization of the entire session state every turn is wasteful when <10% of fields typically change. The dict-building loop runs 20+ `getattr()` calls, then `json.dumps()` the whole thing. |
| **Expected improvement** | Track which fields changed via a dirty set. Only serialize/ persist changed fields for the legacy file. The canonical `SessionManager` path already handles partial updates. |
| **Implementation complexity** | Medium (3 days). Add `_dirty_fields: set[str]` to `MentorSession`. Mark fields dirty in `setattr` overrides or via explicit setters. Serialize only dirty fields for legacy save. |
| **Estimated speed improvement** | **~10–15%** I/O reduction (≈35–55 ms). Most turns only change 1–2 fields, so serializing 2 fields instead of 20 dominates savings. |
| **Risk of regression** | Medium. If `_dirty_fields` is not cleared correctly after save, stale data persists. If a field is modified through raw `__dict__` access (not through tracked setters), the dirty flag is missed. Mitigation: full flush every 5 turns. |
| **Files affected** | `mentor.py` (`MentorSession`, `SessionManager.save`) |

---

### M2 — Build Summary Prompt Once, Cache Per Lifecycle State

| Field | Detail |
|-------|--------|
| **Current behavior** | The `GENERATE_SUMMARY` prompt re-renders the entire project state section every time the LLM is called during summary mode. The `PromptBuilder.build_prompt()` constructs the full 5-section prompt from scratch each turn. |
| **Why it is inefficient** | Once the summary is built and presented, the prompt state ("Empathize Summary" section) does not change until a new summary is generated (only on new extraction). The LLM call during `WAITING_FOR_CONFIRMATION` re-renders the identical summary section. |
| **Expected improvement** | Cache the rendered summary section string when the lifecycle state transitions to `READY_FOR_SUMMARY`. Reuse the cached string for subsequent `WAITING_FOR_CONFIRMATION` turns. Invalidate only when new extractions arrive. |
| **Implementation complexity** | Medium (3 days). Add a `_cached_summary_section: str | None` to the prompt builder. Set on GENERATE_SUMMARY prompt, clear on state change. |
| **Estimated speed improvement** | **~3–5%** per-turn during summary/confirmation phase (≈15–20 ms saved on prompt construction). |
| **Risk of regression** | Low. The summary is immutable once built — re-rendering produces the same string. Caching is safe. |
| **Files affected** | `module4/prompt_builder.py` (`PromptBuilder`, `_render_summary_section`) |

---

### M3 — Lazy-Load Ollama Client in server.py

| Field | Detail |
|-------|--------|
| **Current behavior** | `load_models()` in `server.py` runs in a background daemon thread that eagerly calls `ollama.show()`, `WhisperModel()`, and `torch.hub.load()` at startup. All three models load sequentially, blocking the `200 OK` health response until all finish (or fail). The TTS model (`silero_tts`) loads via `torch.hub.load()` which downloads from GitHub on first run. |
| **Why it is inefficient** | Non-blocking but still wasteful: TTS is not needed for text-only interactions and STT is not needed for text-only interactions. The thread blocks the model-loading order: if Ollama is slow or offline, TTS and STT still wait for it to time out. |
| **Expected improvement** | Load models lazily on first use, not at startup. STT loads on first `/voice` call. TTS loads on first request with `enable_voice=true`. Ollama stays eager (health endpoint needs to report readiness). |
| **Implementation complexity** | Medium (3 days). Replace the `load_models()` thread with per-model lazy-loading wrappers that check `MODELS["stt"] is None` and load on demand. Add a `loading` flag to prevent concurrent duplicate loads. |
| **Estimated speed improvement** | **N/A for per-turn** — this is a startup speed improvement. Reduces startup time from ~15–30s (3 models) to ~2–5s (Ollama only). |
| **Risk of regression** | Medium. First `/voice` request would be slow (~5s) while STT/TTS load. The background thread model made the first request fast at the cost of slower startup. Tradeoff depends on user preference. |
| **Files affected** | `server.py` (`load_models`, `/voice`, `/text` endpoints) |

---

### M4 — Avoid WAV Round-Trip for STT

| Field | Detail |
|-------|--------|
| **Current behavior** | The `/voice` endpoint in `server.py` writes incoming audio bytes to a temp WAV file (`temp_{username}_{uuid}_voice.wav`), calls `faster_whisper`'s `transcribe()` on the file path, then deletes the file in a `finally` block. |
| **Why it is inefficient** | Disk I/O for every voice turn: write ~100 KB, read ~100 KB, delete. In-memory transcription is supported by faster-whisper via `transcribe(audio)` with a numpy array, but the current code passes a file path. |
| **Expected improvement** | Parse the WAV bytes directly into a numpy array using `scipy.io.wavfile.read(io.BytesIO(audio_bytes))` and pass the array to `transcribe()`. Eliminates the temp file entirely. |
| **Implementation complexity** | Very low (1 day). Replace file write with `scipy.io.wavfile.read(BytesIO)` inline. |
| **Estimated speed improvement** | **~5–10%** per voice turn (≈50–150 ms saved on disk I/O). The write/read/delete cycle costs ~100 ms on typical laptop SSDs. |
| **Risk of regression** | Very low. faster-whisper accepts both file paths and numpy arrays. The existing `scipy` import is already present. |
| **Files affected** | `server.py` (`/voice` endpoint) |

---

### M5 — Cache `ordered_candidates` Results

| Field | Detail |
|-------|--------|
| **Current behavior** | `PriorityEngine.decide()` calls `ordered_candidates()` each turn, which iterates all 6 `DEFAULT_RULES`, checks coverage via `report.is_field_complete()` for each, and sorts by priority. The same rules and same coverage pattern repeat across turns (e.g., once `personas` is populated, it stays populated). |
| **Why it is inefficient** | O(n log n) sorting of 6 items is negligible individually, but across thousands of turns the repeated enumeration and attribute access adds up. Additionally, `ordered_candidates` calls `report.is_field_complete()` which does a dict lookup per rule — 6 lookups per turn. |
| **Expected improvement** | Cache the candidate list and invalidate only when the set of missing fields changes. Since fields rarely change on every turn, the candidate list is stable for 80%+ of turns. |
| **Implementation complexity** | Medium (3 days). Add a `_cached_missing_fields: frozenset[StateField]` and `_cached_candidates: List[Rule]` to `PriorityEngine`. On `decide()`, compare current missing fields to cached — if identical, reuse cached candidate list. |
| **Estimated speed improvement** | **~1–2%** per-turn reduction. Minimal absolute savings (~2–5 µs), but measurable at scale (thousands of turns). |
| **Risk of regression** | Low. The cached data is purely derived from immutable inputs. Invalid cache → recompute. Deterministic — same state always produces same candidates. |
| **Files affected** | `module3/priority_engine.py` (`PriorityEngine.decide`) |

---

### M6 — Skip LifecycleManager for Non-Empathize Pods

| Field | Detail |
|-------|--------|
| **Current behavior** | `mentor.py`'s `process_mentor_turn()` instantiates and calls `LifecycleManager.decide()` on EVERY turn, even when the pod is not Empathize (e.g., "general", "Define", "Ideate"). The lifecycle manager is Empathize-specific and always returns `CONTINUE` for non-Empathize states. |
| **Why it is inefficient** | The lifecycle manager creates a `LifecycleDecision`, evaluates confirmation markers, and checks summary-presented flags — all work that is immediately discarded for non-Empathize pods. |
| **Expected improvement** | Gate `LifecycleManager.decide()` call behind a pod check. Only call it when `current_stage == "Empathize"` or `stage == "Empathize_Complete"`. |
| **Implementation complexity** | Very low (1 day). Add an `if` guard around the lifecycle call in `process_mentor_turn()`. |
| **Estimated speed improvement** | **~1–3%** for non-Empathize turns (≈5–10 ms saved). |
| **Risk of regression** | Very low. The lifecycle manager is a no-op for non-Empathize stages — skipping it entirely produces the same outcome. |
| **Files affected** | `mentor.py` (`process_mentor_turn`) |

---

## LOW IMPACT

### L1 — Inline `_FIELD_LABELS` and `_OBJECTIVE_LABELS` Dicts

| Field | Detail |
|-------|--------|
| **Current behavior** | `module4/prompt_builder.py` defines 4 separate label/section dicts as module-level constants: `_FIELD_LABELS` (6 entries), `_SUMMARY_FIELD_LABELS` (6), `_SUMMARY_FIELD_ORDER` (6), `_OBJECTIVE_LABELS` (7). Each dict key is looked up once per prompt build. |
| **Why it is inefficient** | Dict lookups on small (<10 entry) dicts cost ~50 ns each — negligible per turn. At 4 lookups × thousands of turns, the absolute saving is trivial. |
| **Expected improvement** | Replace with tuple-indexed lookups or match/case for the rare case (100k+ turns). Not worth implementing. |
| **Implementation complexity** | Very low (hours). |
| **Estimated speed improvement** | **< 0.1%**. Microseconds per turn. Not measurable in wall time. |
| **Risk of regression** | Very low. Cosmetic refactor only. |
| **Files affected** | `module4/prompt_builder.py` |

---

### L2 — Pre-Compile Regex Patterns in RuleBasedExtractor

| Field | Detail |
|-------|--------|
| **Current behavior** | `RuleBasedExtractor` class defines 7 `_*_PATTERNS` lists with raw regex strings. Each `_first_match()` call compiles the regex via `re.search(pattern, ...)` on every extraction, re-compiling the same regex every turn. |
| **Why it is inefficient** | `re.search()` with a string pattern re-compiles the regex each time. Python caches compiled patterns (up to 512), so in practice this is fast — but there are 30+ patterns that are re-compiled each turn. |
| **Expected improvement** | Pre-compile all patterns using `re.compile()` at module level or class attribute definition time. |
| **Implementation complexity** | Very low (hours). Change each `(pattern_str, handler)` tuple to `(re.compile(pattern_str), handler)`. |
| **Estimated speed improvement** | **< 0.5%**. Regex compilation is cached by Python's `re` module (LRU cache of 512). Most patterns are already cached after first call. |
| **Risk of regression** | Very low. Behavior-identical. |
| **Files affected** | `mentor.py` (`RuleBasedExtractor._*_PATTERNS`) |

---

### L3 — Use `__slots__` on ProjectState

| Field | Detail |
|-------|--------|
| **Current behavior** | `ProjectState` is a `@dataclass` with 6 list-type fields and 1 scalar field. Each instance has a full `__dict__`, consuming ~600+ bytes per instance. New instances are created during session conversion (`_legacy_session_to_project_state`). |
| **Why it is inefficient** | `__dict__` wastes ~40% memory overhead per instance. ProjectState is created/destroyed each turn during the legacy bridge conversion. |
| **Expected improvement** | Add `__slots__` to `ProjectState` to reduce memory footprint. Requires manual `__init__` instead of `@dataclass` auto-generation. |
| **Implementation complexity** | Low-Medium (2 days). Replace `@dataclass` with manual class + `__slots__`. Verify serialization paths still work. |
| **Estimated speed improvement** | **Minimal per-turn**. ~300 bytes saved per instance. Cumulative memory benefit if many sessions are active. |
| **Risk of regression** | Medium. `__slots__` breaks `@dataclass` features like `asdict()`, `astuple()`, and `field()` defaults. All serialization code currently relies on `dataclasses.asdict()` or direct attribute access. Manual serialization methods would be needed. |
| **Files affected** | `memory_extractor.py` (`ProjectState`) |

---

### L4 — Consolidate `strip_model_output` Duplicate

| Field | Detail |
|-------|--------|
| **Current behavior** | `strip_model_output()` is defined in `mentor.py` (lines ~80–95) with regex for `<think>` tags, `</think>` tags, and JSON/fence blocks. The same logic appears inline in `use_ollama.py` (line ~30, ad-hoc `re.sub(r' thinking.*? response', ...)`). |
| **Why it is inefficient** | Duplicate code. The `use_ollama.py` version uses a different, less robust pattern. Code drift risk. |
| **Expected improvement** | Not a performance improvement — a maintainability improvement. Consolidate into a shared utility. |
| **Implementation complexity** | Low (1 day). Move to a shared module or keep in `mentor.py` and import from there. |
| **Estimated speed improvement** | **N/A** — maintainability only. |
| **Risk of regression** | Very low if import-based. The `use_ollama.py` version is dead code per Dead_Code_Report.md. |
| **Files affected** | `mentor.py`, `use_ollama.py` (dead) |

---

### L5 — Skip TF-IDF Retrieval for Short Queries (<3 words)

| Field | Detail |
|-------|--------|
| **Current behavior** | `SemanticHistoryRetriever.retrieve_relevant_context()` always computes TF-IDF cosine similarity against the full inverted index, even when the query is "yes", "okay", "sure" — single-word or very short messages with minimal semantic content. |
| **Why it is inefficient** | `retrieve_relevant_context()` computes `log(1 + N/df)` IDF for every query term, scores every posting, computes vector lengths, and sorts — all to return empty results when the query has <3 non-stopword tokens. |
| **Expected improvement** | Early return when `len(query_words) < 2` after stopword filtering. Most short acknowledgments have zero semantic retrieval value. |
| **Implementation complexity** | Very low (hours). Add `if len(query_words) < 2: return []` at the top of the retrieval function. |
| **Estimated speed improvement** | **~1–2%** per-turn for short messages (cuts ~2–5 µs for queries like "yes", "okay", "I see"). |
| **Risk of regression** | Very low. Short queries with <2 meaningful words cannot match anything reliably — the retrieval threshold (0.15 cosine) already ensures they produce empty results. |
| **Files affected** | `memory.py` (`SemanticHistoryRetriever.retrieve_relevant_context`) |

---

### L6 — Cache Mermaid Diagram Between Turns

| Field | Detail |
|-------|--------|
| **Current behavior** | `generate_mermaid_diagram()` in `server.py` calls `ollama.chat()` with the full conversation history every time the user clicks "Visualize". The entire history (potentially hundreds of turns) is re-sent to the LLM, and the LLM re-generates the diagram from scratch. |
| **Why it is inefficient** | The diagram rarely changes between turns (only when new topics are discussed). The LLM call costs ~3-10 seconds and processes the full history each time. |
| **Expected improvement** | Cache the last generated diagram keyed by `(username, hash(history))`. Invalidate only when new messages are added. Serve cached diagram immediately. |
| **Implementation complexity** | Medium (3 days). Add a `_diagram_cache: dict[str, tuple[str, int]]` mapping username → (diagram, history_length). On visualize request, compare current history length to cached — if same, return cached. |
| **Estimated speed improvement** | **~3–10 seconds** eliminated per visualize click when history hasn't changed. |
| **Risk of regression** | Low. If history has changed, cache is invalidated. Worst case: user sees slightly stale diagram if they edit a past message (not possible in current frontend). |
| **Files affected** | `server.py` (`generate_mermaid_diagram`, `/visualize`), `app.py` (visualize button handler) |

---

## Summary Table

| ID | Opportunity | Impact | Est. Speed | Complexity | Risk | Category |
|----|-------------|--------|------------|------------|------|----------|
| H1 | Eliminate dual session save | **High** | **~30–35%** (~140ms) | Low | Medium | I/O reduction |
| H2 | Cache extraction system prompt | **High** | **~5–8%** (~20-35ms) | Low | Very low | Prompt compression |
| H3 | Skip RuleBasedExtractor with LLM | **High** | **~2–5%** (~10-20ms) | Low | Low | Dead-code avoidance |
| H4 | Singleton KnowledgeBase | **High** | ~1-3% startup | Very low | Very low | Lazy loading |
| M1 | Lazy JSON serialization | Medium | **~10–15%** (~35-55ms) | Medium | Medium | Serialization |
| M2 | Cache summary prompt section | Medium | ~3–5% (~15-20ms) | Medium | Low | Prompt compression |
| M3 | Lazy-load STT/TTS models | Medium | Startup speed | Medium | Medium | Lazy loading |
| M4 | Avoid temp WAV for STT | Medium | **~5–10%** (~50-150ms) | Very low | Very low | I/O reduction |
| M5 | Cache ordered_candidates | Medium | ~1-2% (~2-5µs) | Medium | Low | Caching |
| M6 | Skip LifecycleManager for non-Empathize | Medium | ~1-3% (~5-10ms) | Very low | Very low | Dead-code avoidance |
| L1 | Inline label dicts | Low | <0.1% | Very low | Very low | Micro-opt |
| L2 | Pre-compile regex patterns | Low | <0.5% | Very low | Very low | Micro-opt |
| L3 | `__slots__` on ProjectState | Low | Minimal | Medium | Medium | Object creation |
| L4 | Consolidate strip_model_output | Low | N/A (maintainability) | Low | Very low | Code quality |
| L5 | Skip TF-IDF for short queries | Low | ~1-2% (~2-5µs) | Very low | Very low | Micro-opt |
| L6 | Cache Mermaid diagram | Low | **~3-10s per click** | Medium | Low | Caching |

---

## Recommended Sprint Plan

### Sprint 1 — Quick wins (2–3 days total)

| Priority | Item | Days | Expected gain |
|----------|------|------|---------------|
| 1 | H2 — Cache extraction prompt | 1 | ~5-8% per turn |
| 2 | H4 — Singleton KnowledgeBase | 0.5 | ~1-3% startup |
| 3 | M4 — Remove temp WAV file | 1 | ~5-10% voice turns |
| 4 | M6 — Skip LifecycleManager guard | 0.5 | ~1-3% non-Empathize |

**Cumulative gain**: ~12–24% per turn (text) + ~5-10% (voice), plus faster startup.

### Sprint 2 — I/O and serialization (4–5 days total)

| Priority | Item | Days | Expected gain |
|----------|------|------|---------------|
| 1 | H1 — Eliminate dual session save | 2 | ~30-35% per turn |
| 2 | M1 — Lazy JSON serialization | 3 | ~10-15% I/O savings |

**Cumulative gain**: ~40-50% combined with Sprint 1.

### Sprint 3 — Caching and polish (5–7 days total)

| Priority | Item | Days | Expected gain |
|----------|------|------|---------------|
| 1 | M2 — Cache summary prompt | 3 | ~3-5% during summary phase |
| 2 | M5 — Cache ordered_candidates | 3 | ~1-2% per turn |
| 3 | L1–L6 — Low-impact items | 2 | Marginal |

**Cumulative gain**: ~45-55% combined with Sprints 1+2.

---

## Key Metric Targets

| Metric | Baseline (no Ollama) | Target with all H+M items |
|--------|---------------------|---------------------------|
| Per-turn pipeline time | 453 ms | **~180–250 ms** |
| I/O portion | 79.7% (360 ms) | **~40-50%** (~80-120 ms) |
| Pure computation portion | 20.3% (93 ms) | **~50-60%** (~100-130 ms) |
| Voice turn (transcription + pipeline + TTS) | ~3-8 s | **~2-5 s** |
| Startup time | 15-30 s | **~2-5 s** (Ollama only) |
| Mermaid diagram generation | 3-10 s per click | **< 100 ms** (cached) |

No changes should regress the pipeline's deterministic, read-only Module 3/4/5 contracts. All optimizations are runtime/caching improvements — no architectural redesign required.