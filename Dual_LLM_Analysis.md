# Dual-LLM Architecture Analysis — Mentor Pipeline

**Generated:** 2026-07-29  
**Scope:** Whether the current architecture genuinely requires two LLM calls per turn.  
**Method:** Data-flow tracing through `process_mentor_turn()` (mentor.py:1065–1427), extracting what each stage consumes, produces, and discards.  
**Constraint:** No code redesign. Analysis only.

---

## Current Flow (Condensed)

```
User Message
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ 1. MemoryExtractor (LLM Call 1)                          │
│    Input:  ~22,639 chars (5,659 tok) — 93% static rules  │
│    Output: ExtractionResult { message_type, updates[] }   │
│    Each update: { operation, field, value }  [3 fields]   │
│    LLM also produces but CODE DISCARDS:                   │
│      context, domain, keywords, raw_text,                 │
│      extraction_method, contributing_factors, qualifiers  │
└──────────────────────────────────────────────────────────┘
    │ 8 fields used: message_type + (operation, field, value) × N
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ 2. InferenceEngine (deterministic, currently pass-through)│
│ 3. HypothesisManager (deterministic)                      │
│ 4. StateManager.apply_extraction → ProjectState update    │
│ 5. RuleBasedExtractor (deterministic regex, skipped when  │
│    MemoryExtractor produced MEANINGFUL updates)           │
│ 6. ObjectiveEngine (deterministic → next objective)       │
│ 7. LifecycleManager/Moodule5 (deterministic → decision)   │
│ 8. SummaryBuilder (deterministic → EmpathizeSummary)      │
│ 9. ResponseStrategyEngine (deterministic → strategy)      │
│10. PromptBuilder (deterministic → prompt string)          │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│11. Mentor Response (LLM Call 2)                           │
│    Input:  ~1,877 chars (469 tok) — 77% dynamic data     │
│    Output: Natural language reply (constrained by         │
│            enforce_mentor_reply() filter)                  │
└──────────────────────────────────────────────────────────┘
    │
    ▼
   Reply
```

---

## 1. What Information from the First LLM Is Consumed by Later Stages?

### ExtractionResult — all 8 bytes of it

The downstream pipeline receives exactly:

- `message_type`: one of `{MEANINGFUL, NO_UPDATE, AMBIGUOUS, END}` — consumed by:
  - **StateManager** → decides whether to apply updates or skip
  - **RuleBasedExtractor bypass guard** (line 1194) → skips rule-based extraction when MEANINGFUL
  - **Legacy one-way mirror** (line 1202) → triggers mirror pass only on MEANINGFUL

- `updates[]`: list of `ExtractionUpdate`, each with 3 scalar values:
  - `operation`: `ADD` or `SET` — consumed by **StateManager** to decide append vs overwrite
  - `field`: one of 6 `StateField` values — consumed by **StateManager** to select the ProjectState list/scalar to mutate
  - `value`: string — consumed by **StateManager** as the value to append or assign

That is **3 scalars per update**. The entire downstream pipeline runs on roughly **40–80 bytes of structured data** per turn.

### Downstream consumers of ProjectState (the resulting state after extraction is applied)

| Stage | Reads from ProjectState | Uses |
|-------|----------------------|------|
| **ObjectiveEngine** | `personas`, `problems`, `current_solutions`, `pain_points`, `evidence`, `frequency` | Determine which fields are missing → decide next objective |
| **LifecycleManager** | Same fields + lifecycle tokens | Decide READY_FOR_SUMMARY / WAITING / CONTINUE / TRANSITION |
| **SummaryBuilder** | Same fields | Render structured summary for the prompt |
| **ResponseStrategyEngine** | Same fields + objective | Pick GENERATE_SUMMARY or ASK_QUESTION |
| **PromptBuilder** | Same fields + objective + strategy | Format the prompt for Call 2 |

### The second LLM call

The Mentor Response LLM receives:
- **Role header** (static)
- **Current Objective** (single line, from ObjectiveEngine)
- **Empathize Summary** or **ProjectState** (structured, from SummaryBuilder)
- **Latest Conversation** (previous assistant + user message, from raw_history)
- **Instructions** (static)

It does NOT receive any raw extraction output. The extraction has been fully consumed (→ state update) by the time the Mentor Response prompt is built.

---

## 2. What Information Is Discarded?

### Discarded from the LLM JSON response

The prompt asks the LLM to produce these fields per update:

| JSON Key | Discarded? | Why |
|----------|-----------|-----|
| `operation` | **Kept** | Needed by StateManager |
| `field` | **Kept** | Needed by StateManager |
| `value` | **Kept** | Needed by StateManager |
| `context` | **Discarded** | Not parsed. ExtractionUpdate has no `context` field |
| `domain` | **Discarded** | Not parsed. Always equals `field` value anyway |
| `keywords` | **Discarded** | Not parsed. Not consumed by any stage |
| `raw_text` | **Discarded** | Not parsed. Original text span is in user_message |
| `extraction_method` | **Discarded** | Always `"llm"` — tautological |
| `contributing_factors` | **Discarded** | Not parsed. Used in a few examples but schema doesn't include it |
| `qualifiers` | **Discarded** | Not parsed. Same |

**5 of 8 fields per update are discarded.** The LLM spends inference time generating `context`, `domain`, `keywords`, `raw_text`, and `extraction_method` that are parsed out of the JSON by the validator but never stored.

### Discarded from the examples in the prompt

The output schema in the prompt (and the 20 examples) show an 8-field update object. The code validates all 8 fields exist (for forward compatibility) but only reads 3. The other 5 are parsed into typed dicts for validation, then dropped from the `ExtractionUpdate` dataclass.

---

## 3. Could Any Extraction Be Performed by Deterministic Code Instead?

### What the MemoryExtractor currently does

Given a user message like:
> *"We need a system to help elderly people remember their medication schedules. My mother is 78 and often forgets to take her blood pressure pills. I'm worried she might have a stroke."*

The LLM classifies the message into one of 4 types and extracts structured data from natural language. Let's evaluate each classification and field:

### Message Type Classification

| Type | Example | Deterministic Alternative |
|------|---------|--------------------------|
| **NO_UPDATE** | "hello", "thanks", "ok" | Simple regex/string matching. The existing `RuleBasedExtractor` already does this. |
| **AMBIGUOUS** | "yes", "no", "maybe" | Regex for single-word/contextual replies. Already handled by `RuleBasedExtractor`. |
| **END** | "thanks, bye" | Regex for session-ending phrases. Trivial. |
| **MEANINGFUL** | Any substantive description | This is the hard case. |

**Verdict:** Message type classification is fully replicable with deterministic rules. The `RuleBasedExtractor` (which already exists in the codebase at `memory_extractor.py:540`) covers all four types.

### Field Extraction

| Field | Deterministic Feasibility | Assessment |
|-------|--------------------------|------------|
| **personas** | **Partial** | "elderly people", "students", "caregivers" — many can be matched by noun-phrase patterns. But user might say "my mom" → need to infer "family caregivers" or "elderly parents". NLU adds value. |
| **problems** | **Low** | Requires understanding *what* goes wrong. "She forgets pills" → "medication forgetfulness". Involves paraphrasing, not extraction. |
| **current_solutions** | **Partial** | "sticky notes", "phone alarms" are concrete nouns. Pattern: "[people] use [thing]." Often extractable by keyword/pattern. |
| **pain_points** | **Low** | Emotional signals ("worried", "stressed", "anxious") can be keyword-matched. But nuanced pain requires understanding. |
| **evidence** | **Partial** | "I interviewed 5 students", "research shows 70%" — some patterns are formulaic. But "my mother is 78" requires reasoning to classify as evidence vs persona. |
| **frequency** | **Partial** | "every day", "weekly", "multiple times daily" — highly formulaic. Regex could handle most cases. |

**Verdict:** A well-tuned `RuleBasedExtractor` (which already exists) could handle ~60–70% of extraction cases, especially for formulaic patterns (frequency, current_solutions, NO_UPDATE classification). The remaining ~30–40% require NLU for paraphrasing, disambiguation, and inference — which is what the LLM provides.

### What This Means

The `RuleBasedExtractor` already runs as a fallback (line 1194) when the LLM extraction is skipped or produces no updates. The codebase literally has both paths wired in. The LLM path is the primary path because:

1. The existing rule-based extractor is simple keyword/pattern matching — lower recall
2. The LLM generalizes better to novel phrasings
3. The authors determined that LLM quality justifies the 91-second latency cost

**However**, a properly engineered deterministic extractor (regex + spaCy NER + dependency parsing) would likely match the LLM's recall for this narrow domain at a fraction of the latency.

---

## 4. Could Some Extraction Be Deferred?

### Observation: Extraction feeds state, state feeds the second LLM

Current order:
```
User → MemoryExtractor (LLM) → State update → ... → Mentor Response (LLM)
```

### Alternative: Defer extraction to post-response processing

```
User → RuleBasedExtractor (fast) → partial state update → Mentor Response (LLM) → MemoryExtractor (LLM) → refine state
```

This would:
1. Use deterministic extraction for the first pass → low latency, lower quality
2. Respond to the user quickly with partial understanding
3. Run the LLM extractor asynchronously after the response to refine state
4. Return to the user with the refined state on the next turn

**Problem:** The Mentor Response prompt includes the ProjectState (what's known so far). If the LLM extractor runs *after* the response, the response cannot reference newly extracted information.

**When it works:** For multi-turn conversations, this is acceptable — the extraction runs asynchronously and the next turn benefits from full state. For single-turn conversations, the extraction is wasted (nobody reads the refined state).

### Alternative: Defer only non-critical fields

Some fields are more "nice-to-have":
- `evidence` — often not critical for the first response (the LLM can reference what the user said directly)
- `pain_points` — similarly, the LLM can derive emotional context from the raw conversation

But `personas` and `problems` are needed immediately to shape the response.

**Verdict:** Deferral is possible but adds architectural complexity (async state updates, versioning of state between turns) for marginal benefit. The state is simple enough that deferring individual fields is not clearly worthwhile.

---

## 5. Could the Second LLM Safely Perform Extraction Work?

### Current architecture: strict separation

The architecture docstring (mentor.py:1065–1095) explicitly states:
> *"The application decides WHAT to ask, HOW to respond, and WHERE in the Empathize lifecycle the conversation sits; the LLM only produces natural language."*

The PromptBuilder renders structured ProjectState into the prompt. The Mentor Response LLM receives:
- Already-extracted field values (as a structured Empathize Summary)
- The raw user message (embedded in "Latest Conversation")
- Explicit instructions: *"Reference the known project state only; do not invent information"*

### Could the Mentor Response also extract?

**Technically yes.** The Mentor Response already sees the raw user message. It could also produce structured extraction output in addition to its natural language response. This would mean the second LLM call is dual-purpose: extract + respond.

**Risks:**

1. **Prompt bloat.** To do extraction, the second prompt would need all the extraction rules (currently 93% of the MemoryExtractor prompt). The Mentor Response prompt is currently 1,877 chars — adding the extraction rules would make it ~24,000 chars. This would increase inference time for the second call by ~5–10×.

2. **Response constraint conflict.** The `enforce_mentor_reply()` filter (mentor.py:827) strips replies that give advice, ask >1 question, or exceed 55 words. This filter would need to be aware of JSON extraction output embedded in the reply, or the extraction would need to be in a separate output channel.

3. **Architecture violation.** The core constraint of the architecture ("application does the thinking, LLM does the talking") would be broken. The LLM would now be doing both.

4. **Coupling.** The second prompt is optimized for natural-language output (temperature=0.4, top_p=0.85, max_predict=150). Extraction typically uses temperature=0.0, greedy decoding, and higher max_predict. These parameter regimes don't mix well in a single call.

**Verdict:** Possible but architecturally harmful. The dual-purpose prompt would be larger, slower, harder to maintain, and violate the project's core design principle.

---

## 6. Is There Duplicated Reasoning Between the Two LLM Calls?

### What each LLM does

| Dimension | Call 1: MemoryExtractor | Call 2: Mentor Response |
|-----------|------------------------|------------------------|
| **Reads raw user message** | Yes | Yes (via "Latest Conversation") |
| **Reads structured state** | Yes (CURRENT PROJECT STATE) | Yes (Empathize Summary) |
| **Reads previous assistant message** | Yes (for disambiguation) | Yes (Latest Conversation) |
| **Reads static instructions** | 93% of prompt | 23% of prompt |
| **Task** | Classify + extract → JSON | Reflect + summarize → NL |
| **Output** | Structured JSON | Natural language sentence |
| **Latency** | ~91 s | ~32 s |

### What both read

Both calls read the **raw user message** and the **previous assistant message**. The MemoryExtractor reads it to extract field→value pairs. The Mentor Response reads it to produce a coherent reply.

### Is this duplicated reasoning?

**Partially, but with different purposes:**

1. **Understanding the user's intent.** Both calls must understand what the user said. The MemoryExtractor maps it to one of 4 message types. The Mentor Response maps it to a conversational response strategy (GENERATE_SUMMARY or ASK_QUESTION). These are different tasks — classification vs. NLG — but the underlying comprehension work overlaps.

2. **Familiarity with conversation history.** Both calls receive the previous assistant message. The MemoryExtractor uses it to disambiguate short replies ("yes" → AMBIGUOUS). The Mentor Response uses it to maintain conversational coherence. Same input, different purpose.

3. **Known state.** Both receive the project state. The MemoryExtractor uses it as context to avoid re-extracting known info. The Mentor Response uses it as the source of truth for the reply. Redundant but not harmful — the state is tiny (~500 chars).

### How much time is wasted?

If the two calls were merged, they would share the comprehension effort. The total time is currently ~123 s per turn. In a merged model:
- Single prompt with extraction instructions + NLG instructions
- Prompt would be ~24,000 chars (4× bigger than Call 2's current prompt)
- Would need to produce two outputs: JSON extraction + NL reply
- Inference time would be roughly **Call 1's time** (~91 s) because Call 1 dominates due to prompt size

So merging would save ~32 s (Call 2's time) but result in a ~91 s single call. Net savings: ~26%.

### Summary of Overlap

| Overlap Area | Duplicated? | Cost |
|-------------|-------------|------|
| Tokenizing the same input text | Yes | Negligible (part of prompt processing) |
| Processing the same conversation history | Yes | Negligible (history is small) |
| Understanding user intent | Partially | Different outputs needed (JSON vs NL) |
| Processing extraction rules | No | Call 2 doesn't have extraction rules |
| Generating output tokens | No | Different outputs (JSON vs conversation) |

**Verdict:** Minimal duplication. The primary cost (prompt processing) is dominated by Call 1's 93%-static prompt and Call 2's response generation. The overlapping comprehension of the user message (~200 chars) is a tiny fraction of total compute.

---

## Summary

| Question | Answer |
|----------|--------|
| **1. What from first LLM is consumed later?** | ~40–80 bytes: `message_type` + 3 scalars per update. All other LLM output is discarded. |
| **2. What is discarded?** | 5 of 8 JSON fields per update: `context`, `domain`, `keywords`, `raw_text`, `extraction_method`. Also `contributing_factors` and `qualifiers` where generated. |
| **3. Could extraction be deterministic?** | ~60–70% of cases, especially classification (NO_UPDATE/AMBIGUOUS/END) and formulaic fields (frequency, current_solutions). The remaining 30–40% (paraphrasing "my mom" → "elderly parents", disambiguating evidence vs problem) benefits from NLU. A `RuleBasedExtractor` already exists and runs as fallback. |
| **4. Could extraction be deferred?** | Technically yes (async post-response extraction), but it creates state-versioning complexity. The response cannot reference newly extracted information. Deferred extraction adds latency to the *next* turn, not the current one. |
| **5. Could the second LLM do extraction?** | Technically possible but architecturally harmful. Would bloat the prompt by ~12× (adding all extraction rules), double the second call's inference time, and violate the core constraint ("application does the thinking, LLM does the talking"). Different generation parameters (temperature, max_predict) also make unification awkward. |
| **6. Is reasoning duplicated?** | Minimal. The comprehension of the user message (~200 chars) is processed twice, but this is <1% of total compute. Call 1's work is 93% extraction rule processing; Call 2's work is 77% dynamic data rendering. They use the same input for different purposes. |

### Root Cause: The 93%-Static Prompt

The real issue is not the dual-LLM architecture itself, but the **20-example, 13,563-char comprehensive examples section** that inflates Call 1 to 91 seconds. The two calls serve genuinely different purposes (extraction vs. generation). Merging them would save ~32 s (the second call) but produce a 91 s single call — a net improvement of only 26%, while sacrificing architectural clarity.

A prompt cache for the static 93% of Call 1's prompt would eliminate 91 seconds more effectively than any architectural change.
