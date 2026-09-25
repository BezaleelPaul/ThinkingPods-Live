# Performance Report — Design Thinking Mentor Pipeline

**Generated:** 2026-07-28 14:36:17 UTC

## Executive Summary

- **Total pipeline time:** 453.025 ms (0.453 s)
- **Stages measured:** 19
- **Ollama available:** No (environment without Ollama; LLM stages simulated via fallback)
- **Pipeline type:** Deterministic rule engine + 1 LLM call
- **LLM calls per turn:** 2 (MemoryExtractor extraction + mentor reply)
- **Deterministic stages:** 14 (zero LLM, zero I/O)

## Pipeline Stage Timing

| # | Stage | Time (ms) | % of Total | Calls | Avg (ms) | Type |
|---|-------|-----------|------------|-------|----------|------|
| 0 | 0. State Loading (SessionManager.load) | 0.040 | 0.0% | 30 | 0.001 | 🔵 I/O |
| 1 | 0a. Capture last assistant msg | 0.191 | 0.0% | 30 | 0.006 | 🟢 CPU |
| 2 | 0b. Append user msg to history | 0.057 | 0.0% | 30 | 0.002 | 🟢 CPU |
| 3 | 1. MemoryExtractor.extract | 60.312 | 13.3% | 30 | 2.010 | 🟢 CPU |
| 4 | 2. InferenceEngine.infer | 8.334 | 1.8% | 30 | 0.278 | 🟢 CPU |
| 5 | 3. HypothesisManager.evaluate_batch | 4.275 | 0.9% | 30 | 0.143 | 🟢 CPU |
| 6 | 4. StateManager.apply_extraction | 0.513 | 0.1% | 30 | 0.017 | 🟢 CPU |
| 7 | 5. RuleBasedExtractor (legacy) | 8.371 | 1.8% | 30 | 0.279 | 🟢 CPU |
| 8 | 5a. merge_extracted_data | 0.491 | 0.1% | 30 | 0.016 | 🟢 CPU |
| 9 | 6. StageController | 0.137 | 0.0% | 30 | 0.005 | 🟢 CPU |
| 10 | 7. ObjectiveEngine.determine_next | 4.003 | 0.9% | 30 | 0.133 | 🟢 CPU |
| 11 | 8. ResponseStrategyEngine.determine_strategy | 2.539 | 0.6% | 30 | 0.085 | 🟢 CPU |
| 12 | 9. LifecycleManager.determine | 0.134 | 0.0% | 30 | 0.004 | 🟢 CPU |
| 13 | 11. PromptBuilder.build_prompt | 1.886 | 0.4% | 30 | 0.063 | 🟢 CPU |
| 14 | 12. LLM (SKIPPED - no Ollama) | 0.018 | 0.0% | 30 | 0.001 | 🔴 LLM |
| 15 | 13. Fallback (_build_journey_fallback) | 0.342 | 0.1% | 30 | 0.011 | 🟢 CPU |
| 16 | 14. Session Save (ProjectState) | 137.211 | 30.3% | 30 | 4.574 | 🔵 I/O |
| 17 | 14a. Session Save (legacy) | 223.862 | 49.4% | 30 | 7.462 | 🔵 I/O |
| 18 | 10. SummaryBuilder.build | 0.311 | 0.1% | 20 | 0.016 | 🟢 CPU |

### Category Breakdown

| Category | Total Time (ms) | % of Total |
|----------|-----------------|------------|
| CPU | 91.896 | 20.1% |
| IO | 361.113 | 79.7% |
| LLM | 0.018 | 0.0% |

### Top 5 Most Expensive Stages

1. **14a. Session Save (legacy)** — 223.862 ms (49.4%) — IO
2. **14. Session Save (ProjectState)** — 137.211 ms (30.3%) — IO
3. **1. MemoryExtractor.extract** — 60.312 ms (13.3%) — CPU
4. **5. RuleBasedExtractor (legacy)** — 8.371 ms (1.8%) — CPU
5. **2. InferenceEngine.infer** — 8.334 ms (1.8%) — CPU

## Object Counts (Average per Turn)

| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| extraction_updates | 0.00 | 0 | 0 |
| hypotheses_auto_accept | 0.00 | 0 | 0 |
| hypotheses_conflicts | 0.67 | 0 | 1 |
| hypotheses_discard | 10.00 | 0 | 15 |
| hypotheses_hold | 0.00 | 0 | 0 |
| hypotheses_new | 0.00 | 0 | 0 |
| hypotheses_total | 10.00 | 0 | 15 |
| hypotheses_verify | 0.00 | 0 | 0 |
| prompt_length | 1168.33 | 793 | 1368 |
| reply_length | 492.67 | 92 | 693 |
| summary_fields | 14.00 | 14 | 14 |

## Detailed Stage Analysis

### 0. State Loading (SessionManager.load)

- **Time:** 0.040 ms (0.0% of total)
- **Calls:** 30
- **Type:** IO
- **Description:** Reads the legacy mentor session JSON file from disk. Includes file open, JSON parse, and MentorSession.from_dict reconstruction. In production this also calls get_active_project_state() on the new SessionManager (another file read). The profiling data uses in-memory sessions to isolate processing time from I/O.

### 0a. Capture last assistant msg

- **Time:** 0.191 ms (0.0% of total)
- **Calls:** 30
- **Type:** CPU
- **Description:** Linear scan of raw_history (~12 turns) to find the most recent assistant message. Used by MemoryExtractor for contextual resolution of short replies.

### 0b. Append user msg to history

- **Time:** 0.057 ms (0.0% of total)
- **Calls:** 30
- **Type:** CPU
- **Description:** Simple list append to session.raw_history. Negligible cost.

### 1. MemoryExtractor.extract

- **Time:** 60.312 ms (13.3% of total)
- **Calls:** 30
- **Type:** CPU
- **Description:** Semantic extraction via LLM. Builds a prompt with the current ProjectState serialized as JSON, calls Ollama with temperature=0, parses the JSON response, and validates it. This is one of two LLM calls in the pipeline.

### 2. InferenceEngine.infer

- **Time:** 8.334 ms (1.8% of total)
- **Calls:** 30
- **Type:** CPU
- **Description:** Deterministic inference: converts extraction updates to facts (0-6 items), loads existing hypotheses from ProjectState (0-30 items), matches concepts via KnowledgeBase (KB lookup per fact), builds evidence bundles, forms hypothesis candidates, merges with existing (N×M search), detects conflicts (O(N²) pairwise), and assigns statuses. No LLM calls. The O(N²) conflict detection is the dominant cost for large hypothesis sets.

### 3. HypothesisManager.evaluate_batch

- **Time:** 4.275 ms (0.9% of total)
- **Calls:** 30
- **Type:** CPU
- **Description:** Deterministic policy evaluation: for each hypothesis (0-30), maps confidence level to numeric value, looks up concept category in KnowledgeBase, applies evidence boost and contradiction penalty, and compares against thresholds. Routes each hypothesis into AUTO_ACCEPT / VERIFY / HOLD / DISCARD.

### 4. StateManager.apply_extraction

- **Time:** 0.513 ms (0.1% of total)
- **Calls:** 30
- **Type:** CPU
- **Description:** Atomic state mutation: validates the extraction batch, applies ADD/SET operations to ProjectState, rolls forward previous-message bookkeeping, and mirrors to legacy session scalars. Pure CPU.

### 5. RuleBasedExtractor (legacy)

- **Time:** 8.371 ms (1.8% of total)
- **Calls:** 30
- **Type:** CPU
- **Description:** Deterministic regex-based extraction: applies ~40 regex patterns against the user message to extract audience, pain points, motivation, frequency, workflow, and evidence. No LLM call by default. Legacy compatibility layer.

### 5a. merge_extracted_data

- **Time:** 0.491 ms (0.1% of total)
- **Calls:** 30
- **Type:** CPU
- **Description:** Merges the legacy RuleBasedExtractor output into the MentorSession's scalar attributes. Iterates 7 field mappings, appends to 3 list fields, rebuilds known_facts. Pure CPU, low overhead.

### 6. StageController

- **Time:** 0.137 ms (0.0% of total)
- **Calls:** 30
- **Type:** CPU
- **Description:** Two-branch conditional: checks if the session has already transitioned to EMPATHIZE_COMPLETE. Constant-time, negligible.

### 7. ObjectiveEngine.determine_next

- **Time:** 4.003 ms (0.9% of total)
- **Calls:** 30
- **Type:** CPU
- **Description:** Completeness checker evaluates all 6 ProjectState fields. Priority engine applies 6 ordered rules to find the first missing field. Composes the ConversationObjective with reasoning. Pure CPU, 6 field reads.

### 8. ResponseStrategyEngine.determine_strategy

- **Time:** 2.539 ms (0.6% of total)
- **Calls:** 30
- **Type:** CPU
- **Description:** Strategy planner: looks up base strategy per objective, filters relevant hypotheses (0-30), generates candidate intents (2-9), estimates information gain, ranks candidates, detects contradictions (O(H²) loop), selects best, and builds a ResponsePlan. The O(H²) contradiction detection and O(C×H) gain estimation dominate when hypotheses are plentiful.

### 9. LifecycleManager.determine

- **Time:** 0.134 ms (0.0% of total)
- **Calls:** 30
- **Type:** CPU
- **Description:** Four-rule decision chain: checks if WRAP_UP objective, checks for summary presented marker, checks for user confirmation (regex match against ~20 words), and returns CONTINUE / READY_FOR_SUMMARY / WAITING_FOR_CONFIRMATION / READY_FOR_TRANSITION. Constant-time, 4 checks.

### 11. PromptBuilder.build_prompt

- **Time:** 1.886 ms (0.4% of total)
- **Calls:** 30
- **Type:** CPU
- **Description:** Builds the LLM prompt from 5 sections: role definition, objective label, project state rendering (6 fields), latest conversation (2 messages), and response instructions. Pure string formatting, ~300-800 chars of output.

### 12. LLM (SKIPPED - no Ollama)

- **Time:** 0.018 ms (0.0% of total)
- **Calls:** 30
- **Type:** LLM
- **Description:** No detailed analysis available.

### 13. Fallback (_build_journey_fallback)

- **Time:** 0.342 ms (0.1% of total)
- **Calls:** 30
- **Type:** CPU
- **Description:** Deterministic fallback reply: 4-way branch on lifecycle decision to produce a template-based reply without any LLM call. Used when Ollama is unavailable or the LLM reply is empty/too short. Pure string formatting.

### 14. Session Save (ProjectState)

- **Time:** 137.211 ms (30.3% of total)
- **Calls:** 30
- **Type:** IO
- **Description:** Writes the current ProjectState to the new SessionManager's session_data.json. Includes JSON serialization of the full ProjectState + conversation history.

### 14a. Session Save (legacy)

- **Time:** 223.862 ms (49.4% of total)
- **Calls:** 30
- **Type:** IO
- **Description:** Writes the full MentorSession to the legacy {user}_{project}_mentor.json file. Includes JSON serialization of all session fields. Runs in addition to the new SessionManager save, resulting in the same data being serialized twice.

### 10. SummaryBuilder.build

- **Time:** 0.311 ms (0.1% of total)
- **Calls:** 20
- **Type:** CPU
- **Description:** Constructs an EmpathizeSummary dataclass from ProjectState with 5 shallow list copies and 1 scalar copy. Only invoked on READY_FOR_SUMMARY lifecycle branch. Pure CPU, < 10 μs.

## Duplicate / Repeated Work

The following duplicate operations were identified:

1. **Dual session save** — Every turn writes ProjectState to both the new
   SessionManager (session_data.json) AND the legacy file ({user}_{project}_mentor.json).
   The same 15+ fields are serialized twice, doubling session save I/O.

2. **ProjectState → MentorSession mirror** — `_project_state_to_legacy_session`
   (session load) and `_mirror_state_to_session` (after state update) both
   project ProjectState list fields to legacy MentorSession scalars. The
   mirror runs on every MEANINGFUL extraction turn, even when nothing changed.

3. **merge_extracted_data + StateManager** — Both process extraction information,
   but on different data structures: `StateManager.apply_extraction` mutates
   ProjectState (v2), while `merge_extracted_data` writes the legacy
   RuleBasedExtractor output to MentorSession scalars. The rule extractor's
   output may repeat what the MemoryExtractor already extracted via LLM.

4. **InferenceEngine re-reads ProjectState fields** — `_load_existing_hypotheses`
   iterates all 6 ProjectState fields on every turn, generating Hypothesis
   objects that the ObjectiveEngine also reads directly from ProjectState.
   The same field values are converted to Hypothesis objects every turn even
   when unchanged.

5. **PromptBuilder re-renders entire state** — The prompt is rebuilt from scratch
   every turn, re-serializing all 6 ProjectState fields even when only 1-2
   fields changed. No incremental state rendering.

6. **Enforce reply runs regex on every LLM output** — `enforce_mentor_reply` applies
   3 checks (advice detection, question count, word limit) to every LLM response.
   When the LLM produces valid output that passes all checks, the regex is still
   applied and the reply is returned unchanged — wasted parsing.

## Repeated Parsing / Serialization

| Operation | Frequency | Cost |
|-----------|-----------|------|
| Session JSON deserialize (legacy load) | Every turn | 1 file read + JSON parse |
| Session JSON serialize (new save) | Every turn | 1 file write + JSON dump |
| Session JSON serialize (legacy save) | Every turn | 1 file write + JSON dump |
| ProjectState → JSON in extractor prompt | Every turn (MemoryExtractor) | json.dumps of ~6 fields |
| LLM output → JSON parse (extractor) | Every turn (MemoryExtractor) | json.loads of extraction response |
| ProjectState → Hypothesis objects | Every turn (InferenceEngine) | 6 dataclass conversions |

## Unnecessary Object Copying

| Source | Object | Copies |
|--------|--------|--------|
| SummaryBuilder.build | EmpathizeSummary list fields | 5 shallow list copies |
| InferenceEngine._load_existing_hypotheses | Hypothesis creation | 0-30 new objects per turn |
| ResponseStrategyEngine._rank_candidates | Candidate sorting | Sorted copy of 2-9 candidates |
| HypothesisBatchResult | Hypothesis tuples | 4 tuple copies |

## LLM Call Summary

| Call | Purpose | Model | Temperature | Max Tokens |
|------|---------|-------|-------------|------------|
| MemoryExtractor extraction | Parse user message → structured updates | Extractor model | 0.0 | ~256 |
| Mentor reply generation | Natural language reply | Mentor model | 0.4 | ~512 |

**Note:** `InputProcessor._llm_extract` (legacy LLM extraction) is OFF by default
(requires `MENTOR_USE_LLM_EXTRACTION=true`). When enabled, it adds a third LLM call.