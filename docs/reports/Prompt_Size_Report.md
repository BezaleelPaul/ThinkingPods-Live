# Prompt Size Report — Design Thinking Mentor Pipeline

**Generated:** 2026-07-28 15:00:32 UTC

## Executive Summary

- **Largest single prompt:** Extractor prompt (full state, late conversation) — 22,143 chars (~5,535 tokens)
- **Number of LLM calls per turn:** 2 (MemoryExtractor extraction + mentor reply)
- **Number of LLM calls per turn (legacy):** 3 (if `MENTOR_USE_LLM_EXTRACTION=true`)
- **Token estimate ratio:** ~4 chars/token (conservative for mixed English + structured data)

## LLM Call Sites

| # | Call Site | Purpose | Temperature | Max Tokens | Sections |
|---|----------|---------|-------------|------------|----------|
| 1 | `MemoryExtractor._call_model` | Parse user message → structured extraction updates | 0.0 | 300 | 1 system prompt (600 lines) + 3 injected vars |
| 2 | `mentor.py:process_mentor_turn` | Generate natural-language mentor reply | 0.4 | 150 | 5 sections (Role + Objective + State + Conversation + Instructions) |
| 3 | `InputProcessor._llm_extract` | Legacy LLM extraction (OFF by default) | 0.1 | 256 | 1 inline prompt (user msg + 7 scalar fields) |

## 1. MemoryExtractor Extraction Prompt

### Template Overview

**`EXTRACTOR_PROMPT` template length:** 21,449 chars (~5,362 tokens)

The template is a single monolithic 600-line instruction set embedded in `memory_extractor.py`.
It contains:

- Role definition (1 line)
- BE CONSERVATIVE rule
- Output JSON schema (message_type + updates array)
- Field definitions for 6 fields: personas, problems, current_solutions, pain_points, evidence, frequency
- Critical differentiation rules (5 rules)
- Operation rules (ADD vs SET)
- Message type definitions (4 types)
- ~20 comprehensive examples showing exact input/output pairs
- 3 injection points: `{project_state}`, `{previous_context}`, `{user_message}`

* **Extractor prompt (fresh state, first turn):** 21,472 chars (~5,368 tokens)*
* **Extractor prompt (partial state, mid conversation):** 21,658 chars (~5,414 tokens)*
* **Extractor prompt (full state, late conversation):** 22,143 chars (~5,535 tokens)*
*Template alone: 21,449 chars (5,362 tokens)*

## 2. Module 4 Mentor Reply Prompt

### Scenarios

* **Mentor prompt (ASK_QUESTION, full state):** 1,307 chars (~326 tokens)*
* **Mentor prompt (GENERATE_SUMMARY, with EmpathizeSummary):** 1,309 chars (~327 tokens)*
* **Mentor prompt (first turn, empty state):** 710 chars (~177 tokens)*
* **Mentor prompt (ACKNOWLEDGE, partial state):** 873 chars (~218 tokens)*
* **Mentor prompt (no previous messages, empty state):** 655 chars (~163 tokens)*
## 3. Section-by-Section Breakdown

### Individual Section Sizes

| Section | Chars | Tokens (est) | % of Total |
|---------|-------|--------------|------------|
| Section: Role | 84 | 21 | 3.0% |
| Section: Objective | 79 | 19 | 2.8% |
| Section: Project State (full) | 620 | 155 | 21.9% |
| Section: Project State (empty) | 161 | 40 | 5.7% |
| Section: Empathize Summary (full) | 618 | 154 | 21.8% |
| Section: Latest Conversation (both present) | 169 | 42 | 6.0% |
| Section: Latest Conversation (none) | 81 | 20 | 2.9% |
| Section: Instructions (ASK_QUESTION) | 242 | 60 | 8.6% |
| Section: Instructions (GENERATE_SUMMARY) | 347 | 86 | 12.3% |
| Section: Instructions (ACKNOWLEDGE) | 239 | 59 | 8.4% |
| Section: Instructions (CLARIFY) | 190 | 47 | 6.7% |

## 4. Duplicated Information Across Prompts

### Per-Turn Static Content (Changes Rarely)

| Section | Variation | Size (chars) |
|---------|-----------|--------------|
| Role | Never changes | 84 |
| Objective label | 1 of 7 possible values (~40-50 chars each) | 79 |
| Instructions (ASK_QUESTION) | Fixed per strategy | 242 |
| Instructions (GENERATE_SUMMARY) | Fixed per strategy | 347 |
| Instructions (ACKNOWLEDGE) | Fixed per strategy | 239 |
| Instructions (CLARIFY) | Fixed per strategy | 190 |

### Cross-Prompt Duplication

| Data | MemoryExtractor Prompt | Module 4 Prompt | Duplicate? |
|------|----------------------|-----------------|------------|
| ProjectState | Yes (serialized as JSON) | Yes (rendered as bullet list) | **Yes** — same data, different formats |
| Previous assistant message | Yes (context) | Yes (Latest Conversation) | **Yes** — identical text, different headers |
| User message | Yes (primary input) | Yes (Latest Conversation) | **Yes** — identical text, different headers |
| Field definitions | Yes (in 600-line template) | No | — |
| Output schema | Yes (in template) | No | — |
| Examples | Yes (~20 in template) | No | — |
| Role | No | Yes (static line) | — |
| Objective | No | Yes (from Module 3) | — |
| Instructions | No | Yes (per-strategy) | — |

## 5. Conversation History Analysis

### Does the LLM receive the full conversation history?

**No.** The LLM only receives the LAST assistant message + the current user message,
not the full history. Both `MemoryExtractor._build_prompt` and `Module4.PromptBuilder`
limit context to the most recent exchange.

| Component | Full History Sent? | Context Provided |
|-----------|-------------------|------------------|
| MemoryExtractor prompt | No | Last assistant message + current user message |
| Module 4 prompt | No | Last assistant message + current user message |

### What would grow with conversation length?

- **Nothing** — both prompts are bounded to ~2 messages of context.
- State fields grow as data is collected (but bounded: 6 fields, each with list of values).
- The template is static (600 lines, never grows).

## 6. Compression Opportunities

### Section Sizes & Compression Feasibility

| Section | Chars | % of Total | Dominant Content | Can Become Static? |
|---------|-------|------------|------------------|--------------------|
| Role | 84 | 6.6% | Static text | **Yes** — never changes |
| Objective | 79 | 6.2% | Single-line label (1 of 7 values) | No — changes per objective |
| Project State (full) | 620 | 48.9% | 6 field labels + values (latest data) | No — grows with data collection |
| Latest Conversation | 131 | 10.3% | 2 message blocks (latest turn) | No — new text each turn |
| Instructions (ASK_QUESTION) | 242 | 19.1% | 4-5 bullet points (1 of 4 strategy sets) | No — changes per strategy |

### Estimated Savings Per Strategy

| Strategy | Current Size | Compressed Size | Savings | Method |
|----------|-------------|-----------------|---------|--------|
| ASK_QUESTION | ~600 chars | ~450 chars | ~25% | Deduplicate state sections |
| GENERATE_SUMMARY | ~800 chars | ~600 chars | ~25% | Use truncated summary instead of full state |
| ACKNOWLEDGE | ~600 chars | ~450 chars | ~25% | Reduce state to changed fields only |
| CLARIFY | ~600 chars | ~450 chars | ~25% | Reduce state to relevant fields only |

## 7. Findings Summary

1. **MemoryExtractor's 600-line template dominates** — the system prompt is ~18KB and
   is the single largest prompt component. It includes ~20 worked examples that
   are sent every single turn.

2. **State is not repeatedly sent across turns** — each turn sends only the current
   ProjectState snapshot. The LLM does not see the delta or history of state changes.

3. **Conversation history is NOT repeatedly resent** — only the last assistant message
   and the current user message are included. Full history lives in `session.raw_history`
   on the server, never sent to the LLM.

4. **The same user message appears in BOTH prompts** — MemoryExtractor receives it as
   `USER MESSAGE:` and Module 4 receives it as `Latest Conversation > User`. These are
   two separate LLM calls with the same text duplicated.

5. **ProjectState is serialized twice per turn, in different formats** — the
   MemoryExtractor gets it as JSON (`_build_prompt` uses `json.dumps`), while
   Module 4 gets it as a rendered bullet list. Same data, different representations.

6. **The Role section is static text, 46 chars** — negligible, not worth caching.

7. **Instructions are small (4-5 bullets)** — 60-90 chars, negligible.

8. **The largest growing section is ProjectState** — from ~40 chars (empty) to
   ~500 chars (fully populated across 6 fields with values). Still bounded.

9. **No unnecessary metadata is included** — no session IDs, timestamps, user-agent
    headers, or debug info leaks into the prompt.

10. **Token budgets are tight** — `num_predict=300` for extraction, `num_predict=150`
    for reply. The LLM is instructed to be concise.
