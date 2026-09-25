# Extractor Prompt Analysis — MemoryExtractor

**Generated:** 2026-07-29  
**Source:** `memory_extractor.py:450` (`EXTRACTOR_PROMPT` constant)  
**Rendered prompt:** `_prompt_dumps/call1_msg1_user_qwen2.5_3b.txt` (624 lines, 22,639 chars)  
**Analysis method:** Manual structural reading + cross-reference tracing. No code modified.

---

## 1. Number of Instruction Blocks

The prompt contains **10 distinct instruction blocks** (lines 1–604, all static):

| # | Block Name | Lines | Chars | % of Static | Purpose |
|---|-----------|-------|-------|-------------|---------|
| 1 | System Identity | 1–2 | 159 | 0.8% | Declare role: "semantic extraction unit" |
| 2 | Conservatism Warning | 3–6 | 271 | 1.3% | "BE CONSERVATIVE, HALLUCINATIONS ARE WORSE" |
| 3 | Strict Rules | 8–15 | 448 | 2.1% | 6 formatting/output rules |
| 4 | Output Schema | 16–31 | 439 | 2.1% | JSON structure specification |
| 5 | Field Names Table | 33–40 | 690 | 3.3% | 6 field names with one-line descriptions |
| 6 | Field Definitions with Examples | 42–92 | 3,299 | 15.6% | Per-field detailed guidance |
| 7 | Critical Differentiation Rules | 94–130 | 1,472 | 7.0% | 10 rules distinguishing similar concepts |
| 8 | Operation Rules | 131–137 | 492 | 2.3% | 4 rules about ADD vs SET |
| 9 | Message Type Definitions | 138–144 | 484 | 2.3% | 4 message type descriptions |
| 10 | Comprehensive + Critical Examples | 146–604 | 13,563 | 64.1% | 20 worked examples |

Plus **3 dynamic sections** at the end:

| # | Section | Lines | Chars | % of Total |
|---|---------|-------|-------|------------|
| 11 | CURRENT PROJECT STATE | 606–616 | 564 | 2.5% |
| 12 | PREVIOUS ASSISTANT MESSAGE | 618–619 | 696 | 3.1% |
| 13 | USER MESSAGE | 621–622 | 198 | 0.9% |

---

## 2. Number of Examples

**20 examples**, organized under two sub-headers:

### Block A: "COMPREHENSIVE EXAMPLES" (lines 146–302) — 7 examples

| # | User Message | Output Type | Updates | Purpose |
|---|-------------|-------------|---------|---------|
| 1 | "i want to build a system which will help old people..." | MEANINGFUL | 4 | Multi-field: personas + problem + pain_point + evidence |
| 2 | "students mostly use WhatsApp groups but messages get buried..." | MEANINGFUL | 4 | Multi-field: personas + solution + pain_point + frequency(SET) |
| 3 | "hello" | NO_UPDATE | 0 | Greeting → nothing |
| 4 | "yes" | AMBIGUOUS | 0 | Short reply → ambiguous |
| 5 | "they currently use sticky notes and phone alarms..." | MEANINGFUL | 2 | Solution + pain_point |
| 6 | "it happens every single day, sometimes multiple times" | MEANINGFUL | 1 | Frequency → SET |
| 7 | "thanks, that's all I needed" | END | 0 | Session end |

### Block B: "CRITICAL DISCRIMINATION EXAMPLES" (lines 303–604) — 13 examples

| # | User Message | Output Type | Updates | Purpose |
|---|-------------|-------------|---------|---------|
| 8 | "This is the most commonly faced problem even I have faced it." | MEANINGFUL | 2 | Personal experience → evidence + pain_point |
| 9 | "I have seen students do late submission and lose marks." | MEANINGFUL | 1 | Observation → evidence only |
| 10 | "I want to make people disciplined." | NO_UPDATE | 0 | Vague motivation → nothing |
| 11 | "Students usually set reminders." | MEANINGFUL | 1 | Solution only |
| 12 | "They feel stressed." | MEANINGFUL | 1 | Emotion → pain_point only |
| 13 | "It happens almost every day." | MEANINGFUL | 1 | Frequency → SET (repeat of #6) |
| 14 | "I interviewed five students and they all forget assignments." | MEANINGFUL | 2 | Interview → evidence + problem |
| 15 | "My grandmother struggles with medication timing." | MEANINGFUL | 3 | Personal obs → personas + problem + evidence |
| 16 | "Research shows 70% of students miss deadlines." | MEANINGFUL | 1 | Statistic → evidence |
| 17 | "The problem is late submission and the solution is reminders." | MEANINGFUL | 1 | Explicit problem statement → problem |
| 18 | "Forgetting things is stressful." | MEANINGFUL | 2 | Problem + pain_point combo |
| 19 | "I want to build an app for this." | NO_UPDATE | 0 | Builder intent → nothing (repeat of #10) |
| 20 | "Procrastination causes missed deadlines." | MEANINGFUL | 2 | Problem + evidence combo |

---

## 3. Number of Repeated Rules

**15 distinct rules are stated in 2+ places** (some in 4+):

| Rule | Times Stated | Locations |
|------|-------------|-----------|
| Be conservative / low confidence → NO_UPDATE | **4** | L4–5, L127–129, Ex#10, Ex#19 |
| Output JSON only, no other text | **3** | L9, L624, (implicit in all examples) |
| Never explain reasoning | **1** | L11 |
| Never ask questions | **1** | L12 |
| EVIDENCE is NOT PROBLEM | **4** | L56, L82–84, L98–101, Ex#9 |
| PAIN POINT is NOT EVIDENCE | **3** | L70, L102–104, Ex#8 |
| PROBLEM is NOT SOLUTION | **2** | L62, L106–108 |
| PERSONA is NOT PROBLEM | **2** | L56, L110–112 |
| FREQUENCY is NEVER PAIN POINT | **2** | L92, L114–116 |
| Personal experience → EVIDENCE or PAIN_POINT | **5** | L71, L83, L121–122, Ex#8, Ex#15 |
| Observation → EVIDENCE | **3** | L124–125, Ex#9, Ex#14 |
| Operation: ADD for lists, SET for scalar | **4** | L91, L132–136, Ex#2, Ex#13 |
| SCALAR only for frequency | **2** | L39, L91 |
| Empty updates when nothing to extract | **2** | L14, (implicit in NO_UPDATE examples) |
| Return exactly one message_type | **1** | L13 |

**6 of 15 rules are repeated 3+ times** in different wording.

### Contradictory Instructions

- **Personal experience**: "I faced this problem" → L71 says **PAIN_POINT**, L83 says **EVIDENCE or PAIN_POINT**, L121–122 says **EVIDENCE or PAIN_POINT**. The model receives conflicting guidance — sometimes it's one field, sometimes it's either/both. Ex#8 demonstrates both evidence AND pain_point for personal experience, which is the most permissive interpretation.

---

## 4. Sections That Duplicate Information

### Duplicate Group A: Field boundaries (what goes where)

| Section | Content | Chars |
|---------|---------|-------|
| Field Names Table (L33–40) | 1-line summary per field | 690 |
| Field Definitions (L42–92) | Multi-line per field with examples + keywords + warnings | 3,299 |
| Differentiation Rules (L94–130) | Rules that restate field boundaries | 1,472 |
| Examples (L146–604) | 20 examples that demonstrate field boundaries | 13,563 |

**Total: 19,024 chars** — all teaching the same thing (which text belongs to which field) at increasing levels of detail.

### Duplicate Group B: JSON output format

| Location | Content | Chars |
|----------|---------|-------|
| Output Schema (L16–31) | Formal JSON schema | 439 |
| Example outputs (all MEANINGFUL examples) | Concrete JSON instances | ~7,000 |

Every MEANINGFUL example repeats the same JSON structure with different field values. The structural boilerplate is identical across all examples.

### Duplicate Group C: Operation rules (ADD/SET)

| Location | Content |
|----------|---------|
| L39 (Field Names) | "frequency — SCALAR" |
| L91 (Frequency def) | "SCALAR only — use SET operation" |
| L132–136 (Operation Rules) | Full ADD/SET specification |
| Ex#2, Ex#6, Ex#13 | Frequency demonstrated with SET operation |

### Duplicate Group D: Conservatism

| Location | Content |
|----------|---------|
| L3–6 | Warning banner (3 lines) |
| L127–129 | Rule: low confidence → NO_UPDATE |
| Ex#10 | "I want to make people disciplined" → NO_UPDATE |
| Ex#19 | "I want to build an app for this" → NO_UPDATE |

---

## 5. Sections Never Referenced by Later Instructions

| Section | Why It's Isolated |
|---------|-------------------|
| **System Identity** (L1–2) | Never referenced again. The model is told it's a "semantic extraction unit" but no later section refers back to this role. All subsequent instructions just tell it what to do. |
| **Conservatism Warning** (L3–6) | While the *concept* of conservatism is reinforced at L127–129, the specific banner formatting (with `━━━` borders) is never mentioned again. |
| **Strict Rule "Never explain reasoning"** (L11) | Never enforced or referenced in any example. All examples just show input/output pairs — none show the model attempting to explain, so there's no negative example. |
| **Strict Rule "Never ask questions"** (L12) | Same — never demonstrated, never referenced. |
| **Message Type "END"** (L143) | Only one example (Ex#7) demonstrates END. The definition itself is never referenced elsewhere. |

### Implicit Cross-References

Most sections do reference each other, but the references are **implicit** — they rely on the reader (model) connecting concepts across sections. There are no explicit cross-references like "see the examples below" or "as defined in the previous section." The model must infer that:
- The Field Definitions illustrate the Field Names
- The Differentiation Rules clarify the Field Definitions
- The Examples demonstrate the Differentiation Rules

---

## 6. Examples That Teach the Same Behavior

### Redundant Pair 1: NO_UPDATE for vague motivation (Ex#10 vs Ex#19)

| Ex#10 | Ex#19 |
|-------|-------|
| "I want to make people disciplined." | "I want to build an app for this." |
| → NO_UPDATE | → NO_UPDATE |

Both teach: builder intent / vague motivation → nothing. One could be removed.

### Redundant Pair 2: Frequency → SET (Ex#6 vs Ex#13)

| Ex#6 | Ex#13 |
|------|-------|
| "it happens every single day, sometimes multiple times" | "It happens almost every day." |
| → SET frequency "every day..." | → SET frequency "almost every day" |

Both teach: explicit frequency → SET operation. Nearly identical in structure.

### Redundant Triple: Single-field extraction (Ex#11, Ex#12, Ex#16)

| Ex#11 | Ex#12 | Ex#16 |
|-------|-------|-------|
| "Students usually set reminders." | "They feel stressed." | "Research shows 70%..." |
| → current_solutions only | → pain_points only | → evidence only |

All three teach: single-claim message → one update with one field. Two could be removed.

### Redundant Pair: Personal experience with grandmother (Ex#1 vs Ex#15)

| Ex#1 | Ex#15 |
|------|-------|
| "...help old people...grandma always forgets" | "My grandmother struggles with medication timing." |
| → personas + problem + pain_point + evidence (4 fields) | → personas + problem + evidence (3 fields) |

Both involve elderly grandmothers and medication. The behaviors taught overlap significantly.

### Redundant Example: Evidence-only

Ex#9 ("I have seen students do late submission") and Ex#16 ("Research shows 70%") both demonstrate evidence-only extraction. One covers observation, the other covers statistics — but the structural behavior (single field, single update) is identical.

---

## 7. Candidate Sections for Removal

### Tier 1: Remove safely (no behavioral impact)

| Section | Lines | Chars | Tokens | Reason |
|---------|-------|-------|--------|--------|
| **Field Definitions (L42–92)** | 52 | 3,299 | ~825 | Fully redundant with Field Names Table + examples. The table provides field names; examples demonstrate usage. |
| **Operation Rules (L131–137)** | 8 | 492 | ~123 | Four trivial rules. Examples demonstrate ADD/SET naturally. |
| **Conservatism Warning banner (L3–6)** | 3 | 271 | ~68 | Concept restated at L127–129. Banner formatting is visual noise for an LLM. |
| **CRITICAL DISCRIMINATION header (L303–306)** | 4 | ~200 | ~50 | Label only. Functional content starts at L307. |
| **"OUTPUT (JSON only):" marker (L624)** | 1 | 20 | ~5 | Strict Rules (L9) already say "JSON only." |

**Tier 1 subtotal:** ~11,282 chars / ~2,821 tokens removed.

### Tier 2: Remove or consolidate (review behavioral impact)

| Section | Lines | Chars | Tokens | Assessment |
|---------|-------|-------|--------|------------|
| **Critical Differentiation Rules (L94–130)** | 37 | 1,472 | ~368 | Rules are all demonstrated in examples. But they serve as explicit "cheat sheet" — removing may reduce model accuracy on edge cases. |
| **Examples Ex#10 (vague motivation)** | ~15 | ~300 | ~75 | Redundant with Ex#19. Remove one. |
| **Examples Ex#13 (frequency repeat)** | ~15 | ~350 | ~88 | Redundant with Ex#6. Remove. |
| **Examples Ex#11 or Ex#12 (single-field)** | ~15 | ~350 | ~88 | Remove one of the three single-field examples. |
| **Examples Ex#1 (grandmother multi-field)** | ~45 | ~1,500 | ~375 | Redundant with Ex#15. Remove or consolidate to 3 fields. |
| **Examples Ex#9 (evidence-only)** | ~20 | ~500 | ~125 | Redundant with Ex#16. Remove. |

**Tier 2 subtotal:** ~4,472 chars / ~1,118 tokens removed (with 6 of 20 examples removed).

### Not Recommended for Removal

| Section | Reason to Keep |
|---------|---------------|
| **Output Schema (L16–31)** | Provides exact JSON structure. Essential for format compliance. |
| **Field Names Table (L33–40)** | Dense reference. Only 690 chars to define all 6 fields. |
| **Message Type Definitions (L138–144)** | Compact reference for the 4 output classes. |
| **Strict Rules (L8–15)** | Core behavioral constraints in 448 chars. |
| **System Identity (L1–2)** | Minimal (159 chars). Context-setting. |
| **Examples Ex#3, Ex#4, Ex#7** | Cover NO_UPDATE, AMBIGUOUS, END — each is a distinct message type. Keep. |
| **Examples Ex#2, Ex#5, Ex#8, Ex#14, Ex#17, Ex#18, Ex#20** | Each teaches a unique composite extraction pattern. Keep. |

---

## Token Count Estimate: Current vs. Reduced

| Component | Current (chars) | Current (tokens) | Reduced (chars) | Reduced (tokens) |
|-----------|----------------|-----------------|----------------|-----------------|
| System Identity | 159 | ~40 | 159 | ~40 |
| Conservatism Warning | 271 | ~68 | **0** (remove) | **0** |
| Strict Rules | 448 | ~112 | 448 | ~112 |
| Output Schema | 439 | ~110 | 439 | ~110 |
| Field Names Table | 690 | ~173 | 690 | ~173 |
| Field Definitions | 3,299 | ~825 | **0** (remove) | **0** |
| Differentiation Rules | 1,472 | ~368 | **0** (remove) | **0** |
| Operation Rules | 492 | ~123 | **0** (remove) | **0** |
| Message Types | 484 | ~121 | 484 | ~121 |
| Examples header + 20 examples | 13,563 | ~3,391 | **~5,800** (keep 7) | **~1,450** |
| Dynamic: State + Msg + Context | 1,481 | ~370 | 1,481 | ~370 |
| **Total** | **22,639** | **~5,659** | **~9,501** | **~2,375** |

**Estimated reduction: 58%** (from ~5,659 to ~2,375 tokens per turn).

### Optimized Prompt Skeleton

A minimal viable prompt would contain:

```
1. [System Identity]       (1-2 lines)
2. [Strict Rules]          (7 lines)
3. [Field Names Table]     (9 lines)
4. [Message Types]         (7 lines)
5. [7 diverse examples]:
   - NO_UPDATE (greeting)
   - AMBIGUOUS (short reply)
   - END (session end)
   - MEANINGFUL multi-field (personas + problem + pain_point + evidence)
   - MEANINGFUL single-field (evidence only)
   - MEANINGFUL frequency (SET operation)
   - MEANINGFUL composite (problem + solution or problem + evidence)
6. [Dynamic sections]
```

The Output Schema is removed because the examples demonstrate it. The Field Definitions are removed because the Field Names table + examples suffice. The Differentiation Rules are removed because the examples illustrate the boundaries. Operation Rules are removed because the examples demonstrate ADD/SET.

---

## Summary

| Metric | Value |
|--------|-------|
| Instruction blocks | **10** |
| Examples | **20** |
| Rules repeated 3+ times | **6** |
| Duplicate content sections | **4** (field boundaries, JSON format, ADD/SET rules, conservatism) |
| Unreferenced sections | **5** (identity, banner, "never explain", "never ask", END definition) |
| Redundant example pairs | **5** (NO_UPDATE ×2, frequency ×2, single-field ×3, grandmother ×2, evidence ×2) |
| Current token count | **~5,659** (static: ~5,289 + dynamic: ~370) |
| Estimated reduced token count | **~2,375** (static: ~2,005 + dynamic: ~370) |
| Estimated reduction | **58%** (removing ~12,900 of 22,639 chars) |

**The prompt is 3× larger than necessary.** The 20 examples could be reduced to 7 without losing coverage of any message type or extraction pattern. Three full instruction blocks (Field Definitions, Differentiation Rules, Operation Rules) restate what the examples already demonstrate.
