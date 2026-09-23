# MemoryExtractor Section Analysis

## Source

- **Template constant**: `EXTRACTOR_PROMPT` at `memory_extractor.py:450`
- **Rendered prompt**: `_prompt_dumps/call1_msg1_user_qwen2.5_3b.txt` (624 lines, 22,639 chars / ~5,660 tok)
- **Bug**: duplicate JSON fragment at rendered lines 596–604 ORPHANED after a valid closing `}`

---

## Section Inventory

### **S1 — System Identity** (`memory_extractor.py:452`, rendered line 1, ~20 tok)
> *You are a semantic extraction unit. Your ONLY task is to extract the entities above into the JSON schema below…*

| Criterion | Assessment |
|-----------|------------|
| Purpose | Declare role / scope of work |
| Redundant? | **Yes** — says "entities above" but definitions are *below* (copy-paste artifact). Role is obvious from all following content. |
| Taught by later sections? | Every example shows extraction; the task is implicit in the format. |

---

### **S2 — Conservatism Warning** (rendered lines 3–6, ~45 tok)
> *CRITICAL: BE CONSERVATIVE. IF UNSURE (<80% CONFIDENCE), RETURN NO_UPDATE.*

| Criterion | Assessment |
|-----------|------------|
| Purpose | Hallucination guard; prefer omission over invention |
| Redundant? | **Yes** — Rule 2 in S3 ("never produce conversational responses"), S9 (NO_UPDATE definition), and ~5 examples showing NO_UPDATE all teach the same bias. |
| Taught by later sections? | S3 (rule 2), S9, Examples 3/4/10/19. |

---

### **S3 — Strict Rules** (rendered lines 8–14, ~80 tok)
> *Output ONLY valid JSON. No text before or after. Never explain reasoning…*

| # | Rule | Redundant? | Taught where? |
|---|------|------------|---------------|
| 1 | JSON only | **Yes** | Every example is pure JSON |
| 2 | No conversational | **Yes** | Every example is pure JSON |
| 3 | Never explain reasoning | **Yes** | Not shown in any example |
| 4 | Never ask questions | **Yes** | Not shown in any example |
| 5 | Exactly one message_type | **Yes** | All examples show exactly 1 |
| 6 | Empty updates if nothing | **Yes** | NO_UPDATE examples |

**Verdict**: All 6 rules are demonstrated in the examples. Can be dropped entirely.

---

### **S4 — Output Schema** (rendered lines 16–31, ~160 tok)
> *JSON template with message_type and updates array*

| Criterion | Assessment |
|-----------|------------|
| Purpose | Show the exact JSON shape expected |
| Redundant? | **Yes** — every example (20+) uses the exact same shape. A single annotated example is shown later; this bare template adds zero information. |
| Note | Template uses single‑line `context`/`domain`/`keywords` but examples show multi‑line formatting — contradictory. |

---

### **S5 — Field Names Table** (rendered lines 33–39, ~180 tok)

| Field | Purpose |
|-------|---------|
| personas | WHO — LIST |
| problems | THE CORE PROBLEM — LIST |
| current_solutions | HOW PEOPLE COPE — LIST |
| pain_points | FRUSTRATIONS — LIST |
| evidence | OBSERVED PROOF — LIST |
| frequency | HOW OFTEN — SCALAR |

| Criterion | Assessment |
|-----------|------------|
| Redundant? | **Yes** — S6 (Field Definitions) repeats **every** field with more detail + examples. S6 is itself partially redundant with the worked examples. Compact table could be merged into S6 or removed. |
| Tokens | 6 lines of one‑liners that add no unique information. |

---

### **S6 — Field Definitions** (rendered lines 41–92, ~600 tok)

Blocks:
- **Header** (lines 42–43)
- **PERSONAS** (lines 45–49): 5 lines, 4 bullet types
- **PROBLEMS** (lines 51–56): 6 lines
- **CURRENT_SOLUTIONS** (lines 58–62): 5 lines
- **PAIN_POINTS** (lines 64–72): 9 lines
- **EVIDENCE** (lines 74–84): 11 lines
- **FREQUENCY** (lines 86–92): 7 lines

Each block: definition + examples + trigger keywords + ⚠️ what‑not‑to‑do.

| Criterion | Assessment |
|-----------|------------|
| Redundant? | **Partially**. |
| Keyword lists | The 6 `Keywords:` trigger‑word lists have 0% new info — every keyword is a trivial stem of the field name itself (e.g., "students" → personas, "feel" → pain_points). These add noise, not signal. |
| ⚠️ cross‑field warnings | Repeated verbatim in S7 (Critical Differentiation Rules) AND in the worked examples (Block B). |
| Unique content | The conceptual definition per field is useful background, but the examples already demonstrate correct usage. |
| **Recommendation** | Keep a 1‑line definition per field (merge from S5), drop keyword lists, drop ⚠️ warnings. ~80% reduction possible. |

---

### **S7 — Critical Differentiation Rules** (rendered lines 94–130, ~400 tok)

10 rules:

| # | Rule pattern | Redundant? | Taught by Block B example |
|---|--------------|------------|---------------------------|
| 1 | evidence ≠ problem | **Yes** | Example 9 ("I have seen students…") |
| 2 | pain_point ≠ evidence | **Yes** | Example 12 ("They feel stressed.") |
| 3 | problem ≠ solution | **Yes** | Example 11 ("Students usually set reminders.") |
| 4 | persona ≠ problem | **Yes** | Example 15 ("My grandmother struggles…") |
| 5 | frequency ≠ pain_point | **Yes** | Example 13 ("It happens almost every day.") |
| 6 | personal motivation → pain_point | **Yes** | Example 10 ("I want to make people disciplined.") |
| 7 | personal experience → evidence | **Yes** | Example 8 ("This is the most commonly faced problem…") |
| 8 | observation → evidence | **Yes** | Example 9 ("I have seen students…") |
| 9 | ambiguous → NO_UPDATE | **Yes** | Examples 3/4/10/19 |
| 10 | low confidence → NO_UPDATE | **Yes** | (same) |

| Criterion | Assessment |
|-----------|------------|
| Purpose | Explicitly teach field discrimination |
| Redundant? | **Yes** — every single rule has a matching worked example in Block B. Rules 6+7+8+9+10 map 1:1 to specific Block B examples. Rules 1–5 are also demonstrated. |
| Tokens to save | ~400 tok |

---

### **S8 — Operation Rules** (rendered lines 131–136, ~80 tok)

> *ADD for lists (personas, problems, current_solutions, pain_points, evidence). SET for scalar (frequency).*

| Criterion | Assessment |
|-----------|------------|
| Redundant? | **Yes** — every example with frequency uses SET; all others use ADD. No counterexample exists. |
| Taught by later sections? | Examples 2 (SET for frequency), 6 (SET), 13 (SET). |

---

### **S9 — Message Type Definitions** (rendered lines 138–143, ~95 tok)

| Type | When | Example |
|------|------|---------|
| MEANINGFUL | new info | Examples 1/2/5/6/8/9/11–18/20 |
| NO_UPDATE | greetings, noise | Examples 3/10/19 |
| AMBIGUOUS | "yes"/"no" without context | Example 4 |
| END | session end | Example 7 |

| Criterion | Assessment |
|-----------|------------|
| Redundant? | **Yes** — all 4 types are demonstrated with examples. |
| Ambiguity | The AMBIGUOUS vs NO_UPDATE boundary is still blurry even with the definition. |

---

### **S10 — Examples Block A — "COMPREHENSIVE EXAMPLES"** (rendered lines 145–302, ~1,500 tok)

7 examples covering:

| # | User message | Teaches |
|---|------|---------|
| 1 | "i want to build a system which will help old people…" | Multi‑field extraction (personas+problems+pain_points+evidence) |
| 2 | "students mostly use WhatsApp groups…" | current_solutions + frequency (SET) + pain_points |
| 3 | "hello" → NO_UPDATE | Greeting / noise handling |
| 4 | "yes" → AMBIGUOUS | Short ambiguous reply |
| 5 | "they currently use sticky notes…" | current_solutions + pain_points (ineffective workaround) |
| 6 | "it happens every single day…" | frequency only (SET) |
| 7 | "thanks, that's all I needed" → END | Session termination |

| Criterion | Assessment |
|-----------|------------|
| Keep? | **Required** — provides the basic extraction patterns for every field. |

---

### **S11 — Examples Block B — "CRITICAL DISCRIMINATION EXAMPLES"** (rendered lines 303–604, ~2,400 tok)

⚠️ **Contains a BUG at lines 596–604**: orphan JSON fragment after valid closing `}`.

13 examples:

| # | User message | Teaches | Replaces S7 rule? |
|---|------|---------|-------------------|
| 8 | "This is the most commonly faced problem…" | personal experience → evidence/pain_point | ✅ Rule 7 |
| 9 | "I have seen students do late submission…" | observation → evidence | ✅ Rules 1+8 |
| 10 | "I want to make people disciplined." | vague motivation → NO_UPDATE | ✅ Rule 6 |
| 11 | "Students usually set reminders." | current_solution | ✅ Rule 3 |
| 12 | "They feel stressed." | pain_point | ✅ Rule 2 |
| 13 | "It happens almost every day." | frequency (SET) | ✅ Rule 5 |
| 14 | "I interviewed five students…" | evidence+problem extraction | Partial |
| 15 | "My grandmother struggles…" | persona+problem+evidence | ✅ Rule 4 |
| 16 | "Research shows 70% of students…" | evidence (statistics) | Partial |
| 17 | "The problem is late submission…" | problem+current_solution | ✅ Rule 3 |
| 18 | "Forgetting things is stressful." | problem+pain_point split | ✅ Rules 2 |
| 19 | "I want to build an app for this." | NO_UPDATE (builder intent) | ✅ Rule 10 |
| 20 | "Procrastination causes missed deadlines." | problem+evidence split | ✅ Rules 1 |

| Criterion | Assessment |
|-----------|------------|
| Keep? | **Required** — critical boundary‑case training. |
| Redundancy vs S7 | **Every S7 rule maps to at least one Block B example.** Block B makes S7 fully redundant. |
| Bug | Lines 596–604: extra duplicate evidence block after the `}` on line 595. This corrupts the final example shown to the model. |

---

### **S12 — Decorative Separator** (rendered line 606, ~20 tok)

> *━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━*

| Criterion | Assessment |
|-----------|------------|
| Purpose | Visual delimiter for human readers |
| Redundant? | **Yes** — zero functional value for LLM. ~20 tok wasted. |
| Count | 79 separators spread across the prompt → ~160 tok total (~3% of prompt). |

---

### **S13–S16 — Dynamic Sections** (rendered lines 608–624, ~190 tok combined)

| Section | Content | Keep? |
|---------|---------|-------|
| S13 — Current Project State | `{project_state}` JSON | **Required** |
| S14 — Previous Context | `{previous_context}` | **Required** |
| S15 — User Message | `USER MESSAGE: "{user_message}"` | **Required** |
| S16 — Output instruction | `OUTPUT (JSON only):` | **Required** |

---

## Summary: Classification × Redundancy

| # | Section | Lines | Est. tok | Classification | Reason |
|---|---------|-------|----------|----------------|--------|
| S1 | System Identity | 1 | 20 | **Redundant** | Artifact; wrong "above"; role is implicit |
| S2 | Conservatism Warning | 3–6 | 45 | **Redundant** | Taught by S3,S9, and 5+ examples |
| S3 | Strict Rules (6 rules) | 8–14 | 80 | **Redundant** | Every rule demonstrated in examples |
| S4 | Output Schema | 16–31 | 160 | **Redundant** | Identical shape in all 20+ examples |
| S5 | Field Names Table | 33–39 | 180 | **Redundant** | Subsumed by S6; S6 is itself partially redundant |
| S6 | Field Definitions | 41–92 | 600 | **Helpful** | Provides conceptual framing; keyword lists and ⚠️ warnings are redundant |
| S7 | Critical Differentiation Rules | 94–130 | 400 | **Redundant** | 10/10 rules mapped 1:1 to Block B examples |
| S8 | Operation Rules | 131–136 | 80 | **Redundant** | ADD vs SET taught in 3+ examples |
| S9 | Message Type Definitions | 138–143 | 95 | **Redundant** | 4/4 types demonstrated |
| S10 | Block A (7 examples) | 145–302 | 1,500 | **Required** | Core extraction patterns |
| S11 | Block B (13 examples) | 303–604 | 2,400 | **Required** | Boundary cases; ⚠️ bug at lines 596–604 |
| S12 | Separators | ~9× | ~160 | **Redundant** | Zero functional value |
| S13 | Project State | 608–616 | ~90 | **Required** | Dynamic context |
| S14 | Previous Context | 618–619 | ~70 | **Required** | Dynamic context |
| S15 | User Message | 621–622 | ~30 | **Required** | Input |
| S16 | Output instruction | 624 | ~5 | **Required** | Final delimiter |

---

## Classes

| Class | Count | Sections |
|-------|-------|----------|
| **Required** | 5 | S10 (Block A), S11 (Block B), S13, S14, S15 |
| **Helpful** | 1 | S6 (Field Definitions) — could be trimmed 80% |
| **Redundant** | 10 | S1, S2, S3, S4, S5, S7, S8, S9, S12, S16 |

- **Redundant tokens**: ~1,720 tok (~30% of total prompt).
- **If S6 is trimmed to 1 line/field**: saves ~480 tok → total savings ~2,200 tok (~39%).
- **If S6 is also removed**: saves ~600 tok → total savings ~2,320 tok (~41%).

---

## Findings

1. **The first 9 sections (S1–S9) are fully redundant.** Block A (7 examples) and Block B (13 examples) collectively demonstrate every rule, distinction, operation, message type, and field. A small model (1B–3B) learns at least as well from examples as from declarative rules.

2. **Bug in final example** (S11, lines 596–604): orphan JSON fragment after valid `}`. Source EXTRACTOR_PROMPT has a copy‑paste error (`memory_extractor.py:1047–1055`) that renders an extra evidence block after the Procrastination example closes. This could confuse the model about valid JSON structure or create an implicit bias toward evidence extraction.

3. **Keyword trigger lists** (in S6) are trivially derivable from field names and add only noise.

4. **Separator lines** are the only purely cosmetic element, but across the prompt they account for ~160 tok (~3%).

5. **Separation of concerns** is backwards: declarative rules (S7) describe distinctions that examples (S11) then re‑teach. If examples must stay, the rules add no information. If rules must stay, the discrimination examples are redundant. Keeping both doubles the token cost with zero benefit.

6. **Ideal slim prompt** = S13 (project state) + S14 (context) + S15 (message) + S16 (output instruction) + trimmed S6 (1‑line field defs) + **fewer, higher‑quality examples** (not 20 — a well‑chosen 8–10 would suffice).
