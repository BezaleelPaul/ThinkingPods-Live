# MemoryExtractor Prompt V2 — Design

**Goal**: Reduce from ~5,659 tok to ≤2,200 tok while preserving accuracy, schema compliance, deterministic behavior, and all 7 regression scenarios.

**Method**: Remove entirely redundant instruction blocks (S1–S9, S12), tighten Examples (S10+S11) from 20 to 8, fix the duplicate-JSON bug.

---

## 1. Prompt Outline

```
[Stringent Instruction Block]   ← ~100 tok — the ONLY static instruction section
  - 1-line system identity
  - 3 strict rules
  - 6 field names (1 line each)
  - 1 operation rule
  - 4 message types (1 line each)

[Example Block]                   ← ~1,440 tok — 8 examples
  - Keep compact JSON (8 fields, no contributing_factors/qualifiers)
  - All 7 regression scenarios covered by ≤2 examples each

[Dynamic Sections]                ← ~200 tok — state + context + message
```

**Total**: ~1,740 tok (3,300 tok below target / 69 % reduction from current).

---

## 2. Section Ordering

| Order | Section | Source | Est. tok | Rationale |
|-------|---------|--------|----------|-----------|
| 1 | System Identity + Conservatism | V2 rewrite | 20 | Minimal role declaration; banner dropped |
| 2 | Strict Rules (3 only) | S3 subset | 40 | Keep what is NOT demonstrable via examples |
| 3 | Field Names | S5 trimmed | 80 | 6 compact one-liners; drop ⚠️ noise |
| 4 | Operation Rule | S8 condensed | 15 | 1 line: "ADD for lists, SET for frequency" |
| 5 | Message Types | S9 trimmed | 40 | 4 types, 1 line each, no prose |
| 6 | Examples (8) | S10+S11 selected | 1,440 | See §5 |
| 7 | Dynamic Sections | S13–S16 | 200 | Unchanged: state, context, message, output marker |

**Rationale**: Instructions first (set the frame), then examples (demonstrate the frame), then dynamic context (fill in the blanks). Current order (identity→banner→rules→schema→definitions→rules→rules→types→examples) front-loads too much redundant instruction before the model sees any data.

---

## 3. Rules to Keep

| Rule | Tokens | Why kept |
|------|--------|----------|
| Output ONLY valid JSON. No text before or after. | ~15 | Required for deterministic parsing. Not guaranteed by examples alone. |
| Return exactly one message_type value. | ~10 | Required for schema compliance. An example could imply it, but explicit rule is cheap insurance. |
| Return empty updates array if nothing to extract. | ~15 | Enforces the NO_UPDATE contract. Examples show it but rule prevents drift. |
| ADD for list fields, SET only for frequency. | ~15 | One compact line replaces S8. Essential for deterministic operation selection. |

**Total**: ~55 tok (vs. ~375 tok in S3+S8)

---

## 4. Rules to Remove

| Rule (current) | Source | Tokens saved | Why removed | Why safe |
|----------------|--------|-------------|-------------|----------|
| System Identity (full) | S1 | 20 | Artifact text ("entities above") wrong; role is obvious | Every example shows extraction task |
| Conservatism Warning (banner) | S2 | 45 | Banner is visual noise; concept has no enforcement mechanism | Examples teach conservatism implicitly |
| Never produce conversational responses | S3:R2 | ~12 | No example violates it | Format constraint enforced by "JSON only" rule + all examples |
| Never explain reasoning | S3:R3 | ~12 | No example violates it | Output format rule prevents reasoning text |
| Never ask questions | S3:R4 | ~12 | No example violates it | Covered by "JSON only" rule |
| Output Schema (JSON template) | S4 | 160 | Every example already matches the schema exactly | Examples ARE the schema |
| Field Names Table (warnings) | S5 ☑️ warnings | 100 | ⚠️ lines repeat S7 rules | Warnings are taught by examples |
| Field Definitions (keyword lists) | S6 keywords | 200 | Trivially derivable from field names | "students" → personas, "feel" → pain_points — no model needs this |
| Field Definitions (⚠️ warnings) | S6 warnings | 200 | Duplicate S7 | S7 is itself being removed; examples suffice |
| Field Definitions (examples) | S6 examples | 200 | Duplicate S10/S11 | Block A+B cover every field |
| Critical Differentiation Rules | S7 | 400 | Every rule maps 1:1 to an example in Block B | Example coverage analysis shows 10/10 covered |
| Message Type Definitions (prose) | S9 prose | 55 | 4 types can be listed in 40 tok instead of 95 | Compact listing suffices |
| Decorative separators | S12 | 160 | Zero functional value | No model behavior depends on `━━━━` lines |
| Output marker ("OUTPUT (JSON only):") | S16 | 5 | Already said in rule #1 | Redundant after strict rule |
| contributing_factors, qualifiers (in examples) | Many examples | ~400 | Not in Output Schema; add noise | Schema S4 never includes these fields — examples should match schema |

**Total removal**: ~1,820 tok from instruction blocks + ~400 tok from example noise = ~2,220 tok saved.

---

## 5. Examples to Keep (8 of 20)

### Selection criteria
1. Cover all 7 regression scenarios (R1–R7).
2. No two examples teach the same extraction pattern.
3. Compact user messages (≤10 words).
4. JSON outputs use EXACTLY the 8 schema fields (no `contributing_factors`, no `qualifiers`).
5. Include one multi-example compact block for NO_UPDATE / AMBIGUOUS / END (merge Ex#3 + Ex#4 + Ex#7 into ~60 tok).

| # | User message (style) | Extractions | Updates | Regressions | Replaces current |
|---|---------------------|-------------|---------|-------------|-----------------|
| 1 | Multi-claim (persona + problem + pain + evidence) | personas, problems, pain_points, evidence | 4 | Foundation | Ex#1 (shorter message, fewer tokens) |
| 2 | "students mostly use WhatsApp groups but messages get buried and it's a weekly thing" | current_solutions, pain_points, frequency(SET) | 3 | R5 | Ex#2 (unchanged, it's an efficient multi-pattern example) |
| 3 | Personal experience → evidence + pain_point | evidence, pain_points | 2 | R1 | Ex#8 (shorter user message) |
| 4 | Vague builder motivation → NO_UPDATE | — | 0 | R3, R6 | Ex#10 |
| 5 | Emotional statement → pain_point only | pain_points | 1 | R4 | Ex#12 |
| 6 | Problem + evidence split in same sentence | problems, evidence | 2 | R7 | Ex#20 (shortened, and BUG FIXED) |
| 7 | Compact block: "hello"→NO_UPDATE / "yes"→AMBIGUOUS / "thanks bye"→END | — | 0 | Edge cases | Ex#3 + Ex#4 + Ex#7 in 3 mini-examples (~60 tok total) |
| 8 | Problem + solution distinction | problems, current_solutions | 2 | — | Ex#17 (shorter message) |

### Token estimate (examples)

| Example | Est. tok |
|---------|----------|
| #1 — Multi-field (4 updates) | ~260 |
| #2 — Solutions+freq (3 updates) | ~220 |
| #3 — Personal experience (2 updates) | ~180 |
| #4 — NO_UPDATE (vague) | ~40 |
| #5 — Pain point (1 update) | ~80 |
| #6 — Problem+evidence (2 updates) | ~180 |
| #7 — Compact: NO/U + AMB + END | ~60 |
| #8 — Problem+solution (2 updates) | ~180 |
| Separators | ~40 |
| **Total examples** | **~1,240** |

---

## 6. Examples to Remove (12 of 20)

| Removed example | Reason |
|-----------------|--------|
| Ex#3 "hello" | Merged into compact block #7 |
| Ex#4 "yes" | Merged into compact block #7 |
| Ex#5 "sticky notes and phone alarms" | Behaviorually same as Ex#2 (ineffective solution→pain_point) |
| Ex#6 "every single day" | Behaviorually same as Ex#2 (frequency via SET) |
| Ex#7 "thanks, that's all" | Merged into compact block #7 |
| Ex#9 "I have seen students do late submission" | Behaviorally same as Ex#8 (observation→evidence). Ex#8 also shows pain_point split, making it more informative |
| Ex#11 "Students usually set reminders" | Single-field extraction taught by every MEANINGFUL example |
| Ex#13 "It happens almost every day" | Redundant with Ex#2's frequency extraction |
| Ex#14 "I interviewed five students" | Evidence+problem split already covered by Ex#6 |
| Ex#15 "My grandmother struggles" | Overlap with Ex#1 (both elderly, medication, family observation) |
| Ex#16 "Research shows 70%" | Single-field evidence; taught by Ex#3 |
| Ex#19 "I want to build an app" | Behaviorally same as Ex#10 (vague→NO_UPDATE). Ex#10 is the more informative of the two |

**Net**: 20 → 8 examples.

---

## 7. Estimated Final Token Count

| Component | Current (tok) | V2 (tok) | Savings |
|-----------|--------------|----------|---------|
| Instruction blocks | 2,244 | 195 | 2,049 |
| Examples (static) | 3,391 | 1,240 | 2,151 |
| Dynamic sections | 374 | 374 | 0 |
| **Total** | **~5,659** | **~1,809** | **3,850 (68%)** |

**Safety margin**: 391 tok below the 2,200 ceiling.

If coverage concerns arise, up to ~300 tok of headroom exists to add:
- One more multi-field example (~200 tok)
- Or re-add the Critical Differentiation Rules in compact form (~100 tok for 10 rules at 1 line each)

---

## 8. Risks Introduced

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **R1: Reduced accuracy on personal experience → evidence** | Low | Medium | Ex#3 (V2) directly addresses this. Ex#6 (problem+evidence) reinforces the split. If accuracy drops, add a second personal-experience example. |
| **R2: Confusion on AMBIGUOUS vs NO_UPDATE boundary** | Low | Low | Compact block #7 shows both. The definition in instructions (4 types, 1 line) clarifies. |
| **R3: Model starts adding `contributing_factors` and `qualifiers`** | Very low | Low | No V2 example shows these fields. The strict rule "Output ONLY valid JSON" + examples matching the 8-field schema should extinguish them in 1–2 turns. |
| **R4: Loss of conservatism leads to hallucinated extractions** | Medium | High | This is the primary risk. The current prompt beats conservatism via repetition (4 locations). V2 relies on the "return empty if nothing" rule + NO_UPDATE examples. If the model becomes overly aggressive, re-add a 1-line conservatism instruction before the examples. |
| **R5: Reduced accuracy on infrequent edge cases** | Low | Medium | Removed examples #9 (observation→evidence), #14 (interview→evidence+problem), #16 (statistic→evidence) cover edge variants. If these specific patterns fail, they can be restored as a 3-example "edge cases" block (~400 tok) while staying under 2,200. |
| **R6: Model produces `operation: ADD` on `frequency`** | Very low | Low | Single operation rule explicitly forbids this. Ex#2 shows SET on frequency. No V2 example shows ADD on frequency. |
| **R7: BUG re-introduced (duplicate JSON in example)** | N/A | N/A | V2 explicitly excludes the corrupted Procrastination example. All 8 V2 examples have clean single closure. |

### Risk register summary

| Risk | Mitigation | Contingency |
|------|-----------|-------------|
| R4 — Hallucinated extractions | 1-line conservatism instruction + strict rule | Add "IF UNSURE (<80%) → NO_UPDATE" banner (~45 tok) back |
| R5 — Edge case accuracy | 8 well-chosen examples cover all regression scenarios | Add 3 edge examples back (~400 tok), reduce other example verbosity |
| R1+R2+R3+R6+R7 | All structurally prevented by V2 design | None needed |

---

## 9. Additional Design Notes

### Bug fix (source code)
`memory_extractor.py:1047–1055` contains orphan JSON fields appended after the Procrastination example's closing `}`. V2 omits this example entirely, so the bug is automatically fixed. If the Procrastination example is kept in any form, the source must be corrected to remove the duplicate at lines 1047–1055.

### Example compression strategy
- All examples use exactly the 8 fields from the Output Schema: `operation`, `field`, `value`, `context`, `domain`, `keywords`, `raw_text`, `extraction_method`.
- `context` and `raw_text` should be identical (or `context` slightly more descriptive) to reduce cognitive load — current examples sometimes make them contradictory.
- `keywords` should be 1–3 items (not 3–6 as in some current examples).
- User messages ≤10 words where possible.
- Multi-example compact block (NO_UPDATE + AMBIGUOUS + END) is a single JSON-less example triad in ~60 tok.

### Prompt template string
The new `EXTRACTOR_PROMPT` must use `str.format()` with the same three placeholders (`{project_state}`, `{previous_context}`, `{user_message}`). No changes to the shell code in `memory_extractor.py:439–447`.

### Determinism guarantee
Determinism is preserved because:
- The "JSON only" strict rule forces parseable output.
- The single message_type rule forces unambiguous classification.
- ADD/SET rule forces correct operation.
- Examples demonstrate the exact expected output shape.
- No free-text reasoning or conversational output is shown or permitted.

### Comparison with current prompt

| Dimension | Current (v1) | V2 |
|-----------|-------------|-----|
| Instruction blocks | 10 | 1 (compacted) |
| Examples | 20 | 8 |
| Tokens | ~5,659 | ~1,809 |
| Rules repeated 3×+ | 6 | 0 |
| Contradictory instructions | 1 (personal experience mapping) | 0 |
| Decorative tokens | ~160 | ~40 |
| Bugs | 1 (duplicate Procrastination JSON) | 0 |
| Schema fields in examples | 10 (extras: contributing_factors, qualifiers) | 8 (schema-exact) |
| Regression coverage | 7/7 | 7/7 |
