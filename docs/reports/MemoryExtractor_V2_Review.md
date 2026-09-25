# MemoryExtractor V2 Design — Critical Review

**Review of**: `MemoryExtractor_V2_Design.md`  
**Scope**: Structural completeness, example coverage, regression risks, instruction clarity, schema consistency.

---

## Findings

### F1 — R2 (Observation → Evidence) Uncovered

**Summary**: V2 claims "All 7 regression scenarios covered by ≤2 examples each" but R2 (observation→evidence) has no dedicated example.

**Evidence**:
- V2 Ex#3 ("Personal experience → evidence + pain_point") is based on current Ex#8 ("This is the most commonly faced problem even I have faced it"). This teaches: personal experience → evidence **AND** pain_point.
- R2 requires: third-party observation → evidence **ONLY** (no pain_point). The only training for this is current Ex#9 ("I have seen students do late submission and lose marks."), which V2 removes.
- Personal experience and third-party observation are syntactically different ("I have faced it" vs "I have seen them do it"). A model trained only on personal experience examples will learn to always add pain_point alongside evidence, producing false pain_points for pure observation statements.

| Severity | **HIGH** — Definite regression for R2 |
|----------|----------------------------------------|
| Recommendation | Restore Ex#9 (or a shortened variant) as a new V2 example. Fits in ~150 tok. Remove the comment "Behaviorally same as Ex#8" from the rationale — they teach different outputs (evidence-only vs evidence+paint_point). |

---

### F2 — Problem + Pain_Point Split Missing

**Summary**: V2 Ex#6 teaches problem+evidence split but NOT problem+pain_point split. These are distinct extraction patterns.

**Evidence**:
- V2 Ex#6: "Problem + evidence split in same sentence" (based on Ex#20 "Procrastination causes missed deadlines" → problems + evidence).
- Current Ex#18 "Forgetting things is stressful" → problems + pain_points. Removed in V2.
- The structural difference: "X causes Y" (causal chain → problem + evidence) vs "X is stressful" (emotional qualifier → problem + pain_point). They teach different extraction rules.
- V2 labels Ex#6 as covering "R7 (Multi-field split)" but R7 encompasses both problem+evidence AND problem+pain_point. V2 only covers one.

| Severity | **HIGH** — Extraction failure on common pattern (emotion + problem in one sentence) |
|----------|-----------------------------------------------------------|
| Recommendation | Add Ex#18 (or a shortened variant) as V2 Ex#9. ~180 tok, within the 391 tok headroom. Alternatively, replace V2 Ex#6 with a combined example that shows BOTH split patterns in the same message (e.g., "Procrastination causes stress and missed deadlines" → problem + pain_point + evidence). |

---

### F3 — Multi-Update Bias Risk

**Summary**: Of 5 MEANINGFUL V2 examples, 4 have 2+ updates and 1 has 1 update. The model may learn to always produce multiple updates.

**Evidence**:

| V2 Ex | Updates | Type |
|-------|---------|------|
| #1 | 4 | Multi-field |
| #2 | 3 | Multi-field + frequency |
| #3 | 2 | Personal experience |
| #5 | 1 | Single emotional pain_point |
| #6 | 2 | Problem+evidence |
| #8 | 2 | Problem+solution |

Ratio: 5 multi-update : 1 single-update (among MEANINGFUL).

Risk: Simple single-claim messages ("Students miss deadlines") may receive hallucinated extra fields because the prompt's examples never show a bare 1-update extraction for anything other than pure emotion.

| Severity | **MEDIUM** — Over-extraction; accuracy loss on simple statements |
|----------|------------------------------------------------------------------|
| Recommendation | Keep one additional single-field example. Options: (a) one-update evidence extraction (Ex#16 "Research shows 70%..."), (b) one-update current_solution (Ex#11 "Students usually set reminders"). ~80-150 tok within headroom. |

---

### F4 — Research / Data / Interview → Evidence Untaught

**Summary**: All evidence-as-data patterns removed. Model sees only personal experience as evidence training.

**Removed**:

| Current Ex | Pattern | Removed rationale |
|-----------|---------|-------------------|
| Ex#9 | "I have seen..." → evidence | "Behaviorally same as Ex#8" — but it's observation-only, not personal experience |
| Ex#14 | "I interviewed five students..." → evidence + problem | "Covered by Ex#6" — but Ex#6 teaches causal consequence, not interview data |
| Ex#16 | "Research shows 70%..." → evidence | "Taught by Ex#3" — but Ex#3 teaches personal experience, not research citation |

Three distinct evidence subtypes are collapsed into one: personal narrative/exemplar. The syntactically different patterns (research citation, interview framing, third-party witnessing) have zero representation.

| Severity | **MEDIUM** — Pattern-specific failure on research/data statements |
|----------|---------------------------------------------------------------|
| Recommendation | Restore ONE evidence-diversity example. Best candidate: Ex#16 ("Research shows 70% of students miss deadlines") at ~80 tok (single-update). This teaches research-statistic-as-evidence and adds a one-update MEANINGFUL example (mitigating F3). |

---

### F5 — Builder Intent (R6) Conflated with Vague Motivation (R3)

**Summary**: V2 Ex#4 claims to cover both R3 (vague personal goal) and R6 (builder intent) but only shows vague personal goal.

**Evidence**:
- V2 Ex#4: "Vague builder motivation → NO_UPDATE" based on Ex#10 "I want to make people disciplined."
- Ex#19 "I want to build an app for this." → NO_UPDATE is removed as "behaviorally the same."
- Syntax difference: "I want to make X (happen)" vs "I want to build an app (for this)."
- Both produce NO_UPDATE, so functional behavior IS the same. But the model's NO_UPDATE trigger relies on generalizing from one "I want to..." pattern to another. This is likely safe but untestable without a regression check.

| Severity | **LOW** — Both patterns generalize to NO_UPDATE via "I want to..." + lack of specifics |
|----------|----------------------------------------------------------------------------------------|
| Recommendation | Accept as-is. If R6 regresses, add a 1-line compact example (~20 tok): `"I want to build an app" → NO_UPDATE`. |

---

### F6 — "3 Strict Rules" Under-Specified

**Summary**: V2 §3 lists 4 items in the rules table but the outline in §1 says "3 strict rules." Internal inconsistency + missing rule specification.

**Issue**: The "3 strict rules" are never enumerated. The table shows:
1. Output ONLY valid JSON
2. Return exactly one message_type
3. Return empty updates array
4. ADD for lists, SET for frequency (labeled "operation rule," not strict rule)

Counting: 3 strict + 1 operation = 4 total. The outline's "3 strict rules" description is inaccurate. The implementer cannot determine which 3 are intended.

| Severity | **MEDIUM** — Cannot implement accurately; design must specify |
|----------|---------------------------------------------------------------|
| Recommendation | Clarify: rename outline to "4 rules" or explicitly say "3 strict rules + 1 operation rule." List all 4 in the outline section to match the table. |

---

### F7 — Field Names Table Dropped ⚠️ Warnings Without Replacement

**Summary**: V2 removes the ⚠️ lines from the Field Names Table (S5), attributing them to the (also removed) S7 rules. No replacement teaches the critical "what-not-to-extract" boundaries.

**Removed warnings**:
- "⚠️ NOT a problem, NOT a solution, NOT evidence" (personas)
- "⚠️ NOT a solution, NOT evidence, NOT a pain point" (problems)
- ... etc. for all 6 fields

**Current behavior**: These 12+ warnings in S5+S6+S7 teach the model what NOT to extract for each field. V2 relies entirely on examples to teach these boundaries. With examples reduced from 20 to 8, the model has fewer opportunities to learn negative patterns (what not to extract).

For example: the model must learn that "students" → persona, NOT problem, NOT solution. In V2, only 8 examples total, with most showing concurrent field extractions where field boundaries are clear. The negative boundary training is implicit.

| Severity | **LOW-MEDIUM** — Cross-field confusion risk increases proportionally with example count reduction |
|----------|----------------------------------------------------------------------------------------------------|
| Recommendation | Either (a) keep a 1-line "cross-field cheat sheet" (~30 tok): "Evidence ≠ problem. Pain_point ≠ evidence. Problem ≠ solution. Persona ≠ problem." or (b) accept the risk and monitor; add back only if regression appears. Headroom allows (a). |

---

### F8 — Conservatism Collapses From 4 Mentions to 1

**Summary**: Current conservatism instruction appears in 4 locations (S2 banner, S7 rule 9, Ex#10, Ex#19). V2 reduces to a 5-word phrase in the identity line + the "return empty" rule.

**V2 identity line** (est.): "You are a semantic extraction unit. Be conservative: if unsure, return NO_UPDATE."

The current "IF UNSURE (< 80% CONFIDENCE)" criterion with a numeric threshold is removed entirely. The V2 version is a qualitative instruction with no yardstick. A 1B–3B parameter model may interpret "unsure" differently than the designer intends.

| Severity | **MEDIUM** — Primary risk identified in V2 design's own risk register (R4). Threshold removal makes the guardrail harder to enforce. |
|----------|--------------------------------------------------------------------------------------------------------------------------------------|
| Recommendation | Either (a) keep the numeric threshold: "If <80% confident, return NO_UPDATE" (~15 tok, trivially within headroom) or (b) accept and plan to add it back as the V2 design's contingency plan says. Recommend (a) for safety. |

---

### F9 — AMBIGUOUS Training Relies on Single `"yes"` Examples

**Summary**: The only AMBIGUOUS training is `"yes" → AMBIGUOUS` in the compact block. Other common ambiguous patterns are absent.

**Current examples removed**:
- S7 rule 9 explicitly lists: `"I want to help"`, `"This is a problem"`, `"It's difficult"`, `"Make it better"` → AMBIGUOUS/NO_UPDATE.
- V2 removes S7 entirely. None of these patterns appear in any V2 example.

The AMBIGUOUS vs NO_UPDATE boundary is already the most blurred distinction in V1 (current S9 defines them with overlapping language). V2 makes it worse by showing zero examples of the "this is a problem" / "it's difficult" / "I want to help" pattern.

| Severity | **LOW** — AMBIGUOUS is rarely used in practice; consequences of wrong classification are low (next turn corrects it) |
|----------|----------------------------------------------------------------------------------------------------------------------|
| Recommendation | If headroom allows, add one ambiguous expression as a 1-line example in the compact block (~15 tok): `"This is a problem" → AMBIGUOUS`. Otherwise accept the risk. |

---

## Cross-Reference: 7 Regression Scenarios × V2 Coverage

| Reg. | Pattern | Current Ex (best) | V2 Ex | Coverage | Status |
|------|---------|-------------------|-------|----------|--------|
| R1 | Personal experience → evidence + pain_point (NOT problem) | Ex#8 | #3 | Full | ✅ |
| R2 | Observation → evidence ONLY | Ex#9 | **None** | **None** | **❌ MISSING** |
| R3 | Vague motivation → NO_UPDATE | Ex#10 | #4 | Indirect (generalizes from Ex#10) | ⚠️ Partial |
| R4 | Emotional state → pain_point (NOT problem/evidence) | Ex#12 | #5 | Full | ✅ |
| R5 | Frequency → SET operation | Ex#2, #6, #13 | #2 | Full | ✅ |
| R6 | Builder intent → NO_UPDATE | Ex#19 | #4 | Indirect (generalizes from vague motivation) | ⚠️ Partial |
| R7 | Multi-field split (same sentence → 2+ fields) | Ex#18, #20 | #6 | **Partial** — problem+evidence only, missing problem+pain_point | **❌ PARTIAL** |

**2 regressions fully uncovered, 2 partially covered.**

---

## Cross-Reference: All 15 Extraction Behaviors × V2 Coverage

| Behavior | Current | V2 | Status |
|----------|---------|----|--------|
| 1. Multi-field extraction (4+ fields) | Ex#1, #15 | #1 | ✅ |
| 2. Current solution extraction | Ex#2, #5, #11 | #2, #8 | ✅ |
| 3. Frequency → SET | Ex#2, #6, #13 | #2 | ✅ |
| 4. "But" clause → pain_point | Ex#2, #5 | #2 | ✅ |
| 5. Greeting → NO_UPDATE | Ex#3 | #7 | ✅ |
| 6. Short ambiguous → AMBIGUOUS | Ex#4 | #7 | ✅ |
| 7. Session end → END | Ex#7 | #7 | ✅ |
| 8. Personal experience → evidence + pain_point | Ex#8 | #3 | ✅ |
| 9. Observation → evidence ONLY | Ex#9 | **None** | **❌** |
| 10. Vague motivation → NO_UPDATE | Ex#10 | #4 | ✅ |
| 11. Single-field emotional → pain_point | Ex#12 | #5 | ✅ |
| 12. Interview/research → evidence | Ex#14, #16 | **None** | **❌** |
| 13. Semantic extraction (not keyword) | Ex#17 | #8 | ✅ |
| 14. Problem+pain_point split | Ex#18 | **None** | **❌** |
| 15. Builder intent → NO_UPDATE | Ex#19 | #4 (indirect) | ⚠️ |

**3 fully missing, 1 partially missing, 11 covered.**

---

## Revised Token Estimate With Fixes

To fix F1 (R2), F2 (R18), F3 (single-field), and F4 (research evidence):

| Fix | Tokens | Description |
|-----|--------|-------------|
| Restore Ex#9 variant | ~120 | Third-party observation → evidence only |
| Restore Ex#18 variant | ~180 | Problem + pain_point split |
| Restore Ex#16 variant | ~80 | Research citation → evidence (also fixes single-field gap) |
| Conservancy threshold | ~15 | "If <80% confident, return NO_UPDATE" |
| Cross-field cheat sheet | ~30 | 4 boundary rules, 1 line each |
| **Additional subtotal** | **~425** | |
| Current V2 total | ~1,809 | |
| **Revised total** | **~2,234** | ~34 tok over 2,200 ceiling |

If the ceiling is strict, trade-offs:
- Drop Ex#16 variant (-80 tok) → ~2,154 tok ✅ (accept F4 risk)
- Drop cross-field cheat sheet (-30 tok) → ~2,204 tok (on the boundary)

Neither is ideal but workable.

---

## Summary Table

| # | Issue | Severity | V2 § | Fix cost |
|---|-------|----------|------|----------|
| F1 | R2 (observation→evidence) uncovered | **HIGH** | §5 | ~120 tok |
| F2 | Problem+pain_point split missing | **HIGH** | §5 | ~180 tok |
| F3 | Multi-update bias risk | MEDIUM | §5 | ~80 tok (via F4) |
| F4 | Research/data evidence untaught | MEDIUM | §5 | ~80 tok |
| F5 | Builder intent conflated with vague motivation | LOW | §5 | Accept |
| F6 | "3 strict rules" under-specified | MEDIUM | §1, §3 | Clarify text |
| F7 | Cross-field warnings dropped without replacement | LOW-MEDIUM | §4 | ~30 tok |
| F8 | Conservatism threshold removed | MEDIUM | §1 | ~15 tok |
| F9 | AMBIGUOUS training minimal | LOW | §5 | ~15 tok |

---

## Conclusion

The V2 design achieves its token reduction goal but has **2 HIGH-severity gaps** (R2 uncovered, problem+pain_point split missing) that will cause measurable regression on specific input patterns. The underlying cause is over-zealous deduplication — personal experience, third-party observation, and research citation produce different outputs but were collapsed into one "evidence" category.

The design's 391 tok headroom is sufficient to fix all HIGH and MEDIUM issues (~425 tok total), putting the final prompt at ~2,234 tok — 34 tok over the 2,200 ceiling. A one-example sacrifice (e.g., restoring Ex#16 but not the cross-field cheat sheet) gets it to ~2,154 tok with no regression.

**Without these fixes, the V2 prompt will regress on at least 2 of the 7 regression scenarios and will fail to extract common patterns (emotion + problem in one sentence, third-party observation, research/data statements).**
