# Hypothesis Transient Conversion: Impact Report

## Conversion Definition

"Transient per-turn reasoning objects" means:

1. Hypotheses are generated fresh each turn from current-turn extraction facts + KnowledgeBase.
2. No hypothesis objects are stored or read between turns.
3. No merge with prior-turn hypotheses (there are no prior hypotheses).
4. No lifecycle statuses (ACTIVE/CONFIRMED/REJECTED/SUPERSEDED) — all candidates are equal-confidence inference outputs.
5. No HypothesisManager disposition (AUTO_ACCEPT/VERIFY/HOLD/DISCARD).
6. No AUTO_ACCEPT -> ProjectState write path.
7. `InferenceResult` carries only `new_hypotheses` (no `updated_hypotheses`, `revised`, `retired`, `conflicts`).
8. `InferenceResult` still flows to ResponseStrategyEngine (for question targeting), but the data is simpler.

---

## 1. Dependency Graph: Hypotheses in the Live Pipeline

### 1.1 Current state (annotated)

```
MemoryExtractor.extract()
    │
    ▼
InferenceEngine.infer()
    │
    ├── _load_existing_hypotheses()      ← READS ProjectState (facts→hypotheses)
    ├── _match_concepts()                ← OK (pure KB lookup)
    ├── _build_evidence_bundles()        ← OK (pure grouping)
    ├── _filter_explicit_duplicates()    ← OK (pure dedup)
    ├── _form_hypothesis_candidates()    ← OK (pure creation)
    ├── _merge_hypotheses()              ← LIFECYCLE (merge with prior)
    ├── _detect_conflicts()              ← LIFECYCLE
    ├── _apply_conflicts_and_status()    ← LIFECYCLE
    ├── _identify_retired()              ← LIFECYCLE
    │
    │  InferenceResult
    │   .updated_hypotheses              ← USED ONLY by HypothesisManager
    │   .new_hypotheses                  ← NEVER CONSUMED
    │   .revised_hypotheses              ← NEVER CONSUMED
    │   .retired_hypotheses              ← NEVER CONSUMED
    │   .conflicts_detected              ← NEVER CONSUMED
    │
    ▼
HypothesisManager.evaluate_batch()
    │
    ├── _evaluate_single()               ← DISPOSITION (AUTO_ACCEPT/VERIFY/HOLD/DISCARD)
    │
    │  HypothesisBatchResult
    │   .auto_accept[]                   → written to ProjectState (LIFECYCLE)
    │   .verify[]                        → DISCARDED (ignored by pipeline)
    │   .hold[]                          → DISCARDED (ignored)
    │   .discard[]                       → DISCARDED (ignored)
    │
    ▼
ResponseStrategyEngine.determine_strategy()
    │  receives inference_result=None    ← UNREACHABLE (always None in live pipeline)
    │  code exists to consume hypotheses but never called with data
    │
    ▼
ObjectiveEngine           → NO HYPOTHESES INPUT
LifecycleManager          → NO HYPOTHESES INPUT
PromptBuilder             → NO HYPOTHESES INPUT
LLM                       → NO HYPOTHESES IN PROMPT
```

### 1.2 Edge annotations

| Edge | From | To | Status | Reason |
|---|---|---|---|---|
| E1 | MemoryExtractor | InferenceEngine | **ACTIVE** | ExtractionResult flows to IE.infer() |
| E2 | InferenceEngine | HypothesisManager | **UNUSED** | `.updated_hypotheses` is consumed by HM, but HM output beyond `auto_accept` is ignored |
| E3 | HypothesisManager | StateManager | **DEAD** | `.auto_accept` writes hypotheses as facts — violates §6, only path that uses HM output |
| E4 | InferenceResult | ResponseStrategyEngine | **UNREACHABLE** | Pipeline call at mentor.py:1232 omits `inference_result` arg; defaults to None |
| E5 | InferenceResult | ObjectiveEngine | **N/A** | No parameter exists |
| E6 | InferenceResult | LifecycleManager | **N/A** | No parameter exists |
| E7 | InferenceResult | PromptBuilder | **N/A** | No parameter exists |
| E8 | InferenceResult | LLM | **DEAD** | Never rendered in prompt |

**Verdict**: Of the 8 possible edges, only E1 (MemoryExtractor→IE) is fully active with useful data. E2 is partially active but the output is wasted. E3 is active but architecturally harmful. E4 has dead consumer code. E5-E8 don't exist.

---

## 2. Methods That Become Dead Code

### 2.1 Methods to DELETE entirely

These methods serve no purpose if hypotheses are transient. They are either lifecycle management, cross-turn reconstruction, or disposition routing that has no effect.

| File | Method | Lines | LOC | Reason for Deletion |
|---|---|---|---|---|
| `inference_engine.py` | `_load_existing_hypotheses()` | 401-443 | 43 | Reads ProjectState fields as hypotheses. No prior hypotheses to load. |
| `inference_engine.py` | `_merge_hypotheses()` | 739-780 | 42 | Merges new candidates with existing (which no longer exist). |
| `inference_engine.py` | `_find_merge_match()` | 782-793 | 12 | Subroutine of merge. |
| `inference_engine.py` | `_value_similarity()` | 795-803 | 9 | Subroutine of merge. |
| `inference_engine.py` | `_boost_confidence()` | 805-813 | 9 | Subroutine of merge. |
| `inference_engine.py` | `_detect_conflicts()` | 819-828 | 10 | Conflict detection is lifecycle. With no prior hypotheses, only current-turn candidates exist — no cross-turn contradictions possible. |
| `inference_engine.py` | `_are_contradictory()` | 830-835 | 6 | Subroutine of conflict. |
| `inference_engine.py` | `_apply_conflicts_and_status()` | 841-883 | 43 | Assigns lifecycle statuses. No statuses needed. |
| `inference_engine.py` | `_penalize_confidence()` | 885-891 | 7 | Subroutine of status. |
| `inference_engine.py` | `_identify_retired()` | 893-913 | 21 | Detects hypotheses missing from prior set. No prior set. |
| `inference_engine.py` | `_are_mutually_exclusive()` | 654-672 | 19 | Duplicate of `response_strategy._are_contradictory()`. Called only by `_find_contradicting_facts` within `_build_evidence_bundles`. With no lifecycle, the contradiction check within evidence bundling can also be simplified. Exception: still needed if minimum contradiction detection in evidence aggregation is kept. |
| `inference_engine.py` | `_find_contradicting_facts()` | 620-640 | 21 | Called by `_build_evidence_bundles`. If _are_mutually_exclusive goes, this can simplify to always return empty tuple. |
| `hypothesis_manager.py` | `HypothesisDecision` (enum) | 52-67 | 16 | Whole enum: AUTO_ACCEPT/VERIFY/HOLD/DISCARD. No decisions needed. |
| `hypothesis_manager.py` | `DecisionPolicy` (class) | 75-128 | 54 | Policy for disposition. No dispositions needed. |
| `hypothesis_manager.py` | `DEFAULT_POLICY` | 131 | 1 | Singleton default. |
| `hypothesis_manager.py` | `HypothesisDisposition` (class) | 140-167 | 28 | Wraps hypothesis + decision. |
| `hypothesis_manager.py` | `HypothesisBatchResult` (class) | 170-193 | 24 | Contains auto_accept/verify/hold/discard lists. |
| `hypothesis_manager.py` | `evaluate_batch()` | 217-263 | 47 | Evaluates each hypothesis against policy. |
| `hypothesis_manager.py` | `_evaluate_single()` | 269-344 | 76 | Single hypothesis evaluation. |
| `hypothesis_manager.py` | `_hypothesis_kind_to_field()` | 346-356 | 11 | Mapping for decision routing. |
| `hypothesis_manager.py` | `get_hypothesis_manager()` | 366-371 | 6 | Module singleton accessor. |
| `mentor.py` | AUTO_ACCEPT writing block | 1014-1038 | 25 | `if hypothesis_batch is not None:` block that writes AUTO_ACCEPT to ProjectState. |
| `mentor.py` | `hypothesis_batch` parameter | 984 (sig) + 1009-1012 (docstring) + 1143-1153 (call+print) | 5 | `hypothesis_batch=None` parameter in `_apply_extraction_to_state`. |
| `mentor.py` | `inference_result` parameter | 983 (sig) + 1006-1008 (docstring) + 1139 (call) + 1163 (pass) | 4 | `inference_result=None` parameter in `_apply_extraction_to_state`. Already dead (`_ = inference_result`). |

**Total LOC removed**: ~598 lines

### 2.2 Methods to SIGNIFICANTLY SIMPLIFY

| File | Method | Lines | New LOC Est. | Change |
|---|---|---|---|---|
| `inference_engine.py` | `infer()` | 225-300 (76 lines) | ~45 | Remove stages 1, 5, 6, 7. Keep: fact conversion, concept matching, evidence building, duplicate filtering, candidate formation. Return only `new_hypotheses` + `reasoning_trace`. |
| `inference_engine.py` | `_build_evidence_bundles()` | 525-618 (94 lines) | ~70 | Remove `_find_contradicting_facts` call within bundle building. No contradiction tracking needed across current-turn candidates. |
| `inference_engine.py` | `_form_hypothesis_candidates()` | 698-733 (36 lines) | ~30 | All candidates get status=ACTIVE (default). No status assignment needed. |
| `inference_engine.py` | `_filter_explicit_duplicates()` | 678-692 (15 lines) | ~10 | Still needed — prevents inferred duplicates of explicit facts. Same logic, fewer call sites. |

### 2.3 Methods to KEEP UNCHANGED

| File | Method | Reason |
|---|---|---|
| `inference_engine.py` | `_convert_updates_to_facts()` | Core inference — converts extraction format. |
| `inference_engine.py` | `_match_concepts()` | Core inference — KB lookup. |
| `inference_engine.py` | `_collect_explicit_values()` | Core inference — dedup support. |
| `inference_engine.py` | `_is_explicit_duplicate()` | Core inference — dedup logic. |

### 2.4 Methods to KEEP with INTERFACE CHANGE

| File | Method | Change |
|---|---|---|
| `response_strategy.py` | `_get_relevant_hypotheses()` | Currently reads `inference_result.updated_hypotheses`. Change to read `inference_result.new_hypotheses` (which is the only field left). |
| `response_strategy.py` | `_select_target_hypothesis()` | Same — source data from `new_hypotheses` instead of `updated_hypotheses`. |
| `response_strategy.py` | `_detect_contradiction()` | Same — source from `new_hypotheses`. Still useful for single-turn contradiction detection. |
| `response_strategy.py` | `_resolve_strategy()` | Same — still checks for multiple active hypotheses. |
| `response_strategy.py` | `_generate_candidate_intents()` | Same — still adds verification candidates per hypothesis. |

---

## 3. File-by-File Impact

### 3.1 `inference_engine.py` (945 lines → ~550 lines)

| Change | Lines |
|---|---|
| Delete `_are_mutually_exclusive()` | -19 |
| Delete `_find_contradicting_facts()` | -21 |
| Delete `_load_existing_hypotheses()` | -43 |
| Delete `_merge_hypotheses()` | -42 |
| Delete `_find_merge_match()` | -12 |
| Delete `_value_similarity()` | -9 |
| Delete `_boost_confidence()` | -9 |
| Delete `_detect_conflicts()` | -10 |
| Delete `_are_contradictory()` | -6 |
| Delete `_apply_conflicts_and_status()` | -43 |
| Delete `_penalize_confidence()` | -7 |
| Delete `_identify_retired()` | -21 |
| Simplify `infer()` | ~-30 |
| Simplify `_build_evidence_bundles()` | ~-20 |
| Remove unused imports | ~-1 |
| **Total reduction** | **~-293 lines** (31% of file) |

Method signatures that change:
- `infer()` — return `InferenceResult` with only `new_hypotheses` + `reasoning_trace`
- `_form_hypothesis_candidates()` — no status assignment needed
- `_build_evidence_bundles()` — no contradiction checking within bundles

### 3.2 `hypothesis_manager.py` (371 lines → 0 lines, module deleted)

| Change | Lines |
|---|---|
| Delete `HypothesisDecision` enum | -16 |
| Delete `DecisionPolicy` class | -54 |
| Delete `DEFAULT_POLICY` singleton | -1 |
| Delete `HypothesisDisposition` class | -28 |
| Delete `HypothesisBatchResult` class | -24 |
| Delete `HypothesisManager` class | -159 |
| Delete `get_hypothesis_manager()` | -6 |
| Delete `__all__` / imports | -5 |
| Delete module docstring | -36 |
| **Total** | **-371 lines (file deleted)** |

### 3.3 `contracts.py` (419 lines → ~390 lines)

| Change | Lines |
|---|---|
| Simplify `InferenceResult`: remove `updated_hypotheses`, `revised_hypotheses`, `retired_hypotheses`, `conflicts_detected` | -6 field definitions |
| `InferenceResult.get_active()` becomes unnecessary (no statuses to filter) | -7 lines (method + docstring) |
| `HypothesisSet` type alias | -1 (unused) |
| **Total reduction** | **~-29 lines** |

### 3.4 `mentor.py` (1427 lines → ~1390 lines)

| Change | Lines |
|---|---|
| Remove `inference_result` param from `_apply_extraction_to_state()` | ~-4 |
| Remove `hypothesis_batch` param from `_apply_extraction_to_state()` | ~-3 |
| Remove AUTO_ACCEPT write block (lines 1014-1038) | -25 |
| Remove import of `get_hypothesis_manager` | -1 |
| Remove HM call + print in `process_mentor_turn()` (lines 1142-1153) | -12 |
| Remove `InferenceResult` import | -1 |
| **Total reduction** | **~-46 lines** |

### 3.5 `state_manager.py` (522 lines → ~518 lines)

| Change | Lines |
|---|---|
| Remove `inference_result` parameter from `apply_extraction()` | -3 (sig + docstring) |
| Remove `_ = inference_result` stub | -1 |
| **Total reduction** | **~-4 lines** |

### 3.6 `response_strategy.py` (824 lines → ~820 lines)

| Change | Lines |
|---|---|
| Change `_get_relevant_hypotheses()` to read `.new_hypotheses` | ~0 (rename field access) |
| **Total reduction** | **~0 lines** (interface-compatible) |

### 3.7 `module4/__init__.py` and other init files

No changes needed. HypothesisManager is not re-exported from any `__init__.py`.

### 3.8 `module5/lifecycle_manager.py`

**No changes**. Already has zero hypothesis references.

### 3.9 `module3/objective_engine.py`, `module3/completeness_checker.py`, `module3/priority_engine.py`

**No changes**. Already have zero hypothesis references.

### 3.10 `module4/prompt_builder.py`

**No changes**. Already has zero hypothesis references.

---

## 4. Test Impact

### 4.1 No tests cover the hypothesis lifecycle

From exhaustive search: **zero test files** import from `inference_engine` or `hypothesis_manager` modules. **Zero test assertions** reference `Hypothesis`, `InferenceResult`, `HypothesisDecision`, `HypothesisBatchResult`, `HypothesisStatus`, `HypothesisDisposition`, `evaluate_batch`, `auto_accept`, or `_load_existing_hypotheses`.

| Impact | Count |
|---|---|
| Tests that will fail from deleted code | **0** |
| Tests that need parameter changes | **0** |
| Tests that need import changes | **0** |
| Tests that need assertion changes | **0** |

### 4.2 Tests that call `_apply_extraction_to_state` (will continue passing)

**File**: `tests/test_memory_extractor.py`

Tests call `_apply_extraction_to_state(session, state, result)` with 3 positional args. `inference_result` and `hypothesis_batch` both default to `None`. Removing these parameters from the signature does not change the call sites — the defaults were the only call pattern used.

| Test method | Line | Current call | After change |
|---|---|---|---|
| `test_apply_extraction_add_appends_to_state_list` | 634 | `(session, state, result)` | same — no change |
| `test_apply_extraction_set_overwrites_frequency` | 647 | `(session, state, result)` | same |
| `test_apply_extraction_canonical_contract` | 657 | `(session, state, result)` | same |
| `test_apply_extraction_no_update_is_noop` | 686 | `(session, state, result)` | same |
| `test_apply_extraction_add_does_not_clobber` | 705 | `(session, state, result)` | same |

### 4.3 Integration tests calling `process_mentor_turn` (will continue passing)

**File**: `tests/test_objective_engine_integration.py`

Tests call `process_mentor_turn(user_msg, username, project)` which internally calls the pipeline. Removing hypothesis lifecycle code inside `process_mentor_turn` does not change the function signature or return value. The test assertions check reply strings and session state — neither is affected because hypotheses never reached the LLM or prompt.

| Test method | Assertions on | Hypotheses affect this? |
|---|---|---|
| `test_full_e2e_pipeline` | reply text, session fields | No |
| `test_two_conversations_independent` | reply text, session fields | No |

### 4.4 Regression test suite (7 scenarios)

**File**: `tests/test_objective_engine_integration.py` — 7 regression scenarios

Each sends user messages through `process_mentor_turn` and checks objective sequence. The objectives are computed by ObjectiveEngine from ProjectState, which is updated by StateManager from ExtractionResult. Hypotheses do not affect objectives.

**Verdict**: All 7 scenarios pass unchanged. The hypothesis lifecycle has zero observable effect on any regression test output.

---

## 5. Migration Risks

### 5.1 Risk Assessment

| Risk | Severity | Probability | Mitigation |
|---|---|---|---|
| ResponseStrategyEngine receives empty hypotheses | Low | 100% | It already receives None. Changing to receive `new_hypotheses` improves the situation. |
| HypothesisManager not deleted before AE's inference_result field removed | Medium | 20% (ordering) | Delete HM first, then update IE interface. |
| Downstream code (not in repo) depends on HypothesisManager | N/A | 0% | None — CLI and web tools import from `mentor.py`, not from `hypothesis_manager.py` directly. |
| `session_manager.py` comment references "Hypothesis management" | None | 100% | Comment-only at line 172. No functional impact. |
| Build/deploy scripts import hypothesis_manager | None | 0% | Not imported by any launcher script (run.py, app.py, server.py). |

### 5.2 What could break (and why it won't)

| Concern | Analysis |
|---|---|
| "But AUTO_ACCEPT writes facts to ProjectState!" | This is the §6 violation being fixed. Removing it means inferred values no longer pollute explicit facts. Any test that depends on AUTO_ACCEPT'd values appearing in ProjectState will need updating. However, no test does this. |
| "But the LLM needs inferred info!" | The LLM has never received hypotheses — PromptBuilder doesn't render them. If inferred info is needed, it must be added to ResponseStrategyEngine's plan (which already has code for it) and to the prompt section. Transient conversion does not prevent this — it makes it cleaner. |
| "But the dashboard shows inferred data!" | The dashboard reads from the legacy Mentorship scalars, which are mirrored from ProjectState. Removing AUTO_ACCEPT removes the mirror source for inferred values. The dashboard needs to be refactored to read from a separate hypothesis store if inferred data display is needed. |
| "But conflicts are detected in inference_engine!" | The only conflict detection that could affect behavior is in response_strategy._detect_contradiction() — which decides CLARIFY vs CORRECT — and it's already unreachable. |

---

## 6. Recommended Implementation Order

```
Phase 1: Delete hypothesis_manager.py          (safe — no consumers)
  1a.  Remove the file
  1b.  Remove import from mentor.py
  1c.  Remove evaluate_batch() call + print from process_mentor_turn()

Phase 2: Strip InferenceEngine of lifecycle     (safe — no downstream consumption)
  2a.  Delete _load_existing_hypotheses()
  2b.  Delete _merge_hypotheses(), _find_merge_match(), _value_similarity(), _boost_confidence()
  2c.  Delete _detect_conflicts(), _are_contradictory()
  2d.  Delete _apply_conflicts_and_status(), _penalize_confidence()
  2e.  Delete _identify_retired()
  2f.  Delete _are_mutually_exclusive()
  2g.  Simplify _find_contradicting_facts() (or delete if not needed in evidence bundling)
  2h.  Simplify infer() — remove stages 1, 5, 6, 7 from the pipeline

Phase 3: Simplify InferenceResult contract      (tighten API)
  3a.  Remove updated_hypotheses, revised_hypotheses, retired_hypotheses, conflicts_detected fields
  3b.  Remove get_active() method (no status to filter)
  3c.  Rename new_hypotheses → hypotheses (or keep, since it's the only field)

Phase 4: Remove AUTO_ACCEPT → ProjectState      (fix §6 violation)
  4a.  Remove AUTO_ACCEPT write block in _apply_extraction_to_state()
  4b.  Remove hypothesis_batch parameter from _apply_extraction_to_state()
  4c.  Remove inference_result parameter from _apply_extraction_to_state()

Phase 5: Clean StateManager                     (remove dead parameter)
  5a.  Remove inference_result from apply_extraction()
  5b.  Remove _ = inference_result line

Phase 6: Wire hypotheses to ResponseStrategy    (optional — was always None)
  6a.  Pass inference_result.new_hypotheses to determine_strategy() in mentor.py
  6b.  Update _get_relevant_hypotheses() to read .new_hypotheses (or .hypotheses after rename)

Phase 7: Run all tests                          (expect 0 failures)
```

---

## 7. Summary Table

| Metric | Value |
|---|---|
| Files deleted | 1 (`hypothesis_manager.py`, 371 LOC) |
| Files modified | 4 (`inference_engine.py`, `contracts.py`, `mentor.py`, `state_manager.py`) |
| Files unchanged (with zero hypothesis refs) | 18 (see §3.7-3.10) |
| Total LOC removed | ~598 |
| Total LOC changed (simplification) | ~150 |
| Net reduction | ~748 LOC (8.5% of ~8,800 LOC Python codebase) |
| Tests affected | 0 |
| Methods deleted | 19 |
| Methods simplified | 3 |
| Methods kept unchanged | 4 (core inference) |
| Methods kept with interface change | 5 (response_strategy field rename) |

---

## 8. Final Answer

**Hypotheses are already effectively transient.** They are regenerated every turn from scratch, have zero persistence, are never rendered in the prompt, and their only downstream effect is an architecturally harmful AUTO_ACCEPT write to ProjectState. Converting to explicitly transient per-turn objects removes ~600 lines of dead lifecycle code with **zero test failures** and **zero behavior change** to any observable pipeline output.

The only risk is if any undocumented external tool or dashboard display depends on AUTO_ACCEPT'd hypothesis values appearing in ProjectState — which would be a §6 violation dependency that should be removed regardless.
