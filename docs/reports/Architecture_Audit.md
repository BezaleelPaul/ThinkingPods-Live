# Architecture Audit: Implementation vs Architecture_V1.md

## Methodology

Every component listed in Architecture_V1.md §4 was checked against its implementation for:
- **Existence**: Does the component exist as a class/module?
- **Responsibility match**: Does every declared responsibility have corresponding code? Does the code do anything outside its declared responsibilities?
- **Inputs and outputs**: Do the method signatures match the spec's data contracts?
- **Dependencies**: Are dependencies acyclic and as declared?
- **Architectural rules**: Does the implementation violate any frozen V1 rules?

All 11 components audited. Files read: `memory_extractor.py` (1326 lines), `inference_engine.py` (945), `knowledge_base.py` (677), `hypothesis_manager.py` (371), `state_manager.py` (522), `contracts.py` (419), `session_manager.py` (513), `module3/objective_engine.py` (217), `module3/completeness_checker.py` (144), `module3/priority_engine.py` (267), `module3/rules.py` (235), `module3/objective.py` (266), `module4/response_strategy.py` (824), `module4/prompt_builder.py` (487), `module5/lifecycle_manager.py` (326), `module5/summary_builder.py` (125), `module5/summary.py` (121), `mentor.py` (1427 lines).

---

## Summary of Issues

| # | Severity | Component | Issue |
|---|----------|-----------|-------|
| 1 | **HIGH** | InferenceEngine | Duplicates HypothesisManager — also manages hypothesis lifecycle (merge, conflict, status) |
| 2 | **HIGH** | InferenceEngine / HypothesisManager | Two competing lifecycle schemes: HypothesisManager (AUTO_ACCEPT/VERIFY/HOLD/DISCARD) vs InferenceEngine (ACTIVE/CONFIRMED/REJECTED) — neither is authoritative |
| 3 | **HIGH** | Explicit Facts vs Hypotheses | InferenceEngine reconstructs hypotheses from ProjectState fields, conflating explicit facts with inferred knowledge |
| 4 | **HIGH** | HypothesisManager | Interface does not match spec: takes `(hypotheses, extraction_facts)` not `(prior_hypotheses, new_hypotheses)`; produces dispositions, not merged lifecycle statuses |
| 5 | **MEDIUM** | InferenceEngine | `_are_mutually_exclusive()` hardcodes frequency domain knowledge ("daily"/"weekly"/"monthly") — leaks business logic that belongs in KnowledgeBase |
| 6 | **MEDIUM** | StateManager | `apply_extraction()` accepts `inference_result` parameter but ignores it (`_ = inference_result`) — false dependency |
| 7 | **MEDIUM** | mentor.py (SessionManager legacy) | Two SessionManagers coexist — new in `session_manager.py`, legacy in `mentor.py` — with explicit backward-compatibility shims. Not a bug but technical debt |
| 8 | **MEDIUM** | MemoryExtractor | `ProjectState.apply()` is deprecated but not removed — marked as "not called by live pipeline" |
| 9 | **LOW** | PromptBuilder | Section order differs from spec: spec says ROLE → INSTRUCTIONS → OBJECTIVE → STATE → CONVERSATION; code renders ROLE → OBJECTIVE → STATE → CONVERSATION → INSTRUCTIONS |
| 10 | **LOW** | InferenceEngine | Confidence thresholds (`_HIGH_THRESHOLD=0.80`, `_MEDIUM_THRESHOLD=0.50`) are hardcoded constants, not configurable |
| 11 | **LOW** | InferenceEngine | `_boost_confidence()` uses linear arithmetic (`0.3/0.6/0.9` mapping) with a fixed `+0.1` bump — not spec-compliant confidence propagation |
| 12 | **LOW** | KnowledgeBase | `Concept.matches()` uses substring matching (`normalized in alias.lower()`) — will match "traffic" to any alias containing "traffic" regardless of context |

---

## Detailed Audit by Component

### 1. MemoryExtractor

**File**: `memory_extractor.py`  
**Status**: ✅ PASS (with one legacy-debt finding)

**Existence**: Yes — `MemoryExtractor` class at line 1240.  
**Responsibility match**: Matches spec.
- Extracts explicit facts.  
- Does not infer.  
- Does not ask questions.  
- Falls back conservatively (AMBIGUOUS on parse failure).  
- Does not mutate ProjectState (the extract method receives a snapshot; mutation is StateManager's job).

**Inputs**: `extract(user_message, project_state, previous_assistant_message)` — matches spec.

**Outputs**: `ExtractionResult` with `message_type` (MEANINGFUL/NO_UPDATE/AMBIGUOUS/END), `updates` list — matches spec.

**Dependencies**: `ollama` (lazy), `ProjectState`, `ExtractionValidator` (self-contained). Acyclic.

**Architectural rule violations**: None.

**Issue 8 (MEDIUM)**: `ProjectState.apply()` at line 193 is deprecated ("retained for backwards compatibility only") and documented as not called by the live pipeline. It mutates state inline. The comment says it will be removed — but it still exists, meaning the ProjectState data model is not fully owned by StateManager as the spec requires. Remove it to enforce the single-writer invariant.

---

### 2. InferenceEngine

**File**: `inference_engine.py`  
**Status**: ❌ FAIL (4 issues)

**Existence**: Yes — `InferenceEngine` class at line 190.  
**Responsibility match**: **PARTIALLY**. The spec says:
- "Match extracted facts against KnowledgeBase concepts" — ✅ done at `_match_concepts`
- "Apply inference rules" — ✅ domain-agnostic
- "Assign confidence" — ✅
- "Never mutate state" — ✅ `infer()` takes `mission_memory: ProjectState` but does not mutate it
- "HypothesisManager manages lifecycle" — ❌ InferenceEngine already does lifecycle (merge, conflict, status)

**Issue 1 (HIGH) — InferenceEngine duplicates HypothesisManager's role**:  
The spec (Architecture_V1.md §4.3) says InferenceEngine's output goes to HypothesisManager for lifecycle management. However, `inference_engine.py` performs:
- **Hypothesis merging** at `_merge_hypotheses()` (lines 739–780) — merges new candidates with existing hypotheses from ProjectState
- **Conflict detection** at `_detect_conflicts()` (lines 819–828) — detects contradictory hypothesis pairs
- **Status assignment** at `_apply_conflicts_and_status()` (lines 841–883) — demotes confidence for conflicted hypotheses, assigns CONFIRMED/REJECTED statuses
- **Retirement detection** at `_identify_retired()` (lines 893–913) — marks hypotheses as REJECTED when they disappear

These are all HypothesisManager responsibilities per Architecture_V1.md §4.4. The hypothesis lifecycle pipeline is: InferenceEngine generates → HypothesisManager manages. By doing lifecycle work, InferenceEngine conflates its role with HypothesisManager's.

**Issue 3 (HIGH) — InferenceEngine reconstructs hypotheses from ProjectState, conflating facts with inferences**:  
`_load_existing_hypotheses()` (lines 401–443) reads ProjectState fields (personas, problems, etc.) and constructs Hypothesis objects with `status=CONFIRMED` and `metadata={"source": "memory"}`. This means:
- Explicit user facts (stored in ProjectState) are treated as "existing hypotheses"
- Inferred hypotheses (from KnowledgeBase concept matching) are merged with these reconstructed "memory" hypotheses
- The explicit-facts vs inferred-hypotheses separation (Architecture_V1.md §6) is broken: explicit facts are reclassified as hypotheses during inference

The spec says (§6.1): "Explicit facts are entirely separate from inferred hypotheses" and "Flow: User → MemoryExtractor → ExtractionResult → StateManager → ProjectState". This code path makes hypotheses from ProjectState, mixing the two pipelines.

**Issue 5 (MEDIUM) — Domain knowledge hardcoded in `_are_mutually_exclusive()`**:  
Lines 654–672 contain hardcoded frequency contradiction mappings:
```python
freq_keywords = {
    "daily": ["weekly", "monthly", "yearly", "rarely", "never"],
    "weekly": ["daily", "monthly", "yearly", "rarely", "never"],
    ...
}
```
This business logic belongs in the KnowledgeBase (Architecture_V1.md §4.2: "Store seed concepts for each hypothesis kind"). Hardcoding it in the engine couples domain knowledge to code and violates the "KnowledgeBase externalizes" design principle.

**Issue 10 (LOW) — Hardcoded confidence thresholds**:  
`_HIGH_THRESHOLD = 0.80` and `_MEDIUM_THRESHOLD = 0.50` are class-level constants, not configurable. The spec's "Confidence-based inference" principle suggests tunable thresholds.

**Issue 11 (LOW) — Confidence boost is linear arithmetic, not spec-compliant**:  
`_boost_confidence()` maps `{LOW: 0.3, MEDIUM: 0.6, HIGH: 0.9}` and adds a fixed `+0.1`. This is not specified in Architecture_V1.md but is inconsistent with HypothesisManager's configurable `DecisionPolicy` approach.

---

### 3. KnowledgeBase

**File**: `knowledge_base.py`  
**Status**: ✅ PASS (one minor finding)

**Existence**: Yes — `KnowledgeBase` class at line 186.  
**Responsibility match**: Matches spec.
- Stores seed concepts — ✅ `_SEED_CONCEPTS` with 6 domains
- Provides `find_concepts(text)` — ✅ `find(query)` method
- No hypothesis lifecycle logic — ✅
- No state persistence — ✅

**Inputs**: None at runtime (pre-loaded). `find(query)` takes a string. Matches spec.  
**Outputs**: `Concept` objects. Matches spec.  
**Dependencies**: Self-contained. No imports from other project modules. ✅

**Issue 12 (LOW) — `Concept.matches()` is overly broad**:  
`matches()` at line 168 uses substring matching (`normalized in alias.lower()`). This means query "traffic" matches any alias containing "traffic" (traffic_congestion, rush_hour_traffic, etc.), but also any concept whose alias or name is a substring of the query. E.g., if a concept had alias "cat", query "category" would match. Not a functional bug but an observability concern.

---

### 4. HypothesisManager

**File**: `hypothesis_manager.py`  
**Status**: ❌ FAIL (2 issues)

**Existence**: Yes — `HypothesisManager` class at line 200.  
**Responsibility match**: **PARTIALLY**.

**Issue 2 (HIGH) — HypothesisManager's lifecycle scheme competes with InferenceEngine's**:  
The spec says HypothesisManager assigns statuses: ACTIVE, CONFIRMED, REJECTED, SUPERSEDED. The implementation instead uses a different scheme:  
| Spec | Implementation |
|---|---|
| Input: `(prior_hypotheses, new_hypotheses)` | Input: `(hypotheses, extraction_facts)` — single batch, no prior distinction |
| Output: merged list with statuses ACTIVE/CONFIRMED/REJECTED/SUPERSEDED | Output: batch of dispositions AUTO_ACCEPT/VERIFY/HOLD/DISCARD |
| Merge duplicates, auto-accept HIGH, demote LOW, detect contradictions | Threshold-based decision with eviden ce counts |

Meanwhile, InferenceEngine `infer()` already does merging, conflict detection, and status assignment using the ACTIVE/CONFIRMED/REJECTED scheme. So the output of InferenceEngine already has statuses assigned, then HypothesisManager re-evaluates with a different scheme. The two lifecycle systems are **not composed** — they run sequentially with contradictory semantics.

**Issue 4 (HIGH) — Interface does not match spec**:  
The spec (§4.4) says:
- Inputs: "Prior hypotheses (from previous turn), new hypotheses (from InferenceEngine)"
- Outputs: "Updated hypothesis list with statuses (ACTIVE, CONFIRMED, REJECTED, SUPERSEDED)"

The implementation uses: `evaluate_batch(hypotheses, extraction_facts)` — hypotheses arrive as a single batch (not split into prior/new) and are individually assigned decisions. There is no cross-turn state; each call is independent.

---

### 5. StateManager

**File**: `state_manager.py`  
**Status**: ✅ PASS (one minor finding)

**Existence**: Yes — `StateManager` class at line 330.  
**Responsibility match**: Matches spec.
- Single writer of ProjectState — ✅ `apply_extraction` is the only mutation path
- Atomic batch validation — ✅ `StateValidator.validate_batch` collects all errors before raising
- ADD (list append, dedup, preserve order) and SET (scalar overwrite) — ✅ `_apply_add` and `_apply_set`
- Roll forward previous-message bookkeeping — ✅ lines 470–475

**Inputs**: `ExtractionResult` (validated). Also accepts `InferenceResult` (see issue).  
**Outputs**: Mutated `ProjectState`.  
**Dependencies**: `memory_extractor` (ExtractionResult, ProjectState, etc.) — ✅ acyclic.

**Issue 6 (MEDIUM) — False dependency on `InferenceResult`**:  
`apply_extraction()` accepts `inference_result: InferenceResult | None = None` but explicitly ignores it:  
```python
_ = inference_result  # accepted for pipeline compatibility but ignored
```
A comment says "This preserves identical behavior while wiring the new architecture." This is unfinished wiring — a stub parameter that disguises an unimplemented integration point between StateManager and InferenceEngine/HypothesisManager. According to the spec, StateManager should not consume inference output at all. The parameter should be removed.

---

### 6. ObjectiveEngine

**File**: `module3/objective_engine.py`  
**Status**: ✅ PASS

**Existence**: Yes — `ObjectiveEngine` class at line 56.  
**Responsibility match**: Matches spec.
- Evaluates which fields are missing — ✅ via `CompletenessChecker`
- Applies priority rules — ✅ via `PriorityEngine` + `DEFAULT_RULES`
- Returns WRAP_UP when all complete — ✅
- Deterministic — ✅ pure function of ProjectState
- Does not decide *how* to ask — ✅ returns ConversationObjective only

**Inputs**: `ProjectState` only. Matches spec (spec also lists optional `CompletenessReport` and `PriorityEngine` rules — handled via composition).  
**Outputs**: `ConversationObjective` with objective, missing/completed fields, confidence, reasoning. Matches spec.  
**Dependencies**: `memory_extractor.ProjectState`, `module3.completeness_checker`, `module3.priority_engine`, `module3.rules`, `module3.objective`. Acyclic. ✅

**Issues**: None.

---

### 7. LifecycleManager

**File**: `module5/lifecycle_manager.py`  
**Status**: ✅ PASS

**Existence**: Yes — `LifecycleManager` class at line 211.  
**Responsibility match**: Matches spec.
- CONTINUE when objective ≠ WRAP_UP — ✅
- READY_FOR_SUMMARY when WRAP_UP and no prior summary — ✅
- WAITING_FOR_CONFIRMATION when summary was presented — ✅
- READY_FOR_TRANSITION on user confirmation — ✅
- Strict confirmation lexicon — ✅ `_CONFIRMATION_PATTERN` regex

**Inputs**: `ProjectState`, `ConversationObjective`, `ResponseStrategy`. Matches spec.  
**Outputs**: `LifecycleDecision` enum (CONTINUE/READY_FOR_SUMMARY/WAITING_FOR_CONFIRMATION/READY_FOR_TRANSITION). Matches spec.  
**Dependencies**: `memory_extractor.ProjectState`, `module3.ConversationObjective`, `module4.ResponseStrategy`. Acyclic. ✅

**Issues**: None.

---

### 8. ResponseStrategyEngine

**File**: `module4/response_strategy.py`  
**Status**: ✅ PASS

**Existence**: Yes — `ResponseStrategyEngine` class at line 260.  
**Responsibility match**: Matches spec.
- Maps objective → base strategy — ✅ `_OBJ_TO_STRATEGY`
- Information-gain planning — ✅ full pipeline: generate, estimate, rank, select
- Contradiction → CORRECT — ✅ `_detect_contradiction`
- Ambiguity → CLARIFY — ✅ `_resolve_strategy` checks `count > 1`
- Strategy-specific constraints — ✅ `_STRATEGY_CONSTRAINTS`
- Deterministic — ✅ no state, no I/O, no LLM

**Inputs**: `ConversationObjective`, `ProjectState`, optional `InferenceResult`. Matches spec.  
**Outputs**: `ResponsePlan` with all required fields (strategy, question_intent, expected_checklist_fields, candidate_question_intents, selected_question_intent, expected_information_gain, selection_reason, constraints). Matches spec.  
**Dependencies**: `contracts` (ResponsePlan, Hypothesis, etc.), `memory_extractor.ProjectState`, `module3.ConversationObjective`. Acyclic. ✅

**Issues**: None.

---

### 9. PromptBuilder

**File**: `module4/prompt_builder.py`  
**Status**: ✅ PASS (one minor finding)

**Existence**: Yes — `PromptBuilder` class with `build_prompt` staticmethod at line 338.  
**Responsibility match**: Matches spec.
- Renders five sections — ✅ Role, Current Objective, Known Project State (or Empathize Summary), Latest Conversation, Instructions
- Injects strategy constraints — ✅ via `ResponsePlan.constraints`
- Never mutates inputs — ✅ all reads, no writes

**Inputs**: `ProjectState`, `ConversationObjective`, `ResponsePlan` (or legacy `ResponseStrategy`), previous_messages, optional `EmpathizeSummary`. Matches spec.  
**Outputs**: Prompt string. Matches spec.  
**Dependencies**: `memory_extractor` (ProjectState), `module3` (ConversationObjective), `contracts` (ResponsePlan), `module4.response_strategy` (ResponseStrategy). `module5` referenced only under `TYPE_CHECKING`. ✅ Acyclic.

**Issue 9 (LOW) — Section order differs from spec**:  
Architecture_V1.md §4.9 says the five required sections are in order: "ROLE, INSTRUCTIONS, OBJECTIVE, PROJECT_STATE, LATEST_CONVERSATION". The code renders them as: ROLE, OBJECTIVE, STATE, CONVERSATION, INSTRUCTIONS. INSTRUCTIONS appears last, not second. This does not affect functionality (the LLM receives the same information) but diverges from the documented interface. Recommend updating either the spec or the code to match.

---

### 10. SummaryBuilder

**File**: `module5/summary_builder.py`  
**Status**: ✅ PASS

**Existence**: Yes — `SummaryBuilder` class with `build` staticmethod at line 48.  
**Responsibility match**: Matches spec.
- Defensive copy — ✅ `list(project_state.personas)`, etc.
- Preserves insertion order — ✅
- Includes only six extraction fields — ✅
- Does not validate completeness — ✅ just copies

**Inputs**: `ProjectState`. Matches spec.  
**Outputs**: `EmpathizeSummary` (frozen dataclass with six fields). Matches spec.  
**Dependencies**: `module5.summary.EmpathizeSummary`, `memory_extractor.ProjectState` (lazy import). Acyclic. ✅

**Issues**: None.

---

### 11. SessionManager

**File**: `session_manager.py` (new), `mentor.py:263` (legacy)  
**Status**: ✅ PASS (one finding)

**Existence**: Yes — Two implementations. New in `session_manager.py`, legacy wrapper in `mentor.py:263`.  
**Responsibility match**: The new SessionManager matches the spec's §13.1.
- `create_session` → timestamped folder + empty state — ✅
- `get_active_session` → active ProjectState — ✅
- `switch_session` → archive + activate — ✅
- `reset_runtime_state` → archive + create fresh — ✅
- `list_sessions` → all sessions with metadata — ✅
- `delete_session` → permanent removal — ✅

**Issue 7 (MEDIUM) — Two SessionManagers**:  
The legacy `SessionManager` class inside `mentor.py` (line 263) wraps the new one for backward compatibility, with explicit conversion methods (`_project_state_to_legacy_session`, `_legacy_session_to_project_state`, `load_legacy_session`, `save_legacy_session`). This adds complexity: the data in `sessions/*/session_data.json` may not always be in sync with the legacy `sessions/*_mentor.json` files. The legacy path is retained until the dashboard reads from ProjectState directly (noted in TODO comments). Technical debt.

---

## Cross-Cutting Checks

### A. Explicit facts remain separate from inferred hypotheses

**Result: ❌ FAIL (Issue 3, HIGH)**

`InferenceEngine._load_existing_hypotheses()` reads ProjectState fields and reclassifies explicit user facts as inferred hypotheses (with `status=CONFIRMED`). This breaks the Architecture_V1.md §6 separation: explicit facts are ground truth (user-stated), hypotheses are tentative (system-inferred). Mixing them means:
- A hypothesis that duplicates an explicit fact is not recognized as a duplicate (the "explicit" fact is itself a hypothesis)
- The provenance chain is lost: "did the user say this, or did we infer it?"

The same issue existed in `InferenceEngine._filter_explicit_duplicates()` which tries to prevent this, but then `_load_existing_hypotheses` re-creates hypotheses from state that should remain fact-only.

### B. No downstream component mutates extractor output

**Result: ✅ PASS**

`ExtractionResult` is a frozen dataclass. `ExtractionUpdate` is a frozen dataclass. All components receive them as read-only. StateManager validates and applies the updates to its own ProjectState copy, never mutating the original result object. Good.

### C. KnowledgeBase contains all domain knowledge

**Result: ❌ FAIL (Issue 5, MEDIUM)**

`InferenceEngine._are_mutually_exclusive()` at `inference_engine.py:654` hardcodes frequency domain knowledge ("daily", "weekly", "monthly", "yearly", "rarely", "never", "always", "occasionally") with their mutual-exclusion relationships. This business logic belongs in KnowledgeBase concepts or a dedicated inference rule — not in the engine itself.

### D. No business logic has leaked into PromptBuilder

**Result: ✅ PASS**

`PromptBuilder.build_prompt()` is a pure formatter. It reads `_FIELD_LABELS`, `_OBJECTIVE_LABELS`, `_INSTRUCTIONS` from static dictionaries and renders them. No conditional branching on field values, no priority logic, no inference. All strategy decisions come from `ResponsePlan`. Good.

### E. No duplicate responsibilities

**Result: ❌ FAIL (Issues 1, 2, 4)**

- **InferenceEngine vs HypothesisManager**: Both manage hypothesis lifecycle (merge, conflict, status assignment) with different schemes. InferenceEngine uses ACTIVE/CONFIRMED/REJECTED; HypothesisManager uses AUTO_ACCEPT/VERIFY/HOLD/DISCARD.
- **Two SessionManagers**: New + legacy adapter creates redundant persistence paths.
- **StateManager's unused InferenceResult parameter**: A stub awaiting an integration that may or may not be needed.

---

## Recommended Fixes

(Ordered by severity — do not implement; architectural guidance only.)

| # | Severity | Fix |
|---|----------|-----|
| 1 | HIGH | Remove lifecycle management (merge, conflict, status) from `InferenceEngine.infer()`. Make it a pure generator: `(extraction_result, ProjectState) → new_hypotheses`. Move all merging, status assignment, and conflict detection to HypothesisManager. |
| 2 | HIGH | Unify the two hypothesis lifecycle schemes. Choose one: either ACTIVE/CONFIRMED/REJECTED/SUPERSEDED (arch spec) or AUTO_ACCEPT/VERIFY/HOLD/DISCARD (implementation). The arch spec's scheme is canonical. |
| 3 | HIGH | Remove `_load_existing_hypotheses()` from InferenceEngine. Explicit facts in ProjectState must not be reclassified as hypotheses. HypothesisManager should own any cross-turn hypothesis state. |
| 4 | HIGH | Re-implement `HypothesisManager.evaluate_batch()` to match the spec: accept `(prior_hypotheses, new_hypotheses)`, return merged list with lifecycle statuses. Remove the parallel AUTO_ACCEPT/VERIFY scheme. |
| 5 | MEDIUM | Extract frequency contradiction logic from `InferenceEngine._are_mutually_exclusive()` into KnowledgeBase concept definitions or a dedicated inference rule. |
| 6 | MEDIUM | Remove the `inference_result` parameter from `StateManager.apply_extraction()`. If no component consumes inference output at this point, the interface should not claim to. |
| 7 | MEDIUM | Complete the SessionManager migration. Remove the legacy `MentorSession` class, the legacy `SessionManager` in `mentor.py`, and the `_project_state_to_legacy_session` / `_legacy_session_to_project_state` bridges. Refactor the dashboard to read from `ProjectState` directly. |
| 8 | MEDIUM | Remove `ProjectState.apply()` from `memory_extractor.py`. The live pipeline uses `StateManager.apply_extraction()` exclusively. |
| 9 | LOW | Update either the Architecture_V1.md §4.9 section order or the `PromptBuilder.build_prompt` section ordering to match. |
| 10 | LOW | Make confidence thresholds configurable in InferenceEngine (constructor parameters or a policy object, mirroring HypothesisManager's `DecisionPolicy`). |
| 11 | LOW | Replace `_boost_confidence()` linear arithmetic with a configurable policy. |
| 12 | LOW | Tighten `Concept.matches()` to use exact/prefix matching rather than undirected substring matching. |

---

## Final Verdict

| Component | Status |
|-----------|--------|
| MemoryExtractor | ✅ PASS (with legacy debt) |
| KnowledgeBase | ✅ PASS (minor substring matching concern) |
| **InferenceEngine** | **❌ FAIL — 4 issues (2 HIGH)** |
| **HypothesisManager** | **❌ FAIL — 2 issues (2 HIGH)** |
| StateManager | ✅ PASS (with false dependency) |
| ObjectiveEngine | ✅ PASS |
| LifecycleManager | ✅ PASS |
| ResponseStrategyEngine | ✅ PASS |
| PromptBuilder | ✅ PASS (section order deviation) |
| SummaryBuilder | ✅ PASS |
| SessionManager | ✅ PASS (with legacy debt) |

The most critical architectural violations are in the InferenceEngine ↔ HypothesisManager boundary. The InferenceEngine manages hypothesis lifecycle (merge, conflict, status) in direct competition with HypothesisManager's own lifecycle scheme. The two components have overlapping responsibilities with incompatible status semantics, and the separation of explicit facts from inferred hypotheses (§6) is broken by reconstructing hypotheses from ProjectState.
