# Architecture Fix Design: InferenceEngine ↔ HypothesisManager Boundary Correction

## Scope

Minimal interface and responsibility changes to satisfy Architecture_V1.md. No new components. No new features. No changes to MemoryExtractor, KnowledgeBase, ObjectiveEngine, LifecycleManager, ResponseStrategyEngine, PromptBuilder, or SummaryBuilder.

---

## 1. Responsibility Reassignment

### 1.1. Hypothesis Generation

| Current Owner | Correct Owner |
|---|---|
| InferenceEngine (post-merge) | InferenceEngine (pre-merge) |

**Why**: §4.3 says InferenceEngine "generates hypotheses from explicit facts using knowledge base and deterministic rules." The merge/conflict/status steps are all post-generation lifecycle work assigned to HypothesisManager by §4.4.

**What moves**: `_merge_hypotheses()`, `_detect_conflicts()`, `_apply_conflicts_and_status()`, `_identify_retired()`.

**Impact**: InferenceEngine output changes from a fully-merged status-assigned list to a flat set of candidate hypotheses with confidence only. HypothesisManager gains ~120 lines of moved logic.

---

### 1.2. Cross-Turn Hypothesis State

| Current Owner | Correct Owner |
|---|---|
| InferenceEngine (reads ProjectState) | HypothesisManager (internal store) |

**Why**: §4.4 says HypothesisManager inputs are "(prior hypotheses, new hypotheses)." The word "prior" implies the manager owns/has access to previous-turn output. Currently IE reconstructs "prior" from ProjectState via `_load_existing_hypotheses()`, which breaks §6 (facts vs hypotheses separation).

**What moves**: `_load_existing_hypotheses()` is deleted entirely. Hypotheses never read from ProjectState. HypothesisManager stores and retrieves its own output.

**Impact**: ProjectState stops being a hypothesis store. HypothesisManager gains a `_current_hypotheses: list[Hypothesis]` field and a `get_current_hypotheses()` accessor.

---

### 1.3. Hypothesis Status Assignment

| Current Owner | Correct Owner |
|---|---|
| InferenceEngine (ACTIVE/CONFIRMED/REJECTED) + HypothesisManager (AUTO_ACCEPT/VERIFY/HOLD/DISCARD) — competing | HypothesisManager (ACTIVE/CONFIRMED/REJECTED/SUPERSEDED) — unified |

**Why**: Two incompatible lifecycle schemes run sequentially. IE assigns CONFIRMED to a hypothesis; HM re-evaluates the same hypothesis as VERIFY. Neither output is authoritative — the pipeline ignores both and only checks `AUTO_ACCEPT`. §4.4 defines the canonical scheme: ACTIVE/CONFIRMED/REJECTED/SUPERSEDED.

**What moves**: Delete IE's `_apply_conflicts_and_status()` status logic. Delete HM's AUTO_ACCEPT/VERIFY/HOLD/DISCARD scheme. Implement the spec's ACTIVE/CONFIRMED/REJECTED/SUPERSEDED in HypothesisManager, drawing on the confidence-threshold logic already in `_evaluate_single()`.

**Impact**: `HypothesisDecision` enum and its four values are removed. `HypothesisBatchResult` fields change from `(auto_accept, verify, hold, discard)` to `(hypotheses, new_ids, revised_ids, retired_ids)`.

---

### 1.4. Hypothesis → ProjectState Promotion

| Current Owner | Correct Owner |
|---|---|
| mentor.py `_apply_extraction_to_state()` (writes AUTO_ACCEPT to ProjectState) | Nobody — this is prohibited by §6 |

**Why**: §6 says "Hypotheses influence questions but never auto-commit to state." Writing inferred hypotheses to ProjectState makes them indistinguishable from explicit user facts, breaking provenance, auditability, and rollback safety.

**What moves**: Delete the `if hypothesis_batch is not None:` block in `_apply_extraction_to_state()` (mentor.py lines 1014–1038). Remove the `hypothesis_batch` parameter from `_apply_extraction_to_state()`.

**Impact**: ProjectState contains only explicit extraction facts. Hypotheses flow to downstream components (ResponseStrategyEngine → PromptBuilder) separately. PromptBuilder includes a "We infer:" section in the prompt but NEVER writes inferences to state.

---

### 1.5. StateManager Interface

| Current Owner | Correct Owner |
|---|---|
| StateManager (accepts InferenceResult, ignores it) | StateManager (no InferenceResult parameter) |

**Why**: `apply_extraction()` accepts `inference_result: InferenceResult | None = None` and immediately ignores it (`_ = inference_result`). This is a stub parameter that disguises an unimplemented integration point and creates a false dependency.

**What moves**: Remove the `inference_result` parameter from `StateManager.apply_extraction()`.

**Impact**: No functional change — the parameter was already dead code. All call sites must be updated to drop the argument.

---

## 2. Redesigned Interfaces

### 2.1. InferenceEngine (corrected)

**Purpose**: Generate hypothesis candidates from explicit facts and domain knowledge. Pure function: no state, no I/O, no lifecycle decisions.

```
Input:
    extraction_result: ExtractorExtractionResult
        The current turn's extraction output (from MemoryExtractor).
        Only .updates is consumed (converted to ExtractedFact internally).

    project_state: ProjectState
        Current mission memory. Used ONLY for duplicate filtering
        (never reconstructs hypotheses from state fields).

Output:
    InferenceResult
        .new_hypotheses: tuple[Hypothesis, ...]
            Freshly generated hypothesis candidates.
            All have status=ACTIVE (default). NO lifecycle processing.
        .reasoning_trace: tuple[str, ...]
            Audit trail for debugging.

Processing pipeline (stripped):
    _convert_updates_to_facts()
        → ExtractionUpdate → ExtractedFact

    _collect_explicit_values()
        → Gather explicit facts for dedup (from extraction_result + project_state)

    _match_concepts()
        → ExtractedFact → KnowledgeBase concept matches

    _build_evidence_bundles()
        → Group matches into hypothesis evidence bundles

    _filter_explicit_duplicates()
        → Remove candidates that duplicate explicit facts (§7 rule)

    _form_hypothesis_candidates()
        → Evidence bundles → Hypothesis[] (all status=ACTIVE)

Removed (moved to HypothesisManager):
    _load_existing_hypotheses()        — DELETED
    _merge_hypotheses()                → HypothesisManager
    _detect_conflicts()                → HypothesisManager
    _apply_conflicts_and_status()      → HypothesisManager
    _identify_retired()                → HypothesisManager
```

### 2.2. HypothesisManager (corrected)

**Purpose**: Own the full hypothesis lifecycle: merge prior + new, detect conflicts, assign statuses, and maintain cross-turn state.

```
Input:
    prior_hypotheses: tuple[Hypothesis, ...]
        Hypotheses from previous turn (stored internally, or passed by integrator).
        Empty on first turn of a session.

    new_hypotheses: tuple[Hypothesis, ...]
        Fresh candidates from InferenceEngine (status=ACTIVE, no lifecycle).

Output:
    HypothesisBatchResult
        .hypotheses: tuple[Hypothesis, ...]
            Full merged list with statuses after lifecycle processing.
        .new_ids: tuple[str, ...]
            Hypothesis IDs that were added this turn.
        .revised_ids: tuple[str, ...]
            Hypothesis IDs whose status changed this turn.
        .retired_ids: tuple[str, ...]
            Hypothesis IDs that were REJECTED this turn.

Internal state:
    _current_hypotheses: tuple[Hypothesis, ...]
        The output of the most recent evaluate() call.
        Survives between turns in memory.

Status scheme (matches spec §4.4):
    ACTIVE      — plausible, under consideration
    CONFIRMED   — high confidence, accepted as likely true
    REJECTED    — contradicted or insufficient evidence
    SUPERSEDED  — replaced by a more specific hypothesis of same kind

Processing pipeline (moved from InferenceEngine):
    _merge_hypotheses(prior, new)
        → Merge matching (kind + similar value) pairs.
        → SUPERSEDED older duplicates on merge.
        → New entries start ACTIVE.

    _detect_conflicts(merged)
        → Find contradictory pairs (same kind, mutually exclusive).
        → Mark conflicted hypotheses as ACTIVE (not promotable).

    _assign_statuses(merged, conflicts)
        → HIGH confidence + no conflicts → CONFIRMED
        → LOW confidence + no new evidence → REJECTED
        → Conflicted → stays ACTIVE
        → Merged duplicates → SUPERSEDED

    _boost_confidence(existing, new)  — moved from IE
    _are_mutually_exclusive()         — moved from IE

Removed:
    HypothesisDecision enum (AUTO_ACCEPT/VERIFY/HOLD/DISCARD) — DELETED
    _evaluate_single() — replaced by _assign_statuses()
    evidence_boost / contradiction_penalty — kept but simplified
```

### 2.3. StateManager (corrected)

**Purpose**: Atomically validate and apply explicit extraction facts to ProjectState. Pure mutation target: no hypotheses pass through here.

```
Input:
    result: ExtractionResult
        The current turn's extraction output (from MemoryExtractor).
        .updates are validated and applied.

    previous_assistant_message: str | None
        Rolled forward for conversational context.

    previous_user_message: str | None
        Rolled forward for conversational context.

Output:
    ProjectState (mutated in place)
        Contains ONLY explicit extraction fields + bookkeeping.
        NO hypothesis fields.
        NO inference payload.

Removed:
    inference_result parameter — DELETED
```

### 2.4. Pipeline Integrator (mentor.py, corrected)

**Purpose**: Orchestrate the pipeline. Maintains HypothesisManager's cross-turn state between turns.

```
Corrected flow:

1. MemoryExtractor.extract()
   → ExtractionResult (explicit facts only)

2. InferenceEngine.infer(extraction_result, project_state)
   → InferenceResult.new_hypotheses (fresh candidates, status=ACTIVE)

3. hypothesis_mgr.get_current_hypotheses()        # prior
   hypothesis_mgr.evaluate(prior, new_candidates)  # lifecycle processing
   → HypothesisBatchResult (full merged list with statuses)

4. StateManager.apply_extraction(extraction_result)  # facts only
   NO hypothesis_batch parameter.
   NO hypothesis writing to ProjectState.
   → ProjectState (updated)

5. ObjectiveEngine.determine_next(project_state)     # reads facts only
6. LifecycleManager.determine(...)
7. ResponseStrategyEngine.determine_strategy(...)     # receives hypotheses separately
8. PromptBuilder.build_prompt(project_state, hypotheses, ...)
9. LLM → natural language response

10. hypothesis_mgr stores its output for next turn
    SessionManager saves project_state (facts) + hypotheses (separate file)
```

---

## 3. Contract Changes

### 3.1. InferenceResult (simplified)

| Field | Current | Correct |
|---|---|---|
| `updated_hypotheses` | Full merged list | **Removed** (HM owns merged list) |
| `new_hypotheses` | Subset of updated | **Kept** (the only output) |
| `revised_hypotheses` | Status changes | **Removed** (HM tracks) |
| `retired_hypotheses` | REJECTED entries | **Removed** (HM tracks) |
| `conflicts_detected` | Conflict pairs | **Removed** (HM detects) |
| `reasoning_trace` | Audit trail | **Kept** |

New `InferenceResult`:

```python
@dataclass(frozen=True)
class InferenceResult:
    new_hypotheses: tuple[Hypothesis, ...]     # fresh candidates only
    reasoning_trace: tuple[str, ...]            # audit trail
```

### 3.2. HypothesisBatchResult (redesigned)

| Field | Current | Correct |
|---|---|---|
| `dispositions` | HypothesisDisposition per input | **Removed** |
| `auto_accept` | AUTO_ACCEPT hypotheses | **Removed** (scheme deleted) |
| `verify` | VERIFY hypotheses | **Removed** |
| `hold` | HOLD hypotheses | **Removed** |
| `discard` | DISCARD hypotheses | **Removed** |

New `HypothesisBatchResult`:

```python
@dataclass(frozen=True)
class HypothesisBatchResult:
    hypotheses: tuple[Hypothesis, ...]      # full merged list with statuses
    new_ids: tuple[str, ...]                # added this turn
    revised_ids: tuple[str, ...]            # status changed this turn
    retired_ids: tuple[str, ...]            # REJECTED this turn
```

### 3.3. StateManager.apply_extraction() (simplified)

| Parameter | Current | Correct |
|---|---|---|
| `result` | ExtractionResult | **Kept** (unchanged) |
| `inference_result` | InferenceResult (ignored) | **Removed** |
| `previous_assistant_message` | str \| None | **Kept** |
| `previous_user_message` | str \| None | **Kept** |

### 3.4. HypothesisStatus docstring (corrected)

**Current** (`contracts.py:116-117`):

```python
class HypothesisStatus(str, Enum):
    """Lifecycle status of a hypothesis. Owned by Inference Engine."""
```

**Corrected**:

```python
class HypothesisStatus(str, Enum):
    """Lifecycle status of a hypothesis. Owned by HypothesisManager."""
```

---

## 4. Migration Steps

### Step 1: Delete IE's lifecycle methods

**Files**: `inference_engine.py`

**Actions**:
- Delete `_load_existing_hypotheses()` (lines 401–443)
- Delete `_merge_hypotheses()` (lines 739–780)
- Delete `_find_merge_match()` (lines 782–793)
- Delete `_value_similarity()` (lines 795–803)
- Delete `_boost_confidence()` (lines 805–813)
- Delete `_detect_conflicts()` (lines 819–828)
- Delete `_are_contradictory()` (lines 830–835)
- Delete `_apply_conflicts_and_status()` (lines 841–883)
- Delete `_penalize_confidence()` (lines 885–891)
- Delete `_identify_retired()` (lines 893–913)
- Delete `_are_mutually_exclusive()` (lines 654–672) — also fixes Issue 5 (MEDIUM, hardcoded domain knowledge) — move to KnowledgeBase or HypothesisManager

**Simplify `infer()` method** to:
```
1. _convert_updates_to_facts()
2. _match_concepts(facts)
3. _build_evidence_bundles(matches, facts)
4. _filter_explicit_duplicates(bundles, explicit_values)
5. _form_hypothesis_candidates(filtered_bundles)
6. return InferenceResult(new_hypotheses=..., reasoning_trace=...)
```

### Step 2: Move lifecycle logic to HypothesisManager

**Files**: `hypothesis_manager.py`

**Actions**:
- Add `_current_hypotheses: tuple[Hypothesis, ...] = ()` field
- Add `get_current_hypotheses()` accessor
- Add `evaluate(prior_hypotheses, new_hypotheses)` method
- Rewrite `_evaluate_single()` as `_assign_statuses()` using ACTIVE/CONFIRMED/REJECTED/SUPERSEDED scheme
- Import and integrate merge logic from IE (`_merge_hypotheses`, `_find_merge_match`, `_value_similarity`, `_boost_confidence`)
- Import and integrate conflict detection (`_detect_conflicts`, `_are_contradictory`, `_are_mutually_exclusive`)
- Delete `HypothesisDecision` enum (AUTO_ACCEPT/VERIFY/HOLD/DISCARD)
- Rewrite `HypothesisBatchResult` dataclass (remove dispositions/auto_accept/verify/hold/discard, add hypotheses/new_ids/revised_ids/retired_ids)
- Delete `evaluate_batch()` — replace with `evaluate()`

### Step 3: Clean StateManager

**Files**: `state_manager.py`

**Actions**:
- Remove `inference_result` parameter from `apply_extraction()` signature
- Remove `_ = inference_result` comment block

### Step 4: Clean pipeline integrator

**Files**: `mentor.py`

**Actions**:
- In `_apply_extraction_to_state()`:
  - Remove `hypothesis_batch` parameter
  - Remove the entire `if hypothesis_batch is not None:` block (lines 1014–1038)
  - Remove `inference_result` parameter (no longer passed to StateManager)
- In `process_mentor_turn()`:
  - Change IE call to produce-only: `inference_result = inference_engine.infer(extraction_result, project_state)`
  - Change HM call to receive prior + new:
    ```python
    prior = hypothesis_manager.get_current_hypotheses()
    batch = hypothesis_manager.evaluate(prior, inference_result.new_hypotheses)
    ```
  - Store HM output: `hypothesis_manager._current_hypotheses = batch.hypotheses`
    (or use a setter if formal, or let `evaluate()` store internally)
  - Remove `inference_result` arg from `_apply_extraction_to_state()` call
  - Remove `hypothesis_batch` arg from `_apply_extraction_to_state()` call
  - Pass `batch.hypotheses` (not `inference_result.updated_hypotheses`) to downstream components (ResponseStrategyEngine, PromptBuilder)

### Step 5: Fix contracts

**Files**: `contracts.py`

**Actions**:
- `InferenceResult`: remove `updated_hypotheses`, `revised_hypotheses`, `retired_hypotheses`, `conflicts_detected`. Keep only `new_hypotheses` and `reasoning_trace`.
- `HypothesisStatus` docstring: change "Owned by Inference Engine" to "Owned by HypothesisManager"
- (If `HypothesisBatchResult` is defined here, update it per 3.2; if in `hypothesis_manager.py`, update there)

---

## 5. Dependency Impact Map

| Component | Change Type | Impact |
|---|---|---|
| **InferenceEngine** | Major deletion | -9 methods (~180 lines removed). Input/output unchanged. |
| **HypothesisManager** | Major rewrite | ~120 lines added (merge + conflict + status). Interface changed: `evaluate()` replaces `evaluate_batch()`. |
| **StateManager** | Minor deletion | -1 parameter removed from `apply_extraction()`. |
| **mentor.py** | Moderate rewrite | `_apply_extraction_to_state()` loses 2 params + 25-line block. `process_mentor_turn()` adds HM state management. |
| **contracts.py** | Minor deletion | `InferenceResult` loses 4 fields. `HypothesisStatus` docstring fix. |
| **ObjectiveEngine** | Unchanged | Already reads only ProjectState. Hypotheses were unused. |
| **LifecycleManager** | Unchanged | Already reads only ProjectState + Objective + ResponseStrategy. |
| **ResponseStrategyEngine** | Unchanged interface | Receives hypotheses as optional parameter (unchanged). Data provenance is cleaner. |
| **PromptBuilder** | Unchanged interface | Receives hypotheses as part of plan (unchanged). Can add "Inferred:" section separately from "Known:" section. |
| **SessionManager** | Unchanged | ProjectState persists as before. Hypotheses must be persisted separately (new `hypotheses.json`). |
| **KnowledgeBase** | Unchanged | Still provides concept matching. `_are_mutually_exclusive` logic may move here (optional, Issue 5 fix). |
| **Tests** | ~30 test methods affected | Tests that mock `InferenceResult.updated_hypotheses` or check `HypothesisBatchResult.auto_accept` need update. |

---

## 6. Corrected Pipeline Diagram

```
User Message
    │
    ▼
MemoryExtractor.extract()
    │  ┌──────────────────────────────────┐
    │  │ ExtractionResult                 │
    │  │  - message_type                  │
    │  │  - updates[ExtractionUpdate]     │
    ▼  └────────────┬─────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
InferenceEngine.infer()    StateManager.apply_extraction()
  (facts + KB                (explicit facts only)
   → new hypotheses)          → ProjectState (facts only)
        │                       │
        │  ┌─────────┐          │
        │  │ new     │          │
        ▼  │ hyp[]   │          │
HypothesisManager        ◄──────┘
  (prior + new → merged)          │
  (assign statuses)               │
  (detect conflicts)              │
        │                         │
        │  ┌───────────────────┐  │
        │  │ merged hyp[]      │  │
        │  │ (with statuses)   │  │
        ▼  └───────────────────┘  ▼
ObjectiveEngine ◄──────── ProjectState (facts only)
        │
        ▼
LifecycleManager
        │
        ▼
ResponseStrategyEngine ◄──── hypotheses (from HM)
        │                           │
        ▼                           │
PromptBuilder ◄─────────────────────┘
  (renders facts + hypotheses
   as separate prompt sections)
        │
        ▼
LLM → Natural language reply
        │
        ▼
SessionManager.save():
  project_state.json (facts)
  hypotheses.json    (hypotheses)  ← NEW: separate persistence
```

Key differences from current flow:
- InferenceEngine output is NOT yet merged — it's raw candidates
- HypothesisManager receives BOTH prior (from its store) AND new (from IE)
- StateManager receives ONLY ExtractionResult — no hypotheses pass through
- ProjectState contains ONLY explicit facts — no hypothesis contamination
- Hypotheses persist in a separate `hypotheses.json` — not inside ProjectState
- Next turn: HypothesisManager retrieves its own prior from internal store, NOT from ProjectState
