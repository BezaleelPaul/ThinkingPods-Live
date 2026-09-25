# PR 1 Impact Analysis: HypothesisManager Removal

## Objective

Delete `hypothesis_manager.py` and remove every integration point from `mentor.py`.

**Rationale**: HypothesisManager's outputs are consumed only by the `auto_accept` write path in `_apply_extraction_to_state`, which violates Architecture_V1.md §6 (explicit facts vs. inferred hypotheses separation). No other component reads HypothesisManager's output.

---

## 1. Complete HypothesisManager Call Graph

```
┌─────────────────────────────────────────────────────────────────────────┐
│ hypothesis_manager.py (371 LOC)                                        │
│                                                                         │
│  HypothesisDecision (enum)       ← 4 values: AUTO_ACCEPT/VERIFY/HOLD/DISCARD │
│  DecisionPolicy (dataclass)      ← threshold configuration             │
│  DEFAULT_POLICY (singleton)      ← global default policy instance      │
│  HypothesisDisposition (dataclass) ← single-hypothesis decision result │
│  HypothesisBatchResult (dataclass) ← batch of dispositions             │
│  HypothesisManager (class)       ← evaluator                           │
│    evaluate_batch(hypotheses, extraction_facts) → HypothesisBatchResult│
│    _evaluate_single(hypothesis, facts) → HypothesisDisposition         │
│    _hypothesis_kind_to_field(kind) → str                               │
│  get_hypothesis_manager(policy) → HypothesisManager                    │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ imported by
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ mentor.py                                                               │
│                                                                         │
│  Line 33:  from hypothesis_manager import get_hypothesis_manager        │
│  Line 97-104: Module-level singleton                                   │
│     _HYPOTHESIS_MANAGER = get_hypothesis_manager()                     │
│                                                                         │
│  Lines 1141-1153: process_mentor_turn()                                │
│     hypothesis_manager = get_hypothesis_manager()                      │
│     hypothesis_batch = hypothesis_manager.evaluate_batch(              │
│         hypotheses=inference_result.updated_hypotheses,                │
│         extraction_facts=(),                                            │
│     )                                                                   │
│     print(f"[HypothesisManager] auto_accept={len(...)} ...")           │
│                                                                         │
│  Lines 1159-1164: _apply_extraction_to_state() signature + docstring  │
│     def _apply_extraction_to_state(                                     │
│         ..., hypothesis_batch=None                                     │
│     )                                                                   │
│     The hypothesis_batch is the output of HypothesisManager...         │
│                                                                         │
│  Lines 1014-1038: _apply_extraction_to_state() body                    │
│     if hypothesis_batch is not None:                                    │
│         for decision_result in hypothesis_batch.auto_accept:           │
│             # write hypothesis values to ProjectState                  │
│                                                                         │
│  Line 1164: _apply_extraction_to_state() call                          │
│     _apply_extraction_to_state(..., hypothesis_batch)                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.1 What HypothesisManager returns

| Return Field | Type | Consumed By | How |
|---|---|---|---|
| `.dispositions` | `tuple[HypothesisDisposition]` | **Nothing** | Never accessed |
| `.auto_accept` | `tuple[Hypothesis, ...]` | `_apply_extraction_to_state` lines 1014-1038 | Iterated, values written to ProjectState |
| `.verify` | `tuple[Hypothesis, ...]` | **Nothing** (logged only at 1150) |
| `.hold` | `tuple[Hypothesis, ...]` | **Nothing** (logged only at 1151) |
| `.discard` | `tuple[Hypothesis, ...]` | **Nothing** (logged only at 1152) |
| `has_actionable()` | method | **Nothing** | Never called |

**Only `auto_accept` is consumed**. Verify/hold/discard are logged and discarded. Has_actionable() is never called.

---

## 2. Construction Sites

| File | Line | What | Notes |
|---|---|---|---|
| `hypothesis_manager.py` | 370 | `HypothesisManager(policy)` | Inside `get_hypothesis_manager()`, lazy singleton |
| `hypothesis_manager.py` | 366-371 | `get_hypothesis_manager(policy=None)` | Creates only when `_HYPOTHESIS_MANAGER is None` |
| `mentor.py` | 104 | `_HYPOTHESIS_MANAGER = get_hypothesis_manager()` | Module-level eager init on import |
| `mentor.py` | 1143 | `hypothesis_manager = get_hypothesis_manager()` | In `process_mentor_turn()` — returns existing singleton |

---

## 3. Files Requiring Modification

### 3.1 Files to DELETE

| File | LOC | Reason |
|---|---|---|
| `D:\ThinkingPods\ReqGPT\hypothesis_manager.py` | 371 | Entire module becomes dead code |

### 3.2 Files to MODIFY

| File | LOC Changed | Changes |
|---|---|---|
| `D:\ThinkingPods\ReqGPT\mentor.py` | 6 sites | Remove import, module-level singleton reference, `evaluate_batch()` call + print block, `hypothesis_batch` param from `_apply_extraction_to_state`, AUTO_ACCEPT write block, `hypothesis_batch` arg from the call site |

### 3.3 Files with ZERO changes needed (confirmed)

| File | Reason |
|---|---|
| `tests/test_memory_extractor.py` | Calls `_apply_extraction_to_state(session, state, result)` with 3 positional args; `hypothesis_batch` defaults to `None` — removing the param is source-compatible |
| `tests/test_objective_engine_integration.py` | Calls `process_mentor_turn()` which will no longer invoke HM — no signature change |
| `module4/response_strategy.py` | Imports `Hypothesis` and `InferenceResult` from contracts, NOT from hypothesis_manager |
| `module4/__init__.py` | Does not re-export hypothesis_manager |
| `inference_engine.py` | Imports from contracts, NOT from hypothesis_manager |
| `contracts.py` | Hypothesis, InferenceResult, HypothesisStatus, HypothesisKind are independent of hypothesis_manager |
| `state_manager.py` | Imports InferenceResult from contracts, NOT from hypothesis_manager |
| `session_manager.py` | Comment-only reference ("Hypothesis management") — not a code dependency |
| `dina_direct.py` | Zero references to hypothesis_manager |
| `design_thinking_coach.py` | Zero references to hypothesis_manager |
| `design_thinking_coach_text.py` | Zero references to hypothesis_manager |

---

## 4. Hidden Dependencies and Risks

### 4.1 Risks with `get_hypothesis_manager()` import

The import at `mentor.py:33` is:
```python
from hypothesis_manager import get_hypothesis_manager
```

This will raise `ModuleNotFoundError` at `import mentor` time — not at call time — if the file is deleted without removing the import. **Fix must be applied atomically.**

### 4.2 Risks with `_HYPOTHESIS_MANAGER` module-level variable

`mentor.py:104`:
```python
_HYPOTHESIS_MANAGER = get_hypothesis_manager()
```

This is evaluated at module load (eager init). If the import is removed but the variable assignment line is not, `NameError` on `get_hypothesis_manager`. **Must remove both import and assignment atomically.**

### 4.3 No circular import risk

HypothesisManager imports:
```python
from contracts import ConfidenceLevel, Hypothesis, HypothesisKind, HypothesisStatus
from knowledge_base import ConceptCategory, get_knowledge_base
```

It does NOT import from mentor.py, inference_engine.py, or any module3/4/5 package. No cycle exists.

### 4.4 No re-export risk

No `__init__.py` (mentor, module3, module4, module5, contracts) re-exports from hypothesis_manager. The module is ONLY imported at `mentor.py:33`.

### 4.5 No performance risk

Removing the `evaluate_batch()` call removes the per-turn iteration over all hypotheses. The print statement at 1149-1152 is also removed. This is a pure latency reduction (the HM loop is O(N) on hypothesis count, currently ~0-30 items).

### 4.6 Secondary code that becomes latent

Removing HM also makes the following code unreachable, though it is kept for now:
- `inference_engine.py:_load_existing_hypotheses()` — the only code that loads "prior hypotheses" for HM to merge with
- `inference_engine.py:_merge_hypotheses()` — merges new candidates with existing (which HM consumes as `.updated_hypotheses`)
- `inference_engine.py:_detect_conflicts()`, `_apply_conflicts_and_status()`, `_identify_retired()` — lifecycle steps that assign the statuses HM then re-evaluates

These become latent (not causing an error, but their output goes nowhere). A follow-up PR should strip these from InferenceEngine.

---

## 5. PR 1 Implementation Checklist

### Phase A: Remove hypothesis_manager.py (delete file)

- [ ] Delete `D:\ThinkingPods\ReqGPT\hypothesis_manager.py`

### Phase B: Remove all HypothesisManager references from mentor.py

- [ ] **Line 33**: Remove `from hypothesis_manager import get_hypothesis_manager`
- [ ] **Lines 97-104**: Remove the `# Module 6 — Hypothesis Manager` comment block and `_HYPOTHESIS_MANAGER = get_hypothesis_manager()` assignment
- [ ] **Lines 1141-1153**: Remove the `# HypothesisManager — evaluate hypotheses` comment block, the `get_hypothesis_manager()` call, the `evaluate_batch()` call, and the print statement. Replace with a concise comment.

Change from:
```python
    # HypothesisManager — evaluate hypotheses and decide dispositions
    # (AUTO_ACCEPT / VERIFY / HOLD / DISCARD)
    hypothesis_manager = get_hypothesis_manager()
    hypothesis_batch = hypothesis_manager.evaluate_batch(
        hypotheses=inference_result.updated_hypotheses,
        extraction_facts=(),  # TODO: pass extracted facts when available
    )

    print(
        f"[HypothesisManager] auto_accept={len(hypothesis_batch.auto_accept)} "
        f"verify={len(hypothesis_batch.verify)} hold={len(hypothesis_batch.hold)} "
        f"discard={len(hypothesis_batch.discard)}"
    )
```
To:
```python
    # (HypothesisManager removed — see PR 1)
    # Hypotheses no longer flow through lifecycle management.
```

### Phase C: Remove hypothesis_batch from _apply_extraction_to_state

- [ ] **Line 984**: Remove `hypothesis_batch=None,` parameter from signature
- [ ] **Lines 1009-1012**: Remove docstring paragraph "The hypothesis_batch is the output of HypothesisManager..."
- [ ] **Lines 1013-1038**: Remove the entire `if hypothesis_batch is not None:` block (the AUTO_ACCEPT write to ProjectState)
- [ ] **Line 1164**: Remove `hypothesis_batch,` from the `_apply_extraction_to_state(...)` call

### Phase D: Verify

- [ ] Run `python -c "import mentor"` — should complete without `ModuleNotFoundError` or `NameError`
- [ ] Run `python -m pytest tests/test_memory_extractor.py -v` — all bridge tests pass (they never pass `hypothesis_batch`)
- [ ] Run `python -m pytest tests/test_objective_engine_integration.py -v` — all 7 regression scenarios pass
- [ ] Run `python -c "from inference_engine import get_inference_engine; ie = get_inference_engine()"` — inference_engine still importable

### Phase E: Clean up

- [ ] Remove empty `# Module 6 — Hypothesis Manager` comment section header remaining at line 96-104
- [ ] Verify no remaining references to `hypothesis_manager`, `HypothesisManager`, `hypothesis_batch`, `get_hypothesis_manager`, `_HYPOTHESIS_MANAGER` in `mentor.py`

---

## 6. Summary

| Metric | Value |
|---|---|
| Files deleted | 1 |
| Files modified | 1 |
| Files unaffected | 20+ |
| Lines of code removed | ~410 (371 hypothesis_manager + ~39 mentor) |
| Production code added | 0 |
| Test changes required | 0 |
| Test assertions broken | 0 |
| Import errors if done wrong | `ModuleNotFoundError` at startup time (easily caught in Phase D) |
| Hidden dependencies | None (no re-exports, no circular imports) |
| Per-turn performance gain | ~0.1-0.5 ms (eliminates HM loop + print) |
