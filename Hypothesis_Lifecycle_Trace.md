# Hypothesis Lifecycle Trace: Creation to Consumption

## 0. The Hypothesis Data Model

From `contracts.py:255-287`:

```python
@dataclass(frozen=True)
class Hypothesis:
    kind: HypothesisKind
    value: str
    confidence: ConfidenceLevel
    supporting_facts: tuple[ExtractedFact, ...]
    contradicting_facts: tuple[ExtractedFact, ...]
    related_hypotheses: tuple[str, ...]
    status: HypothesisStatus
    metadata: dict[str, Any]
```

---

## 1. Hypothesis Creation Points (7 distinct paths)

### 1.1. Fresh candidates from inference rules

**File**: `inference_engine.py:698-733` (`_form_hypothesis_candidates()`)

```
Line 713-731: Hypothesis(
    kind=bundle.kind,
    value=bundle.value,
    confidence=conf_to_level(bundle.final_confidence),
    supporting_facts=bundle.supporting_facts,
    contradicting_facts=bundle.contradicting_facts,
    related_hypotheses=(),
    status=HypothesisStatus.ACTIVE,                     # ← always starts ACTIVE
    metadata={
        "concept_id": bundle.concept_id,
        "concept_inference": bundle.concept_inference.value,
        "base_confidence": bundle.base_confidence,
        "fact_boost": bundle.fact_boost,
        "final_confidence": bundle.final_confidence,
        "rationale": bundle.rationale_parts,
    },
)
```

**Trigger**: For each evidence bundle (grouped KB matches) that was not filtered as an explicit duplicate.

**Fields written**: kind, value, confidence, supporting_facts, contradicting_facts, status, metadata (all populated).

---

### 1.2. "Memory" hypotheses from ProjectState reconstruction

**File**: `inference_engine.py:401-443` (`_load_existing_hypotheses()`)

```
Line 430-441: Hypothesis(
    kind=kind,
    value=value.strip(),
    confidence=ConfidenceLevel.HIGH,                    # ← hardcoded HIGH
    supporting_facts=(),                                # ← empty
    contradicting_facts=(),                             # ← empty
    related_hypotheses=(),                              # ← empty
    status=HypothesisStatus.CONFIRMED,                  # ← hardcoded CONFIRMED
    metadata={"source": "memory", "turn_added": "prior"},
)
```

**Trigger**: For every non-empty value in ProjectState fields (personas, problems, etc).

**Purpose**: Pretends explicit user facts in ProjectState are "existing hypotheses" so the merge step can combine them with new candidates. **This is the root cause of the §6 facts-vs-hypotheses violation.**

---

### 1.3. Merged hypothesis copies

**File**: `inference_engine.py:762-773` (`_merge_hypotheses()`, inside `merged[match_idx] = merged_hyp`)

```
Line 763-772: Hypothesis(
    kind=existing_hyp.kind,
    value=existing_hyp.value,
    confidence=new_confidence,                          # ← boosted
    supporting_facts=merged_facts,                      # ← combined
    contradicting_facts=combined,                       # ← combined
    related_hypotheses=existing_hyp.related_hypotheses,
    status=existing_hyp.status,                         # ← preserved
    metadata={**existing_hyp.metadata, "revised_turn": "current"},
)
```

**Trigger**: When a new candidate matches an existing hypothesis (same kind, similar value). A copy is created with combined evidence.

---

### 1.4. Unmatched candidate (no merge)

**File**: `inference_engine.py:777-778` (`_merge_hypotheses()`, else branch)

```
Line 777-778: merged.append(candidate)
    → the candidate Hypothesis from 1.1, unchanged
```

**Trigger**: No existing hypothesis matches the candidate. The candidate from 1.1 is kept as-is.

---

### 1.5. Status-assigned copies

**File**: `inference_engine.py:871-882` (`_apply_conflicts_and_status()`)

```
Line 871-882: Hypothesis(
    kind=hyp.kind,
    value=hyp.value,
    confidence=confidence,                              # ← possibly penalized
    supporting_facts=hyp.supporting_facts,
    contradicting_facts=hyp.contradicting_facts,
    related_hypotheses=hyp.related_hypotheses,
    status=status,                                      # ← CONFIRMED / REJECTED / ACTIVE
    metadata=hyp.metadata,
)
```

**Trigger**: For every hypothesis in the merged list. Confidence may be penalized if conflicted. Status may change: HIGH→CONFIRMED, LOW→REJECTED, conflicted→ACTIVE.

---

### 1.6. Retired (REJECTED) copies

**File**: `inference_engine.py:901-912` (`_identify_retired()`)

```
Line 902-912: Hypothesis(
    kind=h.kind,
    value=h.value,
    confidence=h.confidence,
    supporting_facts=h.supporting_facts,
    contradicting_facts=h.contradicting_facts,
    related_hypotheses=h.related_hypotheses,
    status=HypothesisStatus.REJECTED,                   # ← forced REJECTED
    metadata={**h.metadata, "retired": True},
)
```

**Trigger**: For hypotheses that existed in `existing` (from `_load_existing_hypotheses`) but are absent from `final` (after merge+conflict+status).

---

### 1.7. HypothesisManager disposition copies (modification, not creation)

**File**: `hypothesis_manager.py:331-343` (`_evaluate_single()`)

The HypothesisDisposition wraps the Hypothesis unchanged:
```
Line 147: hypothesis: Hypothesis   # ← same object, not copied
```

The Hypothesis object itself is NOT modified; the disposition carries a reference to the original Hypothesis returned by InferenceEngine.

---

## 2. Hypothesis Modification Points

### 2.1. No in-place mutation (frozen dataclass)

Hypothesis is a frozen dataclass (`@dataclass(frozen=True)` at `contracts.py:254`). Every "modification" actually creates a new Hypothesis object. The following lines create fresh instances:

| Step | File:Line | Creates From |
|---|---|---|
| Merge (existing + candidate) | `inference_engine.py:763-772` | existing hypothesis + candidate fields |
| Status assignment | `inference_engine.py:871-882` | each hypothesis |
| Retirement | `inference_engine.py:901-912` | each retired hypothesis |

The InferenceEngine builds three arbitrary new lists, each containing new Hypothesis objects:

```
infer() output:
    .updated_hypotheses = tuple(final_hypotheses)     ← from _apply_conflicts_and_status
    .new_hypotheses      = tuple(new_hypotheses)       ← from _merge_hypotheses
    .revised_hypotheses  = tuple(revised_hypotheses)   ← from _merge_hypotheses
    .retired_hypotheses  = tuple(retired_hypotheses)   ← from _identify_retired
```

Note: `updated_hypotheses` is a SUPERSET of `new_hypotheses ∪ revised_hypotheses` but the three are produced by different methods and may contain different Hypothesis object identities (due to status copy in step 1.5).

---

## 3. Hypothesis Storage

### 3.1. No persistent hypothesis store

Hypotheses are NOT persisted between turns. At the end of `process_mentor_turn()`, SessionManager saves:

```
mentor.py:1396-1401: new_session_mgr.save_project_state(session_id, project_state)
mentor.py:1404: SessionManager.save(session, ...)    # legacy mentor.json
```

Neither of these persistence calls saves the InferenceResult or the HypothesisManager's output. The `session` object (MentorSession) has no `hypotheses` field. The new session folder structure (`sessions/*/`) has no `hypotheses.json` file.

### 3.2. Cross-turn "state" via ProjectState reconstruction

The ONLY cross-turn hypothesis mechanism is `_load_existing_hypotheses()` which reads ProjectState fields and reifies them as hypotheses. This means:

- `supporting_facts` field is LOST across turns (reconstructed hypotheses have `supporting_facts=()`)
- `contradicting_facts` field is LOST across turns
- `related_hypotheses` field is LOST across turns
- `metadata` field is LOST across turns (replaced with `{"source": "memory", "turn_added": "prior"}`)
- `confidence` is LOST (replaced with hardcoded `HIGH`)
- `status` is LOST (replaced with hardcoded `CONFIRMED`)

Every turn, all hypotheses are effectively regenerated from scratch. Only the explicit fact values survive (as partial `value` strings) — all lifecycle metadata is discarded.

### 3.3. HypothesisManager is stateless

At `hypothesis_manager.py:200-211`:
```python
class HypothesisManager:
    def __init__(self, policy: DecisionPolicy | None = None) -> None:
        self._policy = policy or DEFAULT_POLICY
        self._kb = get_knowledge_base()
```

No `_current_hypotheses` field. No internal cross-turn state. Each `evaluate_batch()` call is independent.

---

## 4. Hypothesis Passing (who receives the Hypothesis objects)

### 4.1. InferenceEngine → InferenceResult

**File**: `inference_engine.py:293-300`
```python
return InferenceResult(
    updated_hypotheses=tuple(final_hypotheses),     # ← all hypotheses with statuses
    new_hypotheses=tuple(new_hypotheses),
    revised_hypotheses=tuple(revised_hypotheses),
    retired_hypotheses=tuple(retired_hypotheses),
    conflicts_detected=tuple(conflicts),
    reasoning_trace=tuple(reasoning_trace),
)
```

### 4.2. InferenceResult → HypothesisManager

**File**: `mentor.py:1143-1147`
```python
hypothesis_batch = hypothesis_manager.evaluate_batch(
    hypotheses=inference_result.updated_hypotheses,   # ← the full flat list
    extraction_facts=(),                               # ← always empty tuple
)
```

### 4.3. HypothesisBatchResult → _apply_extraction_to_state

**File**: `mentor.py:1159-1167`
```python
_apply_extraction_to_state(
    session, project_state, extraction_result,
    inference_result,       # ← also passed, but ignored (_ = inference_result at state_manager.py:450)
    hypothesis_batch,       # ← contains .auto_accept[] written to ProjectState
    ...
)
```

### 4.4. InferenceResult → ResponseStrategyEngine

**File**: `mentor.py:1232-1234` — **NOT PASSED**
```python
candidate_strategy = _RESPONSE_STRATEGY_ENGINE.determine_strategy(
    objective, project_state
    # ← NO inference_result argument
)
```

Despite the signature accepting `Optional[InferenceResult]` at `response_strategy.py:286`, the live pipeline NEVER provides it. The parameter defaults to `None`.

### 4.5. InferenceResult → PromptBuilder

**File**: `mentor.py:1331-1338` — **NOT PASSED**
```python
prompt = build_module4_prompt(
    project_state=project_state,
    conversation_objective=objective,
    response_strategy=response_strategy,
    previous_assistant_message=...,
    previous_user_message=...,
    empathize_summary=empathize_summary,
    # ← NO hypotheses / inference_result parameter
)
```

The PromptBuilder's `build_prompt` signature does not accept InferenceResult (only `response_plan`, which contains strategy info but not hypotheses).

---

## 5. Hypothesis Consumption (fields actually read downstream)

### 5.1. By HypothesisManager

**File**: `hypothesis_manager.py:269-344` (`_evaluate_single()`)

| Field | Read at line | Used For |
|---|---|---|
| `hypothesis.confidence` | 278 | Base confidence mapping (`confidence_map.get`) |
| `hypothesis.metadata` | 282 | `metadata.get("concept_id")` for category baseline adjustment |
| `hypothesis.supporting_facts` | 289 | `len(...)` for evidence boost |
| `hypothesis.contradicting_facts` | 294 | `len(...)` for contradiction penalty |
| `hypothesis.kind` | 304 | Field name lookup for policy overrides |

**Fields NOT read**: `value`, `related_hypotheses`, `status`

Note: HM deliberately ignores InferenceEngine's status assignment (it produces its own decision: AUTO_ACCEPT/VERIFY/HOLD/DISCARD).

### 5.2. By ResponseStrategyEngine (code path exists but is UNREACHABLE in live pipeline)

**File**: `response_strategy.py`

| Method | Field(s) Read | Line | Used For |
|---|---|---|---|
| `_get_relevant_hypotheses()` | `kind.value` | 526 | Filter by objective kind |
| `_select_target_hypothesis()` | `status`, `confidence.value`, `status.value`, `value` | 540-551 | Select the best hypothesis to ask about |
| `_detect_contradiction()` | `kind`, `value`, `confidence.value` | 555-558 | Detect contradictory pairs |
| `_resolve_strategy()` | `status`, `kind` | 592-594 | Decide CLARIFY vs. CORRECT vs. base |
| `_generate_candidate_intents()` | `kind.value`, `value` | 649-657 | Add verification candidates |
| `_estimate_information_gain()` | `kind.value` | 682 | Check if hypothesis matches expected fields |
| `_determine_question_focus()` | `status` | 714 | Get first active hypothesis |
| `_select_conversation_strategy()` | `status` | 745 | Choose follow_up vs. verification vs. exploration |

| Field | Times Read | Purpose |
|---|---|---|
| `kind` | 5 | Filtering, grouping, contradiction detection |
| `status` | 4 | Active/confirmed filtering, ambiguity detection |
| `value` | 3 | Target hypothesis key, contradiction check |
| `confidence` | 2 | Sort order, contradiction winner |

**Fields NOT read**: `supporting_facts`, `contradicting_facts`, `related_hypotheses`, `metadata`

However: ALL of this code is unreachable because `inference_result` defaults to `None` in the pipeline call.

### 5.3. By ObjectiveEngine

**NEVER receives hypotheses.** Signature at `objective_engine.py:112`:
```python
def determine_next(self, state: ProjectState) -> ConversationObjective:
```
Only `ProjectState` is consumed. No hypothesis parameter exists.

### 5.4. By LifecycleManager

**NEVER receives hypotheses.** Signature at `lifecycle_manager.py:233`:
```python
def determine(self, project_state, conversation_objective, response_strategy) -> LifecycleDecision:
```
No hypothesis parameter exists.

### 5.5. By PromptBuilder

**NEVER receives hypotheses.** Signature at `prompt_builder.py:353`:
```python
def build_prompt(project_state, conversation_objective, response_plan, ..., empathize_summary) -> str:
```
No hypothesis parameter. The prompt renders:
- `Known Project State` (from ProjectState — facts only)
- `Empathize Summary` (from EmpathizeSummary — facts only)
- `Current Objective`, `Role`, `Latest Conversation`, `Instructions`

No "Inferred Hypotheses" section exists in any prompt.

### 5.6. By LLM

The LLM prompt (string) is the output of PromptBuilder. Since PromptBuilder never renders hypotheses, the **LLM never sees hypotheses**. The LLM only sees explicit facts (from ProjectState) and the current objective.

---

## 6. Fields Summary: Written vs. Consumed

| Field | Written At | Consumed By | Status |
|---|---|---|---|
| `kind` | all creation sites (1.1-1.6) | HM (§5.1), RSE (unreachable, §5.2) | ✅ First-class |
| `value` | all creation sites | RSE (unreachable, §5.2) | ✅ Partially consumed |
| `confidence` | all creation sites | HM (§5.1), RSE (unreachable, §5.2) | ✅ First-class |
| `supporting_facts` | 1.1 `_form_hypothesis_candidates` | HM _evaluate_single (line 289, `len()` only) | ⚠️ **Only count consumed, not content** |
| `contradicting_facts` | 1.1 `_form_hypothesis_candidates` | HM _evaluate_single (line 294, `len()` only) | ⚠️ **Same — only count consumed** |
| `related_hypotheses` | all creation sites (set to `()`) | **Never read anywhere** | ❌ **Dead field** |
| `status` | 1.1 (ACTIVE), 1.2 (CONFIRMED), 1.5 (varies), 1.6 (REJECTED) | RSE (unreachable, §5.2) | ⚠️ **Effective dead field in pipeline** (RSE unreachable) |
| `metadata` | 1.1 (full), 1.2 (minimal), 1.3-1.6 (merged/passed) | HM _evaluate_single (line 282, `concept_id` only) | ⚠️ **Mostly dead — only concept_id consumed** |

---

## 7. Field-by-Field: Which fields survive the cross-turn reset

Between turns, ALL Hypothesis objects are discarded. The only information that survives is what gets written to ProjectState via AUTO_ACCEPT:

| Field | Survives to Next Turn? | How? |
|---|---|---|
| `kind` | **Only via AUTO_ACCEPT write** | `_apply_extraction_to_state()` maps PERSONA→personas list |
| `value` | **Only via AUTO_ACCEPT write** | Written as string to ProjectState field |
| `confidence` | **Lost** | Not persisted |
| `supporting_facts` | **Lost** | Not persisted |
| `contradicting_facts` | **Lost** | Not persisted |
| `related_hypotheses` | **Lost** | Not persisted |
| `status` | **Lost** | Replaced with CONFIRMED on reload (hardcoded at 1.2) |
| `metadata` | **Lost** | Replaced with `{"source": "memory", "turn_added": "prior"}` |

Next turn, `_load_existing_hypotheses` creates fresh Hypothesis objects from ProjectState strings, losing all confidence, evidence, status, and provenance information.

---

## 8. Turn-by-Turn Lifecycle Trace (Concrete Example)

```
TURN 1: User says "I am a student, I forget to do assignments"

Step A: MemoryExtractor.extract() produces:
    ExtractionUpdate(ADD, personas, "student")
    ExtractionUpdate(ADD, problems, "assignment forgetfulness")

Step B: InferenceEngine._convert_updates_to_facts()
    → ExtractedFact(field="personas", value="student", confidence=HIGH)
    → ExtractedFact(field="problems", value="assignment forgetfulness", confidence=HIGH)

Step C: InferenceEngine._load_existing_hypotheses(project_state)
    → ()  (empty — first turn)

Step D: InferenceEngine._match_concepts(facts)
    → KB matches "student" → Concept("student", inferences={personas: ["student"]})
    → KB matches "forget assignments" → Concept("forgetfulness", inferences={problems: ["forgetfulness"], pain_points: ["missed deadlines"]})

Step E: InferenceEngine._form_hypothesis_candidates(bundles)
    → Hypothesis(PERSONA, "student", confidence=HIGH, facts=1, status=ACTIVE)
    → Hypothesis(PROBLEM, "forgetfulness", confidence=HIGH, facts=1, status=ACTIVE)
    → Hypothesis(PAIN_POINT, "missed deadlines", confidence=MEDIUM, facts=0, status=ACTIVE)

Step F: InferenceEngine._merge_hypotheses(existing=[], new=[above 3])
    → Same 3 (no prior to merge with)

Step G: InferenceEngine._detect_conflicts(3 hypotheses)
    → None (all different kinds)

Step H: InferenceEngine._apply_conflicts_and_status(3, [])
    → Hypothesis(PERSONA, "student", confidence=HIGH, status=CONFIRMED)   ← HIGH→CONFIRMED
    → Hypothesis(PROBLEM, "forgetfulness", confidence=HIGH, status=CONFIRMED)
    → Hypothesis(PAIN_POINT, "missed deadlines", confidence=MEDIUM, status=ACTIVE)

Step I: InferenceResult.updated_hypotheses = [above 3]

Step J: HypothesisManager.evaluate_batch(hypotheses=[above 3], extraction_facts=())
    → Hypothesis(PERSONA, "student"): confidence=0.9, facts=1 → VERIFY (needs ≥2 for AUTO_ACCEPT)
    → Hypothesis(PROBLEM, "forgetfulness"): confidence=0.9, facts=1 → VERIFY
    → Hypothesis(PAIN_POINT, "missed deadlines"): confidence=0.55, facts=0 → HOLD

Step K: _apply_extraction_to_state()
    → HypothesisManager.auto_accept = ()  (none — all were VERIFY/HOLD)
    → StateManager.apply_extraction(extraction_result)
        → project_state.personas = ["student"]
        → project_state.problems = ["assignment forgetfulness"]
        → project_state.frequency = None
        ...

Step L: ProjectState saved:
    {personas: ["student"], problems: ["assignment forgetfulness"], ...}

Step M: ResponseStrategyEngine called WITHOUT inference_result
    → hypotheses have NO effect on strategy, prompt, or LLM


TURN 2: User says "It happens daily, and I miss deadlines"

Step A: MemoryExtractor: ExtractionUpdate(SET, frequency, "daily")

Step B-C: InferenceEngine._load_existing_hypotheses(project_state)
    → Hypothesis(PERSONA, "student", confidence=HIGH, status=CONFIRMED, metadata={source:"memory"})
    → Hypothesis(PROBLEM, "assignment forgetfulness", confidence=HIGH, status=CONFIRMED, metadata={source:"memory"})
      ⚠️ "student" was an explicit fact; now reconstructed as a "memory hypothesis"
      ⚠️ "assignment forgetfulness" confidence/stats LOST from Turn 1
      ⚠️ "student" was a VERIFY; now forced to CONFIRMED

Step D-E: New inference:
    → Hypothesis(FREQUENCY, "daily", confidence=HIGH, status=ACTIVE)
    → Hypothesis(PAIN_POINT, "missed deadlines", confidence=HIGH, status=ACTIVE)  (matches KB, now higher confidence because user repeated it)

Step F: _merge_hypotheses(existing=[memory hyps], new=[above])
    → [(PERSONA, student), (PROBLEM, assignment forgetfulness), (FREQUENCY, daily), (PAIN_POINT, missed deadlines)]

Step H-J: Status + HM evaluation, same as turn 1.

Step L: project_state.frequency = "daily"
        project_state.problems stays = ["assignment forgetfulness"]
        (no new ADD to problems — duplicate)
        project_state.pain_points = ["missed deadlines"]

Step M: Hypotheses still do NOT reach RSE, PromptBuilder, or LLM.
```

---

## 9. Conclusion: First-Class Artifact or Intermediate?

### 9.1. What hypotheses actually affect in the live pipeline

| Downstream Component | Receives Hypotheses? | Actually Uses Them? |
|---|---|---|
| HypothesisManager | YES — evaluates, produces AUTO_ACCEPT | YES — drives ProjectState mutation |
| StateManager | NO (inference_result ignored) | NO |
| ObjectiveEngine | NO | NO |
| LifecycleManager | NO | NO |
| ResponseStrategyEngine | NO (parameter always None) | Code exists for it, but unreachable |
| PromptBuilder | NO | NO |
| LLM | NO | NO |

### 9.2. Answer

**Hypotheses are an intermediate artifact that is fully regenerated every turn and has no downstream impact beyond the AUTO_ACCEPT commit path.**

Evidence:
1. Every field except `value` (as a ProjectState string) is lost between turns.
2. HypothesisManager's decisions only reach ProjectState via the AUTO_ACCEPT→`_apply_extraction_to_state` path, which is a fact-contamination path (§6 violation).
3. ResponseStrategyEngine has a full hypothesis-consumption subsystem (`_get_relevant_hypotheses`, `_select_target_hypothesis`, `_detect_contradiction`, `_resolve_strategy`, `_generate_candidate_intents`, `_estimate_information_gain`) that is NEVER CALLED with non-None data in the live pipeline.
4. ObjectiveEngine, LifecycleManager, PromptBuilder, and the LLM never see hypotheses.
5. `supporting_facts`, `contradicting_facts`, `related_hypotheses`, `metadata['rationale']`, `metadata['final_confidence']` are written but never consumed by any downstream code (HM only reads `len()` of facts and `metadata['concept_id']`).

### 9.3. What would need to change for first-class status

Hypotheses would need to:
1. Be persisted between turns (in `hypotheses.json` or a HypothesisManager store)
2. Actually be passed to ResponseStrategyEngine (add `inference_result` to the `determine_strategy` call in mentor.py)
3. Be rendered in the prompt by PromptBuilder (add a "Inferred Hypotheses" section)
4. Have their lifecycle statuses respected by downstream code (not overwritten by `_load_existing_hypotheses`)

Currently, none of these four conditions is met. The entire hypothesis pipeline runs every turn, produces output that is partially discarded (only AUTO_ACCEPT values survive), and is rebuilt from scratch next turn using only ProjectState strings as seeds.
