# Architecture Root Cause Analysis: InferenceEngine ↔ HypothesisManager Overlap

## 1. Ownership Table (Spec vs. Implementation)

### 1.1. Spec (Architecture_V1.md §4.3, §4.4, §6)

| Responsibility | Owned By | In/Out |
|---|---|---|
| Generate hypotheses from facts + KB | InferenceEngine | `ExtractionResult + KB → Hypothesis[]` |
| Merge priors + new hypotheses | HypothesisManager | `(prior_hypotheses, new_hypotheses) → merged list` |
| Assign status (ACTIVE/CONFIRMED/REJECTED/SUPERSEDED) | HypothesisManager | output of merge |
| Detect contradictions | HypothesisManager | output of merge |
| Auto-accept HIGH → CONFIRMED | HypothesisManager | lifecycle |
| Demote LOW → REJECTED | HypothesisManager | lifecycle |
| Preserve insertion order | HypothesisManager | lifecycle |
| Persist explicit facts only (not hypotheses) | StateManager → ProjectState | `ExtractionResult → ProjectState` |
| Hypothesis cross-turn state | HypothesisManager (inferred), StateManager (facts) | separate stores |

### 1.2. Implementation (Actual)

| Responsibility | Implemented In | Method / Lines |
|---|---|---|
| Generate hypotheses from facts + KB | InferenceEngine | `_match_concepts()`, `_build_evidence_bundles()`, `_form_hypothesis_candidates()` |
| Merge new candidates with existing | InferenceEngine | `_merge_hypotheses()` (line 739) |
| Detect contradictions | InferenceEngine | `_detect_conflicts()` (line 819) |
| Assign status (ACTIVE/CONFIRMED/REJECTED) | InferenceEngine | `_apply_conflicts_and_status()` (line 841) |
| Identify retired (REJECTED) | InferenceEngine | `_identify_retired()` (line 893) |
| Reconstruct hypotheses from ProjectState | InferenceEngine | `_load_existing_hypotheses()` (line 401) |
| Decide AUTO_ACCEPT/VERIFY/HOLD/DISCARD | HypothesisManager | `_evaluate_single()` (line 269) |
| AUTO_ACCEPT applied to ProjectState | mentor.py `_apply_extraction_to_state()` (line 1014) | pipeline integrator |

### 1.3. Overlap Matrix

```
                         InferenceEngine       HypothesisManager
                        ┌────────────────────┬────────────────────┐
Merge hypotheses        │  _merge_hypotheses  │  (none)            │  ❌ Overlap
                        │  (line 739)         │                    │
├───────────────────────┼────────────────────┼────────────────────┤
Conflict detection      │  _detect_conflicts  │  (none)            │  ❌ Overlap
                        │  (line 819)         │                    │
├───────────────────────┼────────────────────┼────────────────────┤
Status assignment       │  _apply_conflicts   │  _evaluate_single  │  ❌ COMPETING
                        │  _and_status        │  (AUTO_ACCEPT/     │  (different
                        │  (ACTIVE/CONFIRMED/ │   VERIFY/HOLD/     │   schemes)
                        │   REJECTED)         │   DISCARD)         │
├───────────────────────┼────────────────────┼────────────────────┤
Hypothesis→State commit │  (none)             │  via pipeline      │  ✅ Split
                        │                     │  integrator        │
└───────────────────────┴────────────────────┴────────────────────┘
```

## 2. Sequence Diagrams

### 2.1. Current Flow (What the Pipeline Actually Does)

```
User Message
    │
    ▼
MemoryExtractor.extract()
    │  ┌──────────────────────────────┐
    │  │ ExtractionResult             │
    │  │  - message_type              │
    │  │  - updates[ExtractionUpdate] │
    ▼  └──────────────────────────────┘
InferenceEngine.infer(extraction_result, project_state)
    │
    ├── _convert_updates_to_facts()        # ExtractionUpdate → ExtractedFact
    ├── _load_existing_hypotheses()         # ⚠️ ProjectState → Hypothesis (CONFIRMED)
    │                                        #    Breaks §6: explicit facts become hypotheses
    ├── _match_concepts()                   # Facts → KB matches
    ├── _build_evidence_bundles()           # Group matches
    ├── _filter_explicit_duplicates()       # Remove inferred if user already said it
    ├── _form_hypothesis_candidates()       # Bundles → Hypothesis (status=ACTIVE)
    ├── _merge_hypotheses()                 # ⚠️ Existing + new = merged list
    │                                        #    Merges "memory" hypotheses (from state)
    │                                        #    with new inference candidates
    ├── _detect_conflicts()                 # ⚠️ Contradiction pairs
    ├── _apply_conflicts_and_status()       # ⚠️ Assigns ACTIVE/CONFIRMED/REJECTED
    ├── _identify_retired()                 # ⚠️ Marks missing as REJECTED
    │
    │  ┌──────────────────────────────────────────────┐
    │  │ InferenceResult                              │
    │  │  - updated_hypotheses (with statuses already │
    │  │    assigned by IE: ACTIVE/CONFIRMED/REJECTED)│
    │  │  - new_hypotheses                            │
    │  │  - revised_hypotheses                        │
    ▼  └──────────────────────────────────────────────┘
HypothesisManager.evaluate_batch(hypotheses=inference_result.updated_hypotheses, ...)
    │
    ├── _evaluate_single()                  # Re-evaluates each hypothesis
    │                                        # ⚠️ Uses DIFFERENT scheme:
    │                                        #    AUTO_ACCEPT/VERIFY/HOLD/DISCARD
    │                                        # ⚠️ Ignores InferenceEngine's statuses
    │
    │  ┌──────────────────────────────────────────────┐
    │  │ HypothesisBatchResult                        │
    │  │  - auto_accept[]    → written to ProjectState │
    │  │  - verify[]         → (ignored)              │
    │  │  - hold[]           → (ignored)              │
    │  │  - discard[]        → (ignored)              │
    ▼  └──────────────────────────────────────────────┘
_apply_extraction_to_state()
    │
    ├── Apply HypothesisManager.auto_accept[] to ProjectState  # writes hypotheses as facts
    ├── StateManager.apply_extraction(extraction_result)        # writes explicit facts
    │     └── inference_result ignored (_ = inference_result)
    ▼
ProjectState (updated)
    │  ┌──────────────────────────────────────────────┐
    │  │ Now contains:                                │
    │  │  - explicit facts (from MemoryExtractor)      │
    │  │  - inferred hypotheses (from HypothesisManager│
    │  │    AUTO_ACCEPT) written as if they were facts │
    │  │ ⚠️ No way to tell provenance apart           │
    ▼  └──────────────────────────────────────────────┘
ObjectiveEngine → LifecycleManager → ResponseStrategyEngine → PromptBuilder → LLM
    │
    ▼
SessionManager.save_project_state()
    │
    ▼  (next turn)
InferenceEngine._load_existing_hypotheses()
    │  ⚠️ Reads the mixed state back as "existing hypotheses"
    │  ⚠️ AUTO_ACCEPTED inferences now indistinguishable from explicit facts
    ▼
...cycle repeats...
```

### 2.2. Intended Flow (Architecture_V1.md §3, §6)

```
User Message
    │
    ▼
MemoryExtractor.extract()
    │  ┌──────────────────────────────┐
    │  │ ExtractionResult             │
    │  │  - facts[ExtractedFact]      │ ← only explicit user statements
    ▼  └──────────────────────────────┘
InferenceEngine.infer(extraction_result, project_state)
    │
    ├── Match facts against KnowledgeBase
    ├── Apply inference rules
    ├── Assign confidence
    ├── Never create hypotheses that duplicate explicit facts (§7 rule)
    └── Never mutate state (§7 rule)
    │
    │  ┌──────────────────────────────────────────────┐
    │  │ InferenceResult                              │
    │  │  - updated_hypotheses (NO statuses assigned, │
    │  │    NO merging with prior, NO conflicts)       │
    │  │    Just new hypotheses with confidence only   │
    ▼  └──────────────────────────────────────────────┘
HypothesisManager.evaluate(prior_hypotheses, new_hypotheses)
    │
    ├── Merge new + priors (SUPERSEDED duplicates)
    ├── Auto-accept HIGH → CONFIRMED
    ├── Demote LOW → REJECTED
    ├── Detect contradictions
    │
    │  ┌──────────────────────────────────────────────┐
    │  │ Merged list with statuses:                   │
    │  │  ACTIVE / CONFIRMED / REJECTED / SUPERSEDED  │
    ▼  └──────────────────────────────────────────────┘
StateManager.apply_extraction(extraction_result)
    │  ┌──────────────────────────────────────────────┐
    │  │ ProjectState (facts ONLY, no hypotheses)     │
    │  │ Hypotheses persist SEPARATELY                │
    ▼  └──────────────────────────────────────────────┘
ObjectiveEngine → LifecycleManager → ResponseStrategyEngine → PromptBuilder → LLM

Hypotheses are NOT written to ProjectState.
Hypotheses influence PromptBuilder directly (shown in prompt as "We infer: ...").
Only user-confirmed facts enter ProjectState via extraction.

  (next turn)
HypothesisManager receives prior_hypotheses from its own store (not from ProjectState)
InferenceEngine generates new hypotheses from fresh extractions only
```

## 3. Key Divergences

### 3.1. Overlap #1: InferenceEngine Does Hypothesis Merging (§4.3→§4.4 violation)

- **Spec says**: InferenceEngine generates hypotheses; HypothesisManager merges (§4.3 vs §4.4).
- **Code**: `InferenceEngine._merge_hypotheses()` at `inference_engine.py:739` merges existing (reconstructed from ProjectState) with new candidates.
- **Root cause**: When `_load_existing_hypotheses()` (line 401) reifies ProjectState fields as `Hypothesis(status=CONFIRMED)` objects, the subsequent merge step conflates two distinct pipelines: explicit-fact persistence and inferred-hypothesis lifecycle. The InferenceEngine was designed as a pure stateless generator, but it now carries cross-turn state through ProjectState.

### 3.2. Overlap #2: InferenceEngine Does Conflict Detection (§4.3→§4.4 violation)

- **Spec says**: HypothesisManager detects contradictions (§4.4).
- **Code**: `InferenceEngine._detect_conflicts()` at `inference_engine.py:819`.
- **Root cause**: Same as 3.1 — once hypotheses are merged within IE, conflict detection must follow immediately or the merged list would be inconsistent. The architectural layering was collapsed.

### 3.3. Overlap #3: InferenceEngine Does Status Assignment (§4.3→§4.4 violation)

- **Spec says**: InferenceEngine "does not decide hypothesis status (ACTIVE/CONFIRMED/REJECTED)" (§4.3 non-responsibilities).
- **Code**: `InferenceEngine._apply_conflicts_and_status()` at `inference_engine.py:841` assigns `CONFIRMED` to HIGH confidence, `REJECTED` to LOW, and `ACTIVE` to conflicted.
- **Root cause**: The scheme used by IE (ACTIVE/CONFIRMED/REJECTED) is exactly the life cycle the spec assigns to HypothesisManager. IE implements it preemptively, so HypothesisManager's own evaluation (`_evaluate_single` producing AUTO_ACCEPT/VERIFY/HOLD/DISCARD) runs on top of an already-status-assigned list with a completely different decision scheme. The two outputs are semantically incompatible.

### 3.4. Competing Scheme #4: Two Incompatible Lifecycle Systems

| Dimension | InferenceEngine (inference_engine.py) | HypothesisManager (hypothesis_manager.py) |
|---|---|---|
| Decision output | ACTIVE, CONFIRMED, REJECTED | AUTO_ACCEPT, VERIFY, HOLD, DISCARD |
| Threshold | HIGH→CONFIRMED, LOW→REJECTED | Confidence ≥0.85 + ≥2 facts → AUTO_ACCEPT |
| Cross-turn state | Reads from ProjectState (reconstructs) | Stateless per batch |
| Evidence counting | Fact boost + contradiction penalty in `_EvidenceBundle` | Evidence boost + contradiction penalty in `_evaluate_single` |
| Purpose | Final status for output | Action recommendation for pipeline |

These two schemes run **sequentially** with contradictory semantics:
1. IE assigns `CONFIRMED` to a hypothesis (meaning: "accepted as true").
2. HM evaluates the same hypothesis and returns `VERIFY` (meaning: "needs confirmation before accepting").
3. The pipeline integrator in `mentor.py:_apply_extraction_to_state` only acts on `AUTO_ACCEPT` — it **ignores** both the IE status and the other HM decisions.

### 3.5. Explicit Facts → Hypotheses → ProjectState Loop (§6 violation)

The complete path where explicit facts become hypotheses and back:

```
1. User says "I am a student"
2. MemoryExtractor → ExtractionUpdate(ADD, personas, "student") → ExtractedFact
3. InferenceEngine._convert_updates_to_facts() → ExtractedFact(field="personas", value="student")
4. InferenceEngine._load_existing_hypotheses(project_state)  ← empty on first turn
5. InferenceEngine._form_hypothesis_candidates() → [Hypothesis(kind=PERSONA, value="student", status=ACTIVE)]
6. InferenceEngine._merge_hypotheses(existing=[], new=[(PERSONA, "student")])
   → merged=[Hypothesis(PERSONA, "student", status=ACTIVE)]
7. InferenceEngine._apply_conflicts_and_status() → status becomes CONFIRMED (HIGH confidence)
8. InferenceResult.updated_hypotheses = [Hypothesis(PERSONA, "student", CONFIRMED)]
9. HypothesisManager._evaluate_single() →
   decision = AUTO_ACCEPT (confidence 0.9 ≥ 0.85, 1 supporting fact meets verify threshold but auto_accept needs 2)
   → policy: 0.9 confidence + only 1 fact → min_auto_accept = 2 → actually VERIFY not AUTO_ACCEPT
10. _apply_extraction_to_state() applies AUTO_ACCEPT (writes to ProjectState.personas)
    AND StateManager.apply_extraction(extraction_result) also writes the explicit fact

11. NEXT TURN:
    InferenceEngine._load_existing_hypotheses(project_state)
    → Reads "student" from project_state.personas
    → Creates Hypothesis(PERSONA, "student", CONFIRMED, source="memory")
    → This hypothesis was ORIGINALLY an explicit fact, now it's a "memory hypothesis"

12. NEW EXTRACTION: user says nothing new, or adds data
    → _merge_hypotheses() merges "memory" hypothesis with new candidates
    → System can no longer distinguish "user said 'student'" from "we inferred 'commuter'"
```

### 3.6. Provenance Loss Chain

```
Turn 1:
  User: "I am a student"
  state.personas = ["student"]  ← explicit fact ✓

Turn 2:
  InferenceEngine._load_existing_hypotheses()
    → Hypothesis(PERSONA, "student", status=CONFIRMED, metadata={source: "memory"})
    ← Now "student" is BOTH an explicit fact (in ProjectState) AND a hypothesis (in InferenceResult)

  New extraction from User: "I commute by bus"
    → Hypothesis(PERSONA, "commuter", status=ACTIVE)

  _merge_hypotheses([(PERSONA, "student", CONFIRMED)], [(PERSONA, "commuter", ACTIVE)])
    → merged [(PERSONA, "student", CONFIRMED), (PERSONA, "commuter", ACTIVE)]

  _detect_conflicts() → none (not mutually exclusive)

Turn 3:
  State.personas = ["student", "commuter"]  ← commuter was AUTO_ACCEPTED
  InferenceEngine._load_existing_hypotheses()
    → Hypothesis(PERSONA, "student", CONFIRMED, source="memory")
    → Hypothesis(PERSONA, "commuter", CONFIRMED, source="memory")
    ← Both now have source="memory". Lost: "student" was user-stated, "commuter" was inferred
    ← Lost: "commuter" was never explicitly confirmed by user
```

### 3.7. Stateless HypothesisManager Gap

The spec's HypothesisManager (§4.4) receives `(prior_hypotheses, new_hypotheses)` — implying cross-turn state. The implementation's `evaluate_batch(hypotheses, extraction_facts)` is stateless: it receives all hypotheses as a single flat batch and has no concept of "prior" vs "new". It cannot detect that a hypothesis existed last turn (and should have its status preserved) vs. appeared this turn (and needs initial evaluation). Cross-turn tracking is implicitly delegated to InferenceEngine via `_load_existing_hypotheses`, which makes IE the de facto hypothesis store.

## 4. Summary of Violations

| # | Rule (Architecture_V1.md) | Violation | Location |
|---|---|---|---|
| 1 | §4.3: IE generates; HM manages lifecycle | IE does merge, conflict, status | `inference_engine.py:739,819,841,893` |
| 2 | §4.3: IE "does not decide hypothesis status" | IE assigns CONFIRMED/REJECTED | `inference_engine.py:841-883` |
| 3 | §4.4: HM "merge new hypotheses with prior ones" | IE does it first with different scheme | `inference_engine.py:739-780` |
| 4 | §4.4: HM "detect contradictions" | IE does it first | `inference_engine.py:819-828` |
| 5 | §6: "Explicit facts entirely separate from inferred hypotheses" | IE reifies ProjectState fields as hypotheses | `inference_engine.py:401-443` |
| 6 | §4.4: HM input = `(prior_hypotheses, new_hypotheses)` | HM input = `(flat_batch, extraction_facts)` | `hypothesis_manager.py:217-221` |
| 7 | §4.4: HM output = `ACTIVE/CONFIRMED/REJECTED/SUPERSEDED` | HM output = `AUTO_ACCEPT/VERIFY/HOLD/DISCARD` | `hypothesis_manager.py:52-67` |
| 8 | §5: Hypothesis status owned by InferenceEngine/HypothesisManager | Docstring says "Lifecycle status of a hypothesis. Owned by Inference Engine" — contradicts spec's §4.4 (HM owns lifecycle) | `contracts.py:116-117` |
| 9 | §4.3: IE "never create hypotheses that duplicate explicit facts" | IE creates explicit duplicates via `_load_existing_hypotheses` — the `_filter_explicit_duplicates` step only filters newly inferred candidates, not the reconstructed memory hypotheses | `inference_engine.py:401-443` |

## 5. Data Ownership: Who Should Own What

| Data | Current Owner | Spec Owner | Gap |
|---|---|---|---|
| Hypothesis status | InferenceEngine + HypothesisManager (competing) | HypothesisManager | 2 owners, 2 schemes |
| Cross-turn hypothesis store | InferenceEngine (via ProjectState reconstruction) | HypothesisManager | HM is stateless; IE carries state |
| Hypothesis-to-fact promotion | `mentor.py` integrator (via AUTO_ACCEPT) | HypothesisManager (output) | Implicit, not component-owned |
| Explicit fact separation | None (mixed in ProjectState) | StateManager (persists), ProjectState (stores) | No provenance tag on facts |
