# Architecture_V1.md

## Version History

### Architecture V1
- Modular reasoning pipeline established
- Explicit facts separated from inferred hypotheses
- KnowledgeBase introduced
- Information-gain question planning added
- Architecture frozen

Future versions should only be created if a fundamental architectural change is required.

---

# 1. Overview

The Design Thinking Mentor backend is an AI-powered system that guides users through the Empathize phase of Design Thinking. It conducts structured conversations to discover personas, problems, current solutions, pain points, evidence, and frequency — the six checklist fields required for a complete Empathize brief.

The architecture is **modular by design**. Each stage of reasoning is isolated in its own component with explicit inputs, outputs, and responsibilities. This modularity enables deterministic reasoning, testability, and independent improvement of each component.

**Architecture V1 is considered frozen.** Future work should focus on improving reasoning quality, conversation quality, knowledge expansion, and user experience rather than restructuring the pipeline.

---

# 2. Design Principles

The following principles guided every architectural decision:

| Principle | Description |
|-----------|-------------|
| **Single responsibility per module** | Each component owns exactly one type of reasoning or data transformation. |
| **Modular pipeline** | Stages are chained sequentially with explicit contracts; no stage reaches into another's internals. |
| **Deterministic reasoning outside the LLM** | All state transitions, hypothesis management, objective selection, and strategy planning are pure functions. The LLM only generates natural language. |
| **LLM responsible only for natural language generation** | The LLM never decides conversation state, strategy, or what information is missing. |
| **Explicit facts separated from inferred knowledge** | What the user said (ExtractedFact) is distinct from what the system infers (Hypothesis). They flow through separate pipelines and are merged only at the prompt. |
| **Explainable reasoning** | Every decision (objective, strategy, hypothesis status) carries a human-readable `reasoning` field. |
| **Confidence-based inference** | Hypotheses carry confidence levels (HIGH/MEDIUM/LOW/UNKNOWN) that propagate through the pipeline. |
| **No monolithic prompts** | PromptBuilder assembles small, focused sections rather than one giant template. |
| **Easy to test** | Every component is a pure function of its inputs; 387+ unit tests cover the pipeline. |
| **Easy to extend** | New objectives, inference rules, or knowledge domains can be added without touching core architecture. |

---

# 3. Complete Pipeline

```
User Input
    ↓
MemoryExtractor  (extracts explicit facts from user message)
    ↓
InferenceEngine  (derives hypotheses from facts + knowledge base)
    ↓
HypothesisManager  (merges, promotes, demotes, rejects hypotheses)
    ↓
StateManager  (persists verified facts into ProjectState)
    ↓
ObjectiveEngine  (decides which checklist field to target next)
    ↓
LifecycleManager  (decides if Empathize is complete, needs summary, or awaiting confirmation)
    ↓
ResponseStrategyEngine  (plans how to ask: which question intent maximizes information gain)
    ↓
PromptBuilder  (assembles the final prompt for the LLM)
    ↓
LLM  (generates natural language response)
```

### Stage Responsibilities

| Stage | Responsibility |
|-------|----------------|
| **MemoryExtractor** | Parse user message → ExtractionResult (explicit facts only, no inference) |
| **InferenceEngine** | ExtractionResult + KnowledgeBase → InferenceResult (hypotheses with confidence) |
| **HypothesisManager** | Prior hypotheses + new hypotheses → merged, status-assigned hypotheses |
| **StateManager** | Validated ExtractionResult → mutated ProjectState (mission memory) |
| **ObjectiveEngine** | ProjectState + hypotheses → ConversationObjective (what to ask next) |
| **LifecycleManager** | Objective + state + summary marker → LifecycleDecision (continue/summary/confirm) |
| **ResponseStrategyEngine** | Objective + state + hypotheses → ResponsePlan (strategy + question intent + info-gain metadata) |
| **PromptBuilder** | ResponsePlan + state + history → prompt string for LLM |
| **LLM** | Prompt → natural language reply (no reasoning, no state decisions) |

---

# 4. Component Responsibilities

## 4.1 MemoryExtractor

**Purpose**: Extract explicit, user-stated facts from natural language.

**Inputs**: User message (string), previous assistant message (optional), previous user message (optional), ProjectState (for context).

**Outputs**: `ExtractionResult` containing `ExtractedFact` list and `message_type` (MEANINGFUL / NO_UPDATE / AMBIGUOUS / END).

**Responsibilities**:
- Parse LLM output into structured `ExtractedFact` objects
- Validate against `ExtractionValidator` (operation/field compatibility, duplicate detection)
- Apply conservative fallback: if confidence < 80% or JSON invalid → `AMBIGUOUS`
- Never perform inference; only extract what the user explicitly said

**Non-responsibilities**:
- Does not infer missing information
- Does not mutate ProjectState (StateManager does)
- Does not decide what to ask next

## 4.2 KnowledgeBase

**Purpose**: Provide domain knowledge for inference rules.

**Inputs**: None at runtime (pre-loaded seed concepts).

**Outputs**: `Concept` objects with `name`, `kind`, `synonyms`, `indicators`, `confidence_boost`.

**Responsibilities**:
- Store seed concepts for each hypothesis kind (PERSONA, PROBLEM, CURRENT_SOLUTION, PAIN_POINT, EVIDENCE, FREQUENCY)
- Provide `find_concepts(text)` to match user language to known concepts
- Enable InferenceEngine to recognize domain patterns without LLM calls

**Non-responsibilities**:
- Does not manage hypothesis lifecycle
- Does not persist state
- Does not generate questions

## 4.3 InferenceEngine

**Purpose**: Generate hypotheses from explicit facts using knowledge base and deterministic rules.

**Inputs**: `ExtractionResult` (from MemoryExtractor), `MissionMemory` (ProjectState for context).

**Outputs**: `InferenceResult` containing `updated_hypotheses` (list of `Hypothesis`).

**Responsibilities**:
- Match extracted facts against KnowledgeBase concepts
- Apply inference rules (e.g., "user mentions forgetfulness → PROBLEM hypothesis + PAIN_POINT hypothesis")
- Assign confidence (HIGH/MEDIUM/LOW) based on rule specificity and evidence
- Never create hypotheses that duplicate explicit extracted facts
- Never mutate state

**Non-responsibilities**:
- Does not decide hypothesis status (ACTIVE/CONFIRMED/REJECTED) — HypothesisManager does
- Does not persist hypotheses
- Does not select conversation objectives

## 4.4 HypothesisManager

**Purpose**: Manage hypothesis lifecycle across turns.

**Inputs**: Prior hypotheses (from previous turn), new hypotheses (from InferenceEngine).

**Outputs**: Updated hypothesis list with statuses (ACTIVE, CONFIRMED, REJECTED, SUPERSEDED).

**Responsibilities**:
- Merge new hypotheses with prior ones (same kind + similar value → SUPERSEDED)
- Auto-accept HIGH confidence hypotheses → CONFIRMED
- Demote LOW confidence hypotheses without support → REJECTED
- Detect contradictions between hypotheses of same kind
- Preserve insertion order for explainability

**Non-responsibilities**:
- Does not generate new hypotheses
- Does not mutate ProjectState
- Does not decide what to ask

## 4.5 StateManager

**Purpose**: Single writer of ProjectState; enforces atomic validation.

**Inputs**: `ExtractionResult` (validated), previous assistant/user messages (for bookkeeping).

**Outputs**: Mutated `ProjectState` (owned instance), `state_dict` for serialization.

**Responsibilities**:
- Apply ADD (append to list fields) and SET (overwrite scalar field) operations
- Enforce atomic batch validation: one invalid update rejects entire batch
- Prevent cross-update conflicts (ADD + SET on same field in one batch)
- Maintain previous_message bookkeeping for conversational context
- Own the ProjectState instance; callers receive the same reference

**Non-responsibilities**:
- Does not validate extraction JSON (ExtractionValidator does)
- Does not perform inference
- Does not decide objectives or strategies

## 4.6 ObjectiveEngine

**Purpose**: Decide which checklist field to target next based on coverage gaps.

**Inputs**: `ProjectState` (current memory), `CompletenessReport` (from CompletenessChecker), optional custom `PriorityEngine` rules.

**Outputs**: `ConversationObjective` (objective enum, confidence, targeted_field, missing_fields, completed_fields, reasoning).

**Responsibilities**:
- Evaluate which of the six Empathize fields are missing
- Apply priority rules (default: PERSONAS > PROBLEMS > CURRENT_SOLUTIONS > PAIN_POINTS > EVIDENCE > FREQUENCY > WRAP_UP)
- Return WRAP_UP when all fields have at least one entry
- Support custom rule registries for different domains
- Deterministic: same state → same objective

**Non-responsibilities**:
- Does not decide *how* to ask (ResponseStrategyEngine does)
- Does not generate questions
- Does not manage hypotheses

## 4.7 LifecycleManager

**Purpose**: Decide conversation lifecycle state for Empathize phase.

**Inputs**: `ConversationObjective`, `ProjectState`, `ResponseStrategy` (from previous turn), user message (for confirmation detection).

**Outputs**: `LifecycleDecision` enum (CONTINUE, READY_FOR_SUMMARY, WAITING_FOR_CONFIRMATION, READY_FOR_TRANSITION).

**Responsibilities**:
- CONTINUE: normal Empathize turn (objective ≠ WRAP_UP)
- READY_FOR_SUMMARY: WRAP_UP objective, no prior summary presented
- WAITING_FOR_CONFIRMATION: summary was presented, user hasn't confirmed
- READY_FOR_TRANSITION: user confirmed summary (exact lexeme: "yes", "y", "confirm", "approved", "looks good", "correct")
- Strict confirmation lexicon: no partial matches ("yes ok" → rejected)

**Non-responsibilities**:
- Does not generate summaries (SummaryBuilder does)
- Does not decide question strategy
- Does not manage hypotheses

## 4.8 ResponseStrategyEngine

**Purpose**: Plan *how* to ask the next question — select the question intent that maximizes information gain.

**Inputs**: `ConversationObjective`, `ProjectState`, optional `InferenceResult` (for hypothesis context).

**Outputs**: `ResponsePlan` containing:
- `strategy`: ResponseStrategy (ASK_QUESTION, GENERATE_SUMMARY, CLARIFY, CORRECT, ACKNOWLEDGE)
- `question_intent`: natural language description of what to ask
- `expected_checklist_fields`: which fields this question targets
- `candidate_question_intents`: all candidates considered with metadata
- `selected_question_intent`: the chosen candidate with full detail
- `expected_information_gain`: integer score
- `selection_reason`: why this candidate was chosen
- `constraints`: max_words, tone, allow_summary for PromptBuilder

**Responsibilities**:
- Map objective → base strategy (ASK_QUESTION for field objectives, GENERATE_SUMMARY for WRAP_UP)
- Detect contradictions → CORRECT strategy
- Detect ambiguity (multiple active hypotheses same kind) → CLARIFY strategy
- Information-gain planning: generate candidates from `_CANDIDATE_INTENTS`, estimate gain per candidate, rank, select best
- Conversation strategy selection (open_exploration, follow_up, verification, quantification, wrap_up)
- Deterministic: same inputs → same plan

**Non-responsibilities**:
- Does not extract facts
- Does not manage hypotheses
- Does not decide *what* is missing (ObjectiveEngine does)
- Does not render prompts (PromptBuilder does)

## 4.9 PromptBuilder

**Purpose**: Assemble the final prompt string sent to the LLM.

**Inputs**: `ProjectState`, `ResponsePlan` (or legacy `ResponseStrategy`), conversation history, previous messages.

**Outputs**: Prompt string (markdown sections).

**Responsibilities**:
- Render five required sections in order: ROLE, INSTRUCTIONS, OBJECTIVE, PROJECT_STATE, LATEST_CONVERSATION
- Inject strategy-specific constraints (max_words, tone, allow_summary)
- Include question intent, focus, and expected fields from ResponsePlan
- Support legacy `response_strategy` parameter for backward compatibility
- Never mutate inputs

**Non-responsibilities**:
- Does not decide strategy or question intent
- Does not perform inference
- Does not manage state

## 4.10 SummaryBuilder

**Purpose**: Create immutable Empathize summary from completed ProjectState.

**Inputs**: `ProjectState` (with all six fields populated).

**Outputs**: `EmpathizeSummary` (frozen dataclass with six fields copied verbatim).

**Responsibilities**:
- Defensive copy: mutate source after build → summary unchanged
- Preserve insertion order of list fields
- Include only the six extraction fields (no bookkeeping fields)

**Non-responsibilities**:
- Does not validate state completeness (LifecycleManager does)
- Does not generate natural language (LLM does)

## 4.11 MissionMemory (ProjectState)

**Purpose**: Canonical, persisted memory of the Empathize conversation.

**Fields**:
- `personas`: list[str] — who is affected
- `problems`: list[str] — what problems they face
- `current_solutions`: list[str] — existing workarounds
- `pain_points`: list[str] — why it's painful
- `evidence`: list[str] — observations, data, research
- `frequency`: str | None — how often the problem occurs
- `previous_user_message`: str | None — bookkeeping
- `previous_assistant_message`: str | None — bookkeeping

**Ownership**: StateManager owns the instance; all other components read-only.

---

# 5. Data Contracts

| Contract | Owner | Creator | Consumers |
|----------|-------|---------|-----------|
| **ExtractionResult** | MemoryExtractor | MemoryExtractor | StateManager, InferenceEngine |
| **ExtractedFact** | MemoryExtractor | MemoryExtractor | StateManager (via ExtractionResult), InferenceEngine |
| **Hypothesis** | InferenceEngine / HypothesisManager | InferenceEngine | HypothesisManager, ObjectiveEngine, ResponseStrategyEngine, PromptBuilder |
| **InferenceResult** | InferenceEngine | InferenceEngine | HypothesisManager, ObjectiveEngine, ResponseStrategyEngine |
| **ResponsePlan** | ResponseStrategyEngine | ResponseStrategyEngine | PromptBuilder, Mentor (for logging) |
| **LifecycleDecision** | LifecycleManager | LifecycleManager | Mentor (controls flow) |
| **ProjectState** | StateManager | StateManager (mutates) | All components (read-only) |
| **ConversationObjective** | ObjectiveEngine | ObjectiveEngine | LifecycleManager, ResponseStrategyEngine |
| **EmpathizeSummary** | SummaryBuilder | SummaryBuilder | Mentor (for PRD export) |

### Contract Details

**ExtractionResult**: `message_type` (MEANINGFUL/NO_UPDATE/AMBIGUOUS/END), `updates` (list of ExtractionUpdate), `confidence`, `reasoning`.

**ExtractedFact**: `field` (StateField), `operation` (ADD/SET), `value`, `confidence`.

**Hypothesis**: `kind` (HypothesisKind), `value`, `confidence` (ConfidenceLevel), `status` (HypothesisStatus), `source` (rule name), `reasoning`.

**InferenceResult**: `updated_hypotheses` (list[Hypothesis]), `extracted_facts` (passthrough), `reasoning`.

**ResponsePlan**: `strategy`, `primary_objective`, `conversation_strategy`, `question_intent`, `expected_checklist_fields`, `reasoning`, `target_hypothesis_key`, `question_focus`, `summary_sections`, `constraints`, `metadata`, `candidate_question_intents`, `selected_question_intent`, `expected_information_gain`, `selection_reason`.

**LifecycleDecision**: CONTINUE | READY_FOR_SUMMARY | WAITING_FOR_CONFIRMATION | READY_FOR_TRANSITION.

**ProjectState**: Six extraction fields + two bookkeeping fields.

**ConversationObjective**: `objective` (Objective enum), `confidence`, `targeted_field` (StateField | None), `missing_fields`, `completed_fields`, `reasoning`.

---

# 6. Knowledge Flow

The system maintains three distinct kinds of knowledge that are **intentionally separated**:

## 6.1 Explicit Facts (ExtractedFact)
- **Source**: User explicitly stated this in conversation
- **Flow**: User → MemoryExtractor → ExtractionResult → StateManager → ProjectState
- **Examples**: "I am a student" → `ExtractedFact(field=personas, op=ADD, value="student")`; "It happens daily" → `ExtractedFact(field=frequency, op=SET, value="daily")`
- **Properties**: High confidence, directly attributable to user, never inferred

## 6.2 Inferred Hypotheses (Hypothesis)
- **Source**: InferenceEngine derives from explicit facts + KnowledgeBase
- **Flow**: ExtractionResult → InferenceEngine → InferenceResult → HypothesisManager → (status assigned) → ResponseStrategyEngine / PromptBuilder
- **Examples**: User says "I forget assignments" → InferenceEngine creates `Hypothesis(kind=PROBLEM, value="assignment forgetfulness", confidence=HIGH)` AND `Hypothesis(kind=PAIN_POINT, value="stress from missed deadlines", confidence=MEDIUM)`
- **Properties**: Carry confidence (HIGH/MEDIUM/LOW), status (ACTIVE/CONFIRMED/REJECTED/SUPERSEDED), reasoning trace, source rule name

## 6.3 Persisted Mission Memory (ProjectState)
- **Source**: Verified and accepted knowledge from explicit facts
- **Flow**: StateManager applies MEANINGFUL ExtractionResult → ProjectState
- **Properties**: Only explicit facts (not hypotheses) are persisted; hypotheses influence questions but never auto-commit to state

### Why Separate?
- **Auditability**: Clear provenance — did the user say this, or did we infer it?
- **Safety**: Inferences can be wrong; only user-confirmed facts become mission memory
- **Explainability**: PromptBuilder can show both "User said: X" and "We infer: Y"
- **Rollback**: Rejecting a hypothesis doesn't corrupt persisted state

---

# 7. Architectural Rules

| Rule | Rationale |
|------|-----------|
| MemoryExtractor never performs inference | Keeps extraction pure; inference is a separate, testable stage |
| InferenceEngine never mutates state | Pure function; enables replay and deterministic testing |
| InferenceEngine never generates hypotheses that duplicate explicit extracted facts | Prevents double-counting; explicit facts are ground truth |
| HypothesisManager never asks questions | Single responsibility: hypothesis lifecycle only |
| ObjectiveEngine decides what information is missing | Separates "what to discover" from "how to discover it" |
| ResponseStrategyEngine decides how to discover that information | Information-gain planning is a distinct optimization problem |
| PromptBuilder converts plans into prompts | No reasoning in prompt assembly; pure rendering |
| LLM never decides conversation state | All state transitions are deterministic application logic |
| Only StateManager persists memory | Single writer prevents race conditions and partial updates |

---

# 8. Design Decisions

### Why Modular Architecture?
Monolithic prompts conflate reasoning with generation. By separating reasoning into discrete, testable stages, we achieve:
- **Debuggability**: Each stage's output is inspectable
- **Testability**: 387+ unit tests verify each component in isolation
- **Replaceability**: Swap LLM, improve one stage without touching others
- **Explainability**: Every decision has a `reasoning` field

### Why No Giant Prompt?
A single prompt would require the LLM to simultaneously:
- Track conversation state
- Decide what's missing
- Plan question strategy
- Generate natural language
This exceeds reliable LLM capability. Modular pipeline keeps LLM focused on generation only.

### Why Deterministic Reasoning?
LLMs are non-deterministic. Conversation state, objective selection, and strategy must be reproducible. Deterministic pipeline ensures same inputs → same outputs, enabling regression testing and user trust.

### Why Confidence Scores?
Not all inferences are equal. Confidence (HIGH/MEDIUM/LOW) propagates through HypothesisManager to affect:
- Auto-accept (HIGH → CONFIRMED)
- Strategy selection (contradiction detection weights by confidence)
- Prompt rendering (HIGH confidence hypotheses shown more prominently)

### Why KnowledgeBase?
Hardcoding inference rules in InferenceEngine would couple domain knowledge to code. KnowledgeBase externalizes:
- Concept synonyms ("student" ↔ "learner" ↔ "pupil")
- Indicators (phrases that trigger concepts)
- Confidence boosts (domain-specific reliability)

### Why HypothesisManager?
Hypotheses accumulate across turns. Without a manager:
- Duplicate hypotheses would pile up
- Contradictions would go undetected
- No lifecycle (ACTIVE → CONFIRMED → SUPERSEDED)
HypothesisManager provides a clean, append-only log of inferred knowledge.

### Why Information-Gain Planning?
Early versions asked one field per question. Information-gain planning evaluates multiple candidate intents per turn and selects the one that covers the most missing checklist fields. This reduces conversation length and improves user experience.

### Why Explicit and Inferred Knowledge Separated?
Mixing them would make it impossible to:
- Know what the user actually said vs. what we guessed
- Safely roll back incorrect inferences
- Explain reasoning to the user ("You mentioned X, so I inferred Y")

---

# 9. Regression Tests

Seven canonical scenarios validate the pipeline end-to-end. Each scenario drives a fresh session through the Empathize phase and asserts correct objective sequence, strategy selection, and information-gain behavior.

| Scenario | Domain | Validated Behavior |
|----------|--------|-------------------|
| **Assignment Procrastination** | Student productivity | PERSONAS → PROBLEMS → CURRENT_SOLUTIONS → PAIN_POINTS → EVIDENCE → FREQUENCY → WRAP_UP |
| **Teacher Attendance** | Education admin | Multi-persona handling (teachers + admins), evidence from interviews |
| **Cancer Detection** | Healthcare | High-stakes domain; evidence prioritization; frequency quantification |
| **Traffic Congestion** | Urban planning | Problem known, missing PERSONAS + PAIN_POINTS + EVIDENCE → info-gain selects "who experiences this" |
| **Restaurant Food Waste** | Hospitality | Current solutions (compost, donation) known; pain points + frequency missing |
| **Medication Reminders** | Health tech | PERSONA + PROBLEM + PAIN_POINT known; CURRENT_SOLUTION missing → info-gain prefers "understand current workflow" over narrow "current solution" |
| **Household Water Wastage** | Sustainability | PROBLEM + FREQUENCY known; CURRENT_SOLUTION missing → info-gain prefers "explore current behavior" |

**Future architectural changes must pass all seven regression scenarios.**

---

# 10. Extension Guidelines

### Adding New Knowledge Domains
1. Extend `KnowledgeBase._SEED_CONCEPTS` with domain concepts
2. Add inference rules in `InferenceEngine._infer_from_facts()`
3. No pipeline changes required

### Adding New Inference Rules
1. Add rule function in `InferenceEngine`
2. Register in rule dispatcher
3. HypothesisManager automatically handles lifecycle

### Adding New Objectives
1. Add enum value to `Objective` in `module3/contracts.py`
2. Add entry to `ObjectiveEngine.DEFAULT_RULES` (priority, field)
3. Add entry to `ResponseStrategyEngine._OBJ_TO_STRATEGY`
4. Add candidate intents to `ResponseStrategyEngine._CANDIDATE_INTENTS`
5. Add fallback to `ResponseStrategyEngine._QUESTION_PLANS`

### Adding New Checklist Fields (If Ever Needed)
1. Add `StateField` enum value
2. Update `ProjectState` dataclass
3. Update `StateManager` validation
4. Update `CompletenessChecker`
5. Update `PriorityEngine` rules
6. Update `ObjectiveEngine` mapping
7. Update `ResponseStrategyEngine` candidate intents

### Improving PromptBuilder
- Modify section rendering in `PromptBuilder._build_*()` methods
- Add new sections by extending `_build_prompt()` sequence
- Constraints from ResponsePlan control tone/length

### Improving ResponseStrategyEngine
- Add candidates to `_CANDIDATE_INTENTS`
- Refine `_estimate_information_gain()` heuristic
- Add conversation strategies to `_STRATEGY_DESCRIPTIONS`
- Core planning algorithm (`_generate → _estimate → _rank → _select`) remains unchanged

---

# 11. Future Work

Improvements that belong **after Architecture V1** — these should not require architectural changes:

- **Expand KnowledgeBase**: More seed concepts, better synonym coverage, domain-specific concept packs
- **Improve PromptBuilder**: Better few-shot examples, dynamic tone adaptation, structured output hints
- **Improve conversation quality**: More natural question phrasing, better follow-up detection, empathy calibration
- **Refine confidence calibration**: Empirical tuning of confidence thresholds, per-rule confidence profiles
- **Improve summaries**: Structured PRD sections, stakeholder-specific views, export formats
- **Improve question phrasing**: Template library, LLM-assisted rephrasing (with deterministic fallback)
- **Add more regression scenarios**: New domains, edge cases, multi-user conversations
- **Add Discovery/Define/Test phases**: New Objective enums, new checklist fields, phase-transition logic

---

# 13. Session Management

## 13.1 Session Lifecycle

The `SessionManager` component handles the complete lifecycle of mentoring sessions independently from the Design Thinking pipeline.

```
┌─────────────────────────────────────────────────────────────────┐
│                        SESSION MANAGER                          │
├─────────────────────────────────────────────────────────────────┤
│  create_session()    →  New timestamped folder + empty state    │
│  get_active_session() →  Returns active ProjectState to pipeline │
│  switch_session()    →  Archive current, activate target         │
│  reset_runtime_state() →  Archive + create fresh session        │
│  archive_session()   →  Mark session as archived (preserved)     │
│  list_sessions()     →  All sessions with metadata               │
│  delete_session()    →  Permanent removal (optional, cautious)   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  MENTORING PIPELINE │
                    │  (unchanged)        │
                    └─────────────────────┘
```

### Lifecycle States

| State | Description |
|-------|-------------|
| **active** | Currently selected session; receives new turns |
| **archived** | Previous session preserved for debugging/history |
| **completed** | Session reached natural end (Empathize complete) |

### Session Folder Structure

Each session gets a unique folder named with timestamp + short UUID:
```
sessions/
    20260726T143215Z_8f3a7c/
        session.json          # SessionMetadata
        project_state.json    # ProjectState (canonical v2 state)
        conversation.json     # User/assistant message history
        hypotheses.json       # Hypothesis history
        inference.json        # InferenceEngine history
        extraction.json       # MemoryExtractor history
        lifecycle.json        # LifecycleManager decisions
```

### Session Metadata (`session.json`)

```json
{
  "session_id": "20260726T143215Z_8f3a7c",
  "created_at": "2026-07-26T14:32:15.123456Z",
  "status": "active",
  "phase": "Empathize",
  "project_title": "Medication Reminder App",
  "username": "User"
}
```

This metadata enables future features:
- Session history page (list with timestamps, project titles)
- Conversation replay
- Search/filter past projects
- Resume from any point

## 13.2 Integration with Mentoring Pipeline

The `SessionManager` sits **outside** the mentoring pipeline. The pipeline remains unchanged:

1. **Frontend** calls `POST /session/new`
2. **SessionManager** archives current session, creates new folder, initializes empty `ProjectState`
3. **SessionManager** returns new `session_id`
4. **Frontend** clears chat, resets checklist
5. **Next user message** flows through pipeline as normal:
   - `MemoryExtractor` reads active `ProjectState` from SessionManager
   - Pipeline proceeds through InferenceEngine → HypothesisManager → StateManager → ObjectiveEngine → LifecycleManager → ResponseStrategyEngine → PromptBuilder → LLM
   - `StateManager` writes updated `ProjectState` back to SessionManager's active session folder

### Key Architectural Boundaries

| Component | Knows About Sessions? |
|-----------|----------------------|
| SessionManager | Yes (owns session lifecycle) |
| MemoryExtractor | No (reads ProjectState) |
| InferenceEngine | No (reads ProjectState) |
| HypothesisManager | No |
| StateManager | No (writes ProjectState) |
| ObjectiveEngine | No |
| LifecycleManager | No |
| ResponseStrategyEngine | No |
| PromptBuilder | No |
| SummaryBuilder | No |
| Frontend | Yes (calls /session/new) |

This separation ensures the Design Thinking reasoning pipeline remains **pure** — it never reasons about sessions, only about project state.

## 13.3 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/session/new` | POST | Create new session (archive current) |
| `/session/list` | GET | List all sessions with metadata |
| `/session/switch` | POST | Switch to a different session |
| `/session/archive` | POST | Archive current active session |
| `/session/active` | GET | Get active session metadata |

## 13.4 Frontend Integration

The Streamlit frontend adds a **"New Session"** button in the sidebar:

```python
if st.button("🆕 New Session"):
    resp = requests.post(f"{BACKEND_URL}/session/new", json={...})
    # Clear local state
    st.session_state.messages = [initial_greeting]
    st.session_state.dt_phase = "Empathize"
    st.rerun()
```

Clicking this:
1. Archives the current session on disk
2. Creates a fresh session folder with empty state
3. Clears the chat UI
4. Resets the Empathize checklist display
5. Returns an empty conversation ready for a new project

No backend restart required. The FastAPI server continues running.

---

# 12. Architecture Summary

| Component | Responsibility | Status |
|-----------|----------------|--------|
| MemoryExtractor | Extract explicit facts from user messages | Stable |
| KnowledgeBase | Provide domain concepts for inference | Stable |
| InferenceEngine | Generate hypotheses from facts + knowledge | Stable |
| HypothesisManager | Manage hypothesis lifecycle across turns | Stable |
| StateManager | Atomic persistence of verified facts | Stable |
| ObjectiveEngine | Select next checklist field to target | Stable |
| LifecycleManager | Decide Empathize conversation lifecycle | Stable |
| ResponseStrategyEngine | Plan question intent via information-gain | Stable |
| PromptBuilder | Render prompts from plans + state | Stable |
| SummaryBuilder | Build immutable Empathize summary | Stable |

---

**Architecture V1 is considered frozen. Future work should focus on improving reasoning quality, conversation quality, knowledge expansion, and user experience rather than restructuring the architecture.**