"""
AI Design Thinking Mentor - Empathize Phase

Architecture: The application does the thinking. The LLM does the talking.

Application responsibilities: logic, state, memory, structure, decision-making.
LLM responsibilities: natural language, tone, empathy, conversation.
"""

import copy
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from timing import TurnProfiler, TurnTiming, summarize_extraction_timing, timed_ollama_chat
from memory_extractor import LIST_FIELDS, MemoryExtractor, MessageType, ProjectState, REQUIRED_FIELDS, StateField, resolve_extractor_model
from state_manager import StateManager
from module3 import (
    ObjectiveContext,
    ObjectiveEngine,
    QuestionFamily,
    QuestionFamilyPlanner,
    SufficiencyChecker,
    classify_question,
    family_label,
    field_of,
    FAMILIES_BY_FIELD,
)
from module4 import ResponseStrategy, ResponseStrategyEngine, build_prompt as build_module4_prompt
from module4.prompt_builder import build_dynamic_prompt
from conversation_brief import build_brief as build_conversation_brief
from module5 import EmpathizeSummary, LifecycleDecision, LifecycleManager, SummaryBuilder, summary_presented_marker

# Pipeline sub-modules — internal-use imports
from session_pipeline import _finalize_session, _prepare_turn
from extraction_pipeline import InputProcessor, RuleBasedExtractor, merge_extracted_to_state
from conversation_pipeline import _apply_extraction_to_state, _build_journey_fallback, enforce_mentor_reply

# Observation-only rule-vs-LLM extraction comparison (Developer Console + audit)
from extraction_comparison import compare_extraction, field_label

# Objective-aware hybrid extraction decision layer (rules first, LLM only
# when the deterministic output does not satisfy the current objective)
from hybrid_extraction import decide_hybrid_extraction, rule_update_dicts

# Measurement-only shadow audit of hybrid skips (opt-in via HYBRID_AUDIT=true;
# NEVER influences state, objectives, lifecycle, prompts, or replies)
from hybrid_audit import enabled as _hybrid_audit_enabled

# Measurement-only accuracy audit of LLM extraction updates (Correct /
# Partially Correct / Incorrect / Missed Opportunity). NEVER influences
# state, objectives, lifecycle, prompts, or replies.
from extraction_accuracy import assess_extraction_accuracy

# Measurement-only audit of the mentor's conversational decisions (objective
# appropriateness, continuity, question quality, transition type). NEVER
# influences state, objectives, lifecycle, prompts, or replies.
from mentor_decision_audit import assess_mentor_decision

# Measurement-only audit of the mentor's reply *style* (opening, length,
# acknowledgment, observation, bridge, question count, summary markers,
# references to prior info, topic restart). NEVER influences state,
# objectives, lifecycle, prompts, extraction, or replies.
from conversation_style_audit import (
    ConversationStyleSummary,
    analyze_reply,
    conversation_style_diagnostics_section,
)
from product_experience_audit import (
    ProductExperienceSummary,
    analyze_product_experience,
    product_experience_diagnostics_section,
)

# Measurement-only observation audit of *failed conversations* (the mentor and
# the user out of sync: meta intents, re-asking resolved fields, abrupt
# transitions, over-exploration, unsupported inferences, memory inconsistencies,
# mismatch between the chosen move and the generated reply). NEVER influences
# state, objectives, lifecycle, prompts, extraction, memory, or replies.
from conversation_failure_audit import (
    ConversationFailureRecord,
    ConversationFailureSummary,
    analyze_conversation_failure,
    conversation_failure_diagnostics_section,
)

# Adaptive Coaching Strategy — determines HOW to continue the conversation
# before generating a reply.  Guidance only; never changes objectives,
# lifecycle, extraction, or ProjectState.
from coaching_strategy import (
    CoachingStrategy,
    coaching_diagnostics_section,
    coaching_instruction_bullet,
    determine_coaching_strategy,
)

# Insight Detection — recognises meaningful user insights (evidence,
# constraints, root-cause hints, learnings, contradictions) so the LLM can
# acknowledge or explore them.  Guidance only; never changes extraction,
# objectives, lifecycle, or ProjectState.
from insight_detection import (
    InsightType,
    detect_insight,
    insight_diagnostics_section,
    insight_instruction_bullet,
)

# Conversational Move Planner — synthesises objective, coaching strategy,
# insight detection, memory, and ProjectState into a single primary
# intention for the next reply.  Guidance only; never changes extraction,
# objectives, lifecycle, or ProjectState.
from conversation_move import (
    ConversationMove,
    conversation_move_diagnostics_section,
    conversation_move_instruction_bullet,
    determine_conversation_move,
)

# Conversational Context Gate — deterministic classifier of what is
# happening in this conversational turn (normal DT exchange, correction,
# topic shift, confusion, ...). Phase 1 recorded it observation-only.
# Phase 2 consumes ONE decision from it: when ``is_dt_paused`` is True
# (activated mode + HIGH confidence), DT question steering is suppressed
# for THIS TURN ONLY — the Objective Engine remains canonical and resumes
# normally on the next turn. HYPOTHETICAL/AMBIGUOUS/UNSAFE stay strictly
# observation-only (no steering change, no behavior change).
from conversation_context import (
    classify_conversation_context,
    conversation_context_diagnostics_section,
    context_instruction_bullet,
    acknowledge_first_instruction_bullet,
    is_dt_paused,
    paused_turn_fallback,
    is_unsafe_request,
    strip_pivot_language,
    SAFETY_REFUSAL_REPLY,
    ContextMode,
)

# Developer-only observation metrics (never read by any decision path)
from conversation_metrics import build_metrics_section, record_turn

# Deterministic recovery analysis for non-linear conversations
from recovery_monitor import RecoveryReport, analyze_recovery

# Deterministic conversational-context memory (open/resolved threads, deferred
# topics, acknowledged/summarized facts, partially answered objectives)
from conversation_memory import (
    _record_user_stated_facts,
    memory_diagnostics_section,
    memory_to_prompt_bullets,
    update_conversation_memory,
)

# Local imports for the answer-satisfaction check
import copy
import re

# Re-exports for backward compatibility (external consumers import from mentor)
from session_pipeline import MentorSession, build_mentor_session  # noqa: F401
from extraction_pipeline import strip_model_output  # noqa: F401
from conversation_pipeline import build_deterministic_fallback  # noqa: F401


# ---------------------------------------------------------------------------
# Answer-satisfaction check: does the latest user message answer the question
# for a given field? Used to prevent asking questions already answered.
# ---------------------------------------------------------------------------

# Frequency cadence tokens (mirrors module3.sufficiency._CADENCE_TOKENS)
_CADENCE_TOKENS: tuple[str, ...] = (
    "daily", "weekly", "monthly", "yearly", "annually", "hourly", "nightly",
    "fortnightly",
    "every day", "every single day", "each day", "every week", "each week",
    "every month", "every morning", "every evening", "every night",
    "every afternoon", "every weekend", "every weekday",
    "all the time", "most days", "most of the time", "most weeks",
    "on most days",
    "often", "sometimes", "always", "constantly", "regularly", "frequently",
    "occasionally", "rarely", "seldom", "usually", "normally", "typically",
    "never",
    "once in a while", "from time to time", "every so often", "now and then",
    "at times",
)

# Quantitative frequency tokens (mirrors module3.sufficiency._QUANTITATIVE_TOKENS)
_QUANTITATIVE_TOKENS: tuple[str, ...] = (
    "times", "per day", "per week", "per month", "per year", "per hour",
    "once a day", "once a week", "once a month", "twice a day", "twice a week",
    "twice a month", "a day", "a week", "a month", "an hour", "a year",
    "couple of", "few times", "several times",
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
)

# Context/situation tokens - answers to "in what situations", "when", "where"
_CONTEXT_TOKENS: tuple[str, ...] = (
    "anywhere", "everywhere", "always", "all situations", "any situation",
    "any context", "universal", "regardless", "no matter", "doesn't matter",
    "works everywhere", "works anywhere", "applies everywhere", "applies anywhere",
)

# Negation tokens - answers like "never", "don't help", "not effective"
_NEGATION_TOKENS: tuple[str, ...] = (
    "don't help", "doesn't help", "not help", "never help", "never works",
    "not effective", "ineffective", "useless", "pointless", "waste",
    "don't work", "doesn't work", "won't work", "can't work",
)

# Motivation/implication tokens - answers to "what motivates", "why"
_MOTIVATION_TOKENS: tuple[str, ...] = (
    "because", "since", "as ", "reason", "motivat", "driven", "drive",
    "want to", "need to", "have to", "must", "can't", "cannot", "unable",
    "no choice", "only way", "forced", "necessary", "essential",
)

_MOTIVATION_TOKENS_EXCLUDING_BECAUSE: tuple[str, ...] = (
    "since", "as ", "reason", "motivat", "driven", "drive",
    "want to", "need to", "have to", "must", "can't", "cannot", "unable",
    "no choice", "only way", "forced", "necessary", "essential",
)


def _matches_any(text: str, tokens: tuple[str, ...]) -> bool:
    """Word-boundary match for any token in the tuple."""
    lowered = text.lower()
    for token in tokens:
        if re.search(r"\b" + re.escape(token) + r"\b", lowered):
            return True
    return False


def _has_digits(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _message_satisfies_field(user_message: str, field: StateField, project_state: ProjectState) -> bool:
    """
    Check if the user's latest message contains an answer that would make
    `field` sufficient. This catches implicit answers that extraction may miss.

    Returns True if the message appears to provide a sufficient answer for the field.

    This is intentionally CONSERVATIVE - we only advance when we have a clear
    signal that the user answered the specific question being asked. The goal
    is to prevent the specific failure modes in the prompt (duplicate questions,
    ignored implicit answers), not to aggressively skip questions.
    """
    if not user_message or not user_message.strip():
        return False

    text = user_message.strip()
    lowered = text.lower()

    # Check message content for field-specific sufficient answers
    if field is StateField.FREQUENCY:
        # Frequency: cadence phrases, quantitative, or explicit "never"
        # Also accept context/situation answers like "works anywhere" for prevalence
        if _has_digits(text) or _matches_any(text, _QUANTITATIVE_TOKENS):
            return True
        if _matches_any(text, _CADENCE_TOKENS):
            return True
        # Explicit "never" / "not at all" answers
        if re.search(r"\b(never|not at all|zero|none)\b", lowered):
            return True
        # Context/situation: "works anywhere", "everywhere", "universal"
        if _matches_any(text, _CONTEXT_TOKENS):
            return True

    elif field is StateField.PAIN_POINTS:
        # Pain points: frustration, stress, worry, or motivation/implication
        # Catches "It stresses me out" or implied causal explanations such as
        # "don't have a voice box", "can't talk because...", "unable to...",
        # "prevents me from...". Do NOT make "because" by itself sufficient.
        if _matches_any(text, _NEGATION_TOKENS):
            return True
        if _matches_any(text, _MOTIVATION_TOKENS_EXCLUDING_BECAUSE):
            return True
        if re.search(r"\b(unable|don't have|do not have|can't prevent|prevents me from)\b", lowered):
            return True
        if re.search(r"\b(stress|frustrat|worry|hurt|pain|annoy|bother|overwhelm|anxiety)\b", lowered):
            return True

    elif field is StateField.CURRENT_SOLUTIONS:
        # Current solutions: negation, universal effectiveness answers,
        # or explicit effectiveness statements
        # Catches "instant pleasures don't help" or "works anywhere" or
        # "reduces the burden" or "makes them start" or "doesn't help" or "helps"
        if _matches_any(text, _NEGATION_TOKENS):
            return True
        if _matches_any(text, _CONTEXT_TOKENS):
            return True
        if re.search(r"\b(reduces? the burden|makes? (them|it) (start|easier|it easier|avoid)|helps?|effective|works)\b", lowered):
            return True

    elif field is StateField.EVIDENCE:
        # Evidence: observation, research, data keywords
        # Recognize interview/interviewed/interviews/interviewing
        if re.search(r"\b(seen|observed|\binterview\w*|research|data|study|survey|witnessed|noticed)\b", lowered):
            return True

    elif field is StateField.PERSONAS:
        # Personas: explicit who statements
        if re.search(r"\b(who|audience|users?|people|personas|target|demographic)\b", lowered):
            return True

    elif field is StateField.PROBLEMS:
        # Problems: explicit problem keywords
        if re.search(r"\b(problem|issue|challenge|forget|miss|struggle|difficult|trouble)\b", lowered):
            return True

    return False


def _make_field_satisfied(state: ProjectState, field: StateField) -> ProjectState:
    """
    Return a shallow copy of state with `field` artificially marked as satisfied.
    For list fields, adds a placeholder. For frequency, sets a generic sufficient value.
    """
    new_state = copy.copy(state)
    # Shallow copy the lists to avoid mutating original
    for f in LIST_FIELDS:
        val = state.get_list(f)
        setattr(new_state, f.value, list(val))
    if field is StateField.FREQUENCY:
        new_state.frequency = "regularly"  # sufficient cadence
    elif field in LIST_FIELDS:
        current = list(state.get_list(field))
        if not current:
            current.append("(inferred from context)")
        setattr(new_state, field.value, current)
    return new_state


# ---------------------------------------------------------------------------
# Module 3 — Objective Engine (the single planning authority)
# ---------------------------------------------------------------------------
_OBJECTIVE_ENGINE: ObjectiveEngine = ObjectiveEngine()

# ---------------------------------------------------------------------------
# Module 3 — Question-family planner (lightweight semantic de-duplication)
# ---------------------------------------------------------------------------
_QUESTION_FAMILY_PLANNER: QuestionFamilyPlanner = QuestionFamilyPlanner()

# ---------------------------------------------------------------------------
# Module 4 — Response Strategy Engine (how to respond)
# ---------------------------------------------------------------------------
_RESPONSE_STRATEGY_ENGINE: ResponseStrategyEngine = ResponseStrategyEngine()

# ---------------------------------------------------------------------------
# Module 5 — Lifecycle Manager + Summary Builder
# ---------------------------------------------------------------------------
_LIFECYCLE_MANAGER: LifecycleManager = LifecycleManager()
_SUMMARY_BUILDER: SummaryBuilder = SummaryBuilder()

# ---------------------------------------------------------------------------
# Stage constants
# ---------------------------------------------------------------------------
STAGE_EMPATHIZE = "Empathize"
STAGE_EMPATHIZE_COMPLETE = "Empathize_Complete"

# ---------------------------------------------------------------------------
# Checklist Manager - read-only Empathize coverage report for dashboards
# ---------------------------------------------------------------------------

class ChecklistManager:
    _FIELD_MAP = [
        ("target_audience", "target audience", StateField.PERSONAS),
        ("pain_point", "pain point", StateField.PROBLEMS),
        ("motivation", "motivation", StateField.PAIN_POINTS),
        ("existing_solution", "existing solution or current workflow", StateField.CURRENT_SOLUTIONS),
        ("frequency", "frequency of the problem", StateField.FREQUENCY),
        ("evidence", "evidence or observations validating the problem", StateField.EVIDENCE),
    ]

    @staticmethod
    def _get_value(project_state: ProjectState, state_field: StateField) -> str | None:
        if state_field in LIST_FIELDS:
            items = getattr(project_state, state_field.value)
            return items[0] if items else None
        return getattr(project_state, state_field.value)

    @staticmethod
    def get_missing_items(project_state: ProjectState):
        missing = []
        for key, description, state_field in ChecklistManager._FIELD_MAP:
            val = ChecklistManager._get_value(project_state, state_field)
            if not val or val in ("null", "None"):
                missing.append((key, description))
        return missing

    @staticmethod
    def is_complete(project_state: ProjectState):
        return len(ChecklistManager.get_missing_items(project_state)) == 0

    @staticmethod
    def get_status_dict(project_state: ProjectState):
        status = {}
        for key, description, state_field in ChecklistManager._FIELD_MAP:
            val = ChecklistManager._get_value(project_state, state_field)
            status[key] = {
                "complete": bool(val and val not in ("null", "None")),
                "value": val if (val and val not in ("null", "None")) else None,
                "description": description,
            }
        return status


# ---------------------------------------------------------------------------
# Main pipeline - process one mentor turn
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Safety boundary (Phase 3) — hard interception before ANY state access
# ---------------------------------------------------------------------------


def _safety_intercept(user_message, model_name, project_name):
    """Build the fixed safe response for an UNSAFE request.

    Runs BEFORE ``_prepare_turn``: nothing is extracted, no ProjectState
    mutation occurs, the unsafe user message is never appended to the
    conversation history, no question family or memory record is created,
    and no LLM call happens. The refusal is short and non-graphic, exposes
    no classifier internals, and never turns into a Design Thinking
    question. The session is left exactly as it was, so normal traffic can
    continue coherently on the next turn.
    """
    t_start = time.perf_counter()
    context = classify_conversation_context(user_message=user_message)
    print(
        "[SafetyBoundary] unsafe request intercepted; "
        "extraction/objective/reply pipeline skipped"
    )
    diagnostics = {
        "Pipeline": {
            "Stage": STAGE_EMPATHIZE,
            "Objective": None,
            "Response Strategy": None,
            "Lifecycle Decision": None,
            "Summary Presented": False,
            "Model": model_name,
        },
        "Extraction": {"message_type": "SAFETY_INTERCEPTED", "updates": []},
        "ConversationContext": {
            **context.to_dict(),
            "Safety Intercepted": True,
        },
    }
    timing = TurnTiming(
        total_ms=(time.perf_counter() - t_start) * 1000,
    )
    return SAFETY_REFUSAL_REPLY, MentorSession(project_name), timing, diagnostics


def process_mentor_turn(user_message, username="User", project_name="MyProject", model_name="llama3.2:1b"):
    storage_project_name = project_name or "MyProject"

    # Phase 1.5 latency audit: one TurnProfiler per turn. Observation-only
    # (~100 ns per span); never persisted, never read by any decision path.
    # The finished profile is attached as ``timing.profile`` (outside
    # ``TurnTiming.to_dict()``, so headers/console output are unchanged).
    prof = TurnProfiler()

    # --- Safety boundary: earliest possible point -----------------------
    # Some entry points call process_mentor_turn directly (CLI voice loops),
    # so the check must live here rather than only in server routing.
    # UNSAFE input never reaches _prepare_turn / extraction / state
    # mutation / objective / question planning / reply generation.
    with _span(prof, "safety"):
        if is_unsafe_request(user_message):
            return _safety_intercept(user_message, model_name, storage_project_name)

    t_start = time.perf_counter()

    t0 = time.perf_counter()
    with _span(prof, "prepare"):
        session_data, project_state, session_id, last_assistant_msg = _prepare_turn(
            user_message, storage_project_name
        )
    t1 = time.perf_counter()

    _capture = {}
    _capture["state_before"] = project_state.to_state_dict()

    # --- Conversational Context Gate: single decision point -------------
    # Classified BEFORE extraction so the pause decision is frozen prior to
    # any state mutation. Diagnostics are refreshed after extraction with
    # extraction-aware signals; that refresh NEVER changes these decisions.
    with _span(prof, "context_gate"):
        gate_context = classify_conversation_context(
            user_message=user_message,
            previous_assistant_message=last_assistant_msg,
            extraction_updates=[],
        )
    dt_paused = is_dt_paused(gate_context)
    _capture["conversation_context"] = gate_context
    _capture["dt_paused"] = dt_paused

    # --- Extraction gating (Phase 5) ------------------------------------
    # Framing alone never suppresses extraction. On paused HYPOTHETICAL /
    # TOPIC_SHIFT / CORRECTION turns, first strip clause-initial pivot
    # language ("forget that,", "let's switch...", ...) and probe what
    # remains with the deterministic extractor: if genuine DT content is
    # present, run the NORMAL extraction path so nothing is lost; only a
    # content-free pivot/framing turn skips extraction.
    _SUPPRESSING_MODES = (
        ContextMode.HYPOTHETICAL,
        ContextMode.TOPIC_SHIFT,
        ContextMode.CORRECTION,
    )
    extraction_suppressed = False
    if dt_paused and gate_context.mode in _SUPPRESSING_MODES:
        probe_text = strip_pivot_language(user_message)
        probe_updates = rule_update_dicts(
            RuleBasedExtractor.extract(probe_text)
        )
        _capture["gate_extraction_probe_updates"] = len(probe_updates)
        if not probe_updates:
            extraction_suppressed = True

    if extraction_suppressed:
        # Content-free pivot/framing turn: keep it out of ProjectState.
        # Per-turn only — the canonical engine resumes next turn.
        with _span(prof, "extraction"):
            _capture["extraction_message_type"] = MessageType.NO_UPDATE.value
            _capture["extraction_updates"] = []
            _capture["gate_extraction_suppressed"] = True
            print(
                f"[ConversationContext] {gate_context.mode.value}/HIGH paused "
                "turn: extraction suppressed (no substantive content after "
                "pivot-language stripping)"
            )
    else:
        with _span(prof, "extraction"):
            _extract_and_update_state(
                user_message, project_state, session_data, model_name, last_assistant_msg, storage_project_name,
                _capture=_capture,
                _prof=prof,
            )
    t2 = time.perf_counter()

    _capture["state_after"] = project_state.to_state_dict()

    # Measurement-only: complete this turn's extraction-accuracy record with
    # the post-extraction state and aggregate it into the session summary.
    _accuracy_record = _capture.get("extraction_accuracy")
    if _accuracy_record:
        _accuracy_record["state_after"] = _capture["state_after"]
        session_data.extraction_accuracy_summary.add_record(_accuracy_record)

    extraction_message_type = _capture.get("extraction_message_type")
    with _span(prof, "objective"):
        objective, candidate_strategy = _determine_objective(
            project_state, session_data,
            latest_user_message=user_message,
            extraction_message_type=extraction_message_type,
            extraction_updates=_capture.get("extraction_updates", [])
        )
    t3 = time.perf_counter()

    with _span(prof, "recovery"):
        _capture["recovery"] = (
            recovery_report := analyze_recovery(
                user_message=user_message,
                state_before=_capture.get("state_before", {}),
                state_after=_capture.get("state_after", {}),
                last_assistant_message=last_assistant_msg,
            )
        ).to_dict()

    # Diagnostics refresh of the Conversational Context Gate with
    # extraction-aware inputs (the identity-signal guard can use this
    # turn's extracted facts). The gate DECISIONS above — safety
    # interception and dt_paused — were frozen before extraction and are
    # never changed here. For every non-hypothetical turn this yields the
    # same mode Phase 2 produced; for suppressed hypothetical turns the
    # mode stays HYPOTHETICAL (its frames are text-only).
    with _span(prof, "context_refresh"):
        _capture["conversation_context"] = classify_conversation_context(
            user_message=user_message,
            previous_assistant_message=last_assistant_msg,
            extraction_updates=_capture.get("extraction_updates") or [],
        )

    with _span(prof, "family"):
        family_plan = _plan_question_family(objective, project_state, session_data)
    if _capture is not None:
        _capture["family_plan"] = family_plan.to_dict()
        _capture["asked_families"] = list(session_data.asked_question_families)

    with _span(prof, "lifecycle"):
        _inject_lifecycle_marker(project_state, session_data)

        lifecycle_decision = _resolve_lifecycle(project_state, objective, candidate_strategy, session_data)

        empathize_summary, response_strategy = _branch_on_lifecycle(
            lifecycle_decision, project_state, candidate_strategy, session_data
        )
    t4 = time.perf_counter()

    # Reconcile the conversational-context memory (open/resolved threads,
    # deferred topics, acknowledged/summarized facts, partially answered
    # objectives) against this turn's state diff. Deterministic; references
    # field keys/statuses only — ProjectState values are never duplicated.
    # Runs before reply generation so the resume hint can reflect the
    # just-completed extraction, and before _finalize_session so the same
    # save persists it.
    with _span(prof, "memory"):
        update_conversation_memory(
            session_data=session_data,
            state_before=_capture.get("state_before", {}),
            state_after=_capture.get("state_after", {}),
            objective=objective,
            recovery=recovery_report,
            lifecycle_decision=lifecycle_decision,
        )

    _gen_timing = {}
    with _span(prof, "generate_reply"):
        reply = _generate_reply(
            project_state, objective, response_strategy, last_assistant_msg,
            user_message, empathize_summary, lifecycle_decision, model_name,
            family_plan=family_plan,
            session_data=session_data,
            _timing=_gen_timing,
            _capture=_capture,
            _prof=prof,
        )
    t5 = time.perf_counter()

    with _span(prof, "finalize"):
        session = _finalize_session(
            reply, lifecycle_decision, session_data, project_state, storage_project_name,
            # Family hygiene: a paused conversational reply (acknowledgment /
            # clarification) must not register as a newly asked DT question
            # family. Normal turns record exactly as before.
            record_family=not dt_paused,
        )

    # Measurement-only audit of the mentor's conversational decision for this
    # turn: was the chosen objective appropriate, did the question follow the
    # user's last answer, is the question repeated / questionnaire-like, and
    # could another objective have produced a better conversation. NEVER feeds
    # a decision — only surfaces in the Developer Console and the session
    # summary. Wrapped so an audit problem can never break the conversation.
    with _span(prof, "audit_decision"):
        try:
            _decision_record = assess_mentor_decision(
                turn_index=len(session_data.turn_metrics) + 1,
                user_message=user_message,
                generated_question=reply,
                objective_value=objective.objective.value,
                objective_confidence=objective.confidence,
                objective_advancement=objective.advancement,
                selection_trace=objective.selection_trace,
                question_family=_capture.get("question_family"),
            family_plan=_capture.get("family_plan") or {},
            guard_blocked=bool(_capture.get("guard_blocked")),
            state_before=_capture.get("state_before", {}),
            state_after=_capture.get("state_after", {}),
            recovery=_capture.get("recovery") or {},
            memory=session_data.conversation_memory.to_dict(),
            lifecycle_decision=lifecycle_decision.value,
            response_strategy=response_strategy.value,
        )
            _capture["mentor_decision"] = _decision_record
            session_data.mentor_decision_summary.add_record(_decision_record)
        except Exception as _exc:
            print(f"[MentorDecisionAudit] failed to record turn: {_exc}")

    # Measurement-only audit of the reply *style* (opening, length,
    # acknowledgment, observation, bridge, question count, summary markers,
    # references to prior info, topic restart). NEVER feeds a decision —
    # only surfaces in the Developer Console and the session summary.
    # Wrapped so an audit problem can never break the conversation.
    with _span(prof, "audit_style"):
        try:
            _style_record = analyze_reply(
                turn_index=len(session_data.turn_metrics) + 1,
                reply=reply,
                previous_reply=last_assistant_msg,
                known_facts=list(getattr(session, "known_facts", []) or []),
            )
            _capture["conversation_style"] = _style_record
            session_data.conversation_style_summary.add_record(_style_record)
        except Exception as _exc:
            print(f"[ConversationStyleAudit] failed to record turn: {_exc}")

    # Measurement-only observation audit of *failed conversations*: whether
    # this turn the mentor and user fell out of sync (meta intent, re-asking a
    # resolved field, abrupt transition, over-exploration, unsupported
    # inference, missed insight, memory inconsistency, social slip, or a
    # mismatch between the chosen move and the generated reply). Runs AFTER
    # the reply is finalised and observes only existing runtime data — it can
    # never influence the reply/objective/extraction/lifecycle/memory/recovery
    # that were already decided. Wrapped so an audit problem can never break
    # the conversation.
    with _span(prof, "audit_failure"):
        try:
            _cf_record = analyze_conversation_failure(
                user_message=user_message,
                mentor_reply=reply,
                objective=getattr(getattr(objective, "objective", None), "value", ""),
                project_state_before=_capture.get("state_before", {}),
                project_state_after=_capture.get("state_after", {}),
                conversation_memory=session_data.conversation_memory.to_dict(),
                conversation_move=_capture.get("conversation_move"),
                coaching_strategy=_capture.get("coaching_strategy"),
                question_family=_capture.get("question_family"),
                style_audit=_capture.get("conversation_style"),
                diagnostics=_capture,
            )
            _capture["conversation_failure"] = _cf_record.to_dict()
            session_data.conversation_failure_summary.add_record(_cf_record)
        except Exception as _exc:
            print(f"[ConversationFailureAudit] failed to record turn: {_exc}")

    # Measurement-only audit of the end-to-end product experience: startup
    # timing, session restoration, Developer Console shape/cost, conversation
    # export size, and per-turn performance timeline. NEVER feeds a decision
    # — only surfaces in the Developer Console and session summary. Wrapped
    # so an audit problem can never break the conversation.
    with _span(prof, "audit_product"):
        try:
            first_turn = session_data.product_experience_summary.turn_count == 0
            # conversation_history has 2 messages appended this turn (user
            # in _prepare_turn + assistant in _finalize_session); record the
            # count at this session's start by backing those out.
            history_at_start = len(session_data.conversation_history) - 2
            metrics_at_start = len(session_data.turn_metrics)
            # Build a minimal timing snapshot from the elapsed-time marks
            # still in scope (TurnTiming is created later). The extraction
            # breakdown is already stored in _capture.
            _t_now = time.perf_counter()
            _shadow_timing = {
                "prepare_ms": (t1 - t0) * 1000,
                "extraction_ms": (t2 - t1) * 1000,
                "objective_ms": (t3 - t2) * 1000,
                "lifecycle_ms": (t4 - t3) * 1000,
                "prompt_ms": _gen_timing.get("prompt_ms", 0),
                "llm_ms": _gen_timing.get("llm_ms", 0),
                "finalize_ms": (_t_now - t5) * 1000,
                "total_ms": (_t_now - t_start) * 1000,
                "extraction_breakdown": _capture.get("extraction_timing"),
            }
            # The product audit needs a diagnostics snapshot of its own; this
            # duplicates the final _build_diagnostics call below (see the
            # "diagnostics_shadow" span) — measured, not yet deduplicated.
            with _span(prof, "diagnostics_shadow"):
                _shadow_diagnostics = _build_diagnostics(
                    session_data, objective, response_strategy,
                    lifecycle_decision, session, project_state, _capture, model_name,
                )
            _pexp_record = analyze_product_experience(
                turn_index=len(session_data.turn_metrics) + 1,
                session_id=session_id,
                conversation_history=session_data.conversation_history,
                timing=_shadow_timing,
                diagnostics=_shadow_diagnostics,
                first_turn=first_turn,
                history_count_at_start=history_at_start,
                turn_metrics_at_start=metrics_at_start,
                startup_timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z") if first_turn else "",
            )
            _capture["product_experience"] = _pexp_record
            session_data.product_experience_summary.add_record(_pexp_record)
        except Exception as _exc:
            print(f"[ProductExperienceAudit] failed to record turn: {_exc}")

    # Observation-only: record this turn's metrics for the Developer Console.
    with _span(prof, "record_turn"):
        record_turn(
            session_data, _capture, objective, response_strategy,
            lifecycle_decision, reply,
        )
    t6 = time.perf_counter()

    timing = TurnTiming(
        prepare_ms=(t1 - t0) * 1000,
        extraction_ms=(t2 - t1) * 1000,
        objective_ms=(t3 - t2) * 1000,
        lifecycle_ms=(t4 - t3) * 1000,
        prompt_ms=_gen_timing.get("prompt_ms", 0),
        llm_ms=_gen_timing.get("llm_ms", 0),
        finalize_ms=(t6 - t5) * 1000,
        extraction_breakdown=_capture.get("extraction_timing"),
        total_ms=(t6 - t_start) * 1000,
    )
    # The finished per-turn profile rides on the timing object (outside
    # ``to_dict()`` so headers, console rendering, and golden assertions on
    # the serialized shape are unchanged). Benchmarks read
    # ``timing.profile``; production callers ignore it.
    timing.profile = prof.to_dict()
    with _span(prof, "diagnostics_final"):
        diagnostics = _build_diagnostics(
            session_data, objective, response_strategy, lifecycle_decision,
            session, project_state, _capture, model_name,
        )
    timing.profile = prof.to_dict()
    return reply, session, timing, diagnostics


# ---------------------------------------------------------------------------
# Pipeline stage helpers
# ---------------------------------------------------------------------------

_FIELD_DISPLAY_LABELS = {
    "personas": "Personas",
    "problems": "Problems",
    "current_solutions": "Current Solutions",
    "pain_points": "Pain Points",
    "evidence": "Evidence",
    "impacts": "Impacts",
    "frequency": "Frequency",
}


def _diff_project_state(before, after):
    """Field-level diff of two ``to_state_dict()`` snapshots. Read-only."""
    added = {}
    updated = {}
    for raw_name, label in _FIELD_DISPLAY_LABELS.items():
        b = before.get(raw_name)
        a = after.get(raw_name)
        if isinstance(b, list):
            new_items = [v for v in (a or []) if v not in (b or [])]
            if new_items:
                added[label] = new_items
        else:
            if a not in (None, "") and a != b:
                updated[label] = f"{b or '—'} → {a}"
    return {"Added": added, "Updated": updated} if (added or updated) else {}


def _confidence_explanation(objective) -> str:
    """Human-readable explanation of the deterministic objective confidence."""
    if objective.confidence == 1.0:
        if objective.is_wrap_up:
            return "All required Empathize fields are populated — deterministic WRAP_UP, full confidence."
        return "Single highest information-gain candidate — deterministic selection, full confidence."
    if objective.confidence == 0.8:
        return "Information-gain tie broken by declared StateField order — minor ambiguity."
    return f"Deterministic confidence {objective.confidence} (product of rule clarity)."


def _advancement_delta(state_before: dict, project_state, objective) -> dict:
    """Per-turn advancement explanation: which fields reached a usable
    sufficiency level this turn (advanced past) vs which are still being
    pursued (stayed active). Read-only projection of existing state —
    no computation that mutates anything."""
    before = SufficiencyChecker.evaluate_dict(state_before)
    after = SufficiencyChecker.evaluate(project_state)

    advanced_past = [
        sf.value
        for sf in REQUIRED_FIELDS
        if not before.is_satisfied(sf) and after.is_satisfied(sf)
    ]
    stayed_active = [
        sf.value for sf in REQUIRED_FIELDS if not after.is_satisfied(sf)
    ]

    if objective.is_wrap_up:
        verdict = "WRAP_UP"
        reason = "All required Empathize fields are populated."
    elif advanced_past:
        verdict = "ADVANCED"
        reason = (
            "Objective advanced past: "
            + ", ".join(advanced_past)
            + ". Still active: "
            + (", ".join(stayed_active) if stayed_active else "none")
            + "."
        )
    else:
        verdict = "STAYED_ACTIVE"
        reason = (
            "Objective stayed active: "
            + (", ".join(stayed_active) if stayed_active else "no required fields")
            + " still need information."
        )

    return {
        "Verdict": verdict,
        "Reason": reason,
        "Advanced Past": advanced_past,
        "Stayed Active": stayed_active,
        "Objective Verdict": objective.advancement,
        "Objective Reason": objective.advancement_reason,
    }


def _build_diagnostics(session_data, objective, response_strategy, lifecycle_decision, session, project_state, capture, model_name):
    """Read-only snapshot of this turn's runtime state. No computation,
    no mutation, no persistence — just existing objects projected to dicts."""

    state_before = capture.get("state_before", {})
    state_after = capture.get("state_after", {})

    project_state_section = {}
    for raw_name, label in _FIELD_DISPLAY_LABELS.items():
        val = state_after.get(raw_name)
        if isinstance(val, list):
            project_state_section[label] = list(val)
        else:
            project_state_section[label] = val

    diagnostics = {
        "Pipeline": {
            "Stage": session_data.current_stage,
            "Objective": objective.objective.value,
            "Response Strategy": response_strategy.value,
            "Lifecycle Decision": lifecycle_decision.value,
            "Summary Presented": session_data.empathize_summary_presented,
            "Model": model_name,
        },
        "Conversation": {
            "Conversation Turns": len(session_data.conversation_history),
            "Known Facts": len(session.known_facts),
            "Assumptions": len(session.assumptions),
            "Unknown Facts": len(session.unknown_facts),
            "Open Questions": len(session.open_questions),
        },
        "ProjectState": project_state_section,
        "StateChanges": _diff_project_state(state_before, state_after),
        "ObjectiveTrace": {
            "Objective": objective.objective.value,
            "Response Strategy": response_strategy.value,
            "Confidence": objective.confidence,
            "Confidence Explanation": _confidence_explanation(objective),
            "Sufficiency": dict(objective.sufficiency),
            "Advancement": _advancement_delta(state_before, project_state, objective),
            "Completed Fields": [sf.value for sf in objective.completed_fields],
            "Missing Fields": [sf.value for sf in objective.missing_fields],
            "Reasoning": list(objective.reasoning),
        },
        "Extraction": {
            "message_type": capture.get("extraction_message_type"),
            "updates": capture.get("extraction_updates", []),
        },
        "ExtractionComparison": _build_extraction_comparison_section(capture),
        "ExtractionAccuracy": _build_extraction_accuracy_section(capture),
        "ExtractionAccuracySummary": _build_extraction_accuracy_summary_section(session_data),
        "MentorDecision": _build_mentor_decision_section(capture),
        "MentorDecisionSummary": _build_mentor_decision_summary_section(session_data),
        "ConversationStyle": _build_conversation_style_section(capture),
        "ConversationStyleSummary": _build_conversation_style_summary_section(session_data),
        "ProductExperience": _build_product_experience_section(capture),
        "ProductExperienceSummary": _build_product_experience_summary_section(session_data),
        "ConversationFailure": _build_conversation_failure_section(capture),
        "ConversationFailureSummary": _build_conversation_failure_summary_section(session_data),
        "HybridExtraction": _build_hybrid_extraction_section(capture),
        "SemanticComplexity": _build_complexity_section(capture),
        "QuestionHistory": list(session_data.previous_questions),
        "QuestionFamilies": _build_question_family_section(capture, session_data),
        "Recovery": _build_recovery_section(capture),
        "ConversationContext": _build_conversation_context_section(capture),
        "Memory": memory_diagnostics_section(session_data),
        "CoachingStrategy": coaching_diagnostics_section(capture),
        "InsightDetection": insight_diagnostics_section(capture),
        "ConversationMove": conversation_move_diagnostics_section(capture),
        "ConversationBrief": dict(capture.get("conversation_brief") or {}),
        "ChecklistReasoning": ChecklistManager.get_status_dict(project_state),
        "ConversationMetrics": build_metrics_section(session_data),
    }

    # Measurement-only shadow audit. Present ONLY when HYBRID_AUDIT=true —
    # otherwise diagnostics stay byte-identical to a non-audited run.
    if _hybrid_audit_enabled():
        diagnostics["HybridAudit"] = _build_hybrid_audit_section(capture)
        diagnostics["HybridSummary"] = _build_hybrid_summary_section(session_data)

    # Information-gain selection trace (candidates + scores + selected/
    # rejected reasons) — present for every non-WRAP_UP objective.
    if objective.selection_trace is not None:
        diagnostics["ObjectiveTrace"]["Selection"] = objective.selection_trace

    rule_based = capture.get("rule_based_extraction")
    if rule_based:
        diagnostics["Extraction"]["rule_based_extraction"] = rule_based

    prompt = capture.get("prompt")
    if prompt:
        diagnostics["Prompt"] = prompt
    return diagnostics


def _build_recovery_section(capture):
    """Developer Console view of the per-turn recovery analysis: what (if
    anything) was detected as a non-linear conversation situation, what the
    deterministic recovery strategy is, and why (not) to ask a clarification.
    Read-only projection of the recovery report produced this turn."""
    return dict(capture.get("recovery") or RecoveryReport().to_dict())


def _build_conversation_context_section(capture):
    """Developer Console view of the Conversational Context Gate: the
    detected mode/confidence/signals plus whether THIS turn's DT question
    steering was paused (Phase 2) and how the Phase 5 extraction guard
    decided. Read-only projection."""
    section = conversation_context_diagnostics_section(capture)
    if section and "dt_paused" in (capture or {}):
        section["DT Steering Paused"] = bool(capture.get("dt_paused"))
    if section and capture.get("gate_extraction_probe_updates") is not None:
        section["Extraction Probe Updates"] = capture.get(
            "gate_extraction_probe_updates")
        section["Extraction Suppressed"] = bool(
            capture.get("gate_extraction_suppressed"))
    return section


def _build_extraction_comparison_section(capture):
    """Developer Console view of the per-turn rule-vs-LLM extraction
    comparison. Read-only projection of the observation-only audit computed
    in ``_extract_and_update_state``."""
    record = capture.get("extraction_comparison")
    if not record:
        return {}
    return {
        "Rule Extractor": [field_label(f) for f in record["rule_fields"]],
        "LLM Extractor": [field_label(f) for f in record["llm_fields"]],
        "Fields That Match": [field_label(f) for f in record["matching_fields"]],
        "Fields Only Found by Rules": [field_label(f) for f in record["rule_only_fields"]],
        "Fields Only Found by LLM": [field_label(f) for f in record["llm_only_fields"]],
        "Decision": {
            "Rule sufficient": "Yes" if record["rule_sufficient"] else "No",
            "Reason": record["reason"],
            "Rule Meaningful": record["rule_meaningful"],
            "LLM Meaningful": record["llm_meaningful"],
            "LLM Applied": record["llm_applied"],
        },
    }


def _build_hybrid_extraction_section(capture):
    """Developer Console view of the objective-aware hybrid extraction
    decision: what the deterministic extractor produced, the current
    objective, whether the rules satisfied it, whether the LLM was invoked,
    and why. Read-only projection of the decision computed in
    ``_extract_and_update_state``."""
    decision = capture.get("hybrid_decision")
    if not decision:
        return {}
    rule_output = [
        f"{field_label(u['field'])}: {u['value']}"
        for u in decision.get("rule_output", [])
    ]
    return {
        "Rule Output": rule_output,
        "Objective": decision.get("objective"),
        "Objective Field": decision.get("objective_field"),
        "Objective Satisfied By Rules": "Yes" if decision.get("satisfied") else "No",
        "LLM Invoked": "Yes" if decision.get("llm_invoked") else "No",
        "Reason": decision.get("reason"),
    }


def _build_complexity_section(capture):
    """Developer Console view of the Semantic Complexity Gate: the message's
    complexity classification, the matched reasons, and the advisory
    skip/run decision. Read-only projection of the advisory classification
    computed in ``decide_hybrid_extraction``."""
    decision = capture.get("hybrid_decision")
    if not decision:
        return {}
    return {
        "Complexity": decision.get("complexity"),
        "Reasons": decision.get("complexity_reason"),
        "Decision": decision.get("complexity_decision"),
        "Gate Enabled": "Yes" if decision.get("gate_enabled") else "No",
    }


def _build_hybrid_audit_section(capture):
    """Measurement-only shadow audit of this turn's hybrid skip: what the
    shadow LLM would have extracted vs what the rules applied, and the risk
    that skipping the LLM missed something useful. Never feeds a decision."""
    record = capture.get("hybrid_audit")
    if not record:
        return {}
    return {
        "Hybrid Decision": record.get("hybrid_decision"),
        "LLM Skipped": "Yes" if record.get("llm_skipped") else "No",
        "Shadow LLM Fields": [
            field_label(f) for f in record.get("shadow_llm_fields", [])
        ],
        "Additional Fields": [
            field_label(f) for f in record.get("additional_fields", [])
        ],
        "Would Change State": "Yes" if record.get("would_change_state") else "No",
        "Would Change Objective": (
            "Yes" if record.get("would_change_objective") else "No"
        ),
        "Risk Level": record.get("risk_level"),
    }


def _build_hybrid_summary_section(session_data):
    """Measurement-only running aggregate of hybrid-skip shadow audits for
    this session. Read-only projection of ``session_data.hybrid_audit_summary``."""
    summary = getattr(session_data, "hybrid_audit_summary", None)
    if summary is None:
        return {}
    return summary.to_display()


def _build_extraction_accuracy_section(capture):
    """Developer Console view of this turn's LLM-extraction accuracy audit:
    the current objective, the user message, every LLM-extracted update with
    its assessment (Correct / Partially Correct / Incorrect / Missed
    Opportunity) and reason, plus any turn-level missed facts. Read-only
    projection of ``capture["extraction_accuracy"]``; never feeds a decision."""
    record = capture.get("extraction_accuracy")
    if not record:
        return {}
    return {
        "Current Objective": record.get("objective"),
        "Message": record.get("user_message"),
        "Extracted Updates": list(record.get("updates", [])),
        "Missed Opportunities": [
            {
                "field": field_label(m.get("field")),
                "fact": m.get("fact"),
                "kind": ", ".join(m.get("kind", [])),
                "attached": (
                    "Yes" if m.get("attached_to_update") is not None else "No"
                ),
            }
            for m in record.get("missed_opportunities", [])
        ],
    }


def _build_extraction_accuracy_summary_section(session_data):
    """Measurement-only running aggregate of per-turn LLM-extraction accuracy
    for this session. Read-only projection of
    ``session_data.extraction_accuracy_summary``."""
    summary = getattr(session_data, "extraction_accuracy_summary", None)
    if summary is None or summary.total() == 0:
        return {}
    return summary.to_display()


def _build_mentor_decision_section(capture):
    """Developer Console view of this turn's mentor-decision audit:
    the chosen objective, the alternatives considered, the confidence,
    whether the question followed the user's last answer, whether the
    question was fresh or repeated, and whether another objective would
    likely have produced a better conversation. Read-only projection of
    ``capture["mentor_decision"]``; never feeds a decision."""
    record = capture.get("mentor_decision")
    if not record:
        return {}
    alt = record.get("alternative_objectives") or []
    return {
        "Current Objective": record.get("objective"),
        "Alternative Objectives": [
            f"{a['objective']} (score {a['score']:.2f})"
            for a in alt
        ],
        "Decision Confidence": record.get("objective_confidence"),
        "Reason": record.get("objective_reason"),
        "Conversation Continuity": record.get("continuity"),
        "Continuity Reason": record.get("continuity_reason"),
        "Transition Type": record.get("transition_type"),
        "Transition Reason": record.get("transition_reason"),
        "Question Quality": record.get("question_quality"),
        "Quality Reason": record.get("quality_reason"),
        "Objective Appropriate": "Yes" if record.get("objective_appropriate") else "No",
        "Acknowledged Information": "Yes" if record.get("acknowledged_information") else "No",
        "Restarted Topic": "Yes" if record.get("restarted_topic") else "No",
        "Better Alternative": (
            record["better_alternative"]["objective"]
            if record.get("better_alternative")
            else None
        ),
        "Generated Question": record.get("generated_question"),
    }


def _build_mentor_decision_summary_section(session_data):
    """Measurement-only running aggregate of per-turn mentor-decision
    assessments for this session. Read-only projection of
    ``session_data.mentor_decision_summary``."""
    summary = getattr(session_data, "mentor_decision_summary", None)
    if summary is None or summary.turn_count == 0:
        return {}
    return summary.to_display()


def _build_conversation_style_section(capture):
    """Measurement-only per-turn projection of the reply-style audit
    (opening, length, acknowledgment, observation, bridge, question count,
    summary markers, references to prior info, topic restart). Read-only
    projection of ``capture["conversation_style"]``; never feeds a decision."""
    record = capture.get("conversation_style")
    return conversation_style_diagnostics_section(record)


def _build_conversation_style_summary_section(session_data):
    """Measurement-only running aggregate of per-turn reply-style audits
    for this session. Read-only projection of
    ``session_data.conversation_style_summary``."""
    summary = getattr(session_data, "conversation_style_summary", None)
    if summary is None or summary.turn_count == 0:
        return {}
    return summary.to_display()


def _build_product_experience_section(capture):
    """Measurement-only per-turn projection of the product-experience audit
    (Developer Console shape/cost, conversation export size, performance
    timeline). Read-only projection of ``capture["product_experience"]``;
    never feeds a decision."""
    record = capture.get("product_experience")
    return product_experience_diagnostics_section(record)


def _build_product_experience_summary_section(session_data):
    """Measurement-only running aggregate of per-turn product-experience
    audits for this session. Read-only projection of
    ``session_data.product_experience_summary``."""
    summary = getattr(session_data, "product_experience_summary", None)
    if summary is None or summary.turn_count == 0:
        return {}
    return summary.to_display()


def _build_conversation_failure_section(capture):
    """Measurement-only per-turn projection of the conversation-failure audit
    (whether this turn the mentor and user fell out of sync). Read-only
    projection of ``capture["conversation_failure"]``; never feeds a decision."""
    record = capture.get("conversation_failure")
    if not record:
        return {}
    return conversation_failure_diagnostics_section(
        ConversationFailureRecord.from_dict(record)
    )


def _build_conversation_failure_summary_section(session_data):
    """Measurement-only running aggregate of per-turn conversation-failure
    audits for this session. Read-only projection of
    ``session_data.conversation_failure_summary``."""
    summary = getattr(session_data, "conversation_failure_summary", None)
    if summary is None or summary.total_turns() == 0:
        return {}
    return summary.to_display()


def _build_question_family_section(capture, session_data):
    """Developer Console view of the question-family tracking:
    the family of the question just asked, the families asked before, and the
    reason(s) a family was skipped this turn. Read-only projection."""
    plan = capture.get("family_plan") or {}

    def _label(value):
        if not value:
            return value
        try:
            return family_label(QuestionFamily(value))
        except (ValueError, AttributeError):
            return value

    section = {
        "Question Family": _label(capture.get("question_family")),
        "Planned Family": _label(plan.get("selected")),
        "Re-ask Sanctioned": plan.get("reask", False),
        "Re-ask Reason": plan.get("reask_reason", ""),
        "Previously Asked Families": [
            _label(v) for v in plan.get("asked_families", [])
        ],
        "Family Skip Reasons": list(plan.get("skip_reasons", [])),
    }
    # If a plan was never computed this turn (defensive), fall back to the
    # session's recorded families so the console still shows history.
    if not plan.get("selected") and not plan.get("skip_reasons"):
        section["Previously Asked Families"] = [
            _label(v) for v in list(session_data.asked_question_families)
        ]
    return section

def _extract_and_update_state(user_message, project_state, session_data, model_name, last_assistant_msg, storage_project_name, _capture=None, _prof=None):
    _timing: dict[str, float] = {}
    _t_total = time.perf_counter()
    extraction_updates: List[dict] = []

    # Measurement-only: snapshot the pre-extraction state so the shadow audit
    # can simulate the LLM path without touching the live state. Only taken
    # when the audit is enabled (zero cost otherwise).
    _audit_state_before = copy.deepcopy(project_state) if _hybrid_audit_enabled() else None

    # Step 1 — deterministic rule extraction. Always runs: it feeds the
    # objective-aware hybrid decision below and the existing fallback path.
    rule_observation = RuleBasedExtractor.extract(user_message, storage_project_name)

    # Step 2 — objective-aware hybrid decision: when the deterministic
    # extraction already satisfies the CURRENT objective, the LLM extraction
    # is skipped and the rules are applied through the existing StateManager
    # merge path. Otherwise the LLM runs exactly as before.
    hybrid_decision = decide_hybrid_extraction(
        rule_observation, project_state, session_data, user_message=user_message
    )
    if _capture is not None:
        _capture["hybrid_decision"] = hybrid_decision

    if hybrid_decision["llm_invoked"]:
        # Step 4 — LLM extraction (today's behavior, unchanged).
        extractor_model = resolve_extractor_model()
        extractor = MemoryExtractor(model_name=extractor_model)
        extraction_result = extractor.extract(
            user_message=user_message,
            project_state=project_state,
            previous_assistant_message=last_assistant_msg,
            _timing=_timing,
            _prof=_prof,
        )
        if _capture is not None:
            _capture["extraction_updates"] = [u.to_dict() for u in extraction_result.updates]
            _capture["extraction_message_type"] = extraction_result.message_type.value
        print(
            f"[MemoryExtractor] type={extraction_result.message_type.value} "
            f"updates={len(extraction_result.updates)} model={extractor_model}"
        )

        # Measurement-only accuracy audit: classify every LLM-extracted update
        # as Correct / Partially Correct / Incorrect / Missed Opportunity.
        # Never feeds a decision; only surfaces in Developer Console diagnostics
        # and the session-level summary. ``state_after`` is filled by the caller.
        if _capture is not None:
            _capture["extraction_accuracy"] = assess_extraction_accuracy(
                user_message=user_message,
                objective=hybrid_decision.get("objective"),
                extraction_result=extraction_result,
                state_before=_capture.get("state_before", {}),
                rule_observation=rule_observation,
            )

        applied = _apply_extraction_to_state(
            project_state,
            extraction_result,
            previous_assistant_message=last_assistant_msg,
            previous_user_message=user_message,
            _timing=_timing,
        )

        if not applied:
            # The LLM extraction produced nothing usable (NO_UPDATE/AMBIGUOUS/
            # END) or its batch was rejected. Run the deterministic fallback so
            # a natural answer (e.g. "It happens every single day.") is still
            # captured and the mentor does not re-ask the same question.
            extracted = InputProcessor.extract_structured_data(
                model_name, user_message, storage_project_name, project_state=project_state, _timing=_timing, _prof=_prof
            )
            if _capture is not None:
                _capture["rule_based_extraction"] = extracted
            merge_extracted_to_state(project_state, session_data, extracted, _timing=_timing)
            # If the rule-based fallback produced meaningful updates after an
            # AMBIGUOUS LLM extraction, mark the message type as MEANINGFUL
            # so the answer-satisfaction check in _determine_objective() runs.
            if _capture is not None and extracted and _capture.get("extraction_message_type") != MessageType.MEANINGFUL.value:
                _capture["extraction_message_type"] = MessageType.MEANINGFUL.value
    else:
        # Step 3 — the deterministic extraction satisfies the current
        # objective: skip the LLM entirely and apply the rules through the
        # existing StateManager merge path.
        applied = True
        extraction_result = None
        if _capture is not None:
            _capture["extraction_message_type"] = MessageType.MEANINGFUL.value
            _capture["extraction_updates"] = rule_update_dicts(rule_observation)
            _capture["rule_based_extraction"] = rule_observation
        print(
            f"[HybridExtraction] objective={hybrid_decision['objective']} "
            f"field={hybrid_decision['objective_field']} satisfied by "
            f"deterministic extraction; LLM extraction skipped"
        )
        merge_extracted_to_state(project_state, session_data, rule_observation, _timing=_timing)

        # Measurement-only shadow audit (HYBRID_AUDIT=true): run the LLM
        # extractor in shadow mode WITHOUT applying its result, to quantify
        # what skipping the LLM may have missed. Never touches state,
        # objectives, lifecycle, prompts, or replies.
        if _hybrid_audit_enabled():
            record = _run_shadow_audit(
                rule_observation,
                project_state,
                session_data,
                user_message,
                last_assistant_msg,
                _audit_state_before,
                _timing,
                _prof=_prof,
            )
            if _capture is not None:
                _capture["hybrid_audit"] = record

    # Observation-only audit: what the deterministic rule extractor would
    # have captured vs what the LLM extractor produced. Never feeds a
    # decision and never changes what is applied to state.
    if _capture is not None:
        _capture["extraction_comparison"] = compare_extraction(
            rule_observation, extraction_result, applied
        )

    if session_data.current_stage != STAGE_EMPATHIZE_COMPLETE:
        session_data.current_stage = STAGE_EMPATHIZE

    if _capture is not None:
        _capture["extraction_timing"] = summarize_extraction_timing(
            _timing, (time.perf_counter() - _t_total) * 1000
        )

    # Record user-stated facts as known context, so the mentor does not
    # re-ask about them later.  This runs after extraction so the _known
    # flags are available for _determine_objective() this turn.
    if user_message and user_message.strip():
        memory = getattr(session_data, "conversation_memory", None)
        if memory is not None:
            user_lower = user_message.strip().lower()
            extracted_field_values = {u.get("field") for u in (extraction_updates or [])}
            # Keywords per REQUIRED_FIELD that indicate the user is stating
            # something about that field voluntarily.  Using keyword sets is
            # more robust than substring-on-the-field-name alone.
            field_keywords: dict[StateField, tuple[str, ...]] = {
                StateField.PERSONAS: (
                    "who", "people", "users", "audience", "target", "personas",
                ),
                StateField.PROBLEMS: (
                    "problem", "issue", "challenge", "trouble", "struggle",
                ),
                StateField.CURRENT_SOLUTIONS: (
                    "solution", "workaround", "currently", "tool", "software",
                ),
                StateField.PAIN_POINTS: (
                    "pain", "frustrat", "stress", "worry", "hurt", "annoy",
                ),
                StateField.EVIDENCE: (
                    "seen", "observed", "research", "data", "studied", "interview",
                ),
                StateField.FREQUENCY: (
                    "frequency", "often", "daily", "weekly", "cadence",
                ),
            }
            for sf, keywords in field_keywords.items():
                if any(kw in user_lower for kw in keywords):
                    if sf.value not in extracted_field_values:
                        _record_user_stated_facts(memory, sf.value)


def _run_shadow_audit(rule_observation, project_state, session_data, user_message, last_assistant_msg, state_before, _timing=None, _prof=None):
    """Run the LLM extractor in shadow mode for one hybrid skip and record
    the comparison. Measurement only: the shadow result is never applied to
    ``project_state`` and never influences any decision."""
    import hybrid_audit

    extractor_model = resolve_extractor_model()
    extractor = MemoryExtractor(model_name=extractor_model)
    shadow = extractor.extract(
        user_message=user_message,
        project_state=project_state,
        previous_assistant_message=last_assistant_msg,
        _timing=_timing,
        _prof=_prof,
    )
    record = hybrid_audit.build_audit_record(
        rule_observation,
        shadow,
        state_before,
        project_state,
        session_data,
        previous_assistant_message=last_assistant_msg,
        previous_user_message=user_message,
    )
    print(
        f"[HybridAudit] shadow LLM type={shadow.message_type.value} "
        f"fields={[field_label(f) for f in record['shadow_llm_fields']]} "
        f"additional={[field_label(f) for f in record['additional_fields']]} "
        f"would_change_state={record['would_change_state']} "
        f"would_change_objective={record['would_change_objective']} "
        f"risk={record['risk_level']}"
    )
    session_data.hybrid_audit_summary.add(record)
    return record


def _determine_objective(project_state, session_data=None, latest_user_message=None, extraction_message_type=None, extraction_updates=None):
    """Run the Objective Engine over the current state, feeding it a
    read-only conversation context (raw user messages + asked question
    families) so information-gain scoring can prefer what the user has
    already discussed and avoid repeat families. Deterministic.

    If `latest_user_message` is provided, extraction was MEANINGFUL, the
    targeted field was NOT captured by extraction, and the message
    heuristically satisfies the targeted field, the objective is re-computed
    with that field artificially marked as satisfied — preventing the mentor
    from asking a question the user just answered (but extraction missed).
    """
    context = None
    if session_data is not None:
        context = ObjectiveContext(
            user_messages=tuple(
                m.get("content", "")
                for m in session_data.conversation_history
                if m.get("role") == "user"
            ),
            asked_families=frozenset(session_data.asked_question_families),
        )

    # Initial objective selection
    objective = _OBJECTIVE_ENGINE.determine_next(project_state, context=context)

    # Answer-satisfaction check: only if extraction was MEANINGFUL (found
    # actual information), the targeted field was NOT captured by extraction
    # (meaning extraction missed it), the question for that field has already
    # been asked (so the user is answering, not volunteering), and the latest
    # user message heuristically satisfies the targeted field.
    if (
        extraction_message_type == "MEANINGFUL"
        and latest_user_message
        and not objective.is_wrap_up
        and objective.targeted_field() is not None
        and session_data is not None
    ):
        targeted = objective.targeted_field()
        # Check if extraction already captured this field
        extracted_fields = {u.get("field") for u in (extraction_updates or [])}
        if targeted.value not in extracted_fields:
            # Only advance if a question family for this field was already asked.
            # This prevents treating volunteered info as an answer to an unasked question.
            families_for_field = FAMILIES_BY_FIELD.get(targeted, ())
            asked_families = set(session_data.asked_question_families)
            family_asked = any(f.value in asked_families for f in families_for_field)
            if family_asked and _message_satisfies_field(latest_user_message, targeted, project_state):
                # Create a temporary state where this field is satisfied
                satisfied_state = _make_field_satisfied(project_state, targeted)
                # Re-run objective selection with the satisfied state
                objective = _OBJECTIVE_ENGINE.determine_next(satisfied_state, context=context)
                print(
                    f"[AnswerSatisfaction] field={targeted.value} satisfied by latest message "
                    f"(extraction missed, family asked); advanced to objective={objective.objective.value}"
                )

    # Category (2): if the user has already stated this fact (known=True
    # in conversation memory) AND a question family for this field was already
    # asked, advance the objective without marking the field satisfied in
    # ProjectState.  This prevents re-asking about facts the user already
    # established, while still distinguishing established context from a direct
    # answer to a question the mentor asked.
    if (
        objective.targeted_field() is not None
        and session_data is not None
    ):
        targeted = objective.targeted_field()
        memory = getattr(session_data, "conversation_memory", None)
        if memory is not None:
            known_for_targeted = any(
                rec.get("field") == targeted.value and rec.get("known") is True
                for rec in memory.open_threads
            )
            if known_for_targeted:
                asked_families = set(session_data.asked_question_families)
                families_for_field = FAMILIES_BY_FIELD.get(targeted, ())
                family_asked = any(f.value in asked_families for f in families_for_field)
                if family_asked:
                    # Advance objective based on established context only,
                    # without marking the field satisfied in ProjectState.
                    objective = _OBJECTIVE_ENGINE.determine_next(
                        project_state, context=context
                    )
                    print(
                        f"[KnownContext] field={targeted.value} known=True "
                        f"family_asked=True; advanced objective to "
                        f"{objective.objective.value} "
                        f"(without field satisfaction in ProjectState)"
                    )

    print(
        f"[ObjectiveEngine] objective={objective.objective.value} "
        f"confidence={objective.confidence} missing="
        f"{[sf.value for sf in objective.missing_fields]} "
        f"completed={[sf.value for sf in objective.completed_fields]}"
    )
    candidate_strategy = _RESPONSE_STRATEGY_ENGINE.determine_strategy(
        objective, project_state
    )
    return objective, candidate_strategy


def _plan_question_family(objective, project_state, session_data):
    """Plan which question family the next question should target.

    Read-only: consults the objective's targeted field, the families already
    asked this session, and the current sufficiency snapshot. Returns a
    ``FamilyPlan`` (selected family + skip reasons + re-ask flag)."""
    targeted = objective.targeted_field()
    sufficiency = SufficiencyChecker.evaluate(project_state)
    plan = _QUESTION_FAMILY_PLANNER.plan(
        field=targeted,
        asked_family_values=session_data.asked_question_families,
        sufficiency=sufficiency,
    )
    if plan.selected is not None:
        print(
            f"[QuestionFamily] field={targeted.value} "
            f"selected={plan.selected.value} reask={plan.reask} "
            f"asked={list(plan.asked_families)}"
        )
    return plan


def _guard_repeated_question(reply, plan, project_state):
    """Enforce: do not ask another question from an already-asked family unless
    the previous answers for that field were insufficient.

    Returns ``(reply, question_family)``. When the LLM ignored the family
    steering and produced a repeat, the reply is replaced with ``None`` so the
    caller substitutes the family-aware deterministic fallback."""
    question_family = classify_question(reply)
    if question_family is None:
        return reply, None
    if plan is None or plan.selected is None:
        return reply, question_family
    if question_family.value not in plan.asked_families:
        return reply, question_family

    field = field_of(question_family)
    sufficiency = SufficiencyChecker.evaluate(project_state)
    if field is not None and sufficiency.is_satisfied(field):
        # Previous answers for this field were sufficient — repeating its
        # family is a semantic duplicate.
        return None, question_family
    if plan.selected != question_family:
        # Answers are still insufficient but a fresh family is available;
        # don't allow the LLM to fall back to a family already asked.
        return None, question_family
    # Every family has been asked and answers remain insufficient — the
    # re-ask is sanctioned by the "unless previous answers were insufficient"
    # rule.
    return reply, question_family


def _inject_lifecycle_marker(project_state, session_data):
    if session_data.empathize_summary_presented:
        StateManager(project_state).set_previous_assistant_message(
            summary_presented_marker(project_state.previous_assistant_message)
        )


def _resolve_lifecycle(project_state, objective, candidate_strategy, session_data):
    lifecycle_decision = _LIFECYCLE_MANAGER.determine(
        project_state, objective, candidate_strategy
    )
    session_data.previous_lifecycle_decision = lifecycle_decision.value
    print(f"[LifecycleManager] decision={lifecycle_decision.value}")
    return lifecycle_decision


def _branch_on_lifecycle(lifecycle_decision, project_state, candidate_strategy, session_data):
    empathize_summary: EmpathizeSummary | None = None
    response_strategy: ResponseStrategy = candidate_strategy

    if lifecycle_decision is LifecycleDecision.CONTINUE:
        pass

    elif lifecycle_decision is LifecycleDecision.READY_FOR_SUMMARY:
        empathize_summary = _SUMMARY_BUILDER.build(project_state)
        response_strategy = ResponseStrategy.GENERATE_SUMMARY
        print(
            f"[SummaryBuilder] summary has "
            f"{sum(len(getattr(empathize_summary, f) or []) for f in (
                'personas', 'problems', 'current_solutions',
                'pain_points', 'evidence'
            ))} list items, frequency={empathize_summary.frequency!r}"
        )

    elif lifecycle_decision is LifecycleDecision.WAITING_FOR_CONFIRMATION:
        response_strategy = ResponseStrategy.ASK_QUESTION

    elif lifecycle_decision is LifecycleDecision.READY_FOR_TRANSITION:
        session_data.current_stage = STAGE_EMPATHIZE_COMPLETE
        session_data.empathize_summary_presented = False
        response_strategy = ResponseStrategy.ASK_QUESTION
        empathize_summary = _SUMMARY_BUILDER.build(project_state)

    else:
        raise RuntimeError(
            f"LifecycleManager returned an unsupported decision: "
            f"{lifecycle_decision!r}"
        )

    print(f"[ResponseStrategyEngine] strategy={response_strategy.value}")
    return empathize_summary, response_strategy


def _family_display_label(value):
    """Human-readable label for a QuestionFamily value (None-safe)."""
    if not value:
        return None
    try:
        return family_label(QuestionFamily(value))
    except (ValueError, KeyError, TypeError):
        return str(value)


def _build_turn_brief(
    _capture, objective, coaching_strategy, lifecycle_decision,
    session_data, family_to_ask, asked_families, user_message,
):
    """Compose the transient Phase 1 Conversation Brief (read-only).

    Every input is an already-computed pipeline signal; this function only
    projects them into the Brief shape. It never mutates state, objectives,
    lifecycle, memory, or any decision.
    """
    capture = _capture or {}
    ctx = capture.get("conversation_context")
    recovery = capture.get("recovery") or {}
    return build_conversation_brief(
        context_mode=getattr(ctx, "mode", None),
        context_confidence=getattr(ctx, "confidence", None),
        acknowledge_first=bool(getattr(ctx, "acknowledge_first", False)),
        recovery=recovery if isinstance(recovery, dict) else {},
        state_before=capture.get("state_before"),
        state_after=capture.get("state_after"),
        objective=objective,
        coaching=(
            coaching_strategy.value
            if coaching_strategy is not None
            else None
        ),
        lifecycle=(
            lifecycle_decision.value
            if lifecycle_decision is not None
            else None
        ),
        fresh_family=family_to_ask.value if family_to_ask is not None else None,
        asked_families=list(asked_families or []),
    )


def _span(profiler, name):
    """Span helper for latency profiling (Phase 1.5 audit).

    No-op context manager when ``profiler`` is None, otherwise a
    ``TurnProfiler`` span. Keeps every call site behavior-identical
    with profiling off.
    """
    if profiler is None:
        return _null_span()
    return profiler.span(name)


@contextmanager
def _null_span():
    yield


def _generate_reply(project_state, objective, response_strategy, last_assistant_msg, user_message, empathize_summary, lifecycle_decision, model_name, family_plan=None, session_data=None, _timing=None, _capture=None, _prof=None):
    if _timing is None:
        _timing = {}

    # Phase 2 gate consumption. On a paused conversational turn the
    # FamilyPlan is still computed upstream and recorded for diagnostics,
    # but NONE of its steering reaches Module 4 (no family guidance, no
    # avoid-families bullet, no DT template fallback). The Objective Engine
    # decision itself is untouched and resumes canonically next turn.
    dt_paused = bool((_capture or {}).get("dt_paused"))
    context_mode = getattr(
        (_capture or {}).get("conversation_context"), "mode", None
    )

    family_to_ask = family_plan.selected if family_plan is not None else None
    asked_families = (
        list(family_plan.asked_families) if family_plan is not None else []
    )
    if dt_paused:
        family_to_ask = None
        asked_families = []

    # Conversational-memory resume hint: unfinished topics (partial answers,
    # deferred topics, acknowledged summary) the LLM should build on instead
    # of restarting. Only for question-asking strategies — a summary reply
    # should never be steered by pending threads. Read-only projection.
    resume_hint = None
    if (
        session_data is not None
        and response_strategy is not ResponseStrategy.GENERATE_SUMMARY
    ):
        resume_hint = memory_to_prompt_bullets(
            session_data, current_target=objective.targeted_field()
        ) or None

    t_prompt_start = time.perf_counter()

    # Adaptive Coaching Strategy: decide HOW to continue the conversation
    # before building the prompt.  Purely deterministic; never changes
    # objectives, lifecycle, extraction, or ProjectState.
    coaching_strategy, coaching_reason = None, ""
    if session_data is not None:
        from session_manager import get_session_manager
        sd = get_session_manager().get_active_session_data()
        memory = getattr(sd, "conversation_memory", None)
        coaching_strategy, coaching_reason = determine_coaching_strategy(
            objective=objective.objective.value,
            objective_advancement=getattr(objective, "advancement", None),
            objective_confidence=getattr(objective, "confidence", None),
            state_before=_capture.get("state_before") if _capture else None,
            state_after=_capture.get("state_after") if _capture else None,
            user_message=user_message,
            previous_assistant_reply=last_assistant_msg or "",
            recovery_category=(
                (_capture.get("recovery") or {}).get("Category")
                if _capture else None
            ),
            memory_open_threads=list(getattr(memory, "open_threads", [])),
            memory_deferred_topics=list(getattr(memory, "deferred_topics", [])),
            memory_resolved_threads=list(getattr(memory, "resolved_threads", [])),
            objective_completed_fields=(
                [sf.value for sf in objective.completed_fields]
                if hasattr(objective, "completed_fields") else None
            ),
            objective_missing_fields=(
                [sf.value for sf in objective.missing_fields]
                if hasattr(objective, "missing_fields") else None
            ),
        )
    if _capture is not None:
        _capture["coaching_strategy"] = coaching_strategy
        _capture["coaching_reason"] = coaching_reason

    coaching_bullet = (
        coaching_instruction_bullet(coaching_strategy)
        if coaching_strategy is not None else None
    )
    extra_bullets = None
    if coaching_bullet:
        extra_bullets = [coaching_bullet]

    # Insight Detection: recognise meaningful user insights before building
    # the prompt.  Purely deterministic guidance; never changes extraction,
    # objectives, lifecycle, or ProjectState.
    insight_type, insight_confidence, insight_explanation, insight_signals = (
        InsightType.NONE, None, "", []
    )
    if _capture is not None:
        insight_type, insight_confidence, insight_explanation, insight_signals = (
            detect_insight(
                user_message=user_message,
                current_objective=objective.objective.value,
                state=_capture.get("state_after"),
                memory=getattr(session_data, "conversation_memory", None),
                recovery_category=(
                    (_capture.get("recovery") or {}).get("Category")
                    if _capture else None
                ),
                extraction_updates=(
                    _capture.get("extraction_updates")
                    if _capture else None
                ),
            )
        )
        _capture["insight_type"] = insight_type
        _capture["insight_confidence"] = insight_confidence
        _capture["insight_explanation"] = insight_explanation
        _capture["insight_signals"] = insight_signals

    insight_bullet = (
        insight_instruction_bullet(insight_type, insight_confidence)
        if insight_type is not None and insight_type is not InsightType.NONE
        else None
    )
    if insight_bullet:
        extra_bullets = list(extra_bullets or []) + [insight_bullet]

    # Conversational Move Planner: synthesise objective, coaching strategy,
    # insight detection, memory, and ProjectState into one primary
    # intention for the next reply.  Purely deterministic guidance; never
    # changes extraction, objectives, lifecycle, or ProjectState.
    conversation_move, move_reason, move_signals = (
        ConversationMove.ELICIT_INFORMATION, "", []
    )
    memory_for_move = getattr(session_data, "conversation_memory", None) \
        if session_data is not None else None
    conversation_move, move_reason, move_signals = determine_conversation_move(
        current_objective=objective.objective.value,
        coaching_strategy=(
            coaching_strategy.value if coaching_strategy is not None else None
        ),
        insight_type=(
            insight_type.value
            if insight_type is not None and insight_type is not InsightType.NONE
            else None
        ),
        insight_confidence=(
            insight_confidence.value if insight_confidence is not None else None
        ),
        user_message=user_message,
        state_before=_capture.get("state_before") if _capture else None,
        state_after=_capture.get("state_after") if _capture else None,
        memory_open_threads=list(getattr(memory_for_move, "open_threads", [])),
        memory_deferred_topics=list(getattr(memory_for_move, "deferred_topics", [])),
        memory_resolved_threads=list(getattr(memory_for_move, "resolved_threads", [])),
        memory_partially_answered=list(getattr(memory_for_move, "partially_answered_objectives", [])),
        objective_advancement=getattr(objective, "advancement", None),
        objective_completed_fields=(
            [sf.value for sf in objective.completed_fields]
            if hasattr(objective, "completed_fields") else None
        ),
        objective_missing_fields=(
            [sf.value for sf in objective.missing_fields]
            if hasattr(objective, "missing_fields") else None
        ),
        recovery_category=(
            (_capture.get("recovery") or {}).get("Category")
            if _capture else None
        ),
        summary_presented=bool(
            getattr(session_data, "empathize_summary_presented", False)
        ) if session_data is not None else False,
    )
    if _capture is not None:
        _capture["conversation_move"] = conversation_move
        _capture["conversation_move_reason"] = move_reason
        _capture["conversation_move_signals"] = move_signals

    move_bullet = conversation_move_instruction_bullet(conversation_move)
    if move_bullet:
        extra_bullets = list(extra_bullets or []) + [move_bullet]

    # Phase 2: one concise GUIDANCE bullet naming the conversational
    # situation (correction / topic shift / direct question / confusion).
    # Guidance only — never a response template. Normal turns get no extra
    # bullet here, so their prompts stay byte-identical.
    context_bullet = (
        context_instruction_bullet(context_mode) if dt_paused else None
    )
    if context_bullet:
        extra_bullets = list(extra_bullets or []) + [context_bullet]

    # Phase 6: MEDIUM-confidence conversational signal -> acknowledge-first
    # framing WITHOUT pausing anything. Extraction, objective, family
    # planning, enforcement, and family recording all stay fully canonical;
    # Module 4 gets one extra uncertainty-acknowledgment bullet and the
    # planned DT question remains the reply's canonical question.
    ctx_now = (_capture or {}).get("conversation_context")
    acknowledge_first = (
        not dt_paused
        and ctx_now is not None
        and bool(getattr(ctx_now, "acknowledge_first", False))
    )
    if acknowledge_first:
        ack_bullet = acknowledge_first_instruction_bullet(ctx_now.mode)
        if ack_bullet:
            extra_bullets = list(extra_bullets or []) + [ack_bullet]

    # Phase 1 dynamic coach: compose the transient Conversation Brief from
    # the deterministic signals already computed above (context mode,
    # recovery, validated state diff, objective, coaching/move, lifecycle,
    # memory). The Brief is advisory only — it never mutates state,
    # completes objectives, or changes the lifecycle. All computations above
    # still run and are still recorded for diagnostics; only the prompt
    # construction changes, and only on non-paused turns (paused turns keep
    # the legacy prompt byte-identical).
    with _span(_prof, "brief"):
        brief = _build_turn_brief(
            _capture=_capture,
            objective=objective,
            coaching_strategy=coaching_strategy,
            lifecycle_decision=lifecycle_decision,
            session_data=session_data,
            family_to_ask=family_to_ask,
            asked_families=asked_families,
            user_message=user_message,
        )
    if _capture is not None:
        _capture["conversation_brief"] = brief.to_dict()

    with _span(_prof, "prompt_build"):
        if dt_paused:
            prompt = build_module4_prompt(
                project_state=project_state,
                conversation_objective=objective,
                response_strategy=response_strategy,
                previous_assistant_message=last_assistant_msg,
                previous_user_message=user_message,
                empathize_summary=empathize_summary,
                question_family=family_to_ask.value if family_to_ask else None,
                previously_asked_families=asked_families,
                resume_hint=resume_hint,
                coaching_bullet=extra_bullets,
            )
            if _capture is not None:
                _capture["prompt"] = prompt
                _capture["prompt_path"] = "legacy"
        else:
            from relevant_context import build_relevant_context, render_relevant_context

            _targeted = objective.targeted_field()
            snippets = build_relevant_context(
                brief=brief,
                state_after=(_capture or {}).get("state_after"),
                memory=getattr(session_data, "conversation_memory", None),
                conversation_history=getattr(session_data, "conversation_history", None),
                last_assistant_message=last_assistant_msg,
                current_target=_targeted.value if _targeted is not None else None,
            )
            prompt = build_dynamic_prompt(
                brief=brief,
                relevant_context_text=render_relevant_context(snippets),
                user_message=user_message,
                previous_assistant_message=last_assistant_msg,
                response_strategy=response_strategy,
                empathize_summary=empathize_summary,
                fresh_family_label=_family_display_label(
                    family_to_ask.value if family_to_ask else None
                ),
                asked_family_labels=[
                    label
                    for label in (
                        _family_display_label(v) for v in (asked_families or [])
                    )
                    if label
                ] or None,
            )
            if _capture is not None:
                _capture["prompt"] = prompt
                _capture["prompt_path"] = "dynamic"

    if dt_paused:
        # Paused conversational turn: the deterministic path must NOT push
        # the questionnaire either. Use the mode's conversational fallback.
        fallback_reply = (
            paused_turn_fallback(context_mode)
            or _build_journey_fallback(
                lifecycle_decision=lifecycle_decision,
                conversation_objective=objective,
                response_strategy=response_strategy,
                project_state=project_state,
                empathize_summary=empathize_summary,
            )
        )
    else:
        fallback_reply = _build_journey_fallback(
            lifecycle_decision=lifecycle_decision,
            conversation_objective=objective,
            response_strategy=response_strategy,
            project_state=project_state,
            empathize_summary=empathize_summary,
            question_family=family_to_ask,
        )
    t_prompt_end = time.perf_counter()
    _timing["prompt_ms"] = (t_prompt_end - t_prompt_start) * 1000

    reply = fallback_reply
    t_llm_start = time.perf_counter()
    try:
        response = timed_ollama_chat(
            _prof,
            purpose="response_generation",
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.4, "top_p": 0.85, "num_predict": 150},
            gated=False,  # the response call always runs (fallback on error)
        )
        llm_reply = strip_model_output(response["message"]["content"])
        reply = enforce_mentor_reply(
            llm_reply,
            fallback_reply,
            allow_summary=(
                response_strategy is ResponseStrategy.GENERATE_SUMMARY
                or objective.is_wrap_up
            ),
            # Paused conversational turns may answer with a pure statement;
            # advice-marker protection and word limits stay active.
            # Phase 1 dynamic coach: statement-licensed situations (correction
            # acknowledgment, side-question answer, confusion repair) also
            # permit a question-less reply; the question remains the normal
            # mechanism everywhere else.
            allow_statement=(dt_paused or brief.statement_allowed),
        )
    except Exception as e:
        print(f"LLM question generation failed, using template fallback: {e}")
    t_llm_end = time.perf_counter()
    _timing["llm_ms"] = (t_llm_end - t_llm_start) * 1000

    # Semantic de-duplication guard: if the produced question repeats a family
    # that was already asked (and the answers for that field were sufficient,
    # or a fresh family is available), fall back to the family-aware template
    # question. This is the hard enforcement behind the prompt-level steering.
    # On a paused conversational turn the guard is bypassed by passing no
    # plan — a natural clarification ("what part feels hardest?") must never
    # be swapped for a questionnaire template.
    with _span(_prof, "guard"):
        guarded_reply, question_family = _guard_repeated_question(
            reply, None if dt_paused else family_plan, project_state
        )
        if _capture is not None:
            _capture["guard_blocked"] = guarded_reply is None
        if guarded_reply is None:
            print(
                f"[QuestionFamily] blocked repeated question "
                f"family={question_family.value if question_family else None}; "
                f"using family-aware fallback"
            )
            guarded_reply = _build_journey_fallback(
                lifecycle_decision=lifecycle_decision,
                conversation_objective=objective,
                response_strategy=response_strategy,
                project_state=project_state,
                empathize_summary=empathize_summary,
                question_family=family_to_ask,
            )
            question_family = classify_question(guarded_reply)

    if _capture is not None:
        _capture["question_family"] = (
            question_family.value if question_family is not None else None
        )

    if not guarded_reply or len(guarded_reply.strip()) <= 2:
        if dt_paused:
            guarded_reply = fallback_reply
        else:
            guarded_reply = _build_journey_fallback(
                lifecycle_decision=lifecycle_decision,
                conversation_objective=objective,
                response_strategy=response_strategy,
                project_state=project_state,
                empathize_summary=empathize_summary,
                question_family=family_to_ask,
            )
    return guarded_reply


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def print_ascii_dashboard(session, project_state=None):
    GREEN, YELLOW, CYAN, BOLD, RESET = "\033[92m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  AI DESIGN THINKING MENTOR (EMPATHIZE){RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"  {BOLD}Project:{RESET} {session.project_name or 'Not established'}")
    print(f"  {BOLD}Stage:{RESET} {session.current_stage}")
    print(f"{CYAN}{'-' * 60}{RESET}")

    ps = project_state or ProjectState()
    for key, item in ChecklistManager.get_status_dict(ps).items():
        mark = f"{GREEN}OK{RESET}" if item["complete"] else f"{YELLOW}MISSING{RESET}"
        val = item["value"] or "Missing"
        print(f"  [{mark}] {item['description'].title():35} : {val}")

    if session.known_facts:
        print(f"\n  {BOLD}Known Facts:{RESET}")
        for fact in session.known_facts:
            print(f"    - {fact}")

    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")
