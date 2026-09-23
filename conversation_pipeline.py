"""
Conversation pipeline — LLM reply guard + deterministic fallback replies.

Responsibilities:
  * `_OBJECTIVE_FALLBACK_QUESTION` — one-line question per Objective
  * `build_deterministic_fallback` — Module 4 fallback (no LifecycleDecision)
  * `_build_summary_fallback` — summary prose from ProjectState
  * `_build_summary_fallback_from_summary` — summary prose from EmpathizeSummary
  * `_build_journey_fallback` — Module 5-aware fallback (full lifecycle routing)
  * `enforce_mentor_reply` — keep the LLM in communicator-only mode
  * `_apply_extraction_to_state` — apply an ExtractionResult to ProjectState
                           via StateManager (Module 2 contract)

Owns NO session persistence, NO extraction logic, NO lifecycle decision-making.
"""

__all__ = [
    "build_deterministic_fallback",
    "enforce_mentor_reply",
    "_build_journey_fallback",
    "_apply_extraction_to_state",
]

from memory_extractor import ExtractionResult, MessageType, ProjectState
from state_manager import StateManager, StateBatchValidationError
from module3 import (
    ConversationObjective,
    FAMILY_FALLBACK_QUESTIONS,
    Objective,
    QuestionFamily,
)
from module4 import ResponseStrategy
from module5 import EmpathizeSummary, LifecycleDecision

_OBJECTIVE_FALLBACK_QUESTION: dict[Objective, str] = {
    Objective.PERSONAS: "That's a meaningful starting point. To focus the design on the right people, who specifically would benefit from this - can you describe the people you're designing for?",
    Objective.PROBLEMS: "That's a clear view of who you're helping. To shape the problem statement precisely, what is the core problem or frustration they're experiencing?",
    Objective.CURRENT_SOLUTIONS: "That's a real gap you're describing. To see what they currently lean on, how do people handle this problem today - what workarounds or tools do they currently use?",
    Objective.PAIN_POINTS: "That's a compelling motivation. To understand what drives you, what personally drew you to work on this problem?",
    Objective.EVIDENCE: "That's a strong signal. To ground the problem in real observations, what have you seen or heard that tells you this is a real problem worth solving?",
    Objective.FREQUENCY: "That's a clear picture of the situation. To understand how urgently this needs solving, how often does the problem occur - daily, weekly, or only in certain situations?",
}


def build_deterministic_fallback(
    conversation_objective: ConversationObjective,
    response_strategy: ResponseStrategy,
    project_state: ProjectState,
    question_family: QuestionFamily | None = None,
) -> str:
    if response_strategy is ResponseStrategy.GENERATE_SUMMARY or \
            conversation_objective.is_wrap_up:
        return _build_summary_fallback(project_state)

    targeted = conversation_objective.targeted_field()

    if question_family is not None:
        family_question = FAMILY_FALLBACK_QUESTIONS.get(question_family)
        if family_question:
            return family_question

    if targeted is not None:
        return _OBJECTIVE_FALLBACK_QUESTION.get(
            conversation_objective.objective,
            "That's a good foundation. To keep the picture accurate, could you tell me more about the people affected and the problem they're facing?",
        )

    return "That's a good foundation. To keep the picture accurate, could you tell me more about the people affected and the problem they're facing?"


def _build_summary_fallback(project_state: ProjectState) -> str:
    parts = []
    if project_state.personas:
        parts.append(f"you're focusing on {', '.join(project_state.personas)}")
    if project_state.problems:
        parts.append(f"who struggle with {', '.join(project_state.problems)}")
    if project_state.pain_points:
        parts.append(
            f"and you're motivated because of {', '.join(project_state.pain_points)}"
        )
    if parts:
        summary = "Here's what I understand so far: " + ", ".join(parts) + "."
    else:
        summary = "Here's what I understand so far: some initial details about your project."
    return f"That's a rich set of insights. {summary} Does that capture things accurately, or would you like to add anything?"


def _build_summary_fallback_from_summary(summary: EmpathizeSummary) -> str:
    parts = []
    if summary.personas:
        parts.append(f"you're focusing on {', '.join(summary.personas)}")
    if summary.problems:
        parts.append(f"who struggle with {', '.join(summary.problems)}")
    if summary.pain_points:
        parts.append(
            f"and you're motivated because of {', '.join(summary.pain_points)}"
        )
    if summary.current_solutions:
        parts.append(f"currently handling it with {', '.join(summary.current_solutions)}")
    if summary.frequency:
        parts.append(f"and running into this {summary.frequency}")
    if summary.evidence:
        parts.append(f"with evidence: {', '.join(summary.evidence)}")
    if parts:
        summary_text = "Here's what I understand so far: " + ", ".join(parts) + "."
    else:
        summary_text = "Here's what I understand so far: some initial details about your project."
    return f"That's a rich set of insights. {summary_text} Does that capture things accurately, or would you like to add anything?"


def _build_journey_fallback(
    *,
    lifecycle_decision: LifecycleDecision,
    conversation_objective: ConversationObjective,
    response_strategy: ResponseStrategy,
    project_state: ProjectState,
    empathize_summary: EmpathizeSummary | None = None,
    question_family: QuestionFamily | None = None,
) -> str:
    if lifecycle_decision is LifecycleDecision.READY_FOR_SUMMARY:
        if empathize_summary is not None:
            return _build_summary_fallback_from_summary(empathize_summary)
        return _build_summary_fallback(project_state)

    if lifecycle_decision is LifecycleDecision.WAITING_FOR_CONFIRMATION:
        return (
            "That's the full summary as I understand it. Does it capture "
            "things accurately, or would you like to refine anything?"
        )

    if lifecycle_decision is LifecycleDecision.READY_FOR_TRANSITION:
        return (
            "Great - that wraps up Empathize nicely. Before we move to the "
            "next stage, is there anything you'd like to revisit or add?"
        )

    return build_deterministic_fallback(
        conversation_objective,
        response_strategy,
        project_state,
        question_family=question_family,
    )


def enforce_mentor_reply(reply, fallback, allow_summary=False, allow_statement=False):
    text = (reply or "").strip()
    if not text:
        return fallback

    lower = text.lower()
    advice_markers = (
        "you should",
        "you could",
        "i suggest",
        "i recommend",
        "try ",
        "build ",
        "implement",
        "add a feature",
        "the solution is",
    )
    if any(marker in lower for marker in advice_markers):
        return fallback

    question_count = text.count("?")
    if question_count == 0:
        # Paused conversational turns (allow_statement=True) may respond
        # with a pure acknowledgment/clarification statement. Advice-marker
        # protection above stays active in ALL cases; word limits below
        # stay active in ALL cases.
        if not allow_statement:
            return fallback
    elif question_count > 1:
        first_question_end = text.find("?")
        text = text[: first_question_end + 1].strip()

    if not allow_summary and len(text.split()) > 55:
        return fallback
    if allow_summary and len(text.split()) > 95:
        return fallback

    return text


def _apply_extraction_to_state(
    state: ProjectState,
    result: ExtractionResult,
    *,
    previous_assistant_message: str | None = None,
    previous_user_message: str | None = None,
    _timing: dict[str, float] | None = None,
) -> bool:
    """
    Apply an ExtractionResult to ProjectState via StateManager (Module 2).

    Returns ``True`` if the batch was applied (MEANINGFUL with at least one
    accepted update), ``False`` if it was a no-op (NO_UPDATE / AMBIGUOUS /
    END), or if the batch failed atomic validation — in which case the
    caller should fall back to the deterministic rule-based extractor
    instead of silently dropping the user's information.
    """
    try:
        manager = StateManager(state)
        manager.apply_extraction(
            result,
            previous_assistant_message=previous_assistant_message,
            previous_user_message=previous_user_message,
            _timing=_timing,
        )
    except StateBatchValidationError:
        # Batch rejected atomically — ProjectState is untouched. Signal the
        # caller so the deterministic fallback can still capture the facts.
        print("[StateValidator] rejected extraction batch; deterministic fallback will run")
        return False

    if result.message_type == MessageType.MEANINGFUL and result.updates:
        for u in result.updates:
            print(
                f"[StateManager] {u.operation.value} {u.field.value} = {u.value!r}"
            )
        return True

    return False
