"""
conversation_move.py — Conversational Move Planner.

Determines the *primary conversational intention* of the next mentor
reply, independent of the information-gathering objective.

The `ObjectiveEngine` (Module 3) decides WHAT information is needed.
The `ResponseStrategyEngine` (Module 4) decides the structural act.
The `CoachingStrategy` decides HOW to continue in a human-coach sense.
The `Insight Detection` layer recognises meaningful user insights.

The Conversational Move Planner synthesises those decisions into a single
primary intention for the reply — the *move* the mentor makes next:
elicit, expand, connect, validate, challenge, summarise, transition, or
close.

Purely deterministic.  NEVER changes objectives, lifecycle, extraction,
or ``ProjectState``.  The move is guidance only — an additional bullet in
the LLM prompt instructions.  Coaching Strategy and Insight Detection are
consumed read-only and never modified.

Values
------
``ELICIT_INFORMATION``
    The main job is to gather new information with a focused question.
``EXPAND_IDEA``
    The user surfaced an idea worth exploring in depth.
``CONNECT_INFORMATION``
    Link the user's latest observation with earlier discoveries.
``VALIDATE_DISCOVERY``
    Acknowledge and validate a meaningful discovery before continuing.
``CHALLENGE_ASSUMPTION``
    Gently examine an assumption the user's statement rests on.
``SUMMARIZE_PROGRESS``
    Recap the key discoveries gathered so far.
``TRANSITION_TOPIC``
    Bridge from a satisfied objective to a different next objective.
``CLOSE_TOPIC``
    Gracefully conclude the current topic.

Architecture note
-----------------
The planner's only consumer is the Prompt Builder (Module 4), which
appends a ``Conversation move: <name>`` / ``Guidance: <bullet>``
instruction block.  No other component ever reads it.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from memory_extractor import LIST_FIELDS, StateField

__all__ = [
    "ConversationMove",
    "determine_conversation_move",
]

# Objective values that denote the terminal empathize phase.
_WRAP_UP_OBJECTIVES = frozenset({"WRAP_UP"})

# Empathize list fields used to count "related facts" in ProjectState.
_RELATED_FACT_FIELDS = tuple(sf.value for sf in StateField)
_LIST_KEYS = frozenset(sf.value for sf in LIST_FIELDS) if hasattr(StateField, 'PERSONAS') else frozenset()


class ConversationMove(str, Enum):
    ELICIT_INFORMATION = "ELICIT_INFORMATION"
    EXPAND_IDEA = "EXPAND_IDEA"
    CONNECT_INFORMATION = "CONNECT_INFORMATION"
    VALIDATE_DISCOVERY = "VALIDATE_DISCOVERY"
    CHALLENGE_ASSUMPTION = "CHALLENGE_ASSUMPTION"
    SUMMARIZE_PROGRESS = "SUMMARIZE_PROGRESS"
    TRANSITION_TOPIC = "TRANSITION_TOPIC"
    CLOSE_TOPIC = "CLOSE_TOPIC"


# ---------------------------------------------------------------------------
# Guidance strings rendered as instruction bullets
# ---------------------------------------------------------------------------

_MOVE_GUIDANCE: dict[ConversationMove, str] = {
    ConversationMove.ELICIT_INFORMATION: (
        "The primary goal of this turn is to gather new information. "
        "Ask a focused question that fills the most important gap."
    ),
    ConversationMove.EXPAND_IDEA: (
        "The user has surfaced an idea worth exploring. Ask one follow-up "
        "that develops the idea — its context, causes, or implications."
    ),
    ConversationMove.CONNECT_INFORMATION: (
        "Help the user connect their latest observation with earlier "
        "discoveries before asking the next question."
    ),
    ConversationMove.VALIDATE_DISCOVERY: (
        "The user has made a meaningful discovery. Acknowledge and "
        "validate it with a brief warm remark before continuing."
    ),
    ConversationMove.CHALLENGE_ASSUMPTION: (
        "The user's statement may rest on an assumption worth examining. "
        "Gently challenge it with a respectful question."
    ),
    ConversationMove.SUMMARIZE_PROGRESS: (
        "Recap the key discoveries gathered so far in one or two "
        "sentences, then keep the conversation moving."
    ),
    ConversationMove.TRANSITION_TOPIC: (
        "Bridge naturally from the completed objective to the next topic, "
        "referencing the last user answer as a connecting point."
    ),
    ConversationMove.CLOSE_TOPIC: (
        "Wrap up the current topic gracefully. Invite the user to add "
        "anything they feel is still missing before moving on."
    ),
}


def guidance_for_move(move: Optional[ConversationMove]) -> Optional[str]:
    """Human-readable guidance bullet for the given move."""
    if move is None:
        return None
    return _MOVE_GUIDANCE.get(move)


# ---------------------------------------------------------------------------
# Change detection helper (mirrors coaching_strategy._detect_changed_fields)
# ---------------------------------------------------------------------------

_STATE_FIELD_KEYS = tuple(sf.value for sf in StateField)


def _detect_changed_fields(before: dict, after: dict) -> set:
    """StateField values whose content changed between snapshots."""
    changed: set = set()
    for key in _STATE_FIELD_KEYS:
        b = before.get(key)
        a = after.get(key)
        if key in _LIST_KEYS:
            new_items = [i for i in (a or []) if i not in (b or [])]
            if new_items:
                changed.add(key)
        else:
            bv = str(b).strip() if b else ""
            av = str(a).strip() if a else ""
            if av and av != bv:
                changed.add(key)
    return changed


def _field_names(items) -> set:
    if not items:
        return set()
    return {r.get("field") for r in items if isinstance(r, dict) and r.get("field")}


def _populated_fact_count(state: dict) -> int:
    """Number of empathize list fields holding at least one usable fact."""
    count = 0
    for key in _RELATED_FACT_FIELDS:
        val = state.get(key)
        if isinstance(val, list) and val:
            count += 1
        elif isinstance(val, (str, int, float)) and str(val).strip():
            count += 1
    return count


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def determine_conversation_move(
    *,
    current_objective: str = "",
    coaching_strategy: Optional[str] = None,
    insight_type: Optional[str] = None,
    insight_confidence: Optional[str] = None,
    user_message: str = "",
    state: Optional[dict] = None,
    state_before: Optional[dict] = None,
    state_after: Optional[dict] = None,
    changed_fields: Optional[set] = None,
    memory_open_threads: Optional[list] = None,
    memory_deferred_topics: Optional[list] = None,
    memory_resolved_threads: Optional[list] = None,
    memory_partially_answered: Optional[list] = None,
    objective_advancement: Optional[str] = None,
    objective_completed_fields: Optional[list] = None,
    objective_missing_fields: Optional[list] = None,
    recovery_category: Optional[str] = None,
    summary_presented: bool = False,
) -> tuple[ConversationMove, str, list[str]]:
    """Determine the primary conversational intention of the next reply.

    Parameters
    ----------
    current_objective:
        The active objective value, e.g. ``"PERSONAS"`` or ``"WRAP_UP"``.
    coaching_strategy:
        Value of the :class:`coaching_strategy.CoachingStrategy` chosen
        for this turn (read-only), e.g. ``"DEEPEN"``.
    insight_type:
        Value of the :class:`insight_detection.InsightType` detected for
        this turn (read-only), e.g. ``"ROOT_CAUSE_HINT"``, ``"NONE"``.
    insight_confidence:
        Value of the insight confidence (``"LOW"`` / ``"MEDIUM"`` /
        ``"HIGH"``), read-only.
    user_message:
        The user's latest raw text.
    state:
        Current ``ProjectState`` as a dict.  When provided alongside
        ``state_before`` / ``state_after`` the explicit diff is preferred.
    state_before:
        Pre-extraction state snapshot (dict).
    state_after:
        Post-extraction state snapshot (dict).
    changed_fields:
        Set of ``StateField`` values that changed this turn.  Computed
        automatically from ``state_before`` / ``state_after`` when absent.
    memory_open_threads:
        List of open-thread records from ``ConversationMemory``.
    memory_deferred_topics:
        List of deferred-topic records from ``ConversationMemory``.
    memory_resolved_threads:
        List of resolved-thread records from ``ConversationMemory``.
    memory_partially_answered:
        List of partially-answered-objective records from memory.
    objective_advancement:
        ``"ADVANCED"``, ``"STAYED_ACTIVE"``, or ``None``.
    objective_completed_fields:
        List of completed fields for the current objective.
    objective_missing_fields:
        List of missing fields for the current objective.
    recovery_category:
        Recovery analysis category, e.g. ``"CONTRADICTION"``.
    summary_presented:
        Whether the empathize summary has already been presented to the
        user this session.

    Returns
    -------
    tuple[ConversationMove, str, list[str]]
        ``(move, reason, supporting_signals)`` — e.g.
        ``(ConversationMove.EXPAND_IDEA, "Deepen a root-cause hint.",
        ["coaching=DEEPEN", "insight=ROOT_CAUSE_HINT"])``.
    """
    signals: list[str] = []

    state_curr = state or {}
    state_b = state_before or state_curr
    state_a = state_after or state_curr

    if changed_fields is None:
        changed = _detect_changed_fields(state_b, state_a)
    else:
        changed = set(changed_fields or set())

    resolved_fields = _field_names(memory_resolved_threads)
    open_fields = _field_names(memory_open_threads)
    deferred_fields = _field_names(memory_deferred_topics)
    missing = set(objective_missing_fields or [])
    completed = set(objective_completed_fields or [])

    insight = (insight_type or "NONE").upper()
    coaching = (coaching_strategy or "").upper()
    advancement = (objective_advancement or "").upper()

    # --- Rule 1: Terminal WRAP_UP → summarise, then close once presented ---
    if current_objective.upper() in _WRAP_UP_OBJECTIVES:
        if summary_presented:
            signals.append("objective=WRAP_UP")
            signals.append("summary_already_presented")
            return (
                ConversationMove.CLOSE_TOPIC,
                "Empathize summary already presented; close the topic.",
                signals,
            )
        signals.append("objective=WRAP_UP")
        return (
            ConversationMove.SUMMARIZE_PROGRESS,
            "WRAP_UP objective requires a progress recap.",
            signals,
        )

    # --- Rule 2: Insight-driven moves (most specific signals) ---
    if insight == "ROOT_CAUSE_HINT" and coaching == "DEEPEN":
        signals.append("coaching=DEEPEN")
        signals.append("insight=ROOT_CAUSE_HINT")
        return (
            ConversationMove.EXPAND_IDEA,
            "User offered a root-cause hint; expand it into the underlying mechanism.",
            signals,
        )
    if insight == "STRONG_EVIDENCE":
        signals.append("insight=STRONG_EVIDENCE")
        return (
            ConversationMove.VALIDATE_DISCOVERY,
            "User provided strong evidence; validate the discovery.",
            signals,
        )
    if insight == "CONTRADICTION":
        signals.append("insight=CONTRADICTION")
        return (
            ConversationMove.CHALLENGE_ASSUMPTION,
            "User's statement conflicts with earlier information; examine the assumption.",
            signals,
        )
    if insight == "USER_LEARNING":
        signals.append("insight=USER_LEARNING")
        return (
            ConversationMove.VALIDATE_DISCOVERY,
            "User expressed a realization; validate the learning.",
            signals,
        )
    if insight == "SURPRISING_OBSERVATION":
        signals.append("insight=SURPRISING_OBSERVATION")
        return (
            ConversationMove.EXPAND_IDEA,
            "User shared a surprising observation; expand on why it matters.",
            signals,
        )
    if insight == "NEW_PATTERN":
        signals.append("insight=NEW_PATTERN")
        return (
            ConversationMove.CONNECT_INFORMATION,
            "User connected ideas; help them see the broader pattern.",
            signals,
        )

    # --- Rule 3: Objective satisfied and next objective differs → transition ---
    transition_wanted = advancement == "ADVANCED" or coaching == "TRANSITION"
    objective_satisfied = not missing and bool(completed or advancement)
    if transition_wanted or objective_satisfied:
        signals.append(f"advancement={advancement or 'none'}")
        if objective_satisfied:
            signals.append("objective_satisfied")
        else:
            signals.append("next_objective_differs")
        return (
            ConversationMove.TRANSITION_TOPIC,
            "Objective satisfied; bridge to the next topic.",
            signals,
        )

    # --- Rule 4: Multiple related facts now exist → connect ---
    fact_count = _populated_fact_count(state_a)
    multiple_facts = len(changed) >= 2 or fact_count >= 3 or len(resolved_fields) >= 2
    if multiple_facts:
        signals.append(f"changed_fields={len(changed)}")
        signals.append(f"populated_fact_fields={fact_count}")
        if resolved_fields:
            signals.append(f"resolved_fields={','.join(sorted(resolved_fields))}")
        return (
            ConversationMove.CONNECT_INFORMATION,
            "Multiple related facts exist; connect them before asking the next question.",
            signals,
        )

    # --- Rule 5: Coaching-strategy fallback ---
    if coaching == "SUMMARIZE_PROGRESS":
        signals.append("coaching=SUMMARIZE_PROGRESS")
        return (
            ConversationMove.SUMMARIZE_PROGRESS,
            "Several objectives complete; recap progress.",
            signals,
        )
    if coaching == "VALIDATE":
        signals.append("coaching=VALIDATE")
        return (
            ConversationMove.VALIDATE_DISCOVERY,
            "Acknowledge the user's observation before continuing.",
            signals,
        )
    if coaching == "DEEPEN":
        signals.append("coaching=DEEPEN")
        return (
            ConversationMove.EXPAND_IDEA,
            "Rich answer; expand the idea with a deeper question.",
            signals,
        )
    if coaching == "CLARIFY" or coaching == "EXPLORE":
        signals.append(f"coaching={coaching}")
        return (
            ConversationMove.ELICIT_INFORMATION,
            "Fresh or unclear territory; elicit the information needed.",
            signals,
        )

    # --- Rule 6: Default ---
    signals.append("default")
    return (
        ConversationMove.ELICIT_INFORMATION,
        "No specific move warranted; gather the next piece of information.",
        signals,
    )


# ---------------------------------------------------------------------------
# Prompt rendering — builds conversation-move instruction bullets
# ---------------------------------------------------------------------------


def conversation_move_instruction_bullet(
    move: Optional[ConversationMove],
) -> Optional[str]:
    """Single instruction bullet describing the planned conversational move.

    Rendered as::

        Conversation move: CONNECT_INFORMATION
        Guidance: Help the user connect their latest observation ...
    """
    if move is None:
        return None
    guidance = _MOVE_GUIDANCE.get(move)
    if guidance is None:
        return None
    return (
        f"Conversation move: {move.value}\n"
        f"Guidance: {guidance}"
    )


def conversation_move_diagnostics_section(capture: dict) -> dict:
    """Developer Console projection of the conversation move decision."""
    move = capture.get("conversation_move")
    if move is None:
        return {}
    reason = capture.get("conversation_move_reason", "")
    signals = capture.get("conversation_move_signals") or []
    return {
        "Move": move.value if hasattr(move, "value") else str(move),
        "Reason": reason,
        "Signals": list(signals),
    }
