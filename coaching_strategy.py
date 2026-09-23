"""
coaching_strategy.py — Adaptive Coaching Strategy layer.

Before generating a reply, the mentor chooses a conversational coaching
move separate from the information-gathering objective.

The existing `ObjectiveEngine` (Module 3) decides WHAT information is
needed.  The `ResponseStrategyEngine` (Module 4) decides the structural
act (ask, summarise, clarify, acknowledge).  The Coaching Strategy
decides HOW to continue the conversation in a human-coach sense.

Purely deterministic.  NEVER changes objectives, lifecycle, extraction,
or ``ProjectState``.  The strategy is guidance only — an additional
bullet in the LLM prompt instructions.

Values
------
``EXPLORE``
    Begin a new objective with an open-ended question.
``CLARIFY``
    The user's answer was vague / incomplete / ambiguous.
``DEEPEN``
    The answer is useful but could reveal richer insight.
``VALIDATE``
    The user made an important observation that should be
    acknowledged before proceeding.
``SUMMARIZE_PROGRESS``
    Several useful answers have been gathered across objectives.
    Briefly recap before the next question.
``TRANSITION``
    The mentor is finishing one objective and starting another.
    Bridge naturally.

Architecture note
-----------------
The Coaching Strategy's only consumer is the Prompt Builder (Module 4),
which appends a ``Current coaching strategy: <name>`` / ``Guidance:
<bullet>`` instruction block.  No other component ever reads it.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from memory_extractor import LIST_FIELDS, StateField

__all__ = [
    "CoachingStrategy",
    "CoachingGuidance",
    "determine_coaching_strategy",
]


class CoachingStrategy(str, Enum):
    EXPLORE = "EXPLORE"
    CLARIFY = "CLARIFY"
    DEEPEN = "DEEPEN"
    VALIDATE = "VALIDATE"
    SUMMARIZE_PROGRESS = "SUMMARIZE_PROGRESS"
    TRANSITION = "TRANSITION"


# ---------------------------------------------------------------------------
# Guidance strings rendered as instruction bullets
# ---------------------------------------------------------------------------

_GUIDANCE: dict[CoachingStrategy, str] = {
    CoachingStrategy.EXPLORE: (
        "Use this turn to begin a fresh topic. Ask an open-ended question "
        "that invites the user to share broadly about the new objective."
    ),
    CoachingStrategy.CLARIFY: (
        "The user's last answer was incomplete, vague, or ambiguous. "
        "Ask a specific clarifying question that zooms in on the unclear "
        "part — reference their own words to show you were listening."
    ),
    CoachingStrategy.DEEPEN: (
        "The user's last answer contains a useful insight. Build on it "
        "with one deeper follow-up question that explores the "
        "motivations, context, or implications beneath the surface."
    ),
    CoachingStrategy.VALIDATE: (
        "Briefly acknowledge the user's observation before asking the "
        "next question — a short warm remark shows you value what they "
        "shared. Then continue with the next question."
    ),
    CoachingStrategy.SUMMARIZE_PROGRESS: (
        "The conversation has gathered several useful discoveries. "
        "Briefly recap what you understand so far in one or two "
        "sentences, then ask the next question so the user feels the "
        "mentor is tracking the overall picture."
    ),
    CoachingStrategy.TRANSITION: (
        "You have enough information about the previous objective. "
        "Smoothly connect it to the next topic so the conversation "
        "feels continuous — reference the last user answer as a "
        "bridge, then lead into the new objective."
    ),
}


def guidance_for(strategy: Optional[CoachingStrategy]) -> Optional[str]:
    """Human-readable guidance bullet for the given strategy."""
    if strategy is None:
        return None
    return _GUIDANCE.get(strategy)


# ---------------------------------------------------------------------------
# Change detection helper (mirrors mentor_decision_audit._changed_fields)
# ---------------------------------------------------------------------------

_STATE_FIELD_KEYS = tuple(sf.value for sf in StateField)
_LIST_KEYS = frozenset(sf.value for sf in LIST_FIELDS) if hasattr(StateField, 'PERSONAS') else frozenset()


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


# ---------------------------------------------------------------------------
# Helper: count how many required fields have a sufficient answer
# ---------------------------------------------------------------------------

def _required_satisfied_count(state: dict) -> int:
    """Number of required Empathize fields that hold a usable answer."""
    from memory_extractor import REQUIRED_FIELDS
    from module3 import SufficiencyChecker
    try:
        sufficiency = SufficiencyChecker.evaluate_dict(state)
        return sum(
            1 for sf in REQUIRED_FIELDS
            if sufficiency.is_satisfied(sf)
        )
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Helper: detect the user's message quality
# ---------------------------------------------------------------------------

def _user_message_quality(
    user_message: str,
    changed_fields: set[str],
    recovery_category: Optional[str],
    objective_advancement: Optional[str],
) -> str:
    """Classify the user's message for coaching purposes."""
    if recovery_category in ("DONT_KNOW", "UNCERTAIN_ANSWER"):
        return "vague"
    if recovery_category == "CONTRADICTION":
        return "contradiction"
    if recovery_category == "TOPIC_CHANGE":
        return "topic_change"
    if not changed_fields:
        if not user_message or len(user_message.strip()) < 10:
            return "vague"
        return "no_update"
    if objective_advancement == "ADVANCED":
        return "advanced"
    if len(changed_fields) >= 2:
        return "rich"
    return "useful"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def determine_coaching_strategy(
    *,
    objective: str,
    objective_advancement: Optional[str] = None,
    objective_confidence: Optional[float] = None,
    state: Optional[dict] = None,
    state_before: Optional[dict] = None,
    state_after: Optional[dict] = None,
    changed_fields: Optional[set[str]] = None,
    user_message: str = "",
    previous_assistant_reply: str = "",
    recovery_category: Optional[str] = None,
    memory_open_threads: Optional[list] = None,
    memory_deferred_topics: Optional[list] = None,
    memory_resolved_threads: Optional[list] = None,
    objective_completed_fields: Optional[list] = None,
    objective_missing_fields: Optional[list] = None,
) -> tuple[CoachingStrategy, str]:
    """Determine the best coaching move for the next turn.

    Parameters
    ----------
    objective:
        The current objective value, e.g. ``"PERSONAS"``.
    objective_advancement:
        ``"ADVANCED"``, ``"STAYED_ACTIVE"``, or ``None``.
    objective_confidence:
        Confidence of the objective selection, e.g. ``1.0``.
    state:
        Current ``ProjectState`` as a dict.  When provided alongside
        ``state_before`` / ``state_after``, the explicit diff is
        preferred; ``state`` is used as a fallback.
    state_before:
        Pre-extraction state snapshot (dict).
    state_after:
        Post-extraction state snapshot (dict).
    changed_fields:
        Set of ``StateField`` values that changed this turn.
        Computed automatically from ``state_before`` / ``state_after``
        when not provided.
    user_message:
        The user's latest raw text.
    previous_assistant_reply:
        The assistant's previous reply.
    recovery_category:
        Recovery analysis category, e.g. ``"DONT_KNOW"``, ``"NONE"``,
        ``"TOPIC_CHANGE"``.
    memory_open_threads:
        List of open-thread records from ``ConversationMemory``.
    memory_deferred_topics:
        List of deferred-topic records from ``ConversationMemory``.
    memory_resolved_threads:
        List of resolved-thread records from ``ConversationMemory``.
    objective_completed_fields:
        List of completed fields for the current objective.
    objective_missing_fields:
        List of missing fields for the current objective.

    Returns
    -------
    tuple[CoachingStrategy, str]
        (strategy, reason_summary) — e.g.
        ``(CoachingStrategy.DEEPEN, "Rich answer; deeper exploration warranted")``.
    """
    # Parse inputs
    state_curr = state or {}
    state_b = state_before or state_curr
    state_a = state_after or state_curr

    if changed_fields is None:
        changed = _detect_changed_fields(state_b, state_a)
    else:
        changed = set(changed_fields or set())

    # Extract field names from memory records (records are dicts with "field" keys)
    def _field_names(items):
        if not items:
            return set()
        return {r.get("field") for r in items if isinstance(r, dict) and r.get("field")}

    resolved = _field_names(memory_resolved_threads)
    open_thread_fields = _field_names(memory_open_threads)
    deferred_set = _field_names(memory_deferred_topics)
    completed = set(objective_completed_fields or [])
    missing = set(objective_missing_fields or [])
    rec_cat = recovery_category or "NONE"

    # --- Rule 1: Summarise after WRAP_UP or every few completed fields ---
    if objective == "WRAP_UP":
        return CoachingStrategy.SUMMARIZE_PROGRESS, "WRAP_UP objective requires a summary."

    satisfied_count = _required_satisfied_count(state_a)
    if satisfied_count >= 3:
        return CoachingStrategy.SUMMARIZE_PROGRESS, (
            f"Several objectives complete ({satisfied_count}/6); recap warranted."
        )

    # --- Rule 2: CLARIFY for vague / contradictory / DONT_KNOW answers ---
    if rec_cat in ("DONT_KNOW", "UNCERTAIN_ANSWER"):
        return CoachingStrategy.CLARIFY, (
            "User could not answer or gave very little detail; needs clarification."
        )
    if not user_message or len(user_message.strip()) < 10:
        return CoachingStrategy.CLARIFY, (
            "User message is very short; needs clarification."
        )

    # --- Rule 3: TRANSITION when the objective advanced to a new field ---
    # ADVANCED means the user's answer satisfied the target field and the
    # objective moved on.  This is a bridge moment, not a deep-dive.
    if objective_advancement == "ADVANCED":
        return CoachingStrategy.TRANSITION, (
            "Objective advanced past the answered field; bridge to the next topic."
        )

    # --- Rule 4: VALIDATE for contradictions and topic changes ---
    if rec_cat == "CONTRADICTION":
        return CoachingStrategy.VALIDATE, (
            "User contradicted themselves; acknowledge the shift before proceeding."
        )
    if rec_cat == "TOPIC_CHANGE":
        return CoachingStrategy.VALIDATE, (
            "User changed topic; acknowledge the new direction before continuing."
        )

    # --- Rule 5: DEEPEN when the user gave rich / useful content ---
    # Detect richness without using ADVANCED advancement (which is now a TRANSITION).
    multiple_changed = len(changed) >= 2
    user_message_long = len(user_message.strip()) >= 30
    if (multiple_changed and user_message_long) or (user_message_long and changed):
        return CoachingStrategy.DEEPEN, (
            "The user shared substantive information; deeper exploration warranted."
        )
    if changed:
        return CoachingStrategy.DEEPEN, (
            f"New information received ({', '.join(sorted(changed))}); "
            "build on it with a deeper question."
        )

    # --- Rule 6: Default EXPLORE for a fresh objective ---
    return CoachingStrategy.EXPLORE, (
        f"Beginning objective '{objective}'; open exploration."
    )


# ---------------------------------------------------------------------------
# Prompt rendering — builds coaching instruction bullets
# ---------------------------------------------------------------------------


def coaching_instruction_bullet(
    strategy: Optional[CoachingStrategy],
) -> Optional[str]:
    """Single instruction bullet describing the current coaching strategy.

    Rendered as::

        Current coaching strategy: DEEPEN
        Guidance: The user's last answer contains a useful insight. ...
    """
    if strategy is None:
        return None
    guidance = _GUIDANCE.get(strategy)
    if guidance is None:
        return None
    return (
        f"Current coaching strategy: {strategy.value}\n"
        f"Guidance: {guidance}"
    )


def coaching_diagnostics_section(capture: dict) -> dict:
    """Developer Console projection of the coaching strategy decision."""
    strategy = capture.get("coaching_strategy")
    if strategy is None:
        return {}
    reason = capture.get("coaching_reason", "")
    return {
        "Selected Strategy": strategy.value if hasattr(strategy, "value") else str(strategy),
        "Reason": reason,
        "Supporting Signals": (
            capture.get("coaching_signals")
            or capture.get("coaching_reason")
            or ""
        ),
    }