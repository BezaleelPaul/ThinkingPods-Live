"""
conversation_brief.py — transient per-turn Conversation Brief (Phase 1 dynamic coach).

Architecture role
-----------------
``process_mentor_turn`` already computes every signal needed to describe
*what just happened this turn* (context mode, recovery category, validated
state diff, objective + advancement, coaching strategy, conversational move,
lifecycle decision, memory threads). This module *composes* those existing
deterministic signals into one small transient structure that tells the
response model what it needs to know — without prescribing the exact reply.

The Brief is:

* transient — built fresh each turn, never persisted to ``SessionData``,
  never a replacement for ``ProjectState``;
* advisory — the response model may choose any conversational move inside
  ``allowed_moves``; nothing in the Brief mutates state, completes an
  objective, or changes the lifecycle;
* deterministic — pure function of its inputs, no LLM, no I/O.

What the Brief does NOT do (by design):

* no new phrase dictionaries — ``turn_type`` reuses ``ContextMode``,
  ``correction``/``uncertainty`` reuse the recovery report;
* no ``if user_is_X`` response scripting — ``allowed_moves`` names a safe
  set (existing ``ConversationMove`` values); the model chooses;
* no truth claims — ``new_facts`` mirrors the *validated* state diff only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "ConversationBrief",
    "build_brief",
    "allowed_moves_for",
    "brief_diagnostics_section",
]


# ---------------------------------------------------------------------------
# Turn-type vocabulary — mirrors ContextMode, rendered in model-friendly form
# ---------------------------------------------------------------------------

TURN_NORMAL = "normal"
TURN_CORRECTION = "correction"
TURN_TOPIC_SHIFT = "topic_shift"
TURN_DIRECT_QUESTION = "direct_question"
TURN_CONFUSED = "confused"
TURN_HYPOTHETICAL = "hypothetical"
TURN_AMBIGUOUS = "ambiguous"

_MODE_TO_TURN_TYPE: Dict[str, str] = {
    "NORMAL_DT": TURN_NORMAL,
    "CORRECTION": TURN_CORRECTION,
    "TOPIC_SHIFT": TURN_TOPIC_SHIFT,
    "DIRECT_QUESTION": TURN_DIRECT_QUESTION,
    "CONFUSED": TURN_CONFUSED,
    "HYPOTHETICAL": TURN_HYPOTHETICAL,
    "AMBIGUOUS": TURN_AMBIGUOUS,
    "UNSAFE": TURN_AMBIGUOUS,  # unreachable: unsafe turns intercept earlier
}


# ---------------------------------------------------------------------------
# Open-need vocabulary — what this turn most needs from the reply
# ---------------------------------------------------------------------------

NEED_DEEPEN = "acknowledge_and_deepen"
NEED_CLARIFY = "clarify"
NEED_REPAIR = "repair"
NEED_CONFIRM_SUMMARY = "confirm_summary"
NEED_SIDE_QUESTION = "answer_side_question"


# ---------------------------------------------------------------------------
# Allowed-move sets — existing ConversationMove values, situation-selected.
# The application defines the safe space; the model chooses within it.
# ---------------------------------------------------------------------------

_MOVE = {
    "ELICIT_INFORMATION": "ELICIT_INFORMATION",
    "EXPAND_IDEA": "EXPAND_IDEA",
    "CONNECT_INFORMATION": "CONNECT_INFORMATION",
    "VALIDATE_DISCOVERY": "VALIDATE_DISCOVERY",
    "CHALLENGE_ASSUMPTION": "CHALLENGE_ASSUMPTION",
    "SUMMARIZE_PROGRESS": "SUMMARIZE_PROGRESS",
    "TRANSITION_TOPIC": "TRANSITION_TOPIC",
    "CLOSE_TOPIC": "CLOSE_TOPIC",
}

# Moves whose realization may legitimately be a pure statement (no question).
# ELICIT_INFORMATION / EXPAND_IDEA / CONNECT_INFORMATION / TRANSITION_TOPIC
# all end in a question by construction, so they never license statements.
STATEMENT_LICENSED_MOVES = frozenset({
    "VALIDATE_DISCOVERY",
    "CHALLENGE_ASSUMPTION",
    "SUMMARIZE_PROGRESS",
    "CLOSE_TOPIC",
})

_NORMAL_BY_COACHING: Dict[str, List[str]] = {
    "DEEPEN": ["EXPAND_IDEA", "CONNECT_INFORMATION", "ELICIT_INFORMATION"],
    "VALIDATE": ["VALIDATE_DISCOVERY", "ELICIT_INFORMATION"],
    "CLARIFY": ["ELICIT_INFORMATION"],
    "EXPLORE": ["ELICIT_INFORMATION", "EXPAND_IDEA"],
    "TRANSITION": ["TRANSITION_TOPIC", "CONNECT_INFORMATION"],
    "SUMMARIZE_PROGRESS": ["SUMMARIZE_PROGRESS", "ELICIT_INFORMATION"],
}

_DEFAULT_NORMAL_MOVES = [
    "VALIDATE_DISCOVERY",
    "CONNECT_INFORMATION",
    "EXPAND_IDEA",
    "CHALLENGE_ASSUMPTION",
    "ELICIT_INFORMATION",
]


def allowed_moves_for(
    turn_type: str,
    coaching: Optional[str] = None,
    lifecycle: Optional[str] = None,
) -> List[str]:
    """Select the safe conversational-move set for this turn.

    Lifecycle overrides everything (summary states have fixed jobs);
    otherwise the turn type selects, with the coaching strategy refining
    normal exploration turns. Returns existing ``ConversationMove`` value
    strings — no new taxonomy.
    """
    lifecycle = (lifecycle or "").upper()
    if lifecycle == "READY_FOR_SUMMARY":
        return ["SUMMARIZE_PROGRESS"]
    if lifecycle == "WAITING_FOR_CONFIRMATION":
        return ["VALIDATE_DISCOVERY", "CLOSE_TOPIC"]
    if lifecycle == "READY_FOR_TRANSITION":
        return ["TRANSITION_TOPIC", "CLOSE_TOPIC"]

    if turn_type == TURN_CONFUSED:
        return ["ELICIT_INFORMATION", "VALIDATE_DISCOVERY"]
    if turn_type == TURN_CORRECTION:
        return [
            "VALIDATE_DISCOVERY",
            "CONNECT_INFORMATION",
            "CHALLENGE_ASSUMPTION",
            "ELICIT_INFORMATION",
        ]
    if turn_type == TURN_TOPIC_SHIFT:
        return ["TRANSITION_TOPIC", "ELICIT_INFORMATION"]
    if turn_type == TURN_DIRECT_QUESTION:
        return ["VALIDATE_DISCOVERY", "ELICIT_INFORMATION", "TRANSITION_TOPIC"]
    if turn_type == TURN_HYPOTHETICAL:
        return ["ELICIT_INFORMATION", "VALIDATE_DISCOVERY"]
    if turn_type == TURN_AMBIGUOUS:
        return ["ELICIT_INFORMATION"]
    # Normal exploration: let the coaching strategy narrow the set.
    coaching_key = (coaching or "").upper()
    if coaching_key in _NORMAL_BY_COACHING:
        return list(_NORMAL_BY_COACHING[coaching_key])
    return list(_DEFAULT_NORMAL_MOVES)


# ---------------------------------------------------------------------------
# State-diff facts (validated only — mirrors the applied state change)
# ---------------------------------------------------------------------------

_LIST_KEYS = (
    "personas",
    "problems",
    "current_solutions",
    "pain_points",
    "evidence",
    "impacts",
)

_FIELD_LABELS: Dict[str, str] = {
    "personas": "Personas",
    "problems": "Problems",
    "current_solutions": "Current Solutions",
    "pain_points": "Pain Points",
    "evidence": "Evidence",
    "impacts": "Impacts",
    "frequency": "Frequency",
}


def field_label(field_value: str) -> str:
    """Human-readable label for a StateField value (fallback: raw value)."""
    return _FIELD_LABELS.get(field_value or "", field_value or "")


def diff_facts(
    before: Optional[dict],
    after: Optional[dict],
    max_facts: int = 6,
    max_per_field: int = 3,
) -> List[dict]:
    """Project the validated state diff into ``[{field, value}]`` facts.

    List fields contribute newly added items; the ``frequency`` scalar
    contributes on change. Read-only; capped so the Brief stays small.
    """
    before = before or {}
    after = after or {}
    facts: List[dict] = []
    for key in _LIST_KEYS:
        b_items = [i for i in (before.get(key) or []) if i]
        new_items = [i for i in (after.get(key) or []) if i and i not in b_items]
        for item in new_items[:max_per_field]:
            facts.append({"field": key, "value": item})
            if len(facts) >= max_facts:
                return facts
    bv = str(before.get("frequency") or "").strip()
    av = str(after.get("frequency") or "").strip()
    if av and av != bv and len(facts) < max_facts:
        facts.append({"field": "frequency", "value": av})
    return facts


# ---------------------------------------------------------------------------
# The Brief
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversationBrief:
    """Transient advisory record describing this turn for the response model.

    ``allowed_moves`` holds ``ConversationMove`` value strings; the model
    chooses within the set. ``statement_allowed`` records whether a
    question-less reply is acceptable this turn (soft default: questions
    remain the normal Design Thinking mechanism).
    """

    turn_type: str = TURN_NORMAL
    new_facts: List[dict] = field(default_factory=list)
    correction: Optional[dict] = None
    uncertainty: bool = False
    open_need: str = NEED_DEEPEN
    allowed_moves: List[str] = field(default_factory=list)
    current_objective: str = ""
    why_now: str = ""
    acknowledge_first: bool = False
    statement_allowed: bool = False
    fresh_family: Optional[str] = None
    asked_families: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "turn_type": self.turn_type,
            "new_facts": [dict(f) for f in self.new_facts],
            "correction": dict(self.correction) if self.correction else None,
            "uncertainty": self.uncertainty,
            "open_need": self.open_need,
            "allowed_moves": list(self.allowed_moves),
            "current_objective": self.current_objective,
            "why_now": self.why_now,
            "acknowledge_first": self.acknowledge_first,
            "statement_allowed": self.statement_allowed,
            "fresh_family": self.fresh_family,
            "asked_families": list(self.asked_families),
        }


def _mode_value(mode) -> str:
    return getattr(mode, "value", mode or "")


def build_brief(
    *,
    context_mode=None,
    context_confidence=None,
    acknowledge_first: bool = False,
    recovery: Optional[dict] = None,
    state_before: Optional[dict] = None,
    state_after: Optional[dict] = None,
    objective=None,
    coaching: Optional[str] = None,
    lifecycle: Optional[str] = None,
    fresh_family: Optional[str] = None,
    asked_families: Optional[list] = None,
) -> ConversationBrief:
    """Assemble the transient Brief from existing deterministic signals.

    All inputs are read-only projections the pipeline already computed.
    No LLM, no I/O, no mutation — same inputs always yield the same Brief.
    """
    turn_type = _MODE_TO_TURN_TYPE.get(_mode_value(context_mode), TURN_NORMAL)

    recovery = recovery or {}
    recovery_category = (recovery.get("Category") or "NONE").upper()
    affected = list(recovery.get("Affected Fields") or [])
    clarification = (
        recovery.get("Clarification Reason")
        or recovery.get("Detected Inconsistency")
        or ""
    )

    correction = None
    if turn_type == TURN_CORRECTION or recovery_category == "CONTRADICTION":
        correction = {
            "fields": affected,
            "note": clarification,
        }

    uncertainty = recovery_category in ("DONT_KNOW", "UNCERTAIN_ANSWER") or (
        turn_type == TURN_CONFUSED
    )

    new_facts = diff_facts(state_before, state_after)

    lifecycle_key = (lifecycle or "").upper()
    if lifecycle_key in (
        "READY_FOR_SUMMARY",
        "WAITING_FOR_CONFIRMATION",
        "READY_FOR_TRANSITION",
    ):
        open_need = NEED_CONFIRM_SUMMARY
    elif turn_type == TURN_DIRECT_QUESTION:
        open_need = NEED_SIDE_QUESTION
    elif turn_type == TURN_CONFUSED:
        open_need = NEED_CLARIFY
    elif turn_type in (TURN_CORRECTION, TURN_TOPIC_SHIFT):
        open_need = NEED_REPAIR
    elif (coaching or "").upper() == "CLARIFY":
        open_need = NEED_CLARIFY
    else:
        open_need = NEED_DEEPEN

    allowed = allowed_moves_for(
        turn_type, coaching=coaching, lifecycle=lifecycle_key or None
    )

    current_objective = ""
    why_now = ""
    if objective is not None:
        current_objective = getattr(
            getattr(objective, "objective", None), "value", ""
        ) or ""
        why_now = (
            getattr(objective, "advancement_reason", "")
            or (list(getattr(objective, "reasoning", []) or [])[:1] or [""])[0]
        )

    # Statement license: a question remains the normal Design Thinking
    # mechanism (normal exploration turns still require one via the output
    # guard). Statements are licensed only where they are conversationally
    # natural: acknowledging a correction, answering a side question,
    # repairing confusion, or reorienting after a topic shift.
    statement_allowed = turn_type in (
        TURN_CORRECTION,
        TURN_DIRECT_QUESTION,
        TURN_CONFUSED,
        TURN_TOPIC_SHIFT,
    ) or open_need in (NEED_REPAIR, NEED_SIDE_QUESTION)

    return ConversationBrief(
        turn_type=turn_type,
        new_facts=new_facts,
        correction=correction,
        uncertainty=uncertainty,
        open_need=open_need,
        allowed_moves=allowed,
        current_objective=current_objective,
        why_now=why_now,
        acknowledge_first=bool(acknowledge_first),
        statement_allowed=bool(statement_allowed),
        fresh_family=fresh_family,
        asked_families=list(asked_families or []),
    )


def brief_diagnostics_section(capture: Optional[dict]) -> dict:
    """Developer Console projection of this turn's transient Brief."""
    if not capture:
        return {}
    return dict(capture.get("conversation_brief") or {})
