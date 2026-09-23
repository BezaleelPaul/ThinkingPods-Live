"""
conversation_memory.py — deterministic conversational-context memory.

Architecture role
-----------------
Observation + state layer for the Empathize v2 pipeline. After each mentor
turn it reconciles the session's conversational memory — *open threads*,
*resolved threads*, *deferred topics*, *acknowledged facts*, *summarized
facts*, and *partially answered objectives* — against the state diff, the
chosen objective, sufficiency, the lifecycle decision, and the recovery
report. 100% deterministic, no LLM, no I/O, no randomness.

The memory is the single conversational-context home on ``SessionData`` and
persists with the session. It NEVER duplicates ``ProjectState`` values: every
record references a ``StateField`` key plus a status/reason, never the
captured value itself. ``ProjectState`` remains the canonical project-
knowledge store; this layer records *what the conversation is doing with it*.

Why this exists
---------------
The mentor must naturally resume unfinished topics instead of restarting
them. The objective engine already knows *which* fields are missing; this
layer records *which topics the user has touched but not finished*, so the
reply path (via :func:`memory_to_prompt_bullets`) can nudge the LLM to build
on partial answers and return to user-raised topics, and so the Developer
Console can visualize thread state.

Record shape
------------
Every record is a JSON-safe dict::

    {"field": <StateField value>, "reason": <str>, "turn": <int>}

* ``open_threads``    — required fields still being pursued (partial answers,
                        the currently-targeted field, or asked-but-unanswered).
* ``resolved_threads``— required fields that reached a usable answer.
* ``deferred_topics`` — fields the user volunteered out of turn while the
                        mentor was pursuing another objective; returned to later.
* ``acknowledged_facts``  — fields the user explicitly confirmed.
* ``summarized_facts``    — fields included in a presented empathy summary.
* ``partially_answered_objectives`` — objectives pursued this turn whose field
                        still lacks a usable answer.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

from memory_extractor import REQUIRED_FIELDS, StateField
from module3 import QuestionFamily, SufficiencyChecker, field_of

__all__ = [
    "ThreadStatus",
    "ConversationMemory",
    "update_conversation_memory",
    "memory_to_prompt_bullets",
    "memory_diagnostics_section",
]


# ---------------------------------------------------------------------------
# ThreadStatus
# ---------------------------------------------------------------------------


class ThreadStatus(str, Enum):
    """Lifecycle of a conversational thread."""

    OPEN = "open"
    """The topic is still being pursued."""
    RESOLVED = "resolved"
    """The topic reached a usable answer."""
    DEFERRED = "deferred"
    """The user raised the topic out of turn; the mentor returns to it later."""


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------


class ConversationMemory:
    """
    Conversational-context memory container.

    Owned by ``SessionData`` (single memory home) and persisted with the
    session. Holds JSON-safe records that reference fields + status/reason
    only — never ``ProjectState`` values.
    """

    def __init__(
        self,
        acknowledged_facts: Optional[List[dict]] = None,
        summarized_facts: Optional[List[dict]] = None,
        open_threads: Optional[List[dict]] = None,
        resolved_threads: Optional[List[dict]] = None,
        deferred_topics: Optional[List[dict]] = None,
        partially_answered_objectives: Optional[List[dict]] = None,
    ) -> None:
        self.acknowledged_facts: List[dict] = list(acknowledged_facts or [])
        self.summarized_facts: List[dict] = list(summarized_facts or [])
        self.open_threads: List[dict] = list(open_threads or [])
        self.resolved_threads: List[dict] = list(resolved_threads or [])
        self.deferred_topics: List[dict] = list(deferred_topics or [])
        self.partially_answered_objectives: List[dict] = list(
            partially_answered_objectives or []
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "acknowledged_facts": self.acknowledged_facts,
            "summarized_facts": self.summarized_facts,
            "open_threads": self.open_threads,
            "resolved_threads": self.resolved_threads,
            "deferred_topics": self.deferred_topics,
            "partially_answered_objectives": self.partially_answered_objectives,
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ConversationMemory":
        d = d or {}
        return cls(
            acknowledged_facts=d.get("acknowledged_facts", []),
            summarized_facts=d.get("summarized_facts", []),
            open_threads=d.get("open_threads", []),
            resolved_threads=d.get("resolved_threads", []),
            deferred_topics=d.get("deferred_topics", []),
            partially_answered_objectives=d.get(
                "partially_answered_objectives", []
            ),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ConversationMemory):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ConversationMemory({self.to_dict()!r})"


# ---------------------------------------------------------------------------
# Deterministic reconciliation helpers
# ---------------------------------------------------------------------------

_FIELD_LABELS: dict[StateField, str] = {
    StateField.PERSONAS: "Personas",
    StateField.PROBLEMS: "Problems",
    StateField.CURRENT_SOLUTIONS: "Current Solutions",
    StateField.PAIN_POINTS: "Pain Points",
    StateField.EVIDENCE: "Evidence",
    StateField.IMPACTS: "Impacts",
    StateField.FREQUENCY: "Frequency",
}


def _field_label(field_value: str) -> str:
    try:
        return _FIELD_LABELS[StateField(field_value)]
    except (ValueError, KeyError):
        return field_value


def _upsert(records: List[dict], field_value: str, reason: str, turn: int, known: bool = False, asked: bool = False) -> None:
    """Add-or-update a record by field key (single record per field)."""
    for rec in records:
        if rec["field"] == field_value:
            rec["reason"] = reason
            rec["turn"] = turn
            rec["known"] = known
            rec["asked"] = asked
            return
    records.append({"field": field_value, "reason": reason, "turn": turn, "known": known, "asked": asked})


def _remove(records: List[dict], field_value: str) -> None:
    records[:] = [r for r in records if r.get("field") != field_value]


def _asked_fields(session_data) -> set[str]:
    """StateField values whose question family has already been asked."""
    asked: set[str] = set()
    for value in getattr(session_data, "asked_question_families", []) or []:
        try:
            field = field_of(QuestionFamily(value))
        except (ValueError, AttributeError):
            field = None
        if field is not None:
            asked.add(field.value)
    return asked


def _record_user_stated_facts(memory: ConversationMemory, field_value: str) -> None:
    """Mark a field as known (user-stated) in the conversation memory.

    Sets ``known=True`` on the record for ``field_value`` across all memory
    lists (open_threads, deferred_topics, etc.).  Does NOT set the field as
    satisfied in ProjectState — it only signals to the mentor that the user
    has already established this fact voluntarily.
    """
    _upsert(memory.open_threads, field_value, "user stated this fact voluntarily; treat as established context", turn=0, known=True, asked=False)
    _upsert(memory.deferred_topics, field_value, "user stated this fact out of turn; revisit after current objective", turn=0, known=True, asked=False)


def _affected_fields(recovery) -> tuple[str, ...]:
    """Normalise a RecoveryReport (or its dict) to the changed field values."""
    if recovery is None:
        return ()
    attr = getattr(recovery, "affected_fields", None)
    if attr is not None and not isinstance(recovery, dict):
        return tuple(attr)
    if isinstance(recovery, dict):
        return tuple(recovery.get("Affected Fields", []) or [])
    return ()


# ---------------------------------------------------------------------------
# update_conversation_memory
# ---------------------------------------------------------------------------


def update_conversation_memory(
    *,
    session_data,
    state_before: dict,
    state_after: dict,
    objective,
    recovery=None,
    lifecycle_decision=None,
) -> ConversationMemory:
    """
    Reconcile the session's conversational memory against one completed turn.

    Deterministic and read-only with respect to ``ProjectState``: it never
    reads or mutates state values, only the field/status references recorded
    in ``SessionData.conversation_memory``. Same inputs -> same resulting
    memory (modulo append order, which is deterministic too).

    Parameters
    ----------
    session_data:
        The ``SessionData`` whose ``conversation_memory`` is reconciled.
    state_before:
        ``ProjectState.to_state_dict()`` snapshot before extraction.
    state_after:
        ``ProjectState.to_state_dict()`` snapshot after extraction.
    objective:
        The :class:`module3.ConversationObjective` selected for this turn.
    recovery:
        Optional :class:`recovery_monitor.RecoveryReport` (or its dict) for
        this turn; supplies the set of fields the user actually touched.
    lifecycle_decision:
        Optional :class:`module5.LifecycleDecision` for this turn; drives
        summary/acknowledgement tracking.

    Returns
    -------
    ConversationMemory
        The reconciled memory (same object as ``session_data.conversation_memory``).
    """
    memory = getattr(session_data, "conversation_memory", None)
    if memory is None:
        memory = ConversationMemory()
        session_data.conversation_memory = memory

    before = SufficiencyChecker.evaluate_dict(state_before or {})
    after = SufficiencyChecker.evaluate_dict(state_after or {})
    turn_index = len(getattr(session_data, "turn_metrics", [])) + 1
    targeted = getattr(objective, "targeted_field", lambda: None)()
    touched = _affected_fields(recovery)

    # 1. Resolved threads — every satisfied required field is no longer open
    #    or deferred, and is recorded as resolved at most once.
    for sf in REQUIRED_FIELDS:
        if not after.is_satisfied(sf):
            continue
        was_satisfied = before.is_satisfied(sf)
        _upsert(
            memory.resolved_threads,
            sf.value,
            (
                "field reached a usable answer this turn"
                if not was_satisfied
                else "field is satisfied"
            ),
            turn_index,
        )
        _remove(memory.open_threads, sf.value)
        _remove(memory.deferred_topics, sf.value)
        _remove(memory.partially_answered_objectives, sf.value)

    asked = _asked_fields(session_data)

    # 2. Open threads — required fields still unsatisfied that the objective
    #    is actively pursuing right now (the current target) or that were asked
    #    about in an earlier turn without a usable answer.
    for sf in REQUIRED_FIELDS:
        if after.is_satisfied(sf):
            continue
        is_target = targeted is not None and sf is targeted
        if is_target:
            _upsert(
                memory.open_threads,
                sf.value,
                "actively pursued this turn; answer not yet sufficient",
                turn_index,
            )
        elif sf.value in asked:
            _upsert(
                memory.open_threads,
                sf.value,
                "asked earlier; still not sufficiently answered",
                turn_index,
            )

    # 3. Deferred topics — fields the user touched this turn that are NOT the
    #    current target and remain unsatisfied: the mentor captures them now
    #    and returns once the current objective is done. (A topic that reached
    #    a usable answer was already moved to resolved above.)
    for field_value in touched:
        try:
            sf = StateField(field_value)
        except ValueError:
            continue
        if after.is_satisfied(sf):
            continue
        if targeted is not None and sf is targeted:
            continue
        _upsert(
            memory.deferred_topics,
            sf.value,
            "user raised this topic out of turn; revisit after the current objective",
            turn_index,
        )

    # 4. Partially answered objectives — a required field whose answer this
    #    turn is only partial (e.g. a frequency with no usable cadence). The
    #    conversation should remember it as incomplete and refine later.
    for sf in REQUIRED_FIELDS:
        if after.level_of(sf).value == "PARTIAL" and sf.value in touched:
            _upsert(
                memory.partially_answered_objectives,
                sf.value,
                "objective pursued this turn; answer only partial",
                turn_index,
            )

    # 5. Summarized / acknowledged facts.
    decision = getattr(lifecycle_decision, "value", lifecycle_decision)
    if decision == "READY_FOR_SUMMARY":
        for sf in REQUIRED_FIELDS:
            if after.is_satisfied(sf):
                _upsert(
                    memory.summarized_facts,
                    sf.value,
                    "included in the empathy summary presented to the user",
                    turn_index,
                )
    if decision == "READY_FOR_TRANSITION":
        for sf in REQUIRED_FIELDS:
            if after.is_satisfied(sf):
                _upsert(
                    memory.acknowledged_facts,
                    sf.value,
                    "user explicitly confirmed in the empathy summary",
                    turn_index,
                )

    return memory


# ---------------------------------------------------------------------------
# Prompt resume hint
# ---------------------------------------------------------------------------


def memory_to_prompt_bullets(
    session_data, current_target=None, max_bullets: int = 4
) -> List[str]:
    """
    Build short instruction bullets that nudge the LLM to resume unfinished
    topics instead of restarting them. Read-only; returns ``[]`` when there
    is nothing to resume.

    ``current_target`` (an optional ``StateField``) is the objective the
    mentor is already actively pursuing — its open thread is not re-listed.
    """
    memory = getattr(session_data, "conversation_memory", None)
    if memory is None:
        return []

    bullets: List[str] = []
    target_value = getattr(current_target, "value", None)

    for rec in memory.open_threads:
        if target_value is not None and rec.get("field") == target_value:
            continue
        label = _field_label(rec.get("field", ""))
        bullets.append(
            f"The user has an unfinished topic on {label}: "
            f"{rec.get('reason', '')}. "
            "Do not restart it from scratch - build on what they already shared."
        )

    for rec in memory.deferred_topics:
        label = _field_label(rec.get("field", ""))
        bullets.append(
            f"The user previously mentioned {label}. "
            "Return to it naturally once the current question is answered."
        )

    if memory.acknowledged_facts:
        bullets.append(
            "The user already acknowledged the empathy summary. "
            "Do not re-ask for its content; only build on refinements they volunteer."
        )

    # 5th bullet: surface user-stated facts the mentor should remember.
    # This nudges the LLM to build on what the user has already established,
    # rather than treating volunteered information as an answer to a question
    # it asked, while still distinguishing established context from a direct
    # answer to a question the mentor actually asked.
    has_known_facts = any(
        rec.get("known") is True
        for rec in memory.open_threads + memory.deferred_topics
    )
    if has_known_facts:
        bullets.append(
            "The user has already shared some facts voluntarily; "
            "do not treat them as direct answers to your questions. "
            "Build on what they've established or treat as remembered context."
        )

    return bullets[:max_bullets]


# ---------------------------------------------------------------------------
# Diagnostics section
# ---------------------------------------------------------------------------


def memory_diagnostics_section(session_data) -> dict[str, Any]:
    """Developer Console view of the session's conversational memory.
    Read-only projection of the persisted container."""
    memory = getattr(session_data, "conversation_memory", None)
    if memory is None:
        return {}
    return {
        "Open Threads": memory.open_threads,
        "Resolved Threads": memory.resolved_threads,
        "Deferred Topics": memory.deferred_topics,
        "Acknowledged Facts": memory.acknowledged_facts,
        "Summarized Facts": memory.summarized_facts,
        "Partially Answered Objectives": memory.partially_answered_objectives,
    }
