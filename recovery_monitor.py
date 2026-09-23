"""
recovery_monitor.py — deterministic recovery from non-linear conversations.

Architecture role
-----------------
Observation + guidance layer for the Empathize v2 pipeline. After each mentor
turn (extraction + state update + objective selection) it classifies the turn
into a known recovery category and produces a *recovery strategy* plus a
*clarification reason*, surfaced in the Developer Console. 100% deterministic,
read-only, no LLM, no state mutation, no I/O.

Recovery categories (non-linear conversation cases):
  * ``DONT_KNOW`` / ``UNCERTAIN_ANSWER`` — the user could not answer; the
    mentor re-asks from a fresh question family.
  * ``CONTRADICTION`` — a scalar (frequency) was overwritten with a different
    value; the newer value wins and the earlier value is not discarded from
    context.
  * ``TOPIC_CHANGE`` — the user supplied facts for a field other than the one
    the mentor just asked about; the facts are captured and the mentor steers
    back to the pending objective.
  * ``MULTIPLE_UNRELATED_FACTS`` — one turn changed three or more distinct
    fields; all facts are captured and the objective advances to the
    highest-priority unmet field.

Guiding contract (from the robustness requirement)
--------------------------------------------------
  * identify inconsistencies,
  * ask clarification only when necessary (never to restate pinned behaviour),
  * avoid restarting the interview (state is never reset by this layer),
  * preserve previously collected valid information.

This module owns NO reply generation, NO objective selection, NO state
mutation, NO session persistence. It only *explains* what already happened —
the existing deterministic machinery (family rotation, SET-overwrite, append-
not-replace) is the actual recovery; this module makes it explicit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from memory_extractor import LIST_FIELDS, StateField
from module3 import SufficiencyChecker, classify_question, field_of

__all__ = [
    "RecoveryCategory",
    "RecoveryStrategy",
    "RecoveryReport",
    "analyze_recovery",
]


# ---------------------------------------------------------------------------
# Public enums
# ---------------------------------------------------------------------------


class RecoveryCategory(str, Enum):
    """The kind of non-linear conversation situation a turn exhibited."""

    NONE = "NONE"
    """Turn was on-topic; nothing to recover from."""

    DONT_KNOW = "DONT_KNOW"
    """User explicitly stated they could not answer."""

    UNCERTAIN_ANSWER = "UNCERTAIN_ANSWER"
    """User gave a hedged / uncertain answer."""

    CONTRADICTION = "CONTRADICTION"
    """A scalar was overwritten with a different value."""

    TOPIC_CHANGE = "TOPIC_CHANGE"
    """User supplied facts for a field other than the one asked about."""

    MULTIPLE_UNRELATED_FACTS = "MULTIPLE_UNRELATED_FACTS"
    """One turn changed three or more distinct fields."""


class RecoveryStrategy(str, Enum):
    """The deterministic recovery action the pipeline takes for a category."""

    NONE = "NONE"

    REPHRASE_FRESH_ANGLE = "REPHRASE_FRESH_ANGLE"
    """Re-ask from a fresh question family instead of repeating the same one."""

    ACCEPT_NEW_VALUE = "ACCEPT_NEW_VALUE"
    """Newer value supersedes the earlier one; earlier info kept in context."""

    STEER_BACK = "STEER_BACK"
    """Capture the new facts and continue pursuing the pending objective."""

    CAPTURE_ALL_CONTINUE = "CAPTURE_ALL_CONTINUE"
    """Apply every fact and advance to the highest-priority unmet field."""


# ---------------------------------------------------------------------------
# RecoveryReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryReport:
    """Immutable per-turn recovery classification + explanation."""

    category: RecoveryCategory = RecoveryCategory.NONE
    strategy: RecoveryStrategy = RecoveryStrategy.NONE
    detected_inconsistency: str = ""
    clarification_reason: str = ""
    affected_fields: tuple[str, ...] = ()
    preserved_facts: tuple[str, ...] = ()
    avoided_restart: bool = True

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe projection for the Developer Console."""
        return {
            "Category": self.category.value,
            "Detected Inconsistency": self.detected_inconsistency,
            "Recovery Strategy": self.strategy.value,
            "Clarification Reason": self.clarification_reason,
            "Affected Fields": list(self.affected_fields),
            "Preserved Facts": list(self.preserved_facts),
            "Avoided Restart": self.avoided_restart,
        }


# ---------------------------------------------------------------------------
# Detection helpers (deterministic, read-only)
# ---------------------------------------------------------------------------

_STATE_FIELD_KEYS: tuple[str, ...] = tuple(sf.value for sf in StateField)
_LIST_KEYS: frozenset[str] = frozenset(sf.value for sf in LIST_FIELDS)

_DONT_KNOW_PATTERNS: tuple[str, ...] = (
    r"\bi don'?t know\b",
    r"\bi do not know\b",
    r"\bdon'?t know\b",
    r"\bdo not know\b",
    r"\bno idea\b",
    r"\bno clue\b",
    r"\bi can'?t say\b",
    r"\bcan'?t say\b",
    r"\bhaven'?t a clue\b",
    r"\bnot really sure\b",
)
_UNCERTAIN_PATTERNS: tuple[str, ...] = (
    r"\bnot sure\b",
    r"\bunsure\b",
    r"\bnot certain\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bprobably\b",
    r"\bi guess\b",
    r"\bhard to say\b",
    r"\bdepends\b",
    r"\bnot exactly\b",
    r"\bi think so\b",
)
_COMPILED_DONT_KNOW = [re.compile(p, re.IGNORECASE) for p in _DONT_KNOW_PATTERNS]
_COMPILED_UNCERTAIN = [re.compile(p, re.IGNORECASE) for p in _UNCERTAIN_PATTERNS]


def _changed_fields(before: dict, after: dict) -> dict[str, tuple]:
    """Which fields actually changed between two state snapshots.

    Returns ``{field_value: ("added", new_items)}`` for list fields and
    ``{field_value: ("updated", (before, after))}`` for scalars. Duplicate
    ADD proposals that changed nothing are ignored (no effective change).
    """
    changed: dict[str, tuple] = {}
    for key in _STATE_FIELD_KEYS:
        b = before.get(key)
        a = after.get(key)
        if key in _LIST_KEYS:
            b_items = [i for i in (b or []) if i]
            a_items = [i for i in (a or []) if i]
            new_items = [i for i in a_items if i not in b_items]
            if new_items:
                changed[key] = ("added", new_items)
        else:
            bv = str(b).strip() if b else ""
            av = str(a).strip() if a else ""
            if av and av != bv:
                changed[key] = ("updated", (bv, av))
    return changed


def _preserved_facts(before: dict, after: dict, changed: dict) -> tuple[str, ...]:
    """Fields whose previously collected values survive the turn unchanged."""
    preserved: list[str] = []
    for key in _STATE_FIELD_KEYS:
        b = before.get(key)
        a = after.get(key)
        if key in _LIST_KEYS:
            b_items = [i for i in (b or []) if i]
            if b_items and all(i in (a or []) for i in b_items):
                preserved.append(key)
        else:
            bv = str(b).strip() if b else ""
            if bv and bv == (str(a).strip() if a else ""):
                preserved.append(key)
    return tuple(preserved)


def _matches(text: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


def _clip(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


# ---------------------------------------------------------------------------
# analyze_recovery
# ---------------------------------------------------------------------------


def analyze_recovery(
    *,
    user_message: str,
    state_before: Optional[dict] = None,
    state_after: Optional[dict] = None,
    last_assistant_message: Optional[str] = None,
) -> RecoveryReport:
    """Classify one completed turn into a recovery category.

    Read-only: consumes state snapshots and the last mentor question, mutates
    nothing, never calls the LLM. The same inputs always produce the same
    :class:`RecoveryReport`.

    Parameters
    ----------
    user_message:
        The user's raw message for this turn.
    state_before:
        ``ProjectState.to_state_dict()`` snapshot taken before extraction.
    state_after:
        ``ProjectState.to_state_dict()`` snapshot taken after extraction.
    last_assistant_message:
        The mentor's previous reply (used to infer what field the user was
        asked about). ``None`` on the first turn.
    """
    before = dict(state_before or {})
    after = dict(state_after or {})
    changed = _changed_fields(before, after)
    preserved = _preserved_facts(before, after, changed)
    lower = (user_message or "").lower()

    # 1. Explicit inability to answer — only when nothing was captured.
    if not changed and _matches(lower, _COMPILED_DONT_KNOW):
        return RecoveryReport(
            category=RecoveryCategory.DONT_KNOW,
            strategy=RecoveryStrategy.REPHRASE_FRESH_ANGLE,
            detected_inconsistency=(
                f"User stated they could not answer: '{_clip(user_message)}'."
            ),
            clarification_reason=(
                "No clarification question needed — the mentor re-asks from a "
                "fresh question family; the interview is not restarted."
            ),
            preserved_facts=preserved,
        )

    # 2. Hedged / uncertain answer — only when nothing was captured.
    if not changed and _matches(lower, _COMPILED_UNCERTAIN):
        return RecoveryReport(
            category=RecoveryCategory.UNCERTAIN_ANSWER,
            strategy=RecoveryStrategy.REPHRASE_FRESH_ANGLE,
            detected_inconsistency=(
                f"User gave an uncertain answer: '{_clip(user_message)}'."
            ),
            clarification_reason=(
                "No clarification question needed — the mentor re-asks from a "
                "fresh question family; the interview is not restarted."
            ),
            preserved_facts=preserved,
        )

    # 3. Contradiction — a scalar that already held a value was overwritten.
    overwrites = [
        k for k, (op, val) in changed.items() if op == "updated" and val[0]
    ]
    if overwrites:
        details = "; ".join(
            f"{k} '{changed[k][1][0]}' -> '{changed[k][1][1]}'"
            for k in overwrites
        )
        return RecoveryReport(
            category=RecoveryCategory.CONTRADICTION,
            strategy=RecoveryStrategy.ACCEPT_NEW_VALUE,
            detected_inconsistency=f"Contradictory answer: {details}.",
            clarification_reason=(
                "No clarification needed — the newer value supersedes the "
                "earlier one; previously collected information is preserved."
            ),
            affected_fields=tuple(overwrites),
            preserved_facts=preserved,
        )

    # 4. Topic change — the user answered a different field than the one the
    #    mentor just asked about, and that field is still not satisfied.
    asked_field = None
    if last_assistant_message:
        family = classify_question(last_assistant_message)
        if family is not None:
            asked_field = field_of(family)
    if (
        asked_field is not None
        and changed
        and asked_field.value not in changed
    ):
        report = SufficiencyChecker.evaluate_dict(after)
        if not report.is_satisfied(asked_field):
            touched = ", ".join(sorted(changed))
            return RecoveryReport(
                category=RecoveryCategory.TOPIC_CHANGE,
                strategy=RecoveryStrategy.STEER_BACK,
                detected_inconsistency=(
                    f"User provided info for '{touched}' while the pending "
                    f"question was about '{asked_field.value}'."
                ),
                clarification_reason=(
                    "No clarification needed — the new facts are captured and "
                    "the mentor steers back to the pending objective."
                ),
                affected_fields=tuple(sorted(changed)),
                preserved_facts=preserved,
            )

    # 5. Multiple unrelated facts in one turn.
    if len(changed) >= 3:
        touched = ", ".join(sorted(changed))
        return RecoveryReport(
            category=RecoveryCategory.MULTIPLE_UNRELATED_FACTS,
            strategy=RecoveryStrategy.CAPTURE_ALL_CONTINUE,
            detected_inconsistency=(
                f"User supplied multiple unrelated facts in one turn "
                f"({len(changed)} fields: {touched})."
            ),
            clarification_reason=(
                "No clarification needed — all facts were captured and the "
                "objective advances to the highest-priority unmet field."
            ),
            affected_fields=tuple(sorted(changed)),
            preserved_facts=preserved,
        )

    return RecoveryReport(
        category=RecoveryCategory.NONE,
        strategy=RecoveryStrategy.NONE,
        detected_inconsistency="",
        clarification_reason="",
        affected_fields=tuple(sorted(changed)),
        preserved_facts=preserved,
    )
