"""
module3.scoring_engine — information-gain scoring for the Objective Engine.

Responsibility
--------------
The legacy :class:`~module3.priority_engine.PriorityEngine` answers one
question: *"which checklist item is missing and declared most important?"*
This scorer keeps that deterministic baseline but asks the information-gain
question instead: *"which single question would reduce uncertainty the most?"*

For each unmet required field the scorer produces a deterministic
:class:`CandidateScore`:

    score(field) = info_value        (intrinsic importance, from rule priority)
                 + unlock_bonus      (prefer fields that unlock more of the interview)
                 + discussion_bonus  (user already talked about it → latent evidence)
                 - repeat_penalty    (a question family for it was already asked)

Why these terms?
----------------
* ``info_value`` — the registry's declared priority normalised to ``[0, 1]``.
  This preserves the canonical discovery order (personas -> problems ->
  frequency -> current_solutions -> pain_points -> evidence) unless history
  gives a strong reason to pivot.
* ``unlock_bonus`` — an objective "unlocks" every later objective in the
  registry's own priority ordering. Central gaps (who / what) unlock more of
  the interview than tail gaps (evidence).
* ``discussion_bonus`` — if the user has already volunteered material about a
  field (detected by per-field keywords over the raw conversation), capturing
  it now has a high probability of success. This is what makes the mentor
  *pivot to what the user just opened up about* instead of blindly re-asking
  the next checklist item.
* ``repeat_penalty`` — if a question family for this field was already asked
  but the field is still unmet, the mentor has already tried and failed; a
  small penalty makes an untouched, comparably-important field win.

Existing evidence / confidence
------------------------------
"Evidence" and "confidence" are intentionally modelled through the two signals
above rather than a numeric state field:

* By construction every candidate has *zero captured evidence* — the
  CoverageReport only emits fields whose content is still missing, and
  ``ProjectState`` does not persist per-fact extraction confidence (that
  confidence is observation-only). Sufficiency therefore cannot discriminate
  among candidates (all are UNKNOWN).
* The closest live evidence signal is *conversation history*: a field the user
  has already discussed carries latent evidence even though extraction has not
  captured it yet — and asking about it now is the highest-confidence next
  step. That is exactly what ``discussion_bonus`` encodes.
* ``repeat_penalty`` encodes the *confidence that a question will yield new
  information*: a family already asked and unanswered is the lowest-confidence
  repeat, so it is mildly demoted.

Determinism / read-only
-----------------------
The scorer is a pure function of ``(rules, coverage, context)``. It never
mutates state, never calls the LLM, and never touches the network. The same
inputs ALWAYS produce the same scores in the same order. Tie-breaks fall back
to declared ``StateField`` enum order, so output is stable across Python runs.

Compatibility
-------------
The weights are small relative to the info-value gaps between adjacent
priorities, so on canonical Empathize states (and with no conversation
context) the scorer reproduces the legacy PriorityEngine ordering exactly.
The regression suite in ``tests/test_objective_scoring.py`` pins this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from memory_extractor import REQUIRED_FIELDS, StateField

from .objective import Objective
from .question_families import FAMILIES_BY_FIELD
from .rules import RuleRegistry

__all__ = [
    "ObjectiveContext",
    "CandidateScore",
    "build_candidate_scores",
    "UNLOCK_WEIGHT",
    "DISCUSSION_WEIGHT",
    "REPEAT_WEIGHT",
]

# ---------------------------------------------------------------------------
# Scoring weights (deterministic constants)
# ---------------------------------------------------------------------------
#
# Each weight is small enough that the canonical priority ordering survives on
# history-less states, but large enough that a genuine conversation signal can
# pivot between two *nearly equal* candidates. The design intent:
#
#   * info gaps between adjacent DEFAULT_RULES priorities are 0.1 (plus up to
#     UNLOCK_WEIGHT), so a single discussion bonus (~0.15) can only reorder
#     adjacent candidates — it can never leapfrog a field that is two or more
#     bands more important.
#   * REPEAT_WEIGHT alone never overturns the info ordering; it only tips
#     ties between fields that are otherwise close.

UNLOCK_WEIGHT: float = 0.04
"""Per unlocked (lower-priority, still-missing) objective bonus."""

DISCUSSION_WEIGHT: float = 0.15
"""Bonus when the user has already discussed this field's topic."""

REPEAT_WEIGHT: float = 0.05
"""Penalty when a question family for this field was already asked."""

# ---------------------------------------------------------------------------
# Conversation context (the only cross-turn input the scorer needs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectiveContext:
    """
    Read-only, deterministic view of the conversation the scorer may consult.

    This deliberately wraps *just* the two signals that influence objective
    selection — the raw user messages and the set of question families already
    asked — so Module 3 stays decoupled from ``session_manager.SessionData``.

    Fields
    ------
    user_messages:
        The raw user messages so far, oldest first (includes the current
        turn's message). Lowercased substring keyword matching decides
        whether a field was "discussed".
    asked_families:
        The ``QuestionFamily.value`` strings already asked this session
        (e.g. ``{"FREQUENCY_ESTIMATE", ...}``). A field whose family was
        asked but whose coverage is still missing is a repeat.

    Frozen so the context is safe to share across threads and trivially
    deterministic — two identical contexts ALWAYS score identically.
    """

    user_messages: Tuple[str, ...] = ()
    asked_families: frozenset = frozenset()

    def __init__(
        self,
        user_messages: Tuple[str, ...] = (),
        asked_families: frozenset = frozenset(),
    ) -> None:
        object.__setattr__(self, "user_messages", tuple(user_messages))
        object.__setattr__(self, "asked_families", frozenset(asked_families))


# ---------------------------------------------------------------------------
# Per-field discussion keywords (lowercased substring match)
# ---------------------------------------------------------------------------
#
# These let the scorer detect "the user already opened up about this topic"
# from raw text. They are intentionally conservative: each keyword is only
# admitted if it maps clearly to one field and does not appear as a false
# positive in the golden conversation fixtures.

_KEYWORDS_BY_FIELD: Dict[StateField, Tuple[str, ...]] = {
    StateField.PERSONAS: (
        "students", "student", "people", "person", "elderly", "parents",
        "parent", "users", "user", "audience", "college",
    ),
    StateField.PROBLEMS: (
        "forget", "forgot", "missing", "missed", "deadline", "submitting",
        "submit late", "trouble", "struggle",
    ),
    StateField.FREQUENCY: (
        "every single day", "every day", "every week", "weekly",
        "twice a week", "three times", "daily", "occurs", "happens",
    ),
    StateField.CURRENT_SOLUTIONS: (
        "pill box", "google sheet", "sticky notes", "planner", "whatsapp",
        "currently", "workaround", "tool",
    ),
    StateField.PAIN_POINTS: (
        "stress", "stresses", "worry", "frustrat", "hurts", "care",
    ),
    StateField.EVIDENCE: (
        "interview", "survey", "seen", "observed", "watched", "research",
        "data", "study",
    ),
}


def _field_discussed(field: StateField, user_messages: Tuple[str, ...]) -> bool:
    """True iff any user message mentions a keyword for ``field``."""
    keywords = _KEYWORDS_BY_FIELD.get(field, ())
    if not keywords or not user_messages:
        return False
    return any(
        keyword in message.lower()
        for message in user_messages
        for keyword in keywords
    )


def _field_repeated(field: StateField, asked_families: frozenset) -> bool:
    """True iff a question family belonging to ``field`` was already asked."""
    families = FAMILIES_BY_FIELD.get(field, ())
    return any(f.value in asked_families for f in families)


# ---------------------------------------------------------------------------
# CandidateScore — one scored candidate objective
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateScore:
    """
    A single scored candidate objective for the current coverage snapshot.

    The score is the sum of the four component signals; higher wins. The
    component values are kept as attributes so the Objective Engine can
    explain *why* a candidate won or lost in the reasoning / Developer
    Console trace (transparency without recomputation).
    """

    field: StateField
    objective: Objective
    score: float
    info_value: float
    unlock_bonus: float
    discussion_bonus: float
    repeat_penalty: float
    discussed: bool
    repeated: bool


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_candidate_scores(
    rules: RuleRegistry,
    coverage,
    context: ObjectiveContext,
) -> List[CandidateScore]:
    """
    Score every unmet required field covered by ``rules``.

    Parameters
    ----------
    rules:
        The rule registry to derive intrinsic importance from (the
        Objective Engine passes its own configured registry, so custom
        registries are honoured).
    coverage:
        A :class:`~module3.objective.CoverageReport` for the current
        ProjectState. Only fields this report marks missing are candidates.
    context:
        The conversation context (user messages + asked families). An empty
        context yields the canonical priority ordering.

    Returns
    -------
    list of CandidateScore
        Sorted by ``(score DESC, declared StateField enum order ASC)`` so
        the selection is deterministic and the head of the list is the
        objective the engine should pursue.

    Notes
    -----
    * Fields with no rule in ``rules`` are skipped (defensive — a valid
      registry covers every required field).
    * ``unlock_bonus`` counts the other *still-missing* fields whose rule
      priority is lower than this field's, in the registry's own ordering.
    """
    priority_by_field = {r.field: r.priority for r in rules}
    max_priority = max(priority_by_field.values(), default=1)
    missing = [sf for sf in REQUIRED_FIELDS if not coverage.is_field_complete(sf)]

    scores: List[CandidateScore] = []
    for sf in missing:
        priority = priority_by_field.get(sf)
        if priority is None:
            continue
        info_value = priority / max_priority
        unlock = sum(
            1
            for other in missing
            if other is not sf and priority_by_field.get(other, 0) < priority
        )
        discussed = _field_discussed(sf, context.user_messages)
        repeated = _field_repeated(sf, context.asked_families)
        unlock_bonus = UNLOCK_WEIGHT * unlock
        discussion_bonus = DISCUSSION_WEIGHT if discussed else 0.0
        repeat_penalty = REPEAT_WEIGHT if repeated else 0.0
        score = (
            info_value
            + unlock_bonus
            + discussion_bonus
            - repeat_penalty
        )
        scores.append(
            CandidateScore(
                field=sf,
                objective=Objective.for_field(sf),
                score=score,
                info_value=info_value,
                unlock_bonus=unlock_bonus,
                discussion_bonus=discussion_bonus,
                repeat_penalty=repeat_penalty,
                discussed=discussed,
                repeated=repeated,
            )
        )

    # Deterministic sort: score DESC, then declared StateField enum order ASC.
    enum_order = {sf: i for i, sf in enumerate(StateField)}
    scores.sort(key=lambda c: (-c.score, enum_order.get(c.field, 0)))
    return scores
