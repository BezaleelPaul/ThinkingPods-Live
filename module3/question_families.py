"""
module3.question_families — lightweight semantic grouping of mentor questions.

Problem
-------
The mentor asks ONE question per turn, but natural-language rephrasing means it
can end up asking the *same kind* of question over and over:

    Mentor turn 1: "How common is this?"
    Mentor turn 2: "How widespread is this?"
    Mentor turn 3: "What percentage experience this?"
    Mentor turn 4: "How frequently does this occur?"

All four collect the same information (frequency) from the same angle
(prevalence/estimate). This module introduces a *semantic family* taxonomy so
the mentor can (a) know which family a produced question belongs to and
(b) plan the next question to come from a family that has NOT been asked yet.

Non-goals
---------
- No embeddings, no vector DBs, no ML. Pure regex/keyword normalisation.
- No mutating ``ProjectState`` or ``SessionData`` — read-only classification
  and planning.

Pieces
------
* :class:`QuestionFamily` — enum of one family per (field, angle) pair, e.g.
  ``FREQUENCY_ESTIMATE`` / ``FREQUENCY_PERCENTAGE`` / ``FREQUENCY_PREVALENCE`` /
  ``FREQUENCY_EXAMPLES``.
* :func:`classify_question` — map a question string to its family (or ``None``).
* :class:`QuestionFamilyPlanner` — decide which family the *next* question
  should target, given the field being pursued, the families already asked,
  and the current sufficiency snapshot.
* :class:`FamilyPlan` — the planner output (selected family, skipped families
  with reasons, whether a re-ask is sanctioned).

The planner encodes the rule:

    Avoid asking another question from the same family unless the previous
    answers for that field were insufficient.

"Sufficient" is defined by :class:`module3.sufficiency.SufficiencyChecker`
(SUFFICIENT/COMPLETE). When the field already holds a usable answer the
objective engine advances away from it anyway; the planner only ever steers a
question for a field that is still being pursued.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from memory_extractor import StateField

from .sufficiency import SufficiencyReport

__all__ = [
    "QuestionFamily",
    "FamilyPlan",
    "QuestionFamilyPlanner",
    "classify_question",
    "family_label",
    "field_of",
    "FAMILIES_BY_FIELD",
    "FAMILY_FALLBACK_QUESTIONS",
]


# ---------------------------------------------------------------------------
# QuestionFamily — semantic family taxonomy
# ---------------------------------------------------------------------------


class QuestionFamily(str, Enum):
    """A semantic family of mentor questions.

    Each value is ``<FIELD>_<ANGLE>``. Two questions belong to the same family
    when they collect the same information from the same angle, even if the
    wording is completely different ("How common is this?" and "How widespread
    is this?" are both ``FREQUENCY_PREVALENCE``).

    ``str`` subclass so values serialise naturally (JSON, diagnostics, session
    persistence) and round-trip cleanly.
    """

    # ---- Frequency -------------------------------------------------------
    FREQUENCY_ESTIMATE = "FREQUENCY_ESTIMATE"
    """How often does it happen? (estimate / cadence)"""
    FREQUENCY_PERCENTAGE = "FREQUENCY_PERCENTAGE"
    """What percentage / how many are affected?"""
    FREQUENCY_PREVALENCE = "FREQUENCY_PREVALENCE"
    """How common / widespread is it?"""
    FREQUENCY_EXAMPLES = "FREQUENCY_EXAMPLES"
    """Give an example of when it happens."""

    # ---- Personas --------------------------------------------------------
    PERSONAS_WHO = "PERSONAS_WHO"
    """Who is the target audience?"""
    PERSONAS_EXAMPLES = "PERSONAS_EXAMPLES"
    """Describe one specific representative user."""
    PERSONAS_DEMOGRAPHICS = "PERSONAS_DEMOGRAPHICS"
    """Age / role / background of the users."""

    # ---- Problems --------------------------------------------------------
    PROBLEMS_CORE = "PROBLEMS_CORE"
    """What is the core problem?"""
    PROBLEMS_EXAMPLES = "PROBLEMS_EXAMPLES"
    """Give an example / scenario of the problem."""
    PROBLEMS_WHY = "PROBLEMS_WHY"
    """Why does the problem matter?"""

    # ---- Current solutions ----------------------------------------------
    SOLUTIONS_CURRENT = "SOLUTIONS_CURRENT"
    """How do they currently handle it?"""
    SOLUTIONS_ATTEMPTS = "SOLUTIONS_ATTEMPTS"
    """What have they already tried?"""
    SOLUTIONS_EFFECTIVENESS = "SOLUTIONS_EFFECTIVENESS"
    """How well do current approaches work?"""

    # ---- Pain points -----------------------------------------------------
    PAIN_FRUSTRATIONS = "PAIN_FRUSTRATIONS"
    """What frustrates / hurts the most?"""
    PAIN_MOTIVATION = "PAIN_MOTIVATION"
    """Why does the builder care?"""
    PAIN_CONSEQUENCES = "PAIN_CONSEQUENCES"
    """What are the consequences?"""

    # ---- Evidence --------------------------------------------------------
    EVIDENCE_OBSERVATIONS = "EVIDENCE_OBSERVATIONS"
    """What have they seen / heard?"""
    EVIDENCE_DATA = "EVIDENCE_DATA"
    """What data / research backs it up?"""
    EVIDENCE_ANECDOTES = "EVIDENCE_ANECDOTES"
    """Have they talked to affected people?"""


# Candidate families per field, in the order the planner prefers to try them.
# The planner picks the FIRST family that has not been asked yet; when every
# family has been asked it re-asks the least-recently-asked one.
FAMILIES_BY_FIELD: Dict[StateField, Tuple[QuestionFamily, ...]] = {
    StateField.PERSONAS: (
        QuestionFamily.PERSONAS_WHO,
        QuestionFamily.PERSONAS_EXAMPLES,
        QuestionFamily.PERSONAS_DEMOGRAPHICS,
    ),
    StateField.PROBLEMS: (
        QuestionFamily.PROBLEMS_CORE,
        QuestionFamily.PROBLEMS_EXAMPLES,
        QuestionFamily.PROBLEMS_WHY,
    ),
    StateField.CURRENT_SOLUTIONS: (
        QuestionFamily.SOLUTIONS_CURRENT,
        QuestionFamily.SOLUTIONS_ATTEMPTS,
        QuestionFamily.SOLUTIONS_EFFECTIVENESS,
    ),
    StateField.PAIN_POINTS: (
        QuestionFamily.PAIN_FRUSTRATIONS,
        QuestionFamily.PAIN_MOTIVATION,
        QuestionFamily.PAIN_CONSEQUENCES,
    ),
    StateField.EVIDENCE: (
        QuestionFamily.EVIDENCE_OBSERVATIONS,
        QuestionFamily.EVIDENCE_DATA,
        QuestionFamily.EVIDENCE_ANECDOTES,
    ),
    StateField.FREQUENCY: (
        QuestionFamily.FREQUENCY_ESTIMATE,
        QuestionFamily.FREQUENCY_PERCENTAGE,
        QuestionFamily.FREQUENCY_PREVALENCE,
        QuestionFamily.FREQUENCY_EXAMPLES,
    ),
}

FIELD_OF_FAMILY: Dict[QuestionFamily, StateField] = {
    family: field
    for field, families in FAMILIES_BY_FIELD.items()
    for family in families
}

# Deterministic fallback question per family — used when the LLM is
# unavailable and by the repeat-guard's replacement reply. Each text is
# written so :func:`classify_question` maps it back to its own family, and
# phrased like an experienced Design Thinking facilitator: it opens with a
# short observation/acknowledgment, states why the question matters, and ends
# with exactly ONE focused question. No filler, no questionnaire feel.
FAMILY_FALLBACK_QUESTIONS: Dict[QuestionFamily, str] = {
    QuestionFamily.FREQUENCY_ESTIMATE:
        "That's a clear picture of the situation. To understand how urgently "
        "this needs solving, how often does the problem occur - daily, "
        "weekly, or only in certain situations?",
    QuestionFamily.FREQUENCY_PERCENTAGE:
        "That's a useful gauge of scale. To see how widely this reaches "
        "people, what percentage of the people you're designing for would "
        "you estimate experiences this problem?",
    QuestionFamily.FREQUENCY_PREVALENCE:
        "That's a helpful perspective. To understand how widely felt this is, "
        "how common or widespread would you say this problem is among those "
        "affected?",
    QuestionFamily.FREQUENCY_EXAMPLES:
        "That's a good start. To make the cadence concrete, "
        "could you give me an example of a time when this problem happened?",
    QuestionFamily.PERSONAS_WHO:
        "That's a meaningful starting point. To focus the design on the right "
        "people, who specifically would benefit from this - can you describe "
        "the people you're designing for?",
    QuestionFamily.PERSONAS_EXAMPLES:
        "Good - I can see the audience taking shape. To make the picture "
        "concrete, can you think of one specific person who would use this - "
        "what are they like?",
    QuestionFamily.PERSONAS_DEMOGRAPHICS:
        "That's a solid start on who they are. To sharpen the persona, what "
        "is the age, role, or background of the people you're designing for?",
    QuestionFamily.PROBLEMS_CORE:
        "That's a clear view of who you're helping. To shape the problem "
        "statement precisely, what is the core problem or frustration they're "
        "experiencing?",
    QuestionFamily.PROBLEMS_EXAMPLES:
        "That's a real problem taking shape. To make it concrete, could you "
        "give me an example of the problem they run into?",
    QuestionFamily.PROBLEMS_WHY:
        "That's a useful angle. To understand why it weighs on people, why "
        "does this problem matter to the people experiencing it?",
    QuestionFamily.SOLUTIONS_CURRENT:
        "That's a real gap you're describing. To see what they currently lean "
        "on, how do people handle this problem today - what workarounds or "
        "tools do they currently use?",
    QuestionFamily.SOLUTIONS_ATTEMPTS:
        "That's a useful detail. To avoid re-suggesting what failed before, "
        "what have they already tried to solve this problem?",
    QuestionFamily.SOLUTIONS_EFFECTIVENESS:
        "That's a good sense of their approach. To see whether it actually "
        "helps, how well do their current approaches actually work?",
    QuestionFamily.PAIN_FRUSTRATIONS:
        "That's a real pain point. To know what hurts most, what frustrates "
        "or hurts the most about the problem right now?",
    QuestionFamily.PAIN_MOTIVATION:
        "That's a compelling motivation. To understand what "
        "drives you, what personally drew you to work on this problem?",
    QuestionFamily.PAIN_CONSEQUENCES:
        "That's a serious concern. To understand the real cost, what happens "
        "to people when they face this problem - what are the consequences?",
    QuestionFamily.EVIDENCE_OBSERVATIONS:
        "That's a strong signal. To ground the problem in real observations, "
        "what have you seen or heard that tells you this is a real problem "
        "worth solving?",
    QuestionFamily.EVIDENCE_DATA:
        "That's a good foundation. To back the problem with hard evidence, do "
        "you have any data, research, or studies that back this up?",
    QuestionFamily.EVIDENCE_ANECDOTES:
        "That's a useful signal. To bring the problem to life, have you "
        "talked to anyone who experienced this problem directly?",
}


# ---------------------------------------------------------------------------
# Classification — question text -> QuestionFamily
# ---------------------------------------------------------------------------

# Priority-ordered (family, regex) pairs. Frequency families come first
# because frequency is the most common repeat offender and its markers are
# the most distinct. Within each field, more specific multi-token patterns
# precede generic ones so e.g. "how common or widespread" is caught before a
# single generic token would mis-classify it.
_PATTERNS: List[Tuple[QuestionFamily, str]] = [
    # --- Frequency -------------------------------------------------------
    (QuestionFamily.FREQUENCY_PERCENTAGE,
     r"\bpercentage\b|\bpercent\b|(?<![a-z])%(?!\d)"),
    (QuestionFamily.FREQUENCY_ESTIMATE,
     r"\bhow often\b|\bhow frequently\b|\bhow regularly\b|\bhow regular\b"
     r"|\bfrequency\b|\bfrequently\b"),
    (QuestionFamily.FREQUENCY_PREVALENCE,
     r"\bhow common\b|\bhow widespread\b|\bhow prevalent\b|\bhow rare\b"
     r"|\bhow typical\b|\bhow widely\b"),
    (QuestionFamily.FREQUENCY_EXAMPLES,
     r"(?=.*\b(?:example|instance|occasion|situation)\b)"
     r"(?=.*\b(?:when|happen|occur|time)\b)"),
    # --- Personas ---------------------------------------------------------
    (QuestionFamily.PERSONAS_EXAMPLES,
     r"\bthink of\b|\bname (?:one|a|me)\b|\bexample of (?:a|one|the) (?:person|user)\b"
     r"|\bpicture\b|\bdescribe one\b"),
    (QuestionFamily.PERSONAS_DEMOGRAPHICS,
     r"\bage\b|\bdemographic\w*\b|\brole\b|\bbackground\b|\boccupation\b|\bjob\b"
     r"|(?:where (?:do|are) they)"),
    (QuestionFamily.PERSONAS_WHO,
     r"^who\b|\bwho specifically\b|\bwho is\b|\bwho are\b|\bbenefit\b"
     r"|\bdesigning for\b|(?:describe the (?:people|users))"
     r"|(?:tell me about (?:the )?(?:people|users))"),
    # --- Problems ---------------------------------------------------------
    (QuestionFamily.PROBLEMS_EXAMPLES,
     r"\bexample\b|\bscenario\b|(?:concrete (?:case|example))"),
    (QuestionFamily.PROBLEMS_WHY,
     r"\bwhy\b|\breason\b|\bbecause\b"),
    (QuestionFamily.PROBLEMS_CORE,
     r"\bcore problem\b|\bmain challenge\b|\bwhat problem\b"
     r"|(?:what (?:is|'s) the (?:issue|problem))|\bcore frustration\b"),
    # --- Current solutions ------------------------------------------------
    (QuestionFamily.SOLUTIONS_ATTEMPTS,
     r"\btried\b|\battempted?\b|\btry\b"),
    (QuestionFamily.SOLUTIONS_EFFECTIVENESS,
     r"\bhow well\b|\bactually work\b|\beffective\b|\bdoes it work\b"
     r"|\bsolve\b|\bimprove\b|\bfix\b"),
    (QuestionFamily.SOLUTIONS_CURRENT,
     r"\bhandle\b|\bworkaround\b|\bcurrently\b|\btoday\b|\btools\b"
     r"|(?:how do they)|(?:what do they)"),
    # --- Pain points ------------------------------------------------------
    (QuestionFamily.PAIN_FRUSTRATIONS,
     r"\bfrustrat\w*\b|\bhurt\w*\b|\bpain\w*\b|\bannoy\w*\b|\bbother\w*\b"),
    (QuestionFamily.PAIN_CONSEQUENCES,
     r"\bconsequence\w*\b|\bimpact\w*\b|\bwhat happens\b|\baffect\w*\b"
     r"|\beffect\w*\b|\bresults\b"),
    (QuestionFamily.PAIN_MOTIVATION,
     r"\bdrew\b|\bmotivat\w*\b|\bdrives\b|\bpassion\w*\b"
     r"|\bwhy does it matter\b|\bcare about\b"),
    # --- Evidence ---------------------------------------------------------
    (QuestionFamily.EVIDENCE_DATA,
     r"\bdata\b|\bresearch\b|\bstud(?:y|ies)\b|\bsurvey\w*\b|\bstat\w*\b"
     r"|\bnumbers?\b|\bmetrics?\b"),
    (QuestionFamily.EVIDENCE_ANECDOTES,
     r"\btalked to\b|\bconversation\w*\b|\binterview\w*\b|\btold\b"
     r"|\bcustomer\w*\b|\bsaid\b|\bheard from\b|\bspoke\b"),
    (QuestionFamily.EVIDENCE_OBSERVATIONS,
     r"\bseen\b|\bheard\b|\bobserved?\b|\bnoticed?\b|\bwitnessed?\b"
     r"|\breal problem\b"),
]

_COMPILED: List[Tuple[QuestionFamily, re.Pattern]] = [
    (family, re.compile(pattern)) for family, pattern in _PATTERNS
]


def classify_question(text: Optional[str]) -> Optional[QuestionFamily]:
    """Return the semantic family of ``text``, or ``None`` if unknown.

    Lightweight normalisation only: lowercases the text and runs priority-
    ordered regexes. No embeddings, no models. ``None`` means the question
    does not clearly belong to any known family (e.g. a summary or a plain
    acknowledgement).
    """
    if not text or not text.strip():
        return None
    lowered = text.lower()
    for family, pattern in _COMPILED:
        if pattern.search(lowered):
            return family
    return None


def field_of(family: QuestionFamily) -> Optional[StateField]:
    """Return the StateField a question family collects information for."""
    return FIELD_OF_FAMILY.get(family)


def family_label(family: QuestionFamily) -> str:
    """Human-readable label, e.g. ``FREQUENCY_ESTIMATE`` -> "Frequency estimate"."""
    words = family.value.lower().split("_")
    words[0] = words[0].capitalize()
    return " ".join(words)


# ---------------------------------------------------------------------------
# FamilyPlan — the planner's output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyPlan:
    """Decision record for the next question's semantic family.

    ``selected`` is the family the next question should target, or ``None``
    when there is nothing sensible to ask for ``field`` (WRAP_UP / the field
    already holds a sufficient answer / no families registered). ``skipped``
    lists the families that were passed over and why — surfaced in the
    Developer Console. When every family for the field has already been asked
    AND the previous answers were still insufficient, ``reask`` is ``True``
    and ``selected`` is the least-recently-asked family (re-asking is then
    sanctioned by the "unless previous answers were insufficient" rule).
    """

    field: Optional[StateField]
    selected: Optional[QuestionFamily]
    asked_families: Tuple[str, ...] = ()
    skip_reasons: Tuple[dict, ...] = ()
    reask: bool = False
    reask_reason: str = ""

    def to_dict(self) -> dict:
        """Serialisable projection for diagnostics / capture."""
        return {
            "field": self.field.value if self.field else None,
            "selected": self.selected.value if self.selected else None,
            "asked_families": list(self.asked_families),
            "skip_reasons": list(self.skip_reasons),
            "reask": self.reask,
            "reask_reason": self.reask_reason,
        }


# ---------------------------------------------------------------------------
# QuestionFamilyPlanner
# ---------------------------------------------------------------------------


class QuestionFamilyPlanner:
    """Stateless planner deciding which question family to ask next.

    Pure function of ``(field, asked_family_values, sufficiency_report)`` —
    no state, no I/O, no LLM. The same inputs always produce the same
    ``FamilyPlan``.
    """

    def plan(
        self,
        field: StateField,
        asked_family_values: Sequence[str],
        sufficiency: SufficiencyReport,
    ) -> FamilyPlan:
        """Plan the next question family for ``field``.

        Parameters
        ----------
        field:
            The StateField the current objective is pursuing (the objective's
            ``targeted_field()``), or ``None``.
        asked_family_values:
            Family value strings asked so far this session
            (``session_data.asked_question_families``), in chronological order.
        sufficiency:
            The current :class:`SufficiencyReport` for ``ProjectState``.

        Returns
        -------
        FamilyPlan
            The selected family (or ``None``) plus the skip/re-ask record.
        """
        if field is None:
            return FamilyPlan(
                field=None,
                selected=None,
                asked_families=tuple(asked_family_values),
                skip_reasons=(),
                reask=False,
                reask_reason="No field targeted this turn.",
            )

        candidates: Tuple[QuestionFamily, ...] = FAMILIES_BY_FIELD.get(field, ())
        asked_tuple = tuple(asked_family_values)

        if not candidates:
            return FamilyPlan(
                field=field,
                selected=None,
                asked_families=asked_tuple,
                skip_reasons=(),
                reask=False,
                reask_reason=f"No candidate families registered for {field.value}.",
            )

        if sufficiency.is_satisfied(field):
            skip_reasons = tuple({
                "family": family.value,
                "reason": (
                    f"{field.value} already holds a sufficient answer; no need "
                    f"to ask another question from this family."
                ),
            } for family in candidates)
            return FamilyPlan(
                field=field,
                selected=None,
                asked_families=asked_tuple,
                skip_reasons=skip_reasons,
                reask=False,
                reask_reason=(
                    f"{field.value} already sufficiently answered; the "
                    f"objective advances."
                ),
            )

        candidate_values = {family.value for family in candidates}
        asked_for_field = [v for v in asked_family_values if v in candidate_values]

        skip_reasons = []
        for family in candidates:
            if family.value in asked_for_field:
                skip_reasons.append({
                    "family": family.value,
                    "reason": (
                        "Family already asked; answers still insufficient — "
                        "preferring a fresh angle."
                    ),
                })
            else:
                return FamilyPlan(
                    field=field,
                    selected=family,
                    asked_families=asked_tuple,
                    skip_reasons=tuple(skip_reasons),
                    reask=False,
                    reask_reason="",
                )

        # Every family for this field has already been asked. Previous answers
        # are still insufficient, so re-asking is sanctioned — choose the least
        # recently asked family for variety.
        try:
            least_recent = next(
                QuestionFamily(v) for v in asked_for_field if v in candidate_values
            )
        except StopIteration:
            least_recent = candidates[0]
        return FamilyPlan(
            field=field,
            selected=least_recent,
            asked_families=asked_tuple,
            skip_reasons=tuple(skip_reasons),
            reask=True,
            reask_reason=(
                f"All families for {field.value} already asked and answers "
                f"remain insufficient — re-asking the least recently asked "
                f"family."
            ),
        )
