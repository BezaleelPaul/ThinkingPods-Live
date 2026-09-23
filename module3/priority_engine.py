"""
module3.priority_engine — rule-evaluation engine producing an Objective.

Responsibility
--------------
Given a :class:`~module3.objective.CoverageReport`, evaluate the supplied
rule registry and decide which conversational objective to pursue next.

The engine has NO business knowledge:

- It does NOT know which fields "matter more" — that lives in the rule
  registry passed to it.
- It does NOT know what an Objective means semantically — that is the
  Prompt Builder's job (Module 4).
- It does NOT know what a "complete Empathize stage" means — it derives
  WRAP_UP purely from the CoverageReport's all-complete boolean.

It owns three small pieces of logic, each implemented as a pure function:

1. If the CoverageReport says all required fields are complete -> return
   WRAP_UP. (Derived global condition; not a rule.)
2. Otherwise, gather candidate rules (those whose targeted field is
   missing), evaluate their priorities, and pick the highest-priority
   candidate.
3. On a priority tie, deterministically pick the one whose StateField is
   declared first in the enum, and lower the confidence accordingly to
   signal "more than one candidate tied" to downstream callers.

Read-only contract
------------------
The PriorityEngine takes a CoverageReport (already a value snapshot) and
returns a :class:`PriorityDecision` namedtuple (objective + winning rule
+ tie metadata + confidence + reasoning). It never touches ProjectState.
"""

from __future__ import annotations

from collections import namedtuple
from typing import List, Optional

from memory_extractor import REQUIRED_FIELDS, StateField

from .objective import CoverageReport, Objective
from .rules import (
    DEFAULT_RULES,
    Rule,
    RuleRegistry,
    ordered_candidates,
    validate_rules,
)

__all__ = ["PriorityEngine", "PriorityDecision"]


# ---------------------------------------------------------------------------
# PriorityDecision — the engine's pure output (objective + tie metadata)
# ---------------------------------------------------------------------------

PriorityDecision = namedtuple(
    "PriorityDecision",
    ["objective", "winning_rule", "tied_rules", "confidence", "reasoning"],
)
"""
PriorityEngine's pure decision record.

Fields
------
objective : Objective
    The chosen objective. ``WRAP_UP`` if every required field is covered.
winning_rule : Optional[Rule]
    The rule that won (if any). ``None`` for the WRAP_UP case, because
    WRAP_UP is a derived condition rather than a field-gap rule. The
    Objective Engine surfaces the winning rule's targeted StateField via
    ConversationObjective.targeted_field() so callers do NOT need to look
    at this field outside the engine itself.
tied_rules : List[Rule]
    Every candidate rule that shared the winning priority (after the
    declared-enumerable tiebreaker). If multiple were tied, the engine
    still returns ONE objective (the declared first one), but downstream
    Confidence/Reasoning reflects the tie. Empty for WRAP_UP.
confidence : float
    Deterministic confidence in [0.0, 1.0]:
      * 1.0 — Exactly one candidate, or the canonical WRAP_UP.
      * 0.8 — Tie broken by declared StateField enum order (the chosen
        winning_rule was first among tied_rules).
reasoning : List[str]
    Ordered human-readable strings explaining the decision. The Objective
    Engine incorporates these into the ConversationObjective.
"""


# ---------------------------------------------------------------------------
# PriorityEngine
# ---------------------------------------------------------------------------


class PriorityEngine:
    """
    Stateless, deterministic evaluator of which Objective to pursue next.

    Pure function over (CoverageReport × rules). No ProjectState input,
    no I/O, no cached state. The same (report, rules) pair ALWAYS returns
    the same :class:`PriorityDecision`.

    Construction
    ------------
    The engine accepts a rule registry (defaulting to
    :data:`module3.rules.DEFAULT_RULES`, the Empathize order). Customising
    the experience for future stages is a constructor argument, so an
    Objective Engine instance can be configured per stage.

    Usage
    -----
    >>> from module3 import PriorityEngine, CoverageReport, Objective
    >>> engine = PriorityEngine()  # uses DEFAULT_RULES
    >>> report = CoverageReport()  # everything missing
    >>> decision = engine.decide(report)
    >>> decision.objective
    <Objective.PERSONAS: 'PERSONAS'>
    """

    # Confidence values are declared here as named constants, not magic
    # numbers — the spec calls out distinct bands ("all matched", "minor
    # ambiguity", "multiple equally valid") and these encode them.
    CONFIDENCE_UNIQUE: float = 1.0
    CONFIDENCE_TIEBREAK: float = 0.8

    def __init__(self, rules: Optional[RuleRegistry] = None) -> None:
        """
        Parameters
        ----------
        rules:
            The rule registry to evaluate. ``None`` (default) selects
            :data:`module3.rules.DEFAULT_RULES`. The registry is validated
            once at construction so a runtime call to :meth:`decide` never
            re-walks the rules for structural errors.
        """
        self._rules: RuleRegistry = list(DEFAULT_RULES) if rules is None else list(rules)
        validate_rules(self._rules)
        # Stable declared-enum order reference; closed-over by decide().
        self._enum_order: dict[StateField, int] = {
            sf: i for i, sf in enumerate(StateField)
        }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def rules(self) -> RuleRegistry:
        """Return a defensive copy of the engine's rule registry."""
        return list(self._rules)

    def decide(self, report: CoverageReport) -> PriorityDecision:
        """
        Choose the next Objective for this CoverageReport.

        Algorithm (stateless, deterministic, no branching on field names):

        1. If ``report.all_complete`` -> ``WRAP_UP`` (confidence 1.0).
        2. Otherwise, compute the candidate rules ordered by priority
           (descending) with declared-enum-order as the tiebreaker
           (:func:`module3.rules.ordered_candidates`).
        3. If only one candidate -> return it (confidence 1.0).
        4. If more than one candidate shared the winning priority (a
           "tie group" at the head of the ordered list), choose the
           first (deterministic declared-enum tiebreak, confidence 0.8).

        The Confidence drop signals ambiguity to downstream callers
        (Objective Engine records reasoning strings reflecting the tie).

        Returns
        -------
        PriorityDecision
            The chosen objective plus all candidate metadata the
            Objective Engine needs to compose ConversationObjective.

        Raises
        ------
        TypeError
            If ``report`` is not a CoverageReport.
        RuntimeError
            If the report is not all-complete but no rule matched. With a
            valid rule registry this is unreachable — every gap covered
            by a rule will have a candidate. The error surfaces
            misconfiguration rather than silently returning WRAP_UP.
        """
        if not isinstance(report, CoverageReport):
            raise TypeError(
                f"PriorityEngine.decide requires a CoverageReport, "
                f"got {type(report).__name__}"
            )

        reasoning: List[str] = []
        total_fields: int = len(REQUIRED_FIELDS)

        # 1. All-complete -> WRAP_UP. Derived condition, not a rule.
        if report.all_complete:
            reasoning.append("All required Empathize fields are populated.")
            reasoning.append(
                f"{report.num_complete}/{total_fields} fields covered; "
                f"selecting WRAP_UP."
            )
            return PriorityDecision(
                objective=Objective.WRAP_UP,
                winning_rule=None,
                tied_rules=[],
                confidence=PriorityEngine.CONFIDENCE_UNIQUE,
                reasoning=reasoning,
            )

        # 2. Otherwise, evaluate the candidate rules via declared data.
        candidates = ordered_candidates(self._rules, report)
        # ordered_candidates returns (rule, index) pairs; project to just
        # the rules themselves, preserving the already-sorted order.
        ordered_rules: List[Rule] = [r for (r, _idx) in candidates]
        if not ordered_rules:
            raise RuntimeError(
                "No candidate rules matched the missing fields. The rule "
                "registry does not cover every required field — please add "
                "a rule for each StateField you want the engine to chase."
            )

        # 3/4. Determine whether the head of the ordered list is solitarily
        # the winner or tied with neighbours on priority.
        winner: Rule = ordered_rules[0]
        head_priority: int = winner.priority
        tied: List[Rule] = [r for r in ordered_rules if r.priority == head_priority]

        if len(tied) == 1:
            reasoning.append(
                f"Selected Objective.{winner.objective.name} (priority "
                f"{winner.priority}) from a single unmet candidate."
            )
            return PriorityDecision(
                objective=winner.objective,
                winning_rule=winner,
                tied_rules=[],
                confidence=PriorityEngine.CONFIDENCE_UNIQUE,
                reasoning=reasoning,
            )

        # 4b. Priority tie among multiple rules -> break by declared enum
        # order (the ordering of `ordered_rules` already applied that
        # tiebreak), record the tie as reduced confidence.
        tied_names = ", ".join(
            f"Objective.{r.objective.name} (priority {r.priority}, "
            f"StateField.{r.field.name})"
            for r in tied
        )
        reasoning.append(
            f"Priority tie among {len(tied)} candidate rules: {tied_names}."
        )
        reasoning.append(
            f"Tiebreak: chose Objective.{winner.objective.name} (declared "
            f"StateField enum order)."
        )
        # Confidence is reduced only for ties at the head priority — i.e.
        # "multiple equally valid candidates". Note this is the canonical
        # band 0.8 from the spec ("minor ambiguity").
        return PriorityDecision(
            objective=winner.objective,
            winning_rule=winner,
            tied_rules=tied,
            confidence=PriorityEngine.CONFIDENCE_TIEBREAK,
            reasoning=reasoning,
        )
