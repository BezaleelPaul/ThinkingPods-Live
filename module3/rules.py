"""
module3.rules — declarative priority rules for the Objective Engine.

Why declarative rules?
----------------------
The :class:`~module3.priority_engine.PriorityEngine` evaluates a list of
:class:`Rule` records to decide the next objective. Rules are pure data:

    Rule(field=<which StateField this rule targets>,
         priority=<higher = more important>,
         objective=<which Objective winning this rule yields>)

Designing the engine around a rule registry — instead of a nested chain of
``if`` statements — keeps the engine free of business logic. Adding a new
conversational objective is a data edit, not a code edit:

    # When you add EVIDENCE in a future Empathize stage, add a Rule.
    DEFAULT_RULES.append(Rule(StateField.EVIDENCE, priority=30,
                              objective=Objective.EVIDENCE))

The :class:`PriorityEngine` only iterates rules and picks the highest-priority
unmet one. It owns no knowledge of which field matters most.

Default rule ordering for the Empathize stage
---------------------------------------------
The Empathize conversation has a canonical discovery order that mirrors how a
human mentor would naturally lead a discovery interview:

    1. personas          (100) — who are we even talking about?
    2. problems          (90) — what's the core challenge they face?
    3. frequency         (80) — how often does it actually happen?
    4. current_solutions (70) — how are they coping today?
    5. pain_points       (60) — why does it matter / how does it hurt?
    6. evidence         (50) — what data backs this up?

Higher priority numbers win. The exact numbers are arbitrary but their
relative ordering is intentional and explicit. To re-prioritise the Empathize
flow, edit this registry — never the engine.

Every entry targets exactly one StateField and one Objective. The engine
also handles the implicit WRAP_UP condition (when ALL rule fields are covered);
that is not a Rule because it is a derived global condition rather than a
per-field gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from memory_extractor import StateField

from .objective import Objective

__all__ = ["Rule", "DEFAULT_RULES", "validate_rules"]


# ---------------------------------------------------------------------------
# Rule dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """
    A single declarative priority rule for the Objective Engine.

    Semantics: "If the CoverageReport says ``field`` is missing, this rule
    is a candidate. Its candidate priority is ``priority`` (higher wins);
    if it wins, the engine returns ``objective``."

    Attributes
    ----------
    field:
        The StateField whose coverage gap this rule reacts to. MUST be one
        of the Empathize state fields (PERSONAS, PROBLEMS,
        CURRENT_SOLUTIONS, PAIN_POINTS, EVIDENCE, FREQUENCY).
    priority:
        An integer priority. Higher priorities win. Ties are resolved by
        declared StateField enum order (stable, deterministic).
    objective:
        The Objective the engine should return when this rule wins.

    Frozen so a rule registry cannot be mutated in place by accident; it
    can be appended to (a new list / a new entry) but each Rule itself is
    immutable.
    """

    field: StateField
    priority: int
    objective: Objective

    def __post_init__(self) -> None:
        """Structural invariants — fail loud rather than emit a bad rule."""
        if not isinstance(self.field, StateField):
            raise TypeError(
                f"Rule.field must be a StateField, "
                f"got {type(self.field).__name__}"
            )
        if not isinstance(self.priority, int) or isinstance(self.priority, bool):
            raise TypeError(
                f"Rule.priority must be an int (not a bool); got "
                f"{self.priority!r}"
            )
        if not isinstance(self.objective, Objective):
            raise TypeError(
                f"Rule.objective must be an Objective, "
                f"got {type(self.objective).__name__}"
            )
        # The Objective name must correspond to the targeted StateField —
        # otherwise a rule book could mismatch Objective and StateField,
        # which would let the engine emit a misleading
        # ConversationObjective.
        if self.objective is not Objective.WRAP_UP:
            expected_objective = Objective.for_field(self.field)
            if expected_objective is not self.objective:
                raise ValueError(
                    f"Rule.objective {self.objective!r} does not match "
                    f"StateField {self.field!r} (expected {expected_objective!r})"
                )


# A rule registry is an ordered list of Rules keyed by targeted field.
# Note: Module-3 rules are keyed by FIELD not by Objective — a single field
# can only have one winning rule per turn, so duplicate field entries
# would be ambiguous and are rejected by validate_rules().
RuleRegistry = List[Rule]


def validate_rules(rules: RuleRegistry) -> None:
    """
    Type-check and validate a rule registry deterministically.

    Raises
    ------
    ValueError
        If two rules target the same StateField (ambiguous priority) or a
        rule has the sentinel ``WRAP_UP`` objective (which is derived, not
        a field-gap rule).
    TypeError
        If any element is not a Rule instance.
    """
    if not isinstance(rules, list):
        raise TypeError(f"rules must be a list, got {type(rules).__name__}")

    seen_fields: dict[StateField, Rule] = {}
    for i, r in enumerate(rules):
        if not isinstance(r, Rule):
            raise TypeError(
                f"rules[{i}] is not a Rule (got {type(r).__name__})"
            )
        if r.objective is Objective.WRAP_UP:
            raise ValueError(
                f"rules[{i}]: WRAP_UP must not be a Rule field-gap entry; "
                f"WRAP_UP is derived from the all-complete condition."
            )
        if r.field in seen_fields:
            raise ValueError(
                f"Duplicate rule for StateField {r.field!r}: "
                f"rules target fields, and one field may only have one "
                f"rule (previous rule had priority "
                f"{seen_fields[r.field].priority})"
            )
        seen_fields[r.field] = r


# ---------------------------------------------------------------------------
# DEFAULT_RULES — the Empathize priority order (see module docstring)
# ---------------------------------------------------------------------------


DEFAULT_RULES: RuleRegistry = [
    Rule(field=StateField.PERSONAS,          priority=100, objective=Objective.PERSONAS),
    Rule(field=StateField.PROBLEMS,          priority=90,  objective=Objective.PROBLEMS),
    Rule(field=StateField.FREQUENCY,         priority=80,  objective=Objective.FREQUENCY),
    Rule(field=StateField.CURRENT_SOLUTIONS, priority=70,  objective=Objective.CURRENT_SOLUTIONS),
    Rule(field=StateField.PAIN_POINTS,       priority=60,  objective=Objective.PAIN_POINTS),
    Rule(field=StateField.EVIDENCE,          priority=50,  objective=Objective.EVIDENCE),
]
"""
Default Empathize rule registry.

The ordering is:

    personas (100) > problems (90) > frequency (80) >
    current_solutions (70) > pain_points (60) > evidence (50)

Higher priority numbers are preferred. To re-prioritise the Empathize
conversation, edit this list rather than modifying the engine. To support
new stages, create a second registry — the engine accepts any registry
via its ``rules`` constructor argument, so multiple rule sets can co-exist.
"""

# Validate the default registry once at import time so a bad edit there
# fails loudly at process start rather than silently producing bad
# objectives at runtime.
validate_rules(DEFAULT_RULES)


def rule_for_field(rules: RuleRegistry, sf: StateField) -> Rule | None:
    """Return the rule targeting ``sf`` in ``rules``, or ``None`` if none."""
    for r in rules:
        if r.field is sf:
            return r
    return None


def ordered_candidates(
    rules: RuleRegistry,
    report,
) -> List[Tuple[Rule, int]]:
    """
    Return the candidate rules whose targeted field is missing in ``report``.

    Sorted by (priority DESC, declared StateField enum order ASC). The
    declared-enum tiebreaker is what makes the engine deterministic when
    two rules share a priority: the StateField declared first in the enum
    wins. (Future stages with intentional ties can rely on this rule.)

    Returns
    -------
    list of (Rule, index_in_returned_list)
        The index is included so callers can detect "this was the only
        candidate" vs "this was the first of N candidates" without
        recounting.

    Implementation note: sorting on a derived (priority, enum-order) key
        rather than on the rule itself guarantees stable ordering across
        Python versions regardless of Rule's dataclass ordering.
    """
    # StateField declared-numerical order is the deterministic tiebreaker.
    enum_order = {sf: i for i, sf in enumerate(StateField)}
    unmet = [r for r in rules if not report.is_field_complete(r.field)]
    unmet.sort(key=lambda r: (-r.priority, enum_order.get(r.field, 0)))
    return [(r, i) for i, r in enumerate(unmet)]
