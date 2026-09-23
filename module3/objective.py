"""
module3.objective — pure data models for the Objective Engine.

This module contains ONLY data structures:

- ``Objective``           — enum of conversational objectives the engine may
                            return.
- ``CoverageReport``      — per-field boolean coverage snapshot of
                            ProjectState.
- ``ConversationObjective`` — the structured engine output (objective +
                            reasoning + confidence + completed/missing field
                            lists).

No business logic lives here. Construction, validation, and rule evaluation
are owned by :mod:`module3.completeness_checker`,
:mod:`module3.priority_engine`, and :mod:`module3.objective_engine`.

The models are deliberately simple and dependency-free so Module 4 (the Prompt
Builder) and downstream stages can consume them without dragging in any
Object-Engine internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from memory_extractor import REQUIRED_FIELDS, StateField

__all__ = [
    "Objective",
    "CoverageReport",
    "ConversationObjective",
]


# ---------------------------------------------------------------------------
# Objective enum
# ---------------------------------------------------------------------------


class Objective(str, Enum):
    """
    The next conversational objective the mentor should pursue.

    Each Empathize-relevant objective corresponds to a single
    ``StateField`` the engine wants to populate next. The trailing
    ``WRAP_UP`` value is a sentinel representing the "all required fields
    gathered — move to summary / close-out" state.

    The enum is a ``str`` subclass so it serialises naturally (to JSON,
    logs, response headers, etc.) and round-trips cleanly through ``==``
    comparisons in Module 4 without the Prompt Builder having to parse
    strings.

    Future stages (Define, Ideate, Prototype, Validation, ...) will extend
    this enum with new objectives — they MUST NOT reuse existing values.
    """

    PERSONAS = "PERSONAS"
    """Gather the target audience / user group for the project."""

    PROBLEMS = "PROBLEMS"
    """Gather the core problem or challenge being experienced."""

    CURRENT_SOLUTIONS = "CURRENT_SOLUTIONS"
    """Gather how people currently handle the problem."""

    PAIN_POINTS = "PAIN_POINTS"
    """Gather frustrations, impact, and why it hurts."""

    EVIDENCE = "EVIDENCE"
    """Gather observations, data, or research validating the problem."""

    IMPACTS = "IMPACTS"
    """Gather consequences / effects of the problem on those affected.

    Supporting-knowledge objective: no default rule targets this field, so
    the engine never selects it during Empathize discovery. It exists to
    keep the StateField ↔ Objective 1:1 invariant complete."""

    FREQUENCY = "FREQUENCY"
    """Gather how often the problem occurs."""

    WRAP_UP = "WRAP_UP"
    """All required Empathize fields are populated — produce the empathy summary."""

    @classmethod
    def for_field(cls, sf: StateField) -> "Objective":
        """
        Return the objective that targets populating ``sf``.

        Each Empathize ``StateField`` corresponds one-to-one with an
        ``Objective`` (same name, by design). This mapping is kept explicit
        in a single place so callers never need to construct the enum by
        string parsing.

        Raises
        ------
        ValueError
            If ``sf`` is not one of the Empathize fields this Objective
            enum targets (defence-in-depth — should be unreachable in
            practice because the engine only feeds it values from
            ``StateField``).
        """
        try:
            return cls(sf.name)
        except ValueError as exc:
            raise ValueError(
                f"No Objective maps to StateField {sf!r}"
            ) from exc


# ---------------------------------------------------------------------------
# CoverageReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageReport:
    """
    Boolean coverage snapshot of a ProjectState's required fields.

    Each field is ``True`` when sufficiently populated (a non-empty list for
    the five LIST_FIELDS state fields, or a non-``None``/non-empty scalar
    for ``frequency``), and ``False`` otherwise.

    This is a *fact report* only. It carries no priority, no reasoning,
    and no notion of "next objective" — selecting which missing field to
    pursue is the :class:`~module3.priority_engine.PriorityEngine`'s job.

    Frozen to make snapshot semantics explicit: once a report is built
    it cannot be sneakily mutated by a downstream component, and a
    coverage report is safely hashable for use as an assert/compare token
    in tests.
    """

    personas: bool = False
    problems: bool = False
    current_solutions: bool = False
    pain_points: bool = False
    evidence: bool = False
    frequency: bool = False

    # ---- convenience accessors ------------------------------------------

    def is_field_complete(self, sf: StateField) -> bool:
        """Return the coverage boolean for a given StateField."""
        return getattr(self, sf.value)

    @property
    def completed_fields(self) -> List[StateField]:
        """Return the list of completed required StateFields, declared order."""
        return [sf for sf in REQUIRED_FIELDS if getattr(self, sf.value)]

    @property
    def missing_fields(self) -> List[StateField]:
        """Return the list of missing required StateFields, declared order."""
        return [sf for sf in REQUIRED_FIELDS if not getattr(self, sf.value)]

    @property
    def all_complete(self) -> bool:
        """True iff every required Empathize field is covered."""
        return all(getattr(self, sf.value) for sf in REQUIRED_FIELDS)

    @property
    def num_complete(self) -> int:
        """Count of covered required fields (0..6)."""
        return sum(1 for sf in REQUIRED_FIELDS if getattr(self, sf.value))

    def as_dict(self) -> dict:
        """Return a plain-dict view keyed by field name (for logging/testing)."""
        return {sf.value: getattr(self, sf.value) for sf in REQUIRED_FIELDS}


# ---------------------------------------------------------------------------
# ConversationObjective — the single Objective Engine output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConversationObjective:
    """
    Structured output of the Objective Engine for one ProjectState snapshot.

    The Prompt Builder (Module 4) consumes this object directly. Specifying
    the next objective as a structured object (not a string) keeps
    conversational-flow decisions in application code rather than in LLM
    prompts.

    Fields
    ------
    objective:
        The selected next objective. ``WRAP_UP`` indicates all required
        Empathize fields are populated.
    missing_fields:
        Fields the CoverageReport identified as unpopulated at this turn.
        Includes the field the objective targets — but is the full list, so
        Module 4 can surface "we already know X / Y, just need Z" context
        in the prompt.
    completed_fields:
        Fields considered covered at this turn. Kept alongside
        ``missing_fields`` so Module 4 can avoid re-asking for known
        information without re-querying ProjectState directly.
    confidence:
        Deterministic confidence in the chosen objective, in ``[0.0, 1.0]``.
        NOT an LLM confidence — it is product of the priority rules'
        clarity:

          * ``1.0`` — exactly one rule matched with no ties, the canonical
            "all-complete" wrap-up, or the canonical "first missing field"
            determinism tiebreak.
          * ``0.8`` — a tie was broken by declared enum order (>1 candidate
            shared the same priority and the order tiebreak chose the first).
          * lower values indicate more ambiguous rule sets (extensible per
            future stages, but the same input always yields the same
            value, so this is NOT a probability).

    reasoning:
        Ordered list of human-readable strings explaining why this
        objective was chosen. The first entry states the trigger, the rest
        state supporting facts. Useful for debugging, analytics, and
        deterministic test assertions. Not shown to the user; the Prompt
        Builder decides what to surface.
    sufficiency:
        Per-required-field sufficiency levels (``{field_value: level_value}``,
        e.g. ``{"frequency": "SUFFICIENT"}``). Serialised from
        ``SufficiencyReport.as_dict()`` so diagnostics can show exactly how
        usable each field's information is.
    advancement:
        The engine's verdict for this turn: ``"WRAP_UP"`` (all required
        fields populated), ``"ADVANCED"`` (the targeted field already holds
        sufficient information, so the mentor moves on), or
        ``"STAYED_ACTIVE"`` (the field is UNKNOWN/PARTIAL and the mentor
        keeps pursuing it).
    advancement_reason:
        Human-readable explanation of the advancement verdict — why the
        objective advanced or why it stayed active.
    selection_trace:
        Optional Developer-Console payload from the information-gain scorer:
        the ranked candidates with their component scores, the selected
        objective, the reason it was selected, and the rejected candidates
        with their reasons. ``None`` for the WRAP_UP case (and for any
        hand-built ConversationObjective) — backward compatible with
        downstream consumers that only read the core fields.

    Frozen to enforce the role of Objective Engine output as an immutable
    decision record for the turn.
    """

    objective: Objective
    missing_fields: List[StateField]
    completed_fields: List[StateField]
    confidence: float = 1.0
    reasoning: List[str] = field(default_factory=list)
    # Sufficiency / advancement (see module3.sufficiency): serializable
    # per-field levels plus the engine's verdict on whether the objective
    # advanced, stayed active, or wrapped up, and a human-readable reason.
    # Defaults keep hand-built ConversationObjective constructions working.
    sufficiency: dict = field(default_factory=dict)
    advancement: str = "STAYED_ACTIVE"
    advancement_reason: str = ""
    selection_trace: Optional[dict] = None

    # ---- post-init normalisation ----------------------------------------

    _ADVANCEMENT_VALUES = ("WRAP_UP", "ADVANCED", "STAYED_ACTIVE")

    def __post_init__(self) -> None:
        """
        Validate the structural invariants the Objective Engine guarantees
        to downstream modules.

        Raises deterministically (no silent recovery) so any caller that
        constructs a ConversationObjective outside the engine — or a bug
        inside it — surfaces immediately rather than corrupting a prompt.
        """
        if not isinstance(self.objective, Objective):
            raise TypeError(
                f"objective must be an Objective, got {type(self.objective).__name__}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must lie in [0.0, 1.0], got {self.confidence!r}"
            )
        if self.advancement not in self._ADVANCEMENT_VALUES:
            raise ValueError(
                f"advancement must be one of {self._ADVANCEMENT_VALUES}, "
                f"got {self.advancement!r}"
            )

    # ---- convenience accessors ------------------------------------------

    @property
    def is_wrap_up(self) -> bool:
        """True iff the engine has decided all Empathize fields are gathered."""
        return self.objective is Objective.WRAP_UP

    def targeted_field(self) -> Optional[StateField]:
        """
        Return the single StateField this objective targets, or ``None``
        for ``WRAP_UP`` (which does not target any one field).
        """
        if self.objective is Objective.WRAP_UP:
            return None
        # For Empathize field-objectives, the Objective name matches the
        # StateField name by design (see Objective.for_field).
        return StateField[self.objective.name]
