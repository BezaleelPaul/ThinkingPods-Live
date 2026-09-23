"""
module3.sufficiency — objective sufficiency classification for Empathize fields.

Responsibility
--------------
Determine, for every required Empathize field, how *usable* the information
captured in ``ProjectState`` actually is. The binary :class:`CoverageReport`
answers one question ("is this field populated?"); sufficiency answers a
richer one:

    "Is this field Unknown, Partial, Sufficient, or Complete?"

The four levels let the mentor behave more conversationally:

* ``UNKNOWN``    — no information captured yet. The objective should stay
                   active and the mentor asks for it fresh.
* ``PARTIAL``    — some signal exists but it is not a usable answer (e.g. a
                   frequency value with no cadence). Flagged in diagnostics
                   so the mentor can clarify instead of assuming.
* ``SUFFICIENT`` — a usable, real-world answer has been captured (a natural
                   cadence phrase like "every single day", or any concrete
                   list item). The objective can advance: the mentor does NOT
                   need to re-ask even though "perfect quantitative evidence"
                   (e.g. "3 times a week") was not provided.
* ``COMPLETE``   — the answer exceeds the minimum bar (a quantitative
                   frequency, or multiple distinct list items).

Design constraints
------------------
* **Read-only.** Sufficiency NEVER mutates ``ProjectState``; Module 2
  (StateManager) remains the sole mutation path.
* **No lifecycle impact.** Coverage (and therefore WRAP_UP / objective
  selection) stays binary. Sufficiency is a classification layer consumed by
  diagnostics and reasoning — it cannot block WRAP_UP or reorder the
  canonical Empathize flow.
* **Deterministic.** Same ``ProjectState`` → same levels, always. No LLM.
* **Frequency-aware but application-owned.** The cadence/quantitative
  vocabulary below lives HERE (the application owns the thinking), not in
  any prompt.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List, Optional

from memory_extractor import (
    LIST_FIELDS,
    ProjectState,
    REQUIRED_FIELDS,
    StateField,
)

__all__ = [
    "SufficiencyLevel",
    "SufficiencyReport",
    "SufficiencyChecker",
]


# ---------------------------------------------------------------------------
# SufficiencyLevel
# ---------------------------------------------------------------------------


class SufficiencyLevel(str, Enum):
    """How usable a captured field's information is."""

    UNKNOWN = "UNKNOWN"
    """No information captured — objective stays active and the mentor asks
    for it from scratch."""

    PARTIAL = "PARTIAL"
    """Some signal exists but it is not a usable answer (e.g. a frequency
    value that conveys no cadence). Diagnostics flag it so the mentor can
    clarify rather than assume or re-ask."""

    SUFFICIENT = "SUFFICIENT"
    """A usable, real-world answer exists. The objective advances even if
    perfect quantitative evidence has not been provided (a natural cadence
    phrase like 'every single day' is enough)."""

    COMPLETE = "COMPLETE"
    """The answer exceeds the minimum bar (quantitative frequency, multiple
    distinct list items)."""


# ---------------------------------------------------------------------------
# Frequency vocabulary (application-owned, deterministic)
# ---------------------------------------------------------------------------

# Phrases that convey a usable cadence in natural language. "every single
# day" is SUFFICIENT even though it is qualitative — the mentor does not need
# to re-ask for a precise count.
_CADENCE_TOKENS: tuple[str, ...] = (
    "daily", "weekly", "monthly", "yearly", "annually", "hourly", "nightly",
    "fortnightly",
    "every day", "every single day", "each day", "every week", "each week",
    "every month", "every morning", "every evening", "every night",
    "every afternoon", "every weekend", "every weekday",
    "all the time", "most days", "most of the time", "most weeks",
    "on most days",
    "often", "sometimes", "always", "constantly", "regularly", "frequently",
    "occasionally", "rarely", "seldom", "usually", "normally", "typically",
    "never",
    "once in a while", "from time to time", "every so often", "now and then",
    "at times",
)

# Markers of a *quantitative* frequency — precise evidence, i.e. COMPLETE.
_QUANTITATIVE_TOKENS: tuple[str, ...] = (
    "times", "per day", "per week", "per month", "per year", "per hour",
    "once a day", "once a week", "once a month", "twice a day", "twice a week",
    "twice a month", "a day", "a week", "a month", "an hour", "a year",
    "couple of", "few times", "several times",
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
)


def _matches_any(text: str, tokens: tuple[str, ...]) -> bool:
    """Word-boundary match, so 'times' never matches inside 'sometimes' and
    'never' never matches inside 'whenever'."""
    lowered = text.lower()
    for token in tokens:
        if re.search(r"\b" + re.escape(token) + r"\b", lowered):
            return True
    return False


def _has_digits(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


# ---------------------------------------------------------------------------
# SufficiencyReport
# ---------------------------------------------------------------------------


class SufficiencyReport:
    """
    Per-field sufficiency snapshot for the required Empathize fields.

    Immutable-by-convention (constructed from a frozen dict, no public
    mutators). Convenience accessors mirror the CoverageReport surface so
    downstream code can treat sufficiency and coverage uniformly.
    """

    __slots__ = ("_levels",)

    def __init__(self, levels: Dict[StateField, SufficiencyLevel]) -> None:
        # Only REQUIRED_FIELDS participate in sufficiency; supporting
        # knowledge (impacts) is deliberately excluded.
        self._levels: Dict[StateField, SufficiencyLevel] = {
            sf: levels.get(sf, SufficiencyLevel.UNKNOWN)
            for sf in REQUIRED_FIELDS
        }

    def level_of(self, field: StateField) -> SufficiencyLevel:
        """Return the sufficiency level for ``field`` (UNKNOWN if absent)."""
        return self._levels.get(field, SufficiencyLevel.UNKNOWN)

    def is_satisfied(self, field: StateField) -> bool:
        """True iff the field has a usable answer (SUFFICIENT or COMPLETE).

        This is the advancement threshold: once a field is satisfied the
        mentor does not need to pursue it further."""
        return self.level_of(field) in (
            SufficiencyLevel.SUFFICIENT,
            SufficiencyLevel.COMPLETE,
        )

    @property
    def satisfied_fields(self) -> List[StateField]:
        """Required fields at SUFFICIENT or COMPLETE, declared order."""
        return [sf for sf in REQUIRED_FIELDS if self.is_satisfied(sf)]

    @property
    def active_fields(self) -> List[StateField]:
        """Required fields still at UNKNOWN or PARTIAL, declared order."""
        return [sf for sf in REQUIRED_FIELDS if not self.is_satisfied(sf)]

    def as_dict(self) -> dict[str, str]:
        """Plain-dict view: ``{field_value: level_value}``, declared order."""
        return {sf.value: self.level_of(sf).value for sf in REQUIRED_FIELDS}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SufficiencyReport):
            return NotImplemented
        return self._levels == other._levels

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SufficiencyReport({self.as_dict()!r})"


# ---------------------------------------------------------------------------
# SufficiencyChecker
# ---------------------------------------------------------------------------


class SufficiencyChecker:
    """
    Stateless, deterministic classifier of ProjectState field sufficiency.

    Mirrors :class:`~module3.completeness_checker.CompletenessChecker`: a
    frozen, value-only view over ProjectState. The same state always yields
    the same :class:`SufficiencyReport`.
    """

    @staticmethod
    def evaluate(state: ProjectState) -> SufficiencyReport:
        """Classify every required field of ``state``."""
        if not isinstance(state, ProjectState):
            raise TypeError(
                f"SufficiencyChecker.evaluate requires a ProjectState, "
                f"got {type(state).__name__}"
            )
        levels: Dict[StateField, SufficiencyLevel] = {}
        for sf in REQUIRED_FIELDS:
            levels[sf] = SufficiencyChecker._classify(state, sf)
        return SufficiencyReport(levels=levels)

    @staticmethod
    def evaluate_dict(state_dict: dict) -> SufficiencyReport:
        """Classify a ``to_state_dict()``-shaped dict (missing keys → UNKNOWN).

        Used by diagnostics to classify a state snapshot without requiring a
        live ProjectState instance."""
        levels: Dict[StateField, SufficiencyLevel] = {}
        for sf in REQUIRED_FIELDS:
            raw = state_dict.get(sf.value)
            levels[sf] = SufficiencyChecker._classify_value(sf, raw)
        return SufficiencyReport(levels=levels)

    @staticmethod
    def _classify(state: ProjectState, sf: StateField) -> SufficiencyLevel:
        if sf in LIST_FIELDS:
            try:
                return SufficiencyChecker._classify_list(state.get_list(sf))
            except TypeError:  # pragma: no cover - defensive
                return SufficiencyLevel.UNKNOWN
        if sf is StateField.FREQUENCY:
            return SufficiencyChecker._classify_frequency(state.frequency)
        return SufficiencyLevel.UNKNOWN  # pragma: no cover - defensive

    @staticmethod
    def _classify_value(sf: StateField, raw) -> SufficiencyLevel:
        if sf in LIST_FIELDS:
            values = raw if isinstance(raw, (list, tuple)) else []
            return SufficiencyChecker._classify_list(
                [v for v in values if isinstance(v, str)]
            )
        if sf is StateField.FREQUENCY:
            return SufficiencyChecker._classify_frequency(raw)
        return SufficiencyLevel.UNKNOWN  # pragma: no cover - defensive

    @staticmethod
    def _classify_list(items: List[str]) -> SufficiencyLevel:
        populated = [i for i in (items or []) if isinstance(i, str) and i.strip()]
        if not populated:
            return SufficiencyLevel.UNKNOWN
        if len(populated) >= 2:
            return SufficiencyLevel.COMPLETE
        return SufficiencyLevel.SUFFICIENT

    @staticmethod
    def _classify_frequency(value: Optional[str]) -> SufficiencyLevel:
        if not value or not str(value).strip():
            return SufficiencyLevel.UNKNOWN
        text = str(value).strip()
        if _has_digits(text) or _matches_any(text, _QUANTITATIVE_TOKENS):
            return SufficiencyLevel.COMPLETE
        if _matches_any(text, _CADENCE_TOKENS):
            return SufficiencyLevel.SUFFICIENT
        return SufficiencyLevel.PARTIAL
