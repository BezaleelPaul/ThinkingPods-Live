"""
module3.completeness_checker — read-only ProjectState coverage evaluation.

Responsibility
--------------
Given a ``ProjectState``, decide for each required Empathize field whether
it is *sufficiently populated* and emit a :class:`CoverageReport`.

This component is deliberately minimal and free of prioritization or
conversational-flow logic. It only answers one question per field:

    "Does ProjectState have something non-trivial for this field yet?"

What "sufficiently populated" means (deterministic, no LLM/no heuristics):

- For a LIST_FIELDS field (personas, problems, current_solutions,
  pain_points, evidence): the list contains at least one non-empty
  string after stripping.
- For the SCALAR_FIELDS field (frequency): the scalar is not ``None`` and
  is non-empty after stripping.

Note: only ``REQUIRED_FIELDS`` gate the CoverageReport. ``impacts`` is
captured as supporting knowledge (see ``NON_REQUIRED_FIELDS``) and is
deliberately excluded from coverage so it never influences objective
selection or WRAP_UP.

Read-only contract
------------------
This module MUST NOT mutate the input ``ProjectState``. The
``CoverageReport`` it returns is a frozen, value-only snapshot. Downstream
components can freely share, cache, or re-evaluate a state and always
obtain the same report.
"""

from __future__ import annotations

from memory_extractor import ProjectState, REQUIRED_FIELDS, StateField

from .objective import CoverageReport

__all__ = ["CompletenessChecker"]


def _is_list_populated(state: ProjectState, sf: StateField) -> bool:
    """
    Return True iff the list-typed state field for ``sf`` contains at
    least one non-empty stripped string.

    Defensive against malformed runtime values: treats any non-list value
    as "not populated" rather than raising — the canonical ProjectState
    always exposes lists for LIST_FIELDS fields, so this branch is
    unreachable in normal operation but keeps the checker total.
    """
    try:
        lst = state.get_list(sf)
    except TypeError:
        return False
    if not isinstance(lst, (list, tuple)):
        return False
    for item in lst:
        if isinstance(item, str) and item.strip():
            return True
    return False


def _is_scalar_populated(state: ProjectState, sf: StateField) -> bool:
    """
    Return True iff the scalar-typed state field for ``sf`` contains a
    non-empty stripped string.

    ``None``, empty string, and whitespace-only values all count as "not
    populated". This matches ProjectState.set_scalar's normalisation
    (which converts stripped-empty to ``None``), so the two never disagree
    about a state's coverage.
    """
    try:
        v = state.get_scalar(sf)
    except TypeError:
        return False
    return isinstance(v, str) and bool(v.strip())


class CompletenessChecker:
    """
    Stateless, deterministic evaluator of ProjectState coverage.

    All logic is exposed as static methods so the checker can be used as a
    free function (``CompletenessChecker.evaluate(state)``) without
    instantiation. No instance state, no caches, no dependency injection
    surface — the same ``ProjectState`` ALWAYS produces the same
    ``CoverageReport``.
    """

    @staticmethod
    def evaluate(state: ProjectState) -> CoverageReport:
        """
        Build a :class:`CoverageReport` from ``state``.

        Parameters
        ----------
        state:
            A canonical :class:`memory_extractor.ProjectState` instance.
            The checker reads only — never writes. Legacy ``MentorSession``
            is NOT accepted; Module 3 is the cut-over point at which
            ProjectState becomes the canonical input.

        Returns
        -------
        CoverageReport
            A frozen snapshot of per-field coverage.

        Raises
        ------
        TypeError
            If ``state`` is not a ``ProjectState``. The engine never
            silently accepts a substitute; the spec forbids any
            ``MentorSession`` or compatibility-bridge usage here.
        """
        if not isinstance(state, ProjectState):
            raise TypeError(
                f"CompletenessChecker.evaluate requires a ProjectState, "
                f"got {type(state).__name__}"
            )

        # Evaluating one field at a time keeps the function small and lets
        # future stages (Define/Ideate/...) reuse the same checker pipeline
        # by extending the StateField enum and the coverage primitives.
        # Only REQUIRED_FIELDS gate Empathize completion; NON_REQUIRED_FIELDS
        # (e.g. impacts) are supporting knowledge and never appear here.
        kwargs: dict[str, bool] = {}
        for sf in REQUIRED_FIELDS:
            from memory_extractor import LIST_FIELDS, SCALAR_FIELDS
            if sf in LIST_FIELDS:
                kwargs[sf.value] = _is_list_populated(state, sf)
            elif sf in SCALAR_FIELDS:
                kwargs[sf.value] = _is_scalar_populated(state, sf)
            else:  # pragma: no cover - defensive: every enum member is one of the two
                kwargs[sf.value] = False

        return CoverageReport(**kwargs)

    @staticmethod
    def for_empty_state() -> CoverageReport:
        """
        Return the coverage report for a freshly-initialised ProjectState.

        Convenience alias for :meth:`evaluate` on a default ProjectState.
        Useful in tests and as a stable ground-truth for "nothing has been
        captured yet" without constructing a ProjectState at the call site.
        """
        return CompletenessChecker.evaluate(ProjectState())
