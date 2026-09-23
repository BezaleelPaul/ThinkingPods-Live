"""
module5.summary_builder — copy a ProjectState into an EmpathizeSummary.

Architecture role
-----------------
::

    ProjectState (canonical, owned by StateManager)
        ↓
    **Summary Builder (Module 5)**   <-- this module
        ↓
    EmpathizeSummary (structured, immutable)
        ↓
    Prompt Builder (Module 4) [GENERATE_SUMMARY path]

The Summary Builder NEVER reads conversation history. It ONLY reads
``ProjectState``. The application owns the summary structure; the
Prompt Builder (Module 4) is the only place that turns it into a prompt;
the LLM turns the prompt into prose.

Determinism
-----------
``SummaryBuilder.build(project_state)`` is a pure function of its
argument: same input -> identical (and ``==``) ``EmpathizeSummary``.
No LLM, no I/O, no randomness, no clock.

Rules (per the Module 5 spec)
-----------------------------
    Never invent information.
    Preserve ordering.
    Copy ProjectState exactly.
    No formatting.
    No natural language.

"Copy exactly" is implemented as a deep copy of every list field (so a
caller mutating the source state cannot retroactively change a built
summary) plus a verbatim copy of the scalar.
"""

from __future__ import annotations

__all__ = ["SummaryBuilder"]


from .summary import EmpathizeSummary


class SummaryBuilder:
    """
    Stateless, deterministic Empathize Summary Builder.

    The single public entry point is :meth:`build`. It reads only the
    extraction-relevant fields of a ``ProjectState`` and returns a frozen
    :class:`EmpathizeSummary`. It performs NO business logic beyond the
    "copy each field defensively" step — ordering, deduplication, and
    value normalisation are all owned elsewhere (ProjectState / Module 2).

    Read-only contract
    -------------------
    ``build`` never mutates the supplied ``ProjectState``. The defensive
    copies it makes are reader-side only — each copied list is a NEW
    list object the Summary owns after this call.

    Why a class and not a free function?
    -------------------------------------
    To mirror :class:`module4.PromptBuilder` and
    :class:`module3.ObjectiveEngine` stylistically — Module 5's Builder
    surface is a small, single-staticmethod class so tests and callers
    that want to swap an instance can. The class holds no state.
    """

    @staticmethod
    def build(project_state) -> EmpathizeSummary:
        """
        Construct an :class:`EmpathizeSummary` from ``project_state``.

        Parameters
        ----------
        project_state:
            A :class:`memory_extractor.ProjectState` instance. The
            builder reads it via the public ``personas / problems /
            current_solutions / pain_points / evidence / frequency``
            attributes — it does NOT call ``get_list`` / ``get_scalar``
            (those raise on misuse); instead it relies on the canonical
            shape ProjectState always exposes, even for partially-built
            states.

        Returns
        -------
        EmpathizeSummary
            Frozen, structured snapshot with copy-exactly semantics.

        Raises
        ------
        TypeError
            If ``project_state`` is not a ProjectState. The Builder is
            a strict ProjectState boundary — no MentorSession
            substitutes accepted.
        """
        # Local import avoids a circular dependency at module import time
        # (module5 has no other reason to import memory_extractor at
        # every module load).
        from memory_extractor import ProjectState

        if not isinstance(project_state, ProjectState):
            raise TypeError(
                f"SummaryBuilder.build requires a ProjectState, "
                f"got {type(project_state).__name__}"
            )

        # Defensive copy of every list-typed field — a built summary must
        # never be retroactively mutated by edits to the source state.
        # ProjectState guarantees list-typed fields are lists; we copy
        # via list() to preserve element identity/ordering without an
        # expensive deepcopy (the elements themselves are immutable str).
        return EmpathizeSummary(
            personas=list(project_state.personas),
            problems=list(project_state.problems),
            current_solutions=list(project_state.current_solutions),
            pain_points=list(project_state.pain_points),
            evidence=list(project_state.evidence),
            # Scalar: verbatim copy. An Optional[str] is immutable — no
            # defensive copy needed.
            frequency=project_state.frequency,
        )
