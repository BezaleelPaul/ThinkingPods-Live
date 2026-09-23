"""
module5.summary — pure structured model for the Empathize summary.

Architecture role
-----------------
::

    ProjectState (Module 1/2)
        ↓
    Summary Builder (Module 5)
        ↓
    **EmpathizeSummary (Module 5)**   <-- this module
        ↓
    Prompt Builder (Module 4)  ← used as the "state payload" for GENERATE_SUMMARY

``EmpathizeSummary`` is the structured snapshot the Summary Builder emits
and the Prompt Builder consumes. It is deliberately NOT natural language;
the LLM turns it into prose. The application owns the structure, the LLM
owns the words.

Design constraints (from the Module 5 spec)
--------------------------------------------
    Never invent information.
    Preserve ordering.
    Copy ProjectState exactly.
    No formatting.
    No natural language.

The fields mirror ``ProjectState``'s six extraction-relevant fields by
design: list fields stay lists (insertion-ordered), frequency stays an
``Optional[str]`` scalar. ``EmpathizeSummary`` exists as a separate type
(rather than reusing ProjectState verbatim) for two reasons:

1.  It is the *physical* communication of "the Empathize summary has been
    built by Module 5" — once an ``EmpathizeSummary`` exists, downstream
    code can be assured all required Empathize fields are offered for
    summarisation in the canonical declared order, without re-querying
    ProjectState for "what fields exist".
2.  It is frozen. ``ProjectState`` is a mutable application-owned data
    container; ``EmpathizeSummary`` is a transport-of-record snapshot that
    persists unchanged across the rest of a turn and across turn boundaries
    (the application stores the summary on the session while waiting for
    user confirmation). A frozen dataclass makes that contract explicit and
    hashable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["EmpathizeSummary"]


# ---------------------------------------------------------------------------
# EmpathizeSummary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmpathizeSummary:
    """
    Structured, immutable Empathize summary built from a ProjectState by
    Module 5's Summary Builder.

    Fields
    ------
    personas, problems, current_solutions, pain_points, evidence:
        Insertion-ordered lists copied verbatim from the source
        ``ProjectState`` list fields. ``EmpathizeSummary`` makes its OWN
        copies at construction time so later mutation of the source
        state cannot retroactively change a built summary.
    frequency:
        The Empathize scalar (how often the problem occurs), or
        ``None`` when the source state had no value. Copied verbatim.

    Frozen to enforce the "role of the summary as an immutable record"
    invariant: once a summary has been built and stored on a session
    pending user confirmation, no caller may mutate it in place. The
    application owns lifecycle; the summary is treated as evidence of
    what the application decided to summarise.

    Immutability note
    -----------------
    Python's ``@dataclass(frozen=True)`` only prevents attribute
    rebinding. The list-typed fields below are still *mutable* at their
    element level unless we copy-and-freeze them. To honour the spec's
    "no hallucinated values / preserve ordering / copy exactly" rules
    we therefore construct each list field as a defensive copy of the
    Source state's list. (Tuples would be stricter, but list round-trips
    cleanly through JSON session persistence — see ``SummaryBuilder``.)
    """

    personas: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    current_solutions: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    frequency: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Normalise defensively: coerce any ``None`` placeholder for a list
        field to an empty list, and defer the rest of the "I must be a
        defensive copy" invariant to the Builder (which is the only
        code path that constructs an EmpathizeSummary in practice).
        """
        # ``dataclass(frozen=True)`` does NOT allow attribute assignment
        # in __post_init__ via self.x = ... — use object.__setattr__ to
        # install the normalised value once.
        for fname in (
            "personas", "problems", "current_solutions",
            "pain_points", "evidence",
        ):
            v = getattr(self, fname)
            if v is None:
                object.__setattr__(self, fname, [])
            elif not isinstance(v, list):
                # Defensive: coerce tuples (and other iterables accepted by
                # mistake) into lists so the type contract is honest.
                object.__setattr__(self, fname, list(v))
