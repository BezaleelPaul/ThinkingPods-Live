"""
module5 — Lifecycle Manager + Summary Builder for the Empathize v2 pipeline.

Architecture role
-----------------
::

    User
        ↓
    Memory Extractor (Module 1)
        ↓
    State Manager (Module 2)
        ↓
    Objective Engine (Module 3)
        ↓
    **Lifecycle Manager (Module 5)**   <-- this package
        ↓
    [Summary Builder (Module 5)         <-- only when READY_FOR_SUMMARY
        ↓                                 builds an EmpathizeSummary]
    **EmpathizeSummary (Module 5)**    <-- structured summary data
        ↓
    Response Strategy Engine (Module 4)
        ↓
    Prompt Builder (Module 4)
        ↓
    LLM

Module 5 owns TWO responsibilities:

1. **Lifecycle Manager** — decides the stage lifecycle
   (:class:`LifecycleDecision`) for the current turn. The decision drives
   the rest of the pipeline: CONTINUE resumes gathering; READY_FOR_SUMMARY
   instructs the integrator to build an :class:`EmpathizeSummary` and force
   ``ResponseStrategy.GENERATE_SUMMARY``; WAITING_FOR_CONFIRMATION holds
   the summary open pending an explicit user confirmation; READY_FOR_TRANSITION
   tells the StageController to perform the stage transition.

2. **Summary Builder** — copies :class:`memory_extractor.ProjectState` into
   a structured, immutable :class:`EmpathizeSummary`. The Summary Builder
   NEVER reads conversation history; it copies ProjectState exactly, with
   no hallucinated values and no natural language. The Prompt Builder
   (Module 4) is the only place that turns the summary into a prompt.

Module 5 does NOT:
  - ask questions,
  - build prompts,
  - update ProjectState,
  - call the LLM,
  - decide stage transitions itself (it returns READY_FOR_TRANSITION; the
    StageController performs the actual stage change),
  - perform multi-stage progression beyond Empathize (future work).

Determinism
-----------
Every public component here is a pure function of its inputs: no LLM, no
randomness, no I/O, no state mutation. ``process_mentor_turn`` can reuse
one process-wide instance of each engine safely.

Cross-turn bookkeeping (the WAITING → READY_FOR_TRANSITION transition)
flows ONLY through ProjectState's existing ``previous_assistant_message``
and ``previous_user_message`` slots (rolled forward by Module 2 every
turn). See ``module5.lifecycle_manager``'s module docstring for the
lifecycle-token protocol.

Public interface
----------------
>>> from module5 import (
...     EmpathizeSummary,
...     LifecycleDecision,
...     LifecycleManager,
...     SummaryBuilder,
...     is_confirmation_message,
...     summary_presented_marker,
... )
"""

from __future__ import annotations

from .lifecycle_manager import (
    LifecycleDecision,
    LifecycleManager,
    is_confirmation_message,
    is_summary_presented_marker,
    summary_presented_marker,
)
from .summary import EmpathizeSummary
from .summary_builder import SummaryBuilder

__all__ = [
    "EmpathizeSummary",
    "LifecycleDecision",
    "LifecycleManager",
    "SummaryBuilder",
    "is_confirmation_message",
    "is_summary_presented_marker",
    "summary_presented_marker",
]
