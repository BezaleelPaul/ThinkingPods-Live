"""
module5.lifecycle_manager — decide the lifecycle state of the Empathize stage.

Architecture role
-----------------
::

    Objective Engine (Module 3)
        ↓
    ConversationObjective
        ↓
    Response Strategy Engine (Module 4)  ← may run in parallel as hint
        ↓
    **Lifecycle Manager (Module 5)**   <-- this module
        ↓
    LifecycleDecision
        ↓
    (back in process_mentor_turn)
       - CONTINUE             → continue gathering (use ResponseStrategy as-is)
       - READY_FOR_SUMMARY    → build EmpathizeSummary + GENERATE_SUMMARY
       - WAITING_FOR_CONFIRMATION → wait for explicit user confirmation
       - READY_FOR_TRANSITION → StageController performs the stage transition

Module 5 sits between the Objective Engine and the Response Strategy
Engine (the spec: Module 5 is what makes the choice of GENERATE_SUMMARY
a *lifecycle* event rather than a one-shot derived flag).

Responsibilities (from the spec)
--------------------------------
The Lifecycle Manager ONLY determines the stage lifecycle. It does NOT:

  * ask questions,
  * build prompts,
  * update ProjectState,
  * call the LLM.

Determinism
-----------
``determine()`` is a pure function of its three arguments: same inputs
-> same LifecycleDecision. No LLM, no randomness, no I/O.

Cross-turn state
----------------
Two of the four possible decisions (WAITING_FOR_CONFIRMATION,
READY_FOR_TRANSITION) depend on cross-turn state — "has the assistant
presented a summary in the previous turn?", "did the user just confirm
the summary?". This state is read from the two bookkeeping slots
``ProjectState.previous_assistant_message`` and
``ProjectState.previous_user_message`` that Module 2 already rolls
forward every turn. No other state source is consulted — the spec
requires ``determine()`` take ONLY ``(ProjectState, ConversationObjective,
ResponseStrategy)``.

The application (``process_mentor_turn``) tags the ProjectState's
``previous_assistant_message`` slot with a deterministic lifecycle
token immediately after an Empathize summary has been emitted, so the
next turn's Lifecycle Manager can distinguish "we presented a summary
last turn" from "we asked a question last turn". The natural-language
reply stored on the session's ``raw_history`` is unaffected by this
tagging; the LLM never sees the token. See the constants below and
``mentor.process_mentor_turn``'s integration step.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from memory_extractor import ProjectState

from module3 import ConversationObjective

# Avoid importing ResponseStrategy at module load in a way that creates
# a hard cross-package dependency at import time — module5 depends on
# module4's enum *value* only for the type-check below. Done lazily is
# unnecessary; module4 has no dependents on module5, so the import is acyclic.
from module4 import ResponseStrategy

__all__ = ["LifecycleDecision", "LifecycleManager"]


# ---------------------------------------------------------------------------
# LifecycleDecision enum
# ---------------------------------------------------------------------------


class LifecycleDecision(str, Enum):
    """
    The Empathize stage's lifecycle state for THIS turn.

    Members
    -------
    CONTINUE:
        The discovery phase is still open — keep gathering information
        using the ResponseStrategyEngine's chosen strategy.
    READY_FOR_SUMMARY:
        The Objective Engine has decided all required fields are gathered
        (``WRAP_UP``) AND no summary has been presented yet this stage.
        The turn MUST build an EmpathizeSummary and produce a summary
        reply (ResponseStrategy = GENERATE_SUMMARY).
    WAITING_FOR_CONFIRMATION:
        An Empathize summary was presented in a previous turn and the user
        has NOT yet explicitly confirmed it. The conversation should hold
        the summary open and ask for confirmation/refinement.
    READY_FOR_TRANSITION:
        The user has explicitly confirmed the presented summary. The
        StageController may now perform the Empathize → next-stage
        transition. Application-owned, never LLM-owned.

    The enum is a ``str`` subclass so it serialises cleanly (session JSON,
    logs) and round-trips through ``==`` without callers parsing strings.
    """

    CONTINUE = "CONTINUE"
    READY_FOR_SUMMARY = "READY_FOR_SUMMARY"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"
    READY_FOR_TRANSITION = "READY_FOR_TRANSITION"


# ---------------------------------------------------------------------------
# Lifecycle tokens kept on ProjectState.previous_* slots (NOT user-visible)
# ---------------------------------------------------------------------------
#
# These constants are THE public contract between mentor.process_mentor_turn
# (the integrator) and module5.LifecycleManager. They live here (in Module 5)
# because Module 5 owns the lifecycle meaning; the integrator installs them
# into ProjectState's previous-message slots per Module 5's protocol.
#
# Tokens are chosen to be (a) deterministic, (b) extremely unlikely to ever
# appear in a real user's natural-language reply or an LLM's natural-language
# reply, and (c) short — they are bookkeeping, not content.

_SUMMARY_PRESENTED_TOKEN = "<<empathize_summary_presented>>"

# Deterministic confirmation lexicon — the user's previous message must
# match this regex (case-insensitive) to count as an explicit confirmation.
# Kept deliberately narrow: the spec explicitly says "The LLM should NEVER
# decide this." So the application's rule is strict and short — any message
# that does not match falls back to WAITING_FOR_CONFIRMATION, which keeps
# the summary open for refinement.
_CONFIRMATION_PATTERN = re.compile(
    r"""^\s*(
          yes                          |
          yep                          |
          yeah                        |
          yea                         |
          correct                     |
          exactly                     |
          right                       |
          true                        |
          confirm(?:ed)?              |
          affirmative                 |
          absolutely                  |
          spot[\s-]?on                |
          looks[\s-]?good             |
          that(?:'s|\sis)\sright      |
          thats\sright                |
          sounds[\s-]?good            |
          go[\s-]?ahead               |
          proceed                     |
          y\b
        )
        [.!?\s]*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_confirmation_message(message: Optional[str]) -> bool:
    """
    Return True iff ``message`` is an explicit, deterministic confirmation
    per the Lifecycle Manager's lexicon. Pure function; no LLM.
    """
    if not message:
        return False
    return _CONFIRMATION_PATTERN.match(message.strip()) is not None


def is_summary_presented_marker(message: Optional[str]) -> bool:
    """
    Return True iff ``message`` carries the "summary was presented last
    turn" lifecycle token installed by the integrator.
    """
    if not message:
        return False
    return message.startswith(_SUMMARY_PRESENTED_TOKEN)


def summary_presented_marker(reply: Optional[str]) -> Optional[str]:
    """
    Construct the marker the Lifecycle Manager expects to find inside
    ``ProjectState.previous_assistant_message`` after a summary reply was
    emitted last turn.

    The marker is ``_SUMMARY_PRESENTED_TOKEN`` ALONE — it is intentionally
    NOT prepended to the natural-language reply, so the LLM never sees it.
    The integrator writes the marker into the ProjectState slot (NOT into
    the user-facing ``raw_history``) and writes the natural-language reply
    into the user-facing ``raw_history`` slot. Thus the two channels stay
    cleanly separated.
    """
    return _SUMMARY_PRESENTED_TOKEN


# ---------------------------------------------------------------------------
# LifecycleManager
# ---------------------------------------------------------------------------


class LifecycleManager:
    """
    Stateless, deterministic Empathize lifecycle decider.

    The single public entry point is :meth:`determine`. It reads ONLY the
    three spec-mandated inputs (ProjectState + ConversationObjective +
    ResponseStrategy) and returns a :class:`LifecycleDecision`, using
    ProjectState's previous-message slots as the cross-turn lifecycle
    channel (see module docstring).

    Why is ``ResponseStrategy`` an input when the rules below only use the
    objective? ------------------------------------------------------------

    The spec mandates the input for forward compatibility: future stages
    (Define/Ideate/...) and future strategy values (ACKNOWLEDGE / CLARIFY)
    will carry lifecycle signals the manager will need to consider. Today
    the Empathize decision is purely a function of the objective plus the
    cross-turn markers; the strategy argument is accepted, type-validated,
    and otherwise unused, kept in the signature so callers can rely on it
    staying stable as Module 5 grows.
    """

    def determine(
        self,
        project_state: ProjectState,
        conversation_objective: ConversationObjective,
        response_strategy: ResponseStrategy,
    ) -> LifecycleDecision:
        """
        Compute the Empathize lifecycle state for this turn.

        Parameters
        ----------
        project_state:
            Canonical :class:`memory_extractor.ProjectState`. Read-only.
            The Lifecycle Manager inspects ``previous_assistant_message``
            and ``previous_user_message`` for cross-turn lifecycle markers
            (see module docstring); it does NOT inspect content fields.
        conversation_objective:
            The :class:`module3.ConversationObjective` produced by
            :class:`module3.ObjectiveEngine` for this turn.
        response_strategy:
            The :class:`module4.ResponseStrategy` produced by
            :class:`module4.ResponseStrategyEngine` for this turn. Accepted
            for forward compatibility (see class docstring) and
            type-validated; not consulted by today's rules.

        Returns
        -------
        LifecycleDecision
            CONTINUE / READY_FOR_SUMMARY / WAITING_FOR_CONFIRMATION /
            READY_FOR_TRANSITION.

        Raises
        ------
        TypeError
            If any input is not of the declared type. The Manager is a
            strict boundary.
        """
        if not isinstance(project_state, ProjectState):
            raise TypeError(
                f"determine requires a ProjectState, "
                f"got {type(project_state).__name__}"
            )
        if not isinstance(conversation_objective, ConversationObjective):
            raise TypeError(
                f"determine requires a ConversationObjective, "
                f"got {type(conversation_objective).__name__}"
            )
        if not isinstance(response_strategy, ResponseStrategy):
            raise TypeError(
                f"determine requires a ResponseStrategy, "
                f"got {type(response_strategy).__name__}"
            )

        # --- Rule 1: discovery still open ----------------------------------
        if not conversation_objective.is_wrap_up:
            return LifecycleDecision.CONTINUE

        # From here on, the Objective Engine has decided all Empathize
        # fields are gathered (WRAP_UP). The lifecycle decision now
        # depends on the cross-turn markers.

        prev_assistant = project_state.previous_assistant_message
        prev_user = project_state.previous_user_message

        # --- Rule 4: summary presented + user explicitly confirmed -------
        if is_summary_presented_marker(prev_assistant) and \
                is_confirmation_message(prev_user):
            return LifecycleDecision.READY_FOR_TRANSITION

        # --- Rule 3: summary presented, user has NOT confirmed yet -------
        if is_summary_presented_marker(prev_assistant):
            return LifecycleDecision.WAITING_FOR_CONFIRMATION

        # --- Rule 2: WRAP_UP fresh — no summary presented yet ------------
        return LifecycleDecision.READY_FOR_SUMMARY

    # ------------------------------------------------------------------
    # Predictability helpers used by tests and the integrator.
    # ------------------------------------------------------------------

    @staticmethod
    def is_terminal(decision: LifecycleDecision) -> bool:
        """True iff ``decision`` indicates the Empathize stage is over."""
        return decision is LifecycleDecision.READY_FOR_TRANSITION

    @staticmethod
    def expects_summary(decision: LifecycleDecision) -> bool:
        """True iff ``decision`` requires an EmpathizeSummary build this turn."""
        return decision is LifecycleDecision.READY_FOR_SUMMARY

    @staticmethod
    def is_open_for_confirmation(decision: LifecycleDecision) -> bool:
        """True iff ``decision`` is in the confirmation-wait sub-state."""
        return decision is LifecycleDecision.WAITING_FOR_CONFIRMATION
