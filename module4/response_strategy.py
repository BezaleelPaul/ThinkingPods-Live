"""
module4.response_strategy — convert a ConversationObjective into a
ResponseStrategy.

Architecture role
-----------------
::

    Objective Engine (Module 3)
        ↓
    ConversationObjective
        ↓
    **Response Strategy Engine (Module 4)**   <-- this module
        ↓
    ResponseStrategy
        ↓
    Prompt Builder (Module 4)
        ↓
    LLM

Module 4 owns ONLY the decision of *how* to respond. It does NOT decide
what information is missing (Module 3), does NOT update memory
(Module 1), does NOT validate state (Module 2), and does NOT perform
lifecycle decisions (Module 5).

Determinism
-----------
The strategy engine is a pure function of
``(ConversationObjective, ProjectState)``: no LLM,
no randomness, no I/O. The same inputs ALWAYS yield the same
``ResponseStrategy``. This keeps *how to respond* in application code;
the LLM only produces natural language from the resulting prompt.

Extensibility
-------------
The ``ResponseStrategy`` enum is intentionally open: future Design
Thinking stages (and lifecycle hooks from Module 5) can add new values
without touching the existing ones.
"""

from __future__ import annotations

from typing import Optional

from contracts import ResponseStrategy
from memory_extractor import ProjectState

from module3 import ConversationObjective, Objective

__all__ = ["ResponseStrategy", "ResponseStrategyEngine"]


# ---------------------------------------------------------------------------
# Objective -> Strategy mapping (deterministic, extensible)
# ---------------------------------------------------------------------------

_OBJ_TO_STRATEGY: dict[str, ResponseStrategy] = {
    "PERSONAS": ResponseStrategy.ASK_QUESTION,
    "PROBLEMS": ResponseStrategy.ASK_QUESTION,
    "CURRENT_SOLUTIONS": ResponseStrategy.ASK_QUESTION,
    "PAIN_POINTS": ResponseStrategy.ASK_QUESTION,
    "EVIDENCE": ResponseStrategy.ASK_QUESTION,
    "IMPACTS": ResponseStrategy.ASK_QUESTION,
    "FREQUENCY": ResponseStrategy.ASK_QUESTION,
    "WRAP_UP": ResponseStrategy.GENERATE_SUMMARY,
}





class ResponseStrategyEngine:
    """
    Deterministic translator of ``ConversationObjective`` into
    ``ResponseStrategy``.

    The engine is stateless: it stores no per-turn state, performs no
    I/O, never calls the LLM, and never mutates the ``ProjectState`` it
    is handed. The same inputs ALWAYS produce the same ``ResponseStrategy``.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def determine_strategy(
        self,
        objective: "ConversationObjective",
        project_state: "ProjectState",
    ) -> ResponseStrategy:
        """Map a Module 3 ConversationObjective to a ResponseStrategy."""
        if not isinstance(objective, ConversationObjective):
            raise TypeError(
                f"determine_strategy requires a ConversationObjective, "
                f"got {type(objective).__name__}"
            )
        if not isinstance(project_state, ProjectState):
            raise TypeError(
                f"determine_strategy requires a ProjectState, "
                f"got {type(project_state).__name__}"
            )

        return _OBJ_TO_STRATEGY.get(
            objective.objective.value, ResponseStrategy.ASK_QUESTION
        )

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @staticmethod
    def is_summary_strategy(strategy: ResponseStrategy) -> bool:
        """True iff ``strategy`` is the summary-producing strategy."""
        return strategy is ResponseStrategy.GENERATE_SUMMARY

    @staticmethod
    def is_ask_strategy(strategy: ResponseStrategy) -> bool:
        """True iff ``strategy`` is a question-producing strategy."""
        return strategy is ResponseStrategy.ASK_QUESTION

    @staticmethod
    def strategy_for_objective(
        objective_value: Objective,
    ) -> Optional[ResponseStrategy]:
        """
        Pure lookup helper — returns the registered strategy for an
        Objective enum value, or ``None`` if none is registered. Used by
        tests and tooling; ``determine_strategy`` is the live entry point.
        """
        return _OBJ_TO_STRATEGY.get(objective_value.value)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_RESPONSE_STRATEGY_ENGINE: "ResponseStrategyEngine" | None = None


def get_response_strategy_engine() -> "ResponseStrategyEngine":
    """Return the process-wide ResponseStrategyEngine instance."""
    global _RESPONSE_STRATEGY_ENGINE
    if _RESPONSE_STRATEGY_ENGINE is None:
        _RESPONSE_STRATEGY_ENGINE = ResponseStrategyEngine()
    return _RESPONSE_STRATEGY_ENGINE