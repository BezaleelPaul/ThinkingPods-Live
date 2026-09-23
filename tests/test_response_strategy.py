"""
tests/test_response_strategy.py

Unit tests for module4.response_strategy (Module 4 — Response Strategy).

Covers:
  - ResponseStrategy enum membership and extensibility hooks
  - Deterministic Objective -> ResponseStrategy mapping for every
    Empathize objective (PERSONAS / PROBLEMS / CURRENT_SOLUTIONS /
    PAIN_POINTS / EVIDENCE / FREQUENCY / WRAP_UP)
  - Two-engine-instances agree (stateless determinism)
  - Same (objective, project_state) -> same strategy across calls
  - Predicate helpers is_ask_strategy / is_summary_strategy
  - strategy_for_objective helper
  - Read-only contract (ProjectState is not mutated)
  - Type rejection (non-ConversationObjective, non-ProjectState)
  - ProjectState is consumed by the engine's public signature
"""

import sys
import os
import unittest

# Allow importing from the project root regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import ProjectState, REQUIRED_FIELDS, StateField  # noqa: E402

from module3 import ConversationObjective, Objective, ObjectiveEngine  # noqa: E402
from module4 import ResponseStrategy, ResponseStrategyEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _empty_state() -> ProjectState:
    return ProjectState()


def _full_state() -> ProjectState:
    return ProjectState(
        personas=["x"],
        problems=["y"],
        current_solutions=["z"],
        pain_points=["q"],
        evidence=["w"],
        frequency="daily",
    )


def _state_missing(field: StateField) -> ProjectState:
    """Build a state where `field` is the ONLY missing field."""
    ps = _full_state()
    if field in (StateField.FREQUENCY,):
        ps.frequency = None
    else:
        setattr(ps, field.value, [])
    return ps


def _objective_for(state: ProjectState) -> ConversationObjective:
    """Use the real Objective Engine to produce a ConversationObjective,
    so the test scenarios mirror the live Module 3 -> Module 4 wiring."""
    return ObjectiveEngine().determine_next(state)


# ---------------------------------------------------------------------------
# ResponseStrategy enum membership / extensibility
# ---------------------------------------------------------------------------


class TestResponseStrategyEnum(unittest.TestCase):

    def test_enum_has_required_members(self):
        for name in ("ASK_QUESTION", "GENERATE_SUMMARY",
                     "ACKNOWLEDGE", "CLARIFY"):
            self.assertTrue(hasattr(ResponseStrategy, name),
                            f"ResponseStrategy must support {name}")

    def test_enum_values_are_unique_strings(self):
        values = {s.value for s in ResponseStrategy}
        self.assertEqual(len(values), len(list(ResponseStrategy)))

    def test_ask_question_is_str(self):
        self.assertEqual(ResponseStrategy.ASK_QUESTION.value, "ASK_QUESTION")

    def test_generate_summary_is_str(self):
        self.assertEqual(ResponseStrategy.GENERATE_SUMMARY.value, "GENERATE_SUMMARY")


# ---------------------------------------------------------------------------
# Objective -> Strategy mapping (the canonical Empathize mapping)
# ---------------------------------------------------------------------------


class TestStrategyMapping(unittest.TestCase):
    """The application owns the objective->strategy translation; the LLM
    never chooses."""

    def test_field_objectives_yield_ask_question(self):
        """Every Empathize required-field objective maps to ASK_QUESTION."""
        engine = ResponseStrategyEngine()
        for sf in REQUIRED_FIELDS:
            # Build a state where sf is the only missing field -> the
            # Objective Engine returns its Objective.
            state = _state_missing(sf)
            obj = _objective_for(state)
            self.assertEqual(obj.objective, Objective.for_field(sf))
            strat = engine.determine_strategy(obj, state)
            self.assertEqual(strat, ResponseStrategy.ASK_QUESTION,
                             f"Objective {obj.objective.name} must map to ASK_QUESTION")

    def test_wrap_up_objective_yields_generate_summary(self):
        """All Empathize fields populated -> WRAP_UP -> GENERATE_SUMMARY."""
        engine = ResponseStrategyEngine()
        obj = _objective_for(_full_state())
        self.assertEqual(obj.objective, Objective.WRAP_UP)
        strat = engine.determine_strategy(obj, _full_state())
        self.assertEqual(strat, ResponseStrategy.GENERATE_SUMMARY)

    def test_empty_state_personas_to_ask_question(self):
        """Empty state -> PERSONAS -> ASK_QUESTION (canonical entry)."""
        engine = ResponseStrategyEngine()
        state = _empty_state()
        obj = _objective_for(state)
        self.assertEqual(obj.objective, Objective.PERSONAS)
        strat = engine.determine_strategy(obj, state)
        self.assertEqual(strat, ResponseStrategy.ASK_QUESTION)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):

    def test_engine_called_twice_same_state_same_strategy(self):
        engine = ResponseStrategyEngine()
        state = _state_missing(StateField.EVIDENCE)
        obj = _objective_for(state)
        a = engine.determine_strategy(obj, state)
        b = engine.determine_strategy(obj, state)
        self.assertEqual(a, b)

    def test_two_engine_instances_same_input_same_strategy(self):
        state = _state_missing(StateField.PAIN_POINTS)
        obj = _objective_for(state)
        a = ResponseStrategyEngine().determine_strategy(obj, state)
        b = ResponseStrategyEngine().determine_strategy(obj, state)
        self.assertEqual(a, b)

    def test_wrap_up_strategy_stable_across_calls(self):
        engine = ResponseStrategyEngine()
        obj = _objective_for(_full_state())
        a = engine.determine_strategy(obj, _full_state())
        b = engine.determine_strategy(obj, _full_state())
        self.assertEqual(a, b)
        self.assertEqual(a, ResponseStrategy.GENERATE_SUMMARY)


# ---------------------------------------------------------------------------
# Predicate helpers
# ---------------------------------------------------------------------------


class TestPredicateHelpers(unittest.TestCase):

    def test_is_ask_strategy_true_for_ask(self):
        self.assertTrue(
            ResponseStrategyEngine.is_ask_strategy(ResponseStrategy.ASK_QUESTION)
        )

    def test_is_ask_strategy_false_for_summary(self):
        self.assertFalse(
            ResponseStrategyEngine.is_ask_strategy(ResponseStrategy.GENERATE_SUMMARY)
        )

    def test_is_summary_strategy_true_for_summary(self):
        self.assertTrue(
            ResponseStrategyEngine.is_summary_strategy(ResponseStrategy.GENERATE_SUMMARY)
        )

    def test_is_summary_strategy_false_for_ask(self):
        self.assertFalse(
            ResponseStrategyEngine.is_summary_strategy(ResponseStrategy.ASK_QUESTION)
        )


# ---------------------------------------------------------------------------
# strategy_for_objective helper
# ---------------------------------------------------------------------------


class TestStrategyForObjectiveHelper(unittest.TestCase):

    def test_strategy_for_every_field_objective_returns_ask(self):
        for sf in StateField:
            self.assertEqual(
                ResponseStrategyEngine.strategy_for_objective(Objective.for_field(sf)),
                ResponseStrategy.ASK_QUESTION,
            )

    def test_strategy_for_wrapup_returns_summary(self):
        self.assertEqual(
            ResponseStrategyEngine.strategy_for_objective(Objective.WRAP_UP),
            ResponseStrategy.GENERATE_SUMMARY,
        )


# ---------------------------------------------------------------------------
# Read-only contract — ProjectState is not mutated
# ---------------------------------------------------------------------------


class TestReadOnly(unittest.TestCase):

    def test_engine_does_not_mutate_project_state_lists(self):
        engine = ResponseStrategyEngine()
        state = ProjectState(personas=["x"], problems=["y"])
        before = list(state.personas)
        obj = _objective_for(state)
        engine.determine_strategy(obj, state)
        self.assertEqual(state.personas, before)

    def test_engine_does_not_mutate_project_state_scalar(self):
        engine = ResponseStrategyEngine()
        state = _full_state()
        obj = _objective_for(state)
        engine.determine_strategy(obj, state)
        self.assertEqual(state.frequency, "daily")

    def test_engine_does_not_mutate_empty_state(self):
        engine = ResponseStrategyEngine()
        state = _empty_state()
        obj = _objective_for(state)
        engine.determine_strategy(obj, state)
        for sf in StateField:
            if sf.value == "frequency":
                self.assertIsNone(getattr(state, sf.value))
            else:
                self.assertEqual(getattr(state, sf.value), [])


# ---------------------------------------------------------------------------
# Type rejections (Module 4 is a strict ProjectState boundary)
# ---------------------------------------------------------------------------


class TestTypeRejection(unittest.TestCase):

    def test_rejects_non_conversation_objective(self):
        engine = ResponseStrategyEngine()
        with self.assertRaises(TypeError):
            engine.determine_strategy(
                Objective.PERSONAS,  # bare enum, not a ConversationObjective
                _empty_state(),
            )

    def test_rejects_non_project_state(self):
        engine = ResponseStrategyEngine()
        obj = _objective_for(_empty_state())
        with self.assertRaises(TypeError):
            engine.determine_strategy(
                obj,
                {"personas": []},  # plain dict is NOT accepted
            )

    def test_rejects_none_objective(self):
        engine = ResponseStrategyEngine()
        with self.assertRaises(TypeError):
            engine.determine_strategy(None, _empty_state())

    def test_rejects_none_state(self):
        engine = ResponseStrategyEngine()
        obj = _objective_for(_empty_state())
        with self.assertRaises(TypeError):
            engine.determine_strategy(obj, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
