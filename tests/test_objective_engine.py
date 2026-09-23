"""
tests/test_objective_engine.py

Unit tests for module3.objective_engine.ObjectiveEngine.

Covers (per the Module 3 spec test categories):
  - empty state
  - one missing field
  - multiple missing fields
  - completed Empathize stage -> WRAP_UP
  - deterministic output (identical input -> identical output)
  - read-only contract (project state must not be mutated)

Also covers:
  - the ConversationObjective dataclass invariants (objectives, confidence,
    reasoning lists, missing/completed fields)
  - the targeted_field() convenience accessor
  - the Objective.for_field() mapping invariant
  - custom rule registry injection (extensibility)
"""

import sys
import os
import unittest
from dataclasses import FrozenInstanceError

# Allow importing from the project root regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import ProjectState, REQUIRED_FIELDS, StateField

from module3 import (
    DEFAULT_RULES,
    ConversationObjective,
    Objective,
    ObjectiveEngine,
    Rule,
)
from module3.objective import CoverageReport


# ---------------------------------------------------------------------------
# Fixture helpers — declarative state construction.
# ---------------------------------------------------------------------------

def _empty_state() -> ProjectState:
    return ProjectState()


def _state_only(**kwargs) -> ProjectState:
    """Build a state populated with the given field->value(s) (lists for list fields)."""
    return ProjectState(**kwargs)


def _full_state() -> ProjectState:
    return ProjectState(
        personas=["Students"],
        problems=["Buried messages"],
        current_solutions=["WhatsApp groups"],
        pain_points=["Overwhelmed"],
        evidence=["Saw students struggle"],
        frequency="Weekly",
    )


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


class TestObjectiveEngine_EmptyState(unittest.TestCase):

    def test_empty_state_yields_personas_objective(self):
        obj = ObjectiveEngine().determine_next(_empty_state())
        self.assertEqual(obj.objective, Objective.PERSONAS)

    def test_empty_state_has_all_fields_missing(self):
        obj = ObjectiveEngine().determine_next(_empty_state())
        self.assertEqual(obj.missing_fields, list(REQUIRED_FIELDS))
        self.assertEqual(obj.completed_fields, [])

    def test_empty_state_targeted_field_is_personas(self):
        obj = ObjectiveEngine().determine_next(_empty_state())
        self.assertEqual(obj.targeted_field(), StateField.PERSONAS)

    def test_empty_state_confidence_is_unique(self):
        obj = ObjectiveEngine().determine_next(_empty_state())
        self.assertEqual(obj.confidence, 1.0)

    def test_empty_state_reasoning_includes_coverage_summary(self):
        obj = ObjectiveEngine().determine_next(_empty_state())
        joined = " ".join(obj.reasoning)
        # Macro summary is the first reasoning entry.
        self.assertTrue(obj.reasoning[0].startswith("Coverage: 0/6"))
        # And a "Missing fields" line enumerating them all.
        self.assertIn("Missing fields:", joined)

    def test_empty_state_returns_conversation_objective_type(self):
        obj = ObjectiveEngine().determine_next(_empty_state())
        self.assertIsInstance(obj, ConversationObjective)


# ---------------------------------------------------------------------------
# One missing field
# ---------------------------------------------------------------------------


class TestObjectiveEngine_OneMissingField(unittest.TestCase):

    def test_one_missing_list_field_yields_its_objective(self):
        # Everything but pain_points covered.
        state = ProjectState(
            personas=["x"],
            problems=["x"],
            current_solutions=["x"],
            pain_points=[],       # the one missing field
            evidence=["x"],
            frequency="Weekly",
        )
        obj = ObjectiveEngine().determine_next(state)
        self.assertEqual(obj.objective, Objective.PAIN_POINTS)
        self.assertEqual(obj.missing_fields, [StateField.PAIN_POINTS])
        self.assertEqual(len(obj.completed_fields), 5)

    def test_one_missing_scalar_field_yields_frequency_objective(self):
        state = ProjectState(
            personas=["x"],
            problems=["x"],
            current_solutions=["x"],
            pain_points=["x"],
            evidence=["x"],
            frequency=None,    # the one missing field
        )
        obj = ObjectiveEngine().determine_next(state)
        self.assertEqual(obj.objective, Objective.FREQUENCY)
        self.assertEqual(obj.missing_fields, [StateField.FREQUENCY])
        self.assertEqual(len(obj.completed_fields), 5)

    def test_one_missing_field_confidence_is_unique(self):
        state = ProjectState(
            personas=["x"], problems=["x"], current_solutions=["x"],
            pain_points=["x"], evidence=["x"], frequency=None,
        )
        obj = ObjectiveEngine().determine_next(state)
        self.assertEqual(obj.confidence, 1.0)

    def test_one_missing_reasoning_records_completion(self):
        state = ProjectState(
            personas=["x"], problems=["x"], current_solutions=["x"],
            pain_points=["x"], evidence=["x"], frequency=None,
        )
        obj = ObjectiveEngine().determine_next(state)
        joined = " ".join(obj.reasoning)
        # Coverage summary reports 5/6 done.
        self.assertIn("Coverage: 5/6", joined)
        # Completed fields enumerated.
        self.assertIn("Completed fields:", joined)


# ---------------------------------------------------------------------------
# Multiple missing fields
# ---------------------------------------------------------------------------


class TestObjectiveEngine_MultipleMissingFields(unittest.TestCase):

    def test_two_missing_returns_higher_priority_field(self):
        # Both pain_points (60) and evidence (50) missing — engine must
        # pick pain_points.
        state = ProjectState(
            personas=["x"],
            problems=["x"],
            current_solutions=["x"],
            pain_points=[],     # missing
            evidence=[],        # missing
            frequency="Weekly",
        )
        obj = ObjectiveEngine().determine_next(state)
        self.assertEqual(obj.objective, Objective.PAIN_POINTS)

    def test_multiple_missing_lists_both_in_missing_fields(self):
        state = ProjectState(
            personas=["x"],
            problems=[],
            current_solutions=[],
            pain_points=["x"],
            evidence=["x"],
            frequency="Weekly",
        )
        obj = ObjectiveEngine().determine_next(state)
        # Engine returns one objective; missing_fields is the full list of
        # what's NOT covered (deterministic enum order).
        self.assertEqual(
            obj.missing_fields,
            [StateField.PROBLEMS, StateField.CURRENT_SOLUTIONS],
        )
        self.assertEqual(obj.completed_fields, [
            StateField.PERSONAS,
            StateField.PAIN_POINTS,
            StateField.EVIDENCE,
            StateField.FREQUENCY,
        ])

    def test_priority_chooses_personas_over_everything_when_all_missing(self):
        # State has frequency (priority 80) AND personas (priority 100)
        # missing along with everything else -> personas wins.
        state = _empty_state()
        obj = ObjectiveEngine().determine_next(state)
        self.assertEqual(obj.objective, Objective.PERSONAS)


# ---------------------------------------------------------------------------
# Completed Empathize stage -> WRAP_UP
# ---------------------------------------------------------------------------


class TestObjectiveEngine_WrapUp(unittest.TestCase):

    def test_full_state_yields_wrap_up(self):
        obj = ObjectiveEngine().determine_next(_full_state())
        self.assertEqual(obj.objective, Objective.WRAP_UP)

    def test_wrap_up_is_wrap_up_predicate(self):
        obj = ObjectiveEngine().determine_next(_full_state())
        self.assertTrue(obj.is_wrap_up)

    def test_wrap_up_has_no_missing_fields(self):
        obj = ObjectiveEngine().determine_next(_full_state())
        self.assertEqual(obj.missing_fields, [])
        self.assertEqual(len(obj.completed_fields), len(REQUIRED_FIELDS))

    def test_wrap_up_targeted_field_is_none(self):
        obj = ObjectiveEngine().determine_next(_full_state())
        self.assertIsNone(obj.targeted_field())

    def test_wrap_up_confidence_unique(self):
        obj = ObjectiveEngine().determine_next(_full_state())
        self.assertEqual(obj.confidence, 1.0)

    def test_wrap_up_reasoning_explains_completion(self):
        obj = ObjectiveEngine().determine_next(_full_state())
        joined = " ".join(obj.reasoning)
        self.assertIn("All required", joined)
        self.assertIn("selecting WRAP_UP", joined)


# ---------------------------------------------------------------------------
# Determinism — identical input -> identical output
# ---------------------------------------------------------------------------


class TestObjectiveEngine_Deterministic(unittest.TestCase):

    def test_empty_state_called_twice_yields_equal_results(self):
        engine = ObjectiveEngine()
        state = _empty_state()
        a = engine.determine_next(state)
        b = engine.determine_next(state)
        # ConversationObjective is a frozen dataclass — == compares fields.
        self.assertEqual(a, b)

    def test_partial_state_called_twice_yields_equal_results(self):
        engine = ObjectiveEngine()
        state = ProjectState(personas=["x"], frequency="Weekly")
        a = engine.determine_next(state)
        b = engine.determine_next(state)
        self.assertEqual(a, b)
        # Reasoning list order must be stable too.
        self.assertEqual(a.reasoning, b.reasoning)

    def test_two_engine_instances_same_state_same_output(self):
        # The engine holds no per-turn state — two different instances on
        # the same state must agree.
        state = ProjectState(
            personas=["x"], problems=["x"],
            current_solutions=[], pain_points=[],
            evidence=["x"], frequency="Weekly",
        )
        a = ObjectiveEngine().determine_next(state)
        b = ObjectiveEngine().determine_next(state)
        self.assertEqual(a, b)

    def test_same_state_through_full_completion_sequence_is_deterministic(self):
        """Walk completion through the five field-objectives sequence — replay."""
        engine = ObjectiveEngine()

        # First pass: start empty, walk forward in priority order.
        first_pass_results = []
        state1 = _empty_state()
        for _ in range(len(REQUIRED_FIELDS)):
            obj = engine.determine_next(state1)
            first_pass_results.append(obj)
            field = obj.targeted_field()
            if field is None:
                break
            # Mark the field as complete by adding a value to the state.
            # Note: a real ProjectState never has a "cover field" convenience
            # method (Module 3 is read-only), so we set the attribute
            # directly here, just for test purposes.
            from memory_extractor import LIST_FIELDS, SCALAR_FIELDS
            if field in LIST_FIELDS:
                setattr(state1, field.value, ["x"])
            else:
                setattr(state1, field.value, "x")

        # Second pass: identical walk — must produce identical output.
        state2 = _empty_state()
        for obj1 in first_pass_results:
            obj2 = engine.determine_next(state2)
            self.assertEqual(obj2, obj1)
            field = obj2.targeted_field()
            if field is None:
                break
            from memory_extractor import LIST_FIELDS, SCALAR_FIELDS
            if field in LIST_FIELDS:
                setattr(state2, field.value, ["x"])
            else:
                setattr(state2, field.value, "x")

    def test_execution_in_priority_order(self):
        """Walking a fresh empty state through completion visits all objectives."""
        engine = ObjectiveEngine()
        state = _empty_state()
        visited: list[Objective] = []
        while True:
            obj = engine.determine_next(state)
            visited.append(obj.objective)
            if obj.is_wrap_up:
                break
            field = obj.targeted_field()
            # Mark the field as complete.
            if field in (StateField.FREQUENCY,):  # scalar fields
                setattr(state, field.value, "x")
            else:
                setattr(state, field.value, ["x"])
        # Expected sequence: default-rule priority order, then WRAP_UP.
        self.assertEqual(visited, [
            Objective.PERSONAS,
            Objective.PROBLEMS,
            Objective.FREQUENCY,
            Objective.CURRENT_SOLUTIONS,
            Objective.PAIN_POINTS,
            Objective.EVIDENCE,
            Objective.WRAP_UP,
        ])


# ---------------------------------------------------------------------------
# Read-only contract
# ---------------------------------------------------------------------------


class TestObjectiveEngine_ReadOnly(unittest.TestCase):

    def test_does_not_mutate_state_lists(self):
        state = ProjectState(personas=["x"])
        before = list(state.personas)
        ObjectiveEngine().determine_next(state)
        self.assertEqual(state.personas, before)

    def test_does_not_mutate_state_scalar(self):
        state = ProjectState(frequency="Weekly")
        ObjectiveEngine().determine_next(state)
        self.assertEqual(state.frequency, "Weekly")

    def test_does_not_mutate_empty_state(self):
        state = _empty_state()
        ObjectiveEngine().determine_next(state)
        for sf in REQUIRED_FIELDS:
            from memory_extractor import LIST_FIELDS, SCALAR_FIELDS
            if sf in LIST_FIELDS:
                self.assertEqual(getattr(state, sf.value), [])
            else:
                self.assertIsNone(getattr(state, sf.value))

    def test_does_not_touch_previous_message_bookkeeping(self):
        """previous_assistant_message / previous_user_message are NOT consulted."""
        state = ProjectState(
            personas=["x"],
            previous_assistant_message="should-be-ignored",
            previous_user_message="should-be-ignored",
        )
        ObjectiveEngine().determine_next(state)
        # Untouched.
        self.assertEqual(state.previous_assistant_message, "should-be-ignored")
        self.assertEqual(state.previous_user_message, "should-be-ignored")


# ---------------------------------------------------------------------------
# ConversationObjective structural invariants
# ---------------------------------------------------------------------------


class TestConversationObjective_Contract(unittest.TestCase):

    def test_confidence_must_be_in_unit_interval(self):
        with self.assertRaises(ValueError):
            ConversationObjective(
                objective=Objective.PERSONAS,
                missing_fields=[StateField.PERSONAS],
                completed_fields=[],
                confidence=1.5,
                reasoning=[],
            )

    def test_confidence_negative_rejected(self):
        with self.assertRaises(ValueError):
            ConversationObjective(
                objective=Objective.PERSONAS,
                missing_fields=[StateField.PERSONAS],
                completed_fields=[],
                confidence=-0.1,
                reasoning=[],
            )

    def test_objective_must_be_enum_value(self):
        with self.assertRaises(TypeError):
            ConversationObjective(
                objective="PERSONAS",  # type: ignore[arg-type]
                missing_fields=[],
                completed_fields=[],
                confidence=1.0,
                reasoning=[],
            )

    def test_is_frozen_dataclass(self):
        obj = ObjectiveEngine().determine_next(_empty_state())
        with self.assertRaises(FrozenInstanceError):
            obj.objective = Objective.WRAP_UP  # type: ignore[misc]

    def test_for_field_round_trips_each_empathize_objective(self):
        """Objective.for_field is the bidirectional mapping with StateField."""
        for sf in StateField:
            obj_objective = Objective.for_field(sf)
            # Round trip: name-of-objective == name-of-statefield.
            self.assertEqual(obj_objective.name, sf.name)

    def test_targeted_field_returns_matching_statefield_for_non_wrapup(self):
        for sf in REQUIRED_FIELDS:
            state = ProjectState()
            # Make every required field except `sf` completed.
            from memory_extractor import LIST_FIELDS, SCALAR_FIELDS
            for other in REQUIRED_FIELDS:
                if other is sf:
                    continue
                if other in LIST_FIELDS:
                    setattr(state, other.value, ["x"])
                else:
                    setattr(state, other.value, "x")
            obj = ObjectiveEngine().determine_next(state)
            self.assertEqual(obj.targeted_field(), sf,
                             f"targeted_field mismatch for objective {obj.objective.name}")


# ---------------------------------------------------------------------------
# Custom rule registry injection (Module 3's extensibility surface)
# ---------------------------------------------------------------------------


class TestObjectiveEngine_CustomRules(unittest.TestCase):

    def test_custom_rules_change_objective_ordering(self):
        """A registry that re-prioritises evidence over personas is honoured."""
        custom_rules = [
            Rule(field=StateField.EVIDENCE, priority=200, objective=Objective.EVIDENCE),
            Rule(field=StateField.PERSONAS, priority=100, objective=Objective.PERSONAS),
            Rule(field=StateField.PROBLEMS, priority=90, objective=Objective.PROBLEMS),
            Rule(field=StateField.FREQUENCY, priority=80, objective=Objective.FREQUENCY),
            Rule(field=StateField.CURRENT_SOLUTIONS, priority=70, objective=Objective.CURRENT_SOLUTIONS),
            Rule(field=StateField.PAIN_POINTS, priority=60, objective=Objective.PAIN_POINTS),
        ]
        engine = ObjectiveEngine(rules=custom_rules)
        obj = engine.determine_next(_empty_state())
        self.assertEqual(obj.objective, Objective.EVIDENCE)
        self.assertEqual(obj.targeted_field(), StateField.EVIDENCE)

    def test_rules_property_returns_engine_registry(self):
        engine = ObjectiveEngine(rules=DEFAULT_RULES)
        self.assertEqual(engine.rules, DEFAULT_RULES)

    def test_engine_default_rules_match_default_object(self):
        """No `rules` arg → engine uses DEFAULT_RULES (the canonical Empathize order)."""
        engine = ObjectiveEngine()
        self.assertEqual(engine.rules, DEFAULT_RULES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
