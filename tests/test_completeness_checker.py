"""
tests/test_completeness_checker.py

Unit tests for module3.completeness_checker.CompletenessChecker.

Covers (per the Module 3 spec test categories):
  - empty ProjectState
  - partially complete state
  - fully complete state

Also covers the read-only contract (the checker must never mutate ProjectState)
and the strict ProjectState-only input boundary (no MentorSession/bridge).
"""

import sys
import os
import unittest

# Allow importing from the project root regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import ProjectState, REQUIRED_FIELDS, StateField

from module3 import CompletenessChecker
from module3.objective import CoverageReport


class TestCompletenessChecker_EmptyState(unittest.TestCase):
    """A fresh ProjectState must produce an all-false CoverageReport."""

    def test_empty_state_all_fields_false(self):
        report = CompletenessChecker.evaluate(ProjectState())
        for sf in REQUIRED_FIELDS:
            self.assertFalse(
                report.is_field_complete(sf),
                f"{sf.value} should be False on empty state"
            )

    def test_empty_state_num_complete_is_zero(self):
        report = CompletenessChecker.evaluate(ProjectState())
        self.assertEqual(report.num_complete, 0)

    def test_empty_state_not_all_complete(self):
        report = CompletenessChecker.evaluate(ProjectState())
        self.assertFalse(report.all_complete)

    def test_empty_state_missing_and_completed_lists(self):
        report = CompletenessChecker.evaluate(ProjectState())
        # All 6 required fields are missing, none are completed (declared order).
        self.assertEqual(report.missing_fields, list(REQUIRED_FIELDS))
        self.assertEqual(report.completed_fields, [])

    def test_for_empty_state_helper_equivalent_to_evaluate_default(self):
        a = CompletenessChecker.for_empty_state()
        b = CompletenessChecker.evaluate(ProjectState())
        self.assertEqual(a, b)

    def test_empty_state_as_dict_has_all_keys_false(self):
        d = CompletenessChecker.for_empty_state().as_dict()
        self.assertEqual(
            sorted(d.keys()),
            sorted([sf.value for sf in REQUIRED_FIELDS]),
        )
        self.assertFalse(any(d.values()))


class TestCompletenessChecker_PartiallyComplete(unittest.TestCase):
    """A state with some fields populated must report those as covered."""

    def test_one_list_field_populated_marks_only_that_field(self):
        state = ProjectState(personas=["Students"])
        report = CompletenessChecker.evaluate(state)
        self.assertTrue(report.personas)
        self.assertTrue(report.is_field_complete(StateField.PERSONAS))
        # Other list-typed required fields stay False.
        for sf in REQUIRED_FIELDS:
            if sf is StateField.PERSONAS:
                continue
            self.assertFalse(report.is_field_complete(sf), f"{sf.value} leaked")

    def test_scalar_frequency_populated(self):
        state = ProjectState(frequency="Weekly")
        report = CompletenessChecker.evaluate(state)
        self.assertTrue(report.frequency)
        self.assertTrue(report.is_field_complete(StateField.FREQUENCY))
        # Lists still empty.
        self.assertFalse(report.personas)
        self.assertFalse(report.problems)

    def test_partial_state_num_complete_counts_correctly(self):
        state = ProjectState(
            personas=["Students"],
            problems=["buried messages"],
            frequency="Weekly",
        )
        report = CompletenessChecker.evaluate(state)
        self.assertEqual(report.num_complete, 3)
        self.assertEqual(len(report.completed_fields), 3)
        # Missing three others.
        self.assertEqual(len(report.missing_fields), 3)
        self.assertFalse(report.all_complete)

    def test_partial_state_completed_list_declared_order(self):
        # Populate fields in a scrambled order — CoverageReport's
        # completed_fields MUST follow StateField's declared enum order,
        # not insertion order.
        state = ProjectState(
            evidence=["data first"],         # 5th enum
            current_solutions=["WhatsApp"],  # 3rd enum
            personas=["students"],           # 1st enum
        )
        report = CompletenessChecker.evaluate(state)
        self.assertEqual(
            report.completed_fields,
            [StateField.PERSONAS, StateField.CURRENT_SOLUTIONS, StateField.EVIDENCE],
        )

    def test_multiple_list_values_treated_as_one_coverage(self):
        """A field with two items is still just 'covered' — coverage is boolean."""
        state = ProjectState(personas=["Students", "Teachers", "Parents"])
        report = CompletenessChecker.evaluate(state)
        self.assertTrue(report.personas)
        self.assertEqual(report.num_complete, 1)

    def test_partial_filled_most_fields_not_wrap_up(self):
        """Five of six fields populated — not all_complete."""
        state = ProjectState(
            personas=["x"],
            problems=["x"],
            current_solutions=["x"],
            pain_points=["x"],
            evidence=["x"],
            frequency=None,  # only frequency missing
        )
        report = CompletenessChecker.evaluate(state)
        self.assertEqual(len(report.completed_fields), 5)
        self.assertEqual(report.missing_fields, [StateField.FREQUENCY])
        self.assertFalse(report.all_complete)


class TestCompletenessChecker_FullyComplete(unittest.TestCase):
    """A state with every field populated reports all-complete."""

    def test_fully_complete_state_all_fields_true(self):
        state = ProjectState(
            personas=["Students"],
            problems=["buried messages"],
            current_solutions=["WhatsApp groups"],
            pain_points=["overwhelmed"],
            evidence=["saw students struggle"],
            frequency="Weekly",
        )
        report = CompletenessChecker.evaluate(state)
        for sf in REQUIRED_FIELDS:
            self.assertTrue(report.is_field_complete(sf), f"{sf.value} not complete")
        self.assertTrue(report.all_complete)
        self.assertEqual(report.num_complete, len(REQUIRED_FIELDS))
        self.assertEqual(report.missing_fields, [])
        self.assertEqual(len(report.completed_fields), len(REQUIRED_FIELDS))

    def test_whitespaced_scalar_still_counts_as_empty(self):
        """frequency='   ' is whitespace-only and must NOT count as covered."""
        # ProjectState.set_scalar would normalise '   ' to None, but if a
        # caller bypasses it (via __init__), CoverageReport still must
        # treat it as not-covered.
        state = ProjectState(frequency="   ")
        report = CompletenessChecker.evaluate(state)
        self.assertFalse(report.frequency)

    def test_whitespace_only_list_value_treated_as_empty(self):
        """A list containing only whitespace strings is NOT covered."""
        state = ProjectState(personas=["   ", "\t"])
        report = CompletenessChecker.evaluate(state)
        self.assertFalse(report.personas)

    def test_mixed_whitespace_and_real_value_makes_field_complete(self):
        """A list with at least one non-empty stripped value is covered."""
        state = ProjectState(personas=["", "Students", "   "])
        report = CompletenessChecker.evaluate(state)
        self.assertTrue(report.personas)


class TestCompletenessChecker_ImpactsNonGating(unittest.TestCase):
    """`impacts` is supporting knowledge: captured but NOT part of coverage."""

    def test_impacts_alone_does_not_count_toward_coverage(self):
        report = CompletenessChecker.evaluate(ProjectState(impacts=["lower grades"]))
        self.assertEqual(report.num_complete, 0)
        self.assertFalse(report.all_complete)

    def test_impacts_does_not_appear_in_coverage_report(self):
        report = CompletenessChecker.evaluate(ProjectState(impacts=["lower grades"]))
        self.assertNotIn("impacts", report.as_dict())
        self.assertNotIn(StateField.IMPACTS, report.completed_fields)
        self.assertNotIn(StateField.IMPACTS, report.missing_fields)

    def test_full_required_fields_still_all_complete_without_impacts(self):
        state = ProjectState(
            personas=["Students"],
            problems=["buried messages"],
            current_solutions=["WhatsApp groups"],
            pain_points=["overwhelmed"],
            evidence=["saw students struggle"],
            frequency="Weekly",
        )
        report = CompletenessChecker.evaluate(state)
        self.assertTrue(report.all_complete)
        self.assertEqual(report.num_complete, len(REQUIRED_FIELDS))


class TestCompletenessChecker_ReadOnly(unittest.TestCase):
    """The checker must never mutate the input ProjectState."""

    def test_evaluate_does_not_mutate_lists(self):
        state = ProjectState(personas=["Students"])
        original = list(state.personas)
        CompletenessChecker.evaluate(state)
        self.assertEqual(state.personas, original)

    def test_evaluate_does_not_mutate_scalar(self):
        state = ProjectState(frequency="Weekly")
        CompletenessChecker.evaluate(state)
        self.assertEqual(state.frequency, "Weekly")

    def test_evaluate_does_not_mutate_empty_state(self):
        state = ProjectState()
        CompletenessChecker.evaluate(state)
        for sf in REQUIRED_FIELDS:
            # No field should have been filled in.
            if hasattr(state, sf.value) and not sf.value.endswith("frequency"):
                self.assertEqual(getattr(state, sf.value), [])
        self.assertIsNone(state.frequency)

    def test_double_evaluate_returns_equal_reports(self):
        """Determinism: same state twice -> equal reports (same input, same output)."""
        state = ProjectState(personas=["x"], frequency="Weekly")
        r1 = CompletenessChecker.evaluate(state)
        r2 = CompletenessChecker.evaluate(state)
        self.assertEqual(r1, r2)


class TestCompletenessChecker_InputBoundary(unittest.TestCase):
    """Module 3 is the strict ProjectState-only boundary."""

    def test_rejects_non_project_state(self):
        with self.assertRaises(TypeError):
            CompletenessChecker.evaluate("not a state")  # type: ignore[arg-type]

    def test_rejects_dict(self):
        """A raw dict (e.g. MentorSession.to_dict output) must NOT be accepted."""
        with self.assertRaises(TypeError):
            CompletenessChecker.evaluate({"personas": ["x"]})  # type: ignore[arg-type]

    def test_rejects_none(self):
        with self.assertRaises(TypeError):
            CompletenessChecker.evaluate(None)  # type: ignore[arg-type]

    def test_returns_coverage_report_type(self):
        out = CompletenessChecker.evaluate(ProjectState())
        self.assertIsInstance(out, CoverageReport)


if __name__ == "__main__":
    unittest.main(verbosity=2)
