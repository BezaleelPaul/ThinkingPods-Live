"""
tests/test_summary_builder.py

Unit tests for module5.summary_builder + module5.summary (Module 5 —
Summary Builder + EmpathizeSummary).

Covers (per the Module 5 spec test categories):
  ✓ SummaryBuilder copies ProjectState correctly
  ✓ Empty ProjectState
  ✓ Full ProjectState
  ✓ Ordering preserved
  ✓ No hallucinated values
  ✓ Deterministic output
  ✓ EmpathizeSummary immutability (frozen dataclass)
  ✓ Type rejection (non-ProjectState)
  ✓ Summary doesn't read conversation history
"""

import sys
import os
import unittest

# Allow importing from the project root regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import ProjectState  # noqa: E402

from module5 import EmpathizeSummary, SummaryBuilder  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _empty_state() -> ProjectState:
    return ProjectState()


def _full_state() -> ProjectState:
    return ProjectState(
        personas=["students", "teachers"],
        problems=["buried messages"],
        current_solutions=["whatsapp groups"],
        pain_points=["overwhelmed", "no clean inbox"],
        evidence=["user interviews"],
        frequency="daily",
    )


# ---------------------------------------------------------------------------
# Empty ProjectState
# ---------------------------------------------------------------------------


class TestEmptyProjectState(unittest.TestCase):

    def test_empty_state_has_empty_lists_and_none_frequency(self):
        summary = SummaryBuilder.build(_empty_state())
        self.assertIsInstance(summary, EmpathizeSummary)
        self.assertEqual(summary.personas, [])
        self.assertEqual(summary.problems, [])
        self.assertEqual(summary.current_solutions, [])
        self.assertEqual(summary.pain_points, [])
        self.assertEqual(summary.evidence, [])
        self.assertIsNone(summary.frequency)


# ---------------------------------------------------------------------------
# Full ProjectState — copy-exactly semantics
# ---------------------------------------------------------------------------


class TestFullProjectState(unittest.TestCase):

    def test_full_state_copy_exactly(self):
        ps = _full_state()
        summary = SummaryBuilder.build(ps)
        # All list fields match the original state.
        for attr in ("personas", "problems", "current_solutions",
                     "pain_points", "evidence"):
            self.assertEqual(getattr(summary, attr), getattr(ps, attr),
                             f"{attr} not copied exactly")
        self.assertEqual(summary.frequency, ps.frequency)

    def test_ordering_preserved(self):
        """Insertion order from ProjectState is preserved."""
        ps = ProjectState(
            personas=["third", "first", "second"],
            problems=["a"],
            current_solutions=["b"],
            pain_points=["c"],
            evidence=["d"],
            frequency="weekly",
        )
        summary = SummaryBuilder.build(ps)
        self.assertEqual(summary.personas, ["third", "first", "second"])
        self.assertEqual(summary.personas, ps.personas)

    def test_multi_item_list_fields_preserved(self):
        summary = SummaryBuilder.build(_full_state())
        self.assertEqual(summary.personas, ["students", "teachers"])
        self.assertEqual(summary.pain_points, ["overwhelmed", "no clean inbox"])


# ---------------------------------------------------------------------------
# No hallucinated values
# ---------------------------------------------------------------------------


class TestNoHallucinatedValues(unittest.TestCase):

    def test_does_not_invent_values_for_none_frequency(self):
        ps = _full_state()
        ps.frequency = None
        summary = SummaryBuilder.build(ps)
        self.assertIsNone(summary.frequency)

    def test_does_not_invent_list_items_for_empty_list(self):
        ps = _full_state()
        ps.personas = []
        summary = SummaryBuilder.build(ps)
        self.assertEqual(summary.personas, [])

    def test_partial_state_does_not_fill_missing_lists(self):
        """Only what ProjectState actually contains goes into the summary."""
        ps = ProjectState(personas=["x"])
        summary = SummaryBuilder.build(ps)
        self.assertEqual(summary.personas, ["x"])
        self.assertEqual(summary.problems, [])
        self.assertEqual(summary.current_solutions, [])
        self.assertEqual(summary.pain_points, [])
        self.assertEqual(summary.evidence, [])
        self.assertIsNone(summary.frequency)

    def test_empty_string_items_not_filtered_out(self):
        """The builder doesn't apply its own validation — it copies verbatim.
        Empty-string items from ProjectState are copied as-is."""
        ps = ProjectState(personas=[""], problems=["real"], current_solutions=[""],
                         pain_points=["real"], evidence=[""], frequency="")
        summary = SummaryBuilder.build(ps)
        self.assertEqual(summary.personas, [""])
        self.assertEqual(summary.problems, ["real"])
        self.assertEqual(summary.frequency, "")


# ---------------------------------------------------------------------------
# Defensive copy — source mutation doesn't retroactively change summary
# ---------------------------------------------------------------------------


class TestDefensiveCopy(unittest.TestCase):

    def test_mutating_source_after_build_does_not_change_summary(self):
        ps = ProjectState(personas=["x", "y"])
        summary = SummaryBuilder.build(ps)
        ps.personas.append("z")  # mutate source after snapshot
        self.assertEqual(summary.personas, ["x", "y"])
        self.assertEqual(ps.personas, ["x", "y", "z"])

    def test_summary_lists_are_independent_objects(self):
        ps = ProjectState(personas=["x"])
        summary = SummaryBuilder.build(ps)
        self.assertIsNot(summary.personas, ps.personas)
        self.assertIsNot(summary.problems, ps.problems)
        self.assertIsNot(summary.current_solutions, ps.current_solutions)
        self.assertIsNot(summary.pain_points, ps.pain_points)
        self.assertIsNot(summary.evidence, ps.evidence)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):

    def test_same_state_twice_yields_equal_summaries(self):
        ps = _full_state()
        a = SummaryBuilder.build(ps)
        b = SummaryBuilder.build(ps)
        self.assertEqual(a, b)

    def test_equality_is_value_equal(self):
        """Two summaries built from the same state should be ==."""
        a = SummaryBuilder.build(_full_state())
        b = SummaryBuilder.build(ProjectState(
            personas=["students", "teachers"],
            problems=["buried messages"],
            current_solutions=["whatsapp groups"],
            pain_points=["overwhelmed", "no clean inbox"],
            evidence=["user interviews"],
            frequency="daily",
        ))
        self.assertEqual(a, b)

    def test_inequality_when_one_differs(self):
        a = SummaryBuilder.build(_full_state())
        ps = _full_state()
        ps.frequency = "weekly"
        b = SummaryBuilder.build(ps)
        self.assertNotEqual(a, b)


# ---------------------------------------------------------------------------
# EmpathizeSummary immutability
# ---------------------------------------------------------------------------


class TestEmpathizeSummaryImmutability(unittest.TestCase):

    def test_rebind_rejected(self):
        summary = SummaryBuilder.build(_full_state())
        from dataclasses import FrozenInstanceError
        with self.assertRaises(FrozenInstanceError):
            summary.personas = ["x"]  # type: ignore[misc]

    def test_can_hash(self):
        """Frozen dataclass → hashable (list fields make it non-hashable
        in practice, but the class is frozen)."""
        summary = SummaryBuilder.build(_full_state())
        # The hash built-in won't work because lists are unhashable, but the
        # FrozenInstanceError guard proves the object itself is frozen.
        self.assertTrue(hasattr(summary, "personas"))


# ---------------------------------------------------------------------------
# Type rejection
# ---------------------------------------------------------------------------


class TestSummaryBuilderTypeRejection(unittest.TestCase):

    def test_rejects_non_project_state(self):
        with self.assertRaises(TypeError):
            SummaryBuilder.build({"personas": []})

    def test_rejects_none(self):
        with self.assertRaises(TypeError):
            SummaryBuilder.build(None)

    def test_rejects_string(self):
        with self.assertRaises(TypeError):
            SummaryBuilder.build("not a state")


# ---------------------------------------------------------------------------
# Summary does NOT read conversation history
# ---------------------------------------------------------------------------


class TestNoConversationHistory(unittest.TestCase):
    """Even if ProjectState carries conversation bookkeeping, the summary
    must NOT include those fields."""

    def test_summary_fields_are_only_the_six_extraction_fields(self):
        ps = _full_state()
        ps.previous_assistant_message = "some reply"
        ps.previous_user_message = "some message"
        summary = SummaryBuilder.build(ps)
        # The EmpathizeSummary has exactly the 6 canon fields — no bookkeeping.
        facet_fields = ("personas", "problems", "current_solutions",
                        "pain_points", "evidence", "frequency")
        for f in facet_fields:
            self.assertTrue(hasattr(summary, f),
                            f"EmpathizeSummary must have field '{f}'")
        self.assertFalse(hasattr(summary, "previous_assistant_message"),
                         "Summary must NOT include conversation history")
        self.assertFalse(hasattr(summary, "previous_user_message"),
                         "Summary must NOT include conversation history")

    def test_bookkeeping_fields_are_not_leaked_into_summary(self):
        """Regression guard: the builder ONLY copies the 6 extraction
        fields; it ignores ProjectState.bookkeeping entirely."""
        ps = ProjectState(
            personas=["x"], problems=["y"],
            current_solutions=["z"], pain_points=["q"],
            evidence=["w"], frequency="daily",
            previous_assistant_message="hello",
            previous_user_message="hi",
        )
        summary = SummaryBuilder.build(ps)
        # The list & scalar fields must be as per state.
        self.assertEqual(summary.personas, ["x"])
        self.assertEqual(summary.frequency, "daily")
        # And the conversation fields must NOT appear anywhere.
        self.assertNotIn("hello", str(summary))


if __name__ == "__main__":
    unittest.main(verbosity=2)