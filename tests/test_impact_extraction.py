"""
Impact-extraction tests — consequences/effects of the problem classified
separately from problems.

`impacts` (StateField.IMPACTS) is supporting project knowledge: it is
captured and persisted but does NOT gate Empathize completion, objective
selection, or WRAP_UP (see REQUIRED_FIELDS / NON_REQUIRED_FIELDS).

Covered here:
  * rule-based extraction of impact statements (positive + negative)
  * merge_extracted_to_state -> ProjectState.impacts
  * duplicate handling via StateManager ADD semantics
  * serialization round-trips (to_state_dict + SessionData persistence)
  * non-gating contract (coverage / objective engine ignore impacts)
"""

import sys
import os
import unittest

# Allow importing from the project root regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from extraction_pipeline import RuleBasedExtractor, merge_extracted_to_state
from memory_extractor import (
    ExtractionResult,
    ExtractionUpdate,
    MessageType,
    Operation,
    ProjectState,
    REQUIRED_FIELDS,
    StateField,
)
from module3 import Objective, ObjectiveEngine
from module3.completeness_checker import CompletenessChecker
from session_manager import SessionData
from state_manager import StateManager


# ---------------------------------------------------------------------------
# Rule-based extraction — positive cases
# ---------------------------------------------------------------------------


class TestImpactRuleBasedExtraction(unittest.TestCase):
    """Causal cues (affects, causes, results in, ...) classify as impact."""

    IMPACT_CASES = {
        "This affects academic performance.": "academic performance suffers",
        "It causes students stress.": "students stress",
        "The delay results in lower grades.": "lower grades",
        "The pressure leads to burnout.": "burnout",
        "This creates anxiety for students.": "anxiety for students",
        "It makes collaboration difficult.": "collaboration difficult",
        "It reduces productivity.": "productivity reduced",
        "It improves focus.": "focus improved",
        "It prevents on-time submissions.": "on-time submissions prevented",
        "It wastes valuable time.": "valuable time wasted",
    }

    def test_causal_cues_yield_impact(self):
        for message, expected in self.IMPACT_CASES.items():
            with self.subTest(message=message):
                result = RuleBasedExtractor.extract(message)
                self.assertIn("impact", result,
                              f"expected impact for: {message!r}")
                self.assertEqual(result["impact"], expected)

    def test_pure_impact_statement_is_not_a_problem(self):
        """A bare consequence ('This affects academic performance.') must not
        be classified as a pain point / problem."""
        result = RuleBasedExtractor.extract("This affects academic performance.")
        self.assertIn("impact", result)
        self.assertNotIn("pain_point", result)

    def test_impact_captured_alongside_problem_when_both_present(self):
        """'Procrastination causes missed deadlines' yields BOTH a problem
        and an impact — the causal consequence is not swallowed."""
        result = RuleBasedExtractor.extract("Procrastination causes missed deadlines.")
        self.assertIn("pain_point", result)
        self.assertEqual(result["impact"], "missed deadlines")

    def test_affects_rephrases_as_suffers(self):
        result = RuleBasedExtractor.extract("This affects academic performance.")
        self.assertEqual(result["impact"], "academic performance suffers")


# ---------------------------------------------------------------------------
# Rule-based extraction — negative cases
# ---------------------------------------------------------------------------


class TestImpactRuleBasedNegative(unittest.TestCase):
    """Statements without a causal cue must NOT be classified as impact."""

    NON_IMPACT_CASES = [
        "The problem is late submission of assignments.",
        "They currently use sticky notes and a pill box.",
        "It happens daily.",
        "I interviewed five students.",
        "I want to help elderly people take their medications on time.",
    ]

    def test_non_causal_statements_do_not_yield_impact(self):
        for message in self.NON_IMPACT_CASES:
            with self.subTest(message=message):
                result = RuleBasedExtractor.extract(message)
                self.assertNotIn("impact", result,
                                 f"unexpected impact for: {message!r}")

    def test_problem_statement_still_yields_pain_point(self):
        result = RuleBasedExtractor.extract("The problem is late submission of assignments.")
        self.assertIn("pain_point", result)
        self.assertNotIn("impact", result)


# ---------------------------------------------------------------------------
# merge_extracted_to_state + duplicate handling
# ---------------------------------------------------------------------------


class TestImpactMergeToState(unittest.TestCase):
    """extracted['impact'] lands in ProjectState.impacts via StateManager."""

    def test_merge_writes_impacts_field(self):
        state = ProjectState()
        session_data = SessionData()
        merge_extracted_to_state(
            state, session_data, {"impact": "academic performance suffers"}
        )
        self.assertEqual(state.impacts, ["academic performance suffers"])

    def test_merge_does_not_touch_other_fields(self):
        state = ProjectState()
        merge_extracted_to_state(
            state, SessionData(), {"impact": "students stress"}
        )
        self.assertEqual(state.problems, [])
        self.assertEqual(state.evidence, [])

    def test_duplicate_impacts_are_deduplicated(self):
        state = ProjectState()
        session_data = SessionData()
        extracted = {"impact": "academic performance suffers"}
        merge_extracted_to_state(state, session_data, extracted)
        merge_extracted_to_state(state, session_data, extracted)
        self.assertEqual(state.impacts, ["academic performance suffers"])

    def test_state_manager_add_dedup_on_impacts(self):
        state = ProjectState()
        mgr = StateManager(state)
        for _ in range(2):
            mgr.apply_extraction(ExtractionResult(
                message_type=MessageType.MEANINGFUL,
                updates=[ExtractionUpdate(
                    Operation.ADD, StateField.IMPACTS, "lower grades"
                )],
            ))
        self.assertEqual(state.impacts, ["lower grades"])

    def test_impacts_is_a_list_typed_field(self):
        state = ProjectState()
        mgr = StateManager(state)
        mgr.apply_extraction(ExtractionResult(
            message_type=MessageType.MEANINGFUL,
            updates=[
                ExtractionUpdate(Operation.ADD, StateField.IMPACTS, "missed deadlines"),
                ExtractionUpdate(Operation.ADD, StateField.IMPACTS, "burnout"),
            ],
        ))
        self.assertEqual(state.impacts, ["missed deadlines", "burnout"])


# ---------------------------------------------------------------------------
# Serialization / persistence
# ---------------------------------------------------------------------------


class TestImpactSerialization(unittest.TestCase):
    """impacts round-trips through every persistence representation."""

    def test_to_state_dict_includes_impacts(self):
        d = ProjectState(impacts=["lower grades"]).to_state_dict()
        self.assertEqual(d["impacts"], ["lower grades"])

    def test_empty_impacts_serialises_as_empty_list(self):
        d = ProjectState().to_state_dict()
        self.assertEqual(d["impacts"], [])

    def test_session_data_dict_round_trip_preserves_impacts(self):
        ps = ProjectState(impacts=["academic performance suffers", "burnout"])
        data = SessionData(project_state=ps)
        restored = SessionData.from_dict(data.to_dict())
        self.assertEqual(restored.project_state.impacts,
                         ["academic performance suffers", "burnout"])

    def test_session_data_missing_impacts_defaults_empty(self):
        restored = SessionData.from_dict({"project_state": {}})
        self.assertEqual(restored.project_state.impacts, [])


# ---------------------------------------------------------------------------
# Non-gating contract — impacts never influence coverage or objectives
# ---------------------------------------------------------------------------


class TestImpactNonGating(unittest.TestCase):
    """impacts is supporting knowledge: it does not move the mentor forward."""

    def test_impacts_alone_does_not_complete_coverage(self):
        report = CompletenessChecker.evaluate(ProjectState(impacts=["lower grades"]))
        self.assertEqual(report.num_complete, 0)
        self.assertFalse(report.all_complete)

    def test_impacts_absent_from_coverage_report_dict(self):
        report = CompletenessChecker.evaluate(ProjectState(impacts=["lower grades"]))
        self.assertNotIn("impacts", report.as_dict())

    def test_impacts_alone_does_not_advance_objective(self):
        obj = ObjectiveEngine().determine_next(ProjectState(impacts=["lower grades"]))
        self.assertEqual(obj.objective, Objective.PERSONAS)

    def test_impacts_never_selected_as_objective(self):
        state = ProjectState(impacts=["lower grades"])
        obj = ObjectiveEngine().determine_next(state)
        self.assertEqual(obj.objective, Objective.PERSONAS)
        self.assertNotEqual(obj.targeted_field(), StateField.IMPACTS)

    def test_full_required_fields_still_wrap_up_with_impacts(self):
        state = ProjectState(
            personas=["students"],
            problems=["missed deadlines"],
            current_solutions=["sticky notes"],
            pain_points=["stress"],
            evidence=["observed late submissions"],
            impacts=["academic performance suffers"],
            frequency="weekly",
        )
        obj = ObjectiveEngine().determine_next(state)
        self.assertEqual(obj.objective, Objective.WRAP_UP)
        self.assertIsNone(obj.targeted_field())

    def test_missing_fields_are_only_required_fields(self):
        report = CompletenessChecker.evaluate(ProjectState(impacts=["lower grades"]))
        self.assertEqual(report.missing_fields, list(REQUIRED_FIELDS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
