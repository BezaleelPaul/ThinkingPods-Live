"""
tests/test_objective_sufficiency.py

Objective sufficiency — the mentor advances past a field once it holds a
usable answer, even when perfect quantitative evidence is missing.

Regression target (repeated frequency questions):

    User: "It happens every single day."
    Mentor: (previously) asks another frequency question.

The fix has three cooperating pieces, all tested here:

  1. The deterministic extractor captures natural cadence phrases
     ("every single day") that the old FREQUENCY_PATTERNS missed.
  2. The fallback runs even when the LLM returned a batch that Module 2
     rejected (otherwise the user's information was silently dropped and
     the objective engine re-selected the same objective).
  3. The objective engine classifies each field as
     Unknown / Partial / Sufficient / Complete and reports, in the
     diagnostics, the reason the objective advanced or stayed active.

Coverage is kept binary (WRAP_UP / lifecycle unchanged): sufficiency is a
read-only classification + explanation layer, never a gate.
"""

import os
import sys
import unittest
from unittest import mock

# Allow importing from the project root regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from extraction_pipeline import RuleBasedExtractor  # noqa: E402
from memory_extractor import (  # noqa: E402
    ExtractionResult,
    ExtractionUpdate,
    MessageType,
    Operation,
    ProjectState,
    REQUIRED_FIELDS,
    StateField,
)
from module3 import (  # noqa: E402
    ConversationObjective,
    Objective,
    ObjectiveEngine,
    SufficiencyChecker,
    SufficiencyLevel,
)


# ---------------------------------------------------------------------------
# Sufficiency classification
# ---------------------------------------------------------------------------


class TestSufficiencyClassification(unittest.TestCase):
    """Each required field maps to a sufficiency level deterministically."""

    def test_natural_cadence_is_sufficient(self):
        checker = SufficiencyChecker()
        for value in ("every single day", "every day", "weekly",
                      "often", "sometimes", "always"):
            report = checker.evaluate(ProjectState(frequency=value))
            self.assertEqual(report.level_of(StateField.FREQUENCY),
                             SufficiencyLevel.SUFFICIENT,
                             f"expected SUFFICIENT for {value!r}")

    def test_quantitative_frequency_is_complete(self):
        checker = SufficiencyChecker()
        for value in ("3 times a week", "twice a day", "once a month",
                      "10 times a day"):
            report = checker.evaluate(ProjectState(frequency=value))
            self.assertEqual(report.level_of(StateField.FREQUENCY),
                             SufficiencyLevel.COMPLETE,
                             f"expected COMPLETE for {value!r}")

    def test_blank_frequency_is_unknown(self):
        checker = SufficiencyChecker()
        for value in (None, "", "   "):
            report = checker.evaluate(ProjectState(frequency=value))
            self.assertEqual(report.level_of(StateField.FREQUENCY),
                             SufficiencyLevel.UNKNOWN)

    def test_vague_frequency_is_partial(self):
        report = SufficiencyChecker().evaluate(
            ProjectState(frequency="whenever they feel like it")
        )
        self.assertEqual(report.level_of(StateField.FREQUENCY),
                         SufficiencyLevel.PARTIAL)

    def test_list_sufficiency_counts(self):
        checker = SufficiencyChecker()
        self.assertEqual(
            checker.evaluate(ProjectState()).level_of(StateField.PERSONAS),
            SufficiencyLevel.UNKNOWN,
        )
        self.assertEqual(
            checker.evaluate(ProjectState(personas=["students"]))
            .level_of(StateField.PERSONAS),
            SufficiencyLevel.SUFFICIENT,
        )
        self.assertEqual(
            checker.evaluate(ProjectState(personas=["students", "teachers"]))
            .level_of(StateField.PERSONAS),
            SufficiencyLevel.COMPLETE,
        )

    def test_report_only_covers_required_fields(self):
        report = SufficiencyChecker().evaluate(ProjectState())
        self.assertEqual(sorted(report.as_dict().keys()),
                         sorted(sf.value for sf in REQUIRED_FIELDS))
        self.assertNotIn("impacts", report.as_dict())

    def test_is_satisfied_threshold_is_sufficient(self):
        report = SufficiencyChecker().evaluate(
            ProjectState(frequency="every single day")
        )
        self.assertTrue(report.is_satisfied(StateField.FREQUENCY))

    def test_report_is_read_only(self):
        state = ProjectState(frequency="weekly")
        before = state.to_state_dict()
        SufficiencyChecker().evaluate(state)
        self.assertEqual(state.to_state_dict(), before)

    def test_evaluate_dict_classifies_snapshots(self):
        report = SufficiencyChecker().evaluate_dict({"frequency": "every single day"})
        self.assertEqual(report.level_of(StateField.FREQUENCY),
                         SufficiencyLevel.SUFFICIENT)
        empty = SufficiencyChecker().evaluate_dict({})
        self.assertEqual(empty.level_of(StateField.FREQUENCY),
                         SufficiencyLevel.UNKNOWN)

    def test_classification_is_deterministic(self):
        state = ProjectState(frequency="every single day", personas=["x"])
        a = SufficiencyChecker().evaluate(state)
        b = SufficiencyChecker().evaluate(state)
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# Rule-based extraction — natural frequency phrases
# ---------------------------------------------------------------------------


class TestNaturalFrequencyExtraction(unittest.TestCase):
    """The deterministic fallback must capture phrases the old pattern missed."""

    def test_every_single_day_is_captured(self):
        result = RuleBasedExtractor.extract("It happens every single day.")
        self.assertEqual(result.get("frequency"), "every single day")

    def test_natural_cadence_variants_are_captured(self):
        for message, expected in (
            ("It happens each day.", "each day"),
            ("It occurs most days.", "most days"),
            ("It's all the time.", "all the time"),
            ("It happens once in a while.", "once in a while"),
            ("It's a weekly thing.", "weekly"),
        ):
            with self.subTest(message=message):
                self.assertEqual(
                    RuleBasedExtractor.extract(message).get("frequency"),
                    expected,
                )

    def test_quantitative_phrase_is_captured(self):
        result = RuleBasedExtractor.extract("About three times a week.")
        self.assertEqual(result.get("frequency"), "three times a week")

    def test_non_frequency_statement_has_no_frequency(self):
        for message in (
            "I want to build a reminder app.",
            "Students feel overwhelmed.",
        ):
            with self.subTest(message=message):
                self.assertNotIn("frequency", RuleBasedExtractor.extract(message))


# ---------------------------------------------------------------------------
# Objective engine advances — natural frequency stops the repeat
# ---------------------------------------------------------------------------


class TestObjectiveAdvancesOnSufficiency(unittest.TestCase):
    """Once frequency holds a natural, usable cadence, the engine must NOT
    re-select FREQUENCY."""

    def test_frequency_no_longer_targeted_after_natural_statement(self):
        # Realistic flow: personas + problems captured earlier, the user just
        # answered the frequency question with "every single day". The engine
        # must move on to CURRENT_SOLUTIONS instead of re-asking frequency.
        state = ProjectState(
            personas=["students"],
            problems=["forgetting medication"],
            frequency="every single day",
        )
        obj = ObjectiveEngine().determine_next(state)
        self.assertNotEqual(obj.objective, Objective.FREQUENCY)
        self.assertEqual(obj.objective, Objective.CURRENT_SOLUTIONS)
        self.assertNotIn(StateField.FREQUENCY, obj.missing_fields)

    def test_qualitative_frequency_still_completes_stage(self):
        # Lifecycle unchanged: a qualitative (sufficient) frequency is enough
        # to WRAP_UP — no quantitative evidence required.
        state = ProjectState(
            personas=["students"], problems=["buried messages"],
            current_solutions=["whatsapp"], pain_points=["stress"],
            evidence=["observed"], frequency="every single day",
        )
        obj = ObjectiveEngine().determine_next(state)
        self.assertTrue(obj.is_wrap_up)

    def test_sufficiency_reported_on_objective(self):
        state = ProjectState(frequency="every single day")
        obj = ObjectiveEngine().determine_next(state)
        self.assertEqual(obj.sufficiency["frequency"], "SUFFICIENT")
        self.assertIn("personas", obj.sufficiency)

    def test_wrap_up_advancement_verdict(self):
        state = ProjectState(
            personas=["students"], problems=["p"], current_solutions=["c"],
            pain_points=["q"], evidence=["e"], frequency="daily",
        )
        obj = ObjectiveEngine().determine_next(state)
        self.assertEqual(obj.advancement, "WRAP_UP")
        self.assertTrue(obj.advancement_reason)

    def test_unknown_target_stays_active_with_reason(self):
        state = ProjectState()  # everything UNKNOWN -> PERSONAS targeted
        obj = ObjectiveEngine().determine_next(state)
        self.assertEqual(obj.advancement, "STAYED_ACTIVE")
        self.assertIn("stays active", obj.advancement_reason.lower())
        self.assertIn("PERSONAS", obj.advancement_reason)

    def test_reasoning_records_sufficiency_and_advancement(self):
        state = ProjectState(frequency="every single day")
        obj = ObjectiveEngine().determine_next(state)
        joined = " ".join(obj.reasoning)
        self.assertIn("Field sufficiency:", joined)
        self.assertIn("FREQUENCY=SUFFICIENT", joined)
        self.assertIn("Advancement:", joined)

    def test_invalid_advancement_value_rejected(self):
        with self.assertRaises(ValueError):
            ConversationObjective(
                objective=Objective.PERSONAS,
                missing_fields=[],
                completed_fields=[],
                advancement="NOT_A_VERDICT",
            )


# ---------------------------------------------------------------------------
# Fallback-skip regression — rejected LLM batch still captured
# ---------------------------------------------------------------------------


class TestRejectedBatchFallsBackToRuleBased(unittest.TestCase):
    """If the LLM returns a MEANINGFUL batch that Module 2 rejects (e.g. ADD
    on scalar frequency), the deterministic fallback must still run — this
    is one of the paths that previously caused repeated frequency questions."""

    def test_invalid_frequency_batch_triggers_deterministic_fallback(self):
        from session_manager import SessionData
        from mentor import _extract_and_update_state

        state = ProjectState()
        session_data = SessionData()
        rejected = ExtractionResult(
            message_type=MessageType.MEANINGFUL,
            updates=[ExtractionUpdate(Operation.ADD, StateField.FREQUENCY, "daily")],
        )
        with mock.patch(
            "mentor.MemoryExtractor.extract",
            return_value=rejected,
        ):
            _extract_and_update_state(
                "It happens every single day.",
                state, session_data, "test-model", None, "proj",
            )
        # The rejected ADD-on-frequency batch is dropped, but the rule-based
        # fallback captures the natural phrase instead of losing it.
        self.assertEqual(state.frequency, "every single day")


# ---------------------------------------------------------------------------
# Diagnostics — reason objective advanced / stayed active
# ---------------------------------------------------------------------------


class TestSufficiencyDiagnostics(unittest.TestCase):
    """The Developer Console explains confidence + advancement per turn."""

    AUDIT_USER = "audit_sufficiency"
    AUDIT_PROJ = "audit_sufficiency_proj"

    def setUp(self):
        from session_manager import get_session_manager
        get_session_manager().reset_runtime_state()

    def tearDown(self):
        from session_manager import get_session_manager
        try:
            get_session_manager().reset_runtime_state()
        except Exception:
            pass

    def _run_turn(self, message):
        import mentor
        with mock.patch(
            "mentor.MemoryExtractor.extract",
            return_value=ExtractionResult(MessageType.AMBIGUOUS, []),
        ):
            reply, _sess, _timing, diagnostics = mentor.process_mentor_turn(
                message, username=self.AUDIT_USER, project_name=self.AUDIT_PROJ
            )
        return reply, diagnostics

    def test_natural_frequency_advances_objective_with_explanation(self):
        """'It happens every single day.' → frequency reaches SUFFICIENT,
        the objective advances past it, and the diagnostics explain why."""
        reply, diagnostics = self._run_turn("It happens every single day.")

        trace = diagnostics["ObjectiveTrace"]
        # The engine advanced past frequency: it is not the active objective.
        self.assertNotEqual(trace["Objective"], "FREQUENCY")
        # Sufficiency classification is exposed per field.
        self.assertEqual(trace["Sufficiency"]["frequency"], "SUFFICIENT")
        # Advancement block explains the per-turn delta.
        advancement = trace["Advancement"]
        self.assertEqual(advancement["Verdict"], "ADVANCED")
        self.assertIn("frequency", advancement["Advanced Past"])
        # Confidence is explained, not just reported as a number.
        self.assertIn("Confidence Explanation", trace)
        self.assertTrue(trace["Confidence Explanation"])
        # Objective-level verdict/reason present.
        self.assertIn("Objective Verdict", advancement)
        self.assertIn("Objective Reason", advancement)
        # A reply was still produced (conversation continues).
        self.assertTrue(reply and reply.strip())

    def test_empty_state_stays_active_with_reason(self):
        _, diagnostics = self._run_turn("hello")
        advancement = diagnostics["ObjectiveTrace"]["Advancement"]
        self.assertEqual(advancement["Verdict"], "STAYED_ACTIVE")
        self.assertIn("Objective Reason", advancement)


if __name__ == "__main__":
    unittest.main(verbosity=2)
