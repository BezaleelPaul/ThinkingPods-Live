"""
tests/test_recovery_monitor.py

Robustness regression suite for non-linear conversations.

Covers the five recovery categories — topic changes, contradictory answers,
uncertain answers, "I don't know", and multiple unrelated facts — plus the
deterministic guarantees: read-only analysis, preserved previously collected
facts, no interview restart, and a Developer-Console Recovery section that
always carries `Detected Inconsistency`, `Recovery Strategy`, and
`Clarification Reason`.

Tests are deterministic: unit tests exercise `analyze_recovery` directly with
hand-built state snapshots (no LLM, no session persistence); integration tests
replay live `process_mentor_turn` turns with a stubbed MemoryExtractor and a
raising ollama stub (deterministic fallback path).
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import (  # noqa: E402
    ExtractionResult,
    ExtractionUpdate,
    MessageType,
    Operation,
    StateField,
)
from recovery_monitor import (  # noqa: E402
    RecoveryCategory,
    RecoveryStrategy,
    analyze_recovery,
)


class _RaisingOllama:
    """Stand-in for the ``ollama`` module that always fails, forcing the
    deterministic family-aware fallback path."""

    @staticmethod
    def chat(**kwargs):
        raise RuntimeError("ollama disabled in tests")


def _state(**overrides):
    base = {
        "personas": [],
        "problems": [],
        "current_solutions": [],
        "pain_points": [],
        "evidence": [],
        "impacts": [],
        "frequency": None,
    }
    base.update(overrides)
    return base


def _updated(before, after):
    """The ``{field: ("added"/"updated", ...)}`` diff between snapshots."""
    return analyze_recovery(
        user_message="irrelevant",
        state_before=before,
        state_after=after,
    )


# ---------------------------------------------------------------------------
# Unit tests — direct analyze_recovery behaviour
# ---------------------------------------------------------------------------


class TestDontKnowRecovery(unittest.TestCase):
    def test_explicit_dont_know_detected(self):
        report = analyze_recovery(
            user_message="I don't know yet, let me think",
            state_before=_state(),
            state_after=_state(),
            last_assistant_message="Who would benefit from this?",
        )
        self.assertEqual(report.category, RecoveryCategory.DONT_KNOW)
        self.assertEqual(report.strategy, RecoveryStrategy.REPHRASE_FRESH_ANGLE)
        self.assertTrue(report.detected_inconsistency)
        self.assertTrue(report.clarification_reason)
        self.assertTrue(report.avoided_restart)

    def test_dont_know_variants(self):
        for text in ("I don't know", "no idea", "can't say", "haven't a clue"):
            with self.subTest(text=text):
                report = analyze_recovery(
                    user_message=text,
                    state_before=_state(),
                    state_after=_state(),
                )
                self.assertEqual(report.category, RecoveryCategory.DONT_KNOW)

    def test_dont_know_with_facts_is_not_dont_know(self):
        """If the turn actually captured facts, it is not classified as
        DONT_KNOW even if the phrase appears."""
        report = analyze_recovery(
            user_message="I don't know the exact count, but it happens daily",
            state_before=_state(),
            state_after=_state(frequency="daily"),
        )
        self.assertNotEqual(report.category, RecoveryCategory.DONT_KNOW)


class TestUncertainRecovery(unittest.TestCase):
    def test_uncertain_answer_detected(self):
        report = analyze_recovery(
            user_message="Not sure honestly",
            state_before=_state(),
            state_after=_state(),
        )
        self.assertEqual(report.category, RecoveryCategory.UNCERTAIN_ANSWER)
        self.assertEqual(report.strategy, RecoveryStrategy.REPHRASE_FRESH_ANGLE)
        self.assertTrue(report.detected_inconsistency)
        self.assertTrue(report.clarification_reason)

    def test_uncertain_variants(self):
        for text in ("not sure", "maybe", "probably", "it depends", "i guess so"):
            with self.subTest(text=text):
                report = analyze_recovery(
                    user_message=text,
                    state_before=_state(),
                    state_after=_state(),
                )
                self.assertEqual(report.category, RecoveryCategory.UNCERTAIN_ANSWER)

    def test_uncertain_with_facts_is_not_uncertain(self):
        report = analyze_recovery(
            user_message="I'm not sure but students probably",
            state_before=_state(),
            state_after=_state(personas=["students"]),
        )
        self.assertNotEqual(report.category, RecoveryCategory.UNCERTAIN_ANSWER)


class TestContradictionRecovery(unittest.TestCase):
    def test_scalar_overwrite_detected(self):
        before = _state(frequency="every single day")
        after = _state(frequency="three times a week")
        report = _updated(before, after)
        self.assertEqual(report.category, RecoveryCategory.CONTRADICTION)
        self.assertEqual(report.strategy, RecoveryStrategy.ACCEPT_NEW_VALUE)
        self.assertEqual(report.affected_fields, ("frequency",))
        self.assertIn("frequency", report.detected_inconsistency)
        self.assertTrue(report.clarification_reason)

    def test_first_scalar_value_is_not_a_contradiction(self):
        report = _updated(_state(), _state(frequency="every single day"))
        self.assertNotEqual(report.category, RecoveryCategory.CONTRADICTION)

    def test_unchanged_scalar_is_not_a_contradiction(self):
        report = _updated(
            _state(frequency="daily"), _state(frequency="daily")
        )
        self.assertEqual(report.category, RecoveryCategory.NONE)


class TestTopicChangeRecovery(unittest.TestCase):
    LAST_PROBLEM_QUESTION = "What is the core problem or frustration they're experiencing?"

    def test_topic_change_detected(self):
        """User adds a persona while the pending question was about problems."""
        before = _state(personas=["students"])
        after = _state(personas=["students", "parents"])
        report = analyze_recovery(
            user_message="Actually parents are affected too",
            state_before=before,
            state_after=after,
            last_assistant_message=self.LAST_PROBLEM_QUESTION,
        )
        self.assertEqual(report.category, RecoveryCategory.TOPIC_CHANGE)
        self.assertEqual(report.strategy, RecoveryStrategy.STEER_BACK)
        self.assertIn("personas", report.affected_fields)
        self.assertIn("problems", report.detected_inconsistency)
        self.assertEqual(report.preserved_facts, ("personas",))
        self.assertTrue(report.avoided_restart)

    def test_no_topic_change_when_question_was_answered(self):
        before = _state(personas=["students"])
        after = _state(personas=["students"], problems=["late submissions"])
        report = analyze_recovery(
            user_message="They keep submitting late",
            state_before=before,
            state_after=after,
            last_assistant_message=self.LAST_PROBLEM_QUESTION,
        )
        self.assertEqual(report.category, RecoveryCategory.NONE)

    def test_no_topic_change_without_last_question(self):
        report = analyze_recovery(
            user_message="parents are affected too",
            state_before=_state(personas=["students"]),
            state_after=_state(personas=["students", "parents"]),
            last_assistant_message=None,
        )
        self.assertEqual(report.category, RecoveryCategory.NONE)

    def test_no_topic_change_when_asked_field_already_satisfied(self):
        """If the pending field is already satisfied there is nothing to
        steer back to — the extra fact is simply captured."""
        before = _state(personas=["students"], problems=["forgetting"])
        after = _state(
            personas=["students", "parents"], problems=["forgetting"]
        )
        report = analyze_recovery(
            user_message="parents are affected too",
            state_before=before,
            state_after=after,
            last_assistant_message=self.LAST_PROBLEM_QUESTION,
        )
        self.assertNotEqual(report.category, RecoveryCategory.TOPIC_CHANGE)


class TestMultipleFactsRecovery(unittest.TestCase):
    def test_three_plus_fields_detected(self):
        before = _state()
        after = _state(
            personas=["students"],
            problems=["missing deadlines"],
            pain_points=["stress"],
            evidence=["interviewed 10 students"],
        )
        report = _updated(before, after)
        self.assertEqual(report.category, RecoveryCategory.MULTIPLE_UNRELATED_FACTS)
        self.assertEqual(report.strategy, RecoveryStrategy.CAPTURE_ALL_CONTINUE)
        self.assertEqual(len(report.affected_fields), 4)
        self.assertIn("4 fields", report.detected_inconsistency)
        self.assertTrue(report.clarification_reason)

    def test_two_fields_are_not_multiple(self):
        before = _state()
        after = _state(personas=["students"], problems=["missing deadlines"])
        report = _updated(before, after)
        self.assertNotEqual(
            report.category, RecoveryCategory.MULTIPLE_UNRELATED_FACTS
        )


class TestPreservationAndDeterminism(unittest.TestCase):
    def test_preserved_facts_listed(self):
        before = _state(personas=["students"], frequency="daily")
        after = _state(
            personas=["students"], problems=["late"], frequency="daily"
        )
        report = _updated(before, after)
        self.assertIn("personas", report.preserved_facts)
        self.assertIn("frequency", report.preserved_facts)
        self.assertNotIn("problems", report.preserved_facts)

    def test_never_suggests_restart(self):
        """Across every category the analysis never signals a restart."""
        cases = [
            (RecoveryCategory.DONT_KNOW, "I don't know", _state(), _state()),
            (
                RecoveryCategory.UNCERTAIN_ANSWER,
                "not sure",
                _state(),
                _state(),
            ),
            (
                RecoveryCategory.CONTRADICTION,
                "it happens weekly",
                _state(frequency="daily"),
                _state(frequency="weekly"),
            ),
            (
                RecoveryCategory.TOPIC_CHANGE,
                "parents are affected too",
                _state(personas=["students"]),
                _state(personas=["students", "parents"]),
            ),
            (
                RecoveryCategory.MULTIPLE_UNRELATED_FACTS,
                "students miss deadlines and it stresses them out",
                _state(),
                _state(personas=["a"], problems=["b"], pain_points=["c"]),
            ),
        ]
        for category, message, before, after in cases:
            with self.subTest(category=category.value):
                report = analyze_recovery(
                    user_message=message,
                    state_before=before,
                    state_after=after,
                    last_assistant_message="What is the core problem?",
                )
                self.assertEqual(report.category, category)
                self.assertTrue(report.avoided_restart)

    def test_deterministic_and_json_round_trip(self):
        import json

        args = dict(
            user_message="Actually parents are affected too",
            state_before=_state(personas=["students"]),
            state_after=_state(personas=["students", "parents"]),
            last_assistant_message="What is the core problem or frustration they're experiencing?",
        )
        a = analyze_recovery(**args).to_dict()
        b = analyze_recovery(**args).to_dict()
        self.assertEqual(a, b)
        self.assertEqual(json.loads(json.dumps(a)), a)


# ---------------------------------------------------------------------------
# Live pipeline — Recovery section in the Developer Console
# ---------------------------------------------------------------------------


def _extraction(message_type=MessageType.MEANINGFUL, updates=()):
    return ExtractionResult(
        message_type=message_type,
        updates=[
            ExtractionUpdate(
                operation=u[0], field=u[1], value=u[2]
            )
            for u in updates
        ],
    )


def _run_turn(user, extraction):
    import mentor

    # COMPLEXITY_GATE=false keeps this suite pinned to the deterministic
    # rule-based extractor the recovery assertions were written against.
    # The gate's own behavior is covered in test_complexity_gate.py.
    with mock.patch.dict(os.environ, {"COMPLEXITY_GATE": "false"}):
        with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama()}):
            with mock.patch(
                "mentor.MemoryExtractor.extract", return_value=extraction
            ):
                return mentor.process_mentor_turn(
                    user,
                    username="recovery_user",
                    project_name="RecoveryProject",
                    model_name="test-model",
                )


class TestRecoveryPipelineDiagnostics(unittest.TestCase):
    def setUp(self):
        from session_manager import get_session_manager

        get_session_manager().reset_runtime_state()

    def tearDown(self):
        from session_manager import get_session_manager

        try:
            get_session_manager().reset_runtime_state()
        except Exception:
            pass

    def test_recovery_section_always_present(self):
        _reply, _sess, _timing, diagnostics = _run_turn(
            "hello", _extraction(MessageType.AMBIGUOUS)
        )
        recovery = diagnostics["Recovery"]
        for key in (
            "Detected Inconsistency",
            "Recovery Strategy",
            "Clarification Reason",
            "Avoided Restart",
        ):
            self.assertIn(key, recovery)
        self.assertTrue(recovery["Avoided Restart"])

    def test_dont_know_turn_surfaces_recovery(self):
        _reply, _sess, _timing, diagnostics = _run_turn(
            "I don't know yet, let me think",
            _extraction(MessageType.AMBIGUOUS),
        )
        recovery = diagnostics["Recovery"]
        self.assertEqual(recovery["Recovery Strategy"], "REPHRASE_FRESH_ANGLE")
        self.assertTrue(recovery["Detected Inconsistency"])
        self.assertTrue(recovery["Clarification Reason"])
        self.assertEqual(recovery["Affected Fields"], [])

    def test_contradiction_turn_surfaces_recovery_and_preserves_state(self):
        # Turn 1: seed a frequency.
        _reply, _sess, _timing, _diag = _run_turn(
            "It happens every single day",
            _extraction(
                updates=[
                    (Operation.ADD, StateField.PERSONAS, "students"),
                    (Operation.SET, StateField.FREQUENCY, "every single day"),
                ]
            ),
        )
        # Turn 2: user corrects the frequency.
        _reply, _sess, _timing, diagnostics = _run_turn(
            "Actually it's more like three times a week",
            _extraction(
                updates=[
                    (Operation.SET, StateField.FREQUENCY, "three times a week")
                ]
            ),
        )
        recovery = diagnostics["Recovery"]
        self.assertEqual(recovery["Recovery Strategy"], "ACCEPT_NEW_VALUE")
        self.assertIn("frequency", recovery["Detected Inconsistency"])
        self.assertIn("frequency", recovery["Affected Fields"])
        # The earlier scalar was overwritten; the persona survived untouched.
        self.assertIn("personas", recovery["Preserved Facts"])
        self.assertEqual(
            diagnostics["ProjectState"]["Frequency"], "three times a week"
        )

    def test_topic_change_turn_surfaces_recovery(self):
        # Turn 1: persona established, mentor moves on to problems.
        _reply, _sess, _timing, _diag = _run_turn(
            "I want to help students",
            _extraction(
                updates=[(Operation.ADD, StateField.PERSONAS, "students")]
            ),
        )
        # Turn 2: user adds a persona instead of answering the problem q.
        _reply, _sess, _timing, diagnostics = _run_turn(
            "Actually parents are affected too",
            _extraction(
                updates=[(Operation.ADD, StateField.PERSONAS, "parents")]
            ),
        )
        recovery = diagnostics["Recovery"]
        self.assertEqual(recovery["Recovery Strategy"], "STEER_BACK")
        self.assertIn("personas", recovery["Detected Inconsistency"])
        self.assertIn("personas", recovery["Affected Fields"])
        # Both personas preserved — the interview did not restart. (The first
        # persona came from the deterministic rule extractor, which normalises
        # "students" to "college students".)
        self.assertEqual(
            diagnostics["ProjectState"]["Personas"],
            ["college students", "parents"],
        )

    def test_multiple_facts_turn_surfaces_recovery(self):
        # No deterministic audience cue and no problem cue, so the current
        # objective is NOT satisfied by rules -> the LLM path runs and
        # applies all four canned updates, which the recovery monitor flags
        # as multi-fact.
        _reply, _sess, _timing, diagnostics = _run_turn(
            "It stresses them out and I interviewed ten of them",
            _extraction(
                updates=[
                    (Operation.ADD, StateField.PERSONAS, "students"),
                    (Operation.ADD, StateField.PROBLEMS, "missing deadlines"),
                    (Operation.ADD, StateField.PAIN_POINTS, "stress"),
                    (Operation.ADD, StateField.EVIDENCE, "interviewed 10 students"),
                ]
            ),
        )
        recovery = diagnostics["Recovery"]
        self.assertEqual(
            recovery["Recovery Strategy"], "CAPTURE_ALL_CONTINUE"
        )
        self.assertIn("4 fields", recovery["Detected Inconsistency"])
        self.assertEqual(len(recovery["Affected Fields"]), 4)

    def test_recovery_section_is_json_serialisable(self):
        import json

        _reply, _sess, _timing, diagnostics = _run_turn(
            "I don't know yet", _extraction(MessageType.AMBIGUOUS)
        )
        round_tripped = json.loads(json.dumps(diagnostics))
        self.assertEqual(
            round_tripped["Recovery"]["Recovery Strategy"],
            diagnostics["Recovery"]["Recovery Strategy"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
