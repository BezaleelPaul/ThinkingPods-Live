"""
tests/test_extraction_accuracy.py

Regression tests for the measurement-only LLM Extraction Accuracy Audit
(``extraction_accuracy`` + its wiring in ``mentor``):

  * the deterministic per-update classifier labels every LLM-extracted
    update as ``Correct`` / ``Partially Correct`` (Duplicate, Too generic) /
    ``Incorrect`` (Wrong field, Should have been …) / ``Missed Opportunity``
    (missed explicit/implicit fact) — with the required reason keys only;
  * ``assess_extraction_accuracy`` builds a JSON-safe per-turn record
    carrying user message, current objective, extracted updates, state
    before, and state after;
  * missed facts detected via the rule-extractor proxy attach to the first
    Correct update of the same field, or stay turn-level when no update
    exists;
  * ``ExtractionAccuracySummary`` aggregation, per-field precision, display
    projection, ``merge`` and persistence round-trip;
  * the live pipeline surfaces the Developer Console sections without
    changing state, objectives, or replies (measurement only).

All tests are deterministic (mocked ollama + mocked MemoryExtractor).
"""

import copy
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import extraction_accuracy  # noqa: E402
from extraction_accuracy import (  # noqa: E402
    CORRECT,
    INCORRECT,
    MISSED_OPPORTUNITY,
    PARTIALLY_CORRECT,
    REASON_DUPLICATE,
    REASON_MISSED_EXPLICIT,
    REASON_MISSED_IMPLICIT,
    REASON_SHOULD_EVIDENCE,
    REASON_SHOULD_IMPACT,
    REASON_SHOULD_MOTIVATION,
    REASON_SHOULD_PAIN_POINT,
    REASON_TOO_GENERIC,
    REASON_WRONG_FIELD,
    ExtractionAccuracySummary,
    assess_extraction_accuracy,
    classify_update,
)
from memory_extractor import (  # noqa: E402
    ExtractionResult,
    ExtractionUpdate,
    MessageType,
    Operation,
    ProjectState,
    StateField,
)
from session_manager import SessionData  # noqa: E402

import mentor  # noqa: E402


class _RaisingOllama:
    """Stand-in for the ``ollama`` module that always fails, forcing the
    deterministic family-aware fallback path."""

    @staticmethod
    def chat(**kwargs):
        raise RuntimeError("ollama disabled in tests")


def _meaningful(*updates) -> ExtractionResult:
    return ExtractionResult(
        message_type=MessageType.MEANINGFUL,
        updates=[
            ExtractionUpdate(
                operation=Operation(u[0]),
                field=StateField(u[1]),
                value=u[2],
            )
            for u in updates
        ],
    )


def _empty_state() -> dict:
    return {
        "personas": [],
        "problems": [],
        "current_solutions": [],
        "pain_points": [],
        "evidence": [],
        "impacts": [],
        "frequency": None,
    }


class TestClassifyUpdate(unittest.TestCase):
    """Deterministic per-update classification."""

    def _u(self, op, field, value):
        return ExtractionUpdate(
            operation=Operation(op), field=StateField(field), value=value
        )

    def test_correct_fresh_specific_fact(self):
        label, reasons = classify_update(
            self._u("ADD", "problems", "forgetting to take medication on time"),
            _empty_state(),
        )
        self.assertEqual(label, CORRECT)
        self.assertEqual(reasons, [])

    def test_duplicate_persona_is_partially_correct(self):
        state = _empty_state()
        state["personas"] = ["students"]
        label, reasons = classify_update(
            self._u("ADD", "personas", "Students"), state
        )
        self.assertEqual(label, PARTIALLY_CORRECT)
        self.assertEqual(reasons, [REASON_DUPLICATE])

    def test_duplicate_frequency_is_partially_correct(self):
        state = _empty_state()
        state["frequency"] = "weekly"
        label, reasons = classify_update(
            self._u("SET", "frequency", "weekly"), state
        )
        self.assertEqual(label, PARTIALLY_CORRECT)
        self.assertEqual(reasons, [REASON_DUPLICATE])

    def test_generic_value_is_partially_correct(self):
        label, reasons = classify_update(
            self._u("ADD", "problems", "the problem"), _empty_state()
        )
        self.assertEqual(label, PARTIALLY_CORRECT)
        self.assertEqual(reasons, [REASON_TOO_GENERIC])

    def test_observation_misfiled_as_problem_is_incorrect(self):
        label, reasons = classify_update(
            self._u("ADD", "problems", "I interviewed 5 students who all lose marks"),
            _empty_state(),
        )
        self.assertEqual(label, INCORRECT)
        self.assertEqual(reasons, [REASON_SHOULD_EVIDENCE])

    def test_motivation_misfiled_is_incorrect(self):
        label, reasons = classify_update(
            self._u("ADD", "problems", "I want to help students be more disciplined"),
            _empty_state(),
        )
        self.assertEqual(label, INCORRECT)
        self.assertEqual(reasons, [REASON_SHOULD_MOTIVATION])

    def test_emotional_state_misfiled_is_incorrect(self):
        label, reasons = classify_update(
            self._u("ADD", "problems", "they feel stressed and anxious"),
            _empty_state(),
        )
        self.assertEqual(label, INCORRECT)
        self.assertEqual(reasons, [REASON_SHOULD_PAIN_POINT])

    def test_consequence_misfiled_is_incorrect(self):
        label, reasons = classify_update(
            self._u("ADD", "problems", "this causes missed deadlines and lost marks"),
            _empty_state(),
        )
        self.assertEqual(label, INCORRECT)
        self.assertEqual(reasons, [REASON_SHOULD_IMPACT])

    def test_wrong_field_generic_reason(self):
        label, reasons = classify_update(
            self._u("ADD", "personas", "using sticky notes to track tasks"),
            _empty_state(),
        )
        self.assertEqual(label, INCORRECT)
        self.assertIn(REASON_WRONG_FIELD, reasons)

    def test_ambiguous_value_not_called_wrong(self):
        """A value with signals for multiple fields ties -> not misfielded."""
        label, reasons = classify_update(
            self._u("ADD", "problems", "students forget assignments"),
            _empty_state(),
        )
        self.assertEqual(label, CORRECT)


class TestAssessExtractionAccuracy(unittest.TestCase):
    """Per-turn record construction."""

    def test_record_shape(self):
        record = assess_extraction_accuracy(
            "students forget assignments",
            "PERSONAS",
            _meaningful(("ADD", "personas", "students")),
            _empty_state(),
            {"target_audience": "students"},
        )
        self.assertTrue(record["llm_ran"])
        self.assertEqual(record["message_type"], "MEANINGFUL")
        self.assertEqual(record["objective"], "PERSONAS")
        self.assertEqual(record["user_message"], "students forget assignments")
        self.assertEqual(record["state_before"]["personas"], [])
        self.assertEqual(record["state_after"], {})
        self.assertEqual(len(record["updates"]), 1)
        entry = record["updates"][0]
        self.assertEqual(entry["label"], CORRECT)
        self.assertIn("operation", entry)
        self.assertIn("field", entry)
        self.assertIn("value", entry)

    def test_empty_result_still_records(self):
        record = assess_extraction_accuracy(
            "hello",
            "PERSONAS",
            ExtractionResult(MessageType.NO_UPDATE, []),
            _empty_state(),
        )
        self.assertEqual(record["updates"], [])
        self.assertEqual(record["missed_opportunities"], [])

    def test_explicit_miss_attaches_to_first_correct_update(self):
        record = assess_extraction_accuracy(
            "students keep forgetting assignments and missing deadlines",
            "PROBLEMS",
            _meaningful(
                ("ADD", "personas", "students"),
                ("ADD", "problems", "forgetting assignments"),
            ),
            _empty_state(),
            {"target_audience": "students", "pain_point": "missing deadlines"},
        )
        entries = {e["field"]: e for e in record["updates"]}
        self.assertEqual(entries["problems"]["label"], MISSED_OPPORTUNITY)
        self.assertEqual(
            entries["problems"]["reasons"], [REASON_MISSED_EXPLICIT]
        )
        self.assertEqual(entries["personas"]["label"], CORRECT)
        self.assertTrue(any(
            m["attached_to_update"] is not None
            for m in record["missed_opportunities"]
        ))

    def test_unattached_miss_stays_turn_level(self):
        record = assess_extraction_accuracy(
            "students use sticky notes to cope with the workload",
            "PERSONAS",
            _meaningful(("ADD", "personas", "students")),
            _empty_state(),
            {"target_audience": "students", "existing_solution": "sticky notes"},
        )
        self.assertEqual(len(record["updates"]), 1)
        self.assertEqual(record["updates"][0]["label"], CORRECT)
        missed = [
            m for m in record["missed_opportunities"]
            if m["attached_to_update"] is None
        ]
        self.assertTrue(missed)
        self.assertEqual(missed[0]["field"], "current_solutions")

    def test_implicit_miss_detected(self):
        record = assess_extraction_accuracy(
            "students feel stressed about assignments",
            "PERSONAS",
            _meaningful(("ADD", "personas", "students")),
            _empty_state(),
            {"target_audience": "students"},
        )
        kinds = [k for m in record["missed_opportunities"] for k in m["kind"]]
        self.assertIn("implicit", kinds)

    def test_record_is_json_serialisable(self):
        import json
        record = assess_extraction_accuracy(
            "students forget assignments every day",
            "PROBLEMS",
            _meaningful(
                ("ADD", "personas", "students"),
                ("ADD", "problems", "forgetting assignments"),
                ("SET", "frequency", "every day"),
            ),
            _empty_state(),
        )
        round_tripped = json.loads(json.dumps(record))
        self.assertEqual(round_tripped["updates"], record["updates"])


class TestExtractionAccuracySummary(unittest.TestCase):
    """Aggregation, precision, display, merge, persistence."""

    def test_add_record_aggregates_labels_and_reasons(self):
        summary = ExtractionAccuracySummary()
        record = assess_extraction_accuracy(
            "students forget assignments and miss deadlines",
            "PERSONAS",
            _meaningful(
                ("ADD", "personas", "students"),
                ("ADD", "problems", "the problem"),
                ("ADD", "problems", "I interviewed 5 students"),
                ("ADD", "problems", "forgetting assignments"),
            ),
            _empty_state(),
            {"pain_point": "missing deadlines"},
        )
        summary.add_record(record)
        # personas -> Correct; "the problem" -> Too generic; the interview
        # line -> Should have been evidence; the first Correct problems
        # update absorbs the missed explicit fact ("missing deadlines").
        self.assertEqual(summary.correct, 1)
        self.assertEqual(summary.partially_correct, 1)
        self.assertEqual(summary.incorrect, 1)
        self.assertEqual(summary.missed, 1)
        self.assertIn(REASON_TOO_GENERIC, summary.reason_counts)
        self.assertIn(REASON_SHOULD_EVIDENCE, summary.reason_counts)
        self.assertIn(REASON_MISSED_EXPLICIT, summary.reason_counts)

    def test_per_field_precision(self):
        summary = ExtractionAccuracySummary()
        record = assess_extraction_accuracy(
            "students forget assignments every day",
            "PERSONAS",
            _meaningful(
                ("ADD", "personas", "students"),
                ("ADD", "problems", "forgetting assignments"),
                ("SET", "frequency", "every day"),
            ),
            _empty_state(),
        )
        summary.add_record(record)
        precision = summary.per_field_precision()
        self.assertEqual(precision["Audience"]["classified"], 1)
        self.assertEqual(precision["Audience"]["correct"], 1)
        self.assertEqual(precision["Audience"]["pct"], 100.0)
        self.assertEqual(precision["Problems"]["classified"], 1)
        self.assertEqual(precision["Problems"]["pct"], 100.0)
        self.assertEqual(precision["Frequency"]["classified"], 1)
        self.assertEqual(precision["Impact"]["classified"], 0)

    def test_motivation_bucket(self):
        summary = ExtractionAccuracySummary()
        record = assess_extraction_accuracy(
            "I want to make students more disciplined",
            "PERSONAS",
            _meaningful(
                ("ADD", "pain_points", "I want to help students build discipline"),
                ("ADD", "personas", "students"),
            ),
            _empty_state(),
        )
        summary.add_record(record)
        precision = summary.per_field_precision()
        self.assertEqual(precision["Motivation"]["classified"], 1)
        self.assertEqual(precision["Motivation"]["correct"], 1)

    def test_merge(self):
        a = ExtractionAccuracySummary()
        b = ExtractionAccuracySummary()
        a.correct = 3
        a.field_counts["personas"] = 3
        a.field_correct["personas"] = 3
        b.correct = 2
        b.field_counts["problems"] = 2
        b.field_correct["problems"] = 2
        a.merge(b)
        self.assertEqual(a.correct, 5)
        self.assertEqual(a.field_counts["personas"], 3)
        self.assertEqual(a.field_counts["problems"], 2)

    def test_persistence_round_trip(self):
        summary = ExtractionAccuracySummary()
        record = assess_extraction_accuracy(
            "students forget assignments",
            "PROBLEMS",
            _meaningful(("ADD", "problems", "the problem")),
            _empty_state(),
        )
        summary.add_record(record)
        restored = ExtractionAccuracySummary.from_dict(summary.to_dict())
        self.assertEqual(restored.to_dict(), summary.to_dict())

    def test_session_data_persists_summary(self):
        sd = SessionData()
        record = assess_extraction_accuracy(
            "students forget assignments",
            "PROBLEMS",
            _meaningful(("ADD", "problems", "forgetting assignments")),
            _empty_state(),
        )
        sd.extraction_accuracy_summary.add_record(record)
        restored = SessionData.from_dict(sd.to_dict())
        self.assertEqual(
            restored.extraction_accuracy_summary.to_dict(),
            sd.extraction_accuracy_summary.to_dict(),
        )


class TestAccuracyWiringLivePipeline(unittest.TestCase):
    """The live pipeline surfaces the sections; measurement only."""

    USER = "accuracy_user"
    PROJ = "AccuracyProject"

    def setUp(self):
        from session_manager import get_session_manager

        get_session_manager().reset_runtime_state()

    def tearDown(self):
        from session_manager import get_session_manager

        try:
            get_session_manager().reset_runtime_state()
        except Exception:
            pass

    def _run_turn(self, user_message, patch_extract, gate="true"):
        with mock.patch.dict(os.environ, {"COMPLEXITY_GATE": gate}):
            with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama()}):
                with mock.patch(
                    "mentor.MemoryExtractor.extract", side_effect=patch_extract
                ):
                    return mentor.process_mentor_turn(
                        user_message,
                        username=self.USER,
                        project_name=self.PROJ,
                        model_name="test-model",
                    )

    def _active_summary(self):
        from session_manager import get_session_manager

        return get_session_manager().get_active_session_data().extraction_accuracy_summary

    def test_llm_turn_surfaces_section(self):
        def _patched(*args, **kwargs):
            return _meaningful(
                ("ADD", "personas", "students"),
                ("ADD", "problems", "forgetting assignments"),
                ("ADD", "problems", "I interviewed 5 students"),
            )

        reply, _session, _timing, diagnostics = self._run_turn(
            "students keep forgetting assignments, I interviewed 5 students",
            _patched,
        )
        section = diagnostics["ExtractionAccuracy"]
        self.assertTrue(section)
        self.assertEqual(section["Current Objective"], "PERSONAS")
        self.assertIn("students", section["Message"])
        updates = section["Extracted Updates"]
        self.assertEqual(len(updates), 3)
        by_value = {u["value"]: u for u in updates}
        self.assertEqual(by_value["students"]["label"], CORRECT)
        self.assertEqual(
            by_value["forgetting assignments"]["label"], CORRECT
        )
        misfield = next(u for u in updates if "interviewed" in u["value"])
        self.assertEqual(misfield["label"], INCORRECT)
        self.assertEqual(misfield["reasons"], [REASON_SHOULD_EVIDENCE])

        summary = diagnostics["ExtractionAccuracySummary"]
        self.assertGreaterEqual(summary["Correct"], 2)
        self.assertGreaterEqual(summary["Incorrect"], 1)
        self.assertGreaterEqual(self._active_summary().total(), 3)
        self.assertTrue(reply and reply.strip())

    def test_skip_turn_has_no_section(self):
        """When the hybrid decision skips the LLM, there is nothing to audit."""

        def _patched(*args, **kwargs):
            raise AssertionError("LLM extractor must not run on a skip")

        reply, _session, _timing, diagnostics = self._run_turn(
            "I want to help elderly people take their medications on time",
            _patched,
            gate="false",
        )
        self.assertEqual(diagnostics["ExtractionAccuracy"], {})
        self.assertEqual(self._active_summary().total(), 0)
        self.assertIn("ExtractionAccuracy", diagnostics)

    def test_audit_changes_nothing_about_the_pipeline(self):
        """The accuracy audit is observation-only: the turn's state,
        objective, and reply are exactly the live behavior."""

        def _patched(*args, **kwargs):
            return _meaningful(
                ("ADD", "personas", "students"),
                ("ADD", "problems", "forgetting assignments"),
            )

        reply, _session, _timing, diagnostics = self._run_turn(
            "students keep forgetting assignments every single day",
            _patched,
        )
        from session_manager import get_session_manager

        state = get_session_manager().get_active_session_data().project_state.to_state_dict()
        self.assertEqual(state["personas"], ["students"])
        self.assertEqual(state["problems"], ["forgetting assignments"])
        # The audit never mutated state with anything beyond the applied LLM
        # updates (state_after recorded, not applied by the audit).
        self.assertNotIn("evidence", state["problems"])
        self.assertTrue(reply and reply.strip())
        self.assertIn("ExtractionAccuracy", diagnostics)


if __name__ == "__main__":
    unittest.main()
