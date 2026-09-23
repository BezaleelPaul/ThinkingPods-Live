"""
tests/test_conversation_metrics.py

Developer-only conversation quality metrics (observation-only). These tests
verify the metrics module and the Developer Console section WITHOUT touching
any decision path:

  * Pure unit tests for ``build_turn_metric`` / ``aggregate_metrics`` with
    hand-built records (no LLM, no persistence).
  * A live-pipeline integration test (fully mocked runtime) asserting the
    ``ConversationMetrics`` diagnostics section appears, accumulates across
    turns, and survives the JSON wire contract (X-Diagnostics).

The metrics are a pure function of per-turn records the pipeline already
produces; they must never influence a reply, objective, or state change.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import (  # noqa: E402
    ExtractionResult,
    MessageType,
    REQUIRED_FIELDS,
    ProjectState,
    StateField,
)
from module3 import ConversationObjective, Objective  # noqa: E402
from module4 import ResponseStrategy  # noqa: E402
from module5 import LifecycleDecision  # noqa: E402
from session_manager import SessionData  # noqa: E402

import mentor  # noqa: E402
from conversation_metrics import (  # noqa: E402
    aggregate_metrics,
    build_metrics_section,
    build_turn_metric,
)
from session_pipeline import MentorSession  # noqa: E402


_REQUIRED_VALUES = {sf.value for sf in REQUIRED_FIELDS}


class _RaisingOllama:
    """Stand-in for the ``ollama`` module that always fails, forcing the
    deterministic family-aware fallback path."""

    @staticmethod
    def chat(**kwargs):
        raise RuntimeError("ollama disabled in tests")


def _record(**overrides):
    """Hand-built per-turn metric record with sensible defaults."""
    base = {
        "turn_index": 1,
        "stage": "Empathize",
        "objective": Objective.PERSONAS.value,
        "advancement": "STAYED_ACTIVE",
        "strategy": ResponseStrategy.ASK_QUESTION.value,
        "lifecycle": LifecycleDecision.CONTINUE.value,
        "question_family": None,
        "family_reask": False,
        "family_skip_count": 0,
        "guard_blocked": False,
        "extraction_updates": 1,
        "rule_based_extraction": False,
        "completed_fields": [],
        "missing_fields": sorted(_REQUIRED_VALUES),
        "prompt_tokens": 100,
        "response_tokens": 20,
    }
    base.update(overrides)
    return base


def _objective(value=Objective.PERSONAS, completed=(), missing=None):
    return ConversationObjective(
        objective=value,
        missing_fields=[StateField(v) for v in (missing or (_REQUIRED_VALUES - set(completed)))],
        completed_fields=[StateField(v) for v in completed],
    )


class TestAggregateMetrics(unittest.TestCase):
    """Pure aggregation over hand-built records — all 12 metrics."""

    def test_turns_objectives_and_avg(self):
        metrics = aggregate_metrics([
            _record(turn_index=1, objective="PERSONAS", completed_fields=["personas"]),
            _record(turn_index=2, objective="PROBLEMS", completed_fields=["personas", "problems"]),
            _record(turn_index=3, objective="CURRENT_SOLUTIONS", completed_fields=["personas", "problems", "current_solutions"]),
        ])
        self.assertEqual(metrics["Turns"], 3)
        self.assertEqual(metrics["Objectives Completed"], 3)
        self.assertEqual(metrics["Avg Turns per Objective"], 1.0)
        self.assertEqual(metrics["Checklist Completion Rate"], 0.5)

    def test_repeated_question_attempts(self):
        metrics = aggregate_metrics([
            _record(turn_index=1, family_reask=False),
            _record(turn_index=2, family_reask=True),
            _record(turn_index=3, family_reask=False),
            _record(turn_index=4, family_reask=True),
        ])
        self.assertEqual(metrics["Repeated Question Attempts"], 2)

    def test_semantic_duplicates_prevented(self):
        metrics = aggregate_metrics([
            _record(turn_index=1, guard_blocked=True),
            _record(turn_index=2, guard_blocked=False),
            _record(turn_index=3, guard_blocked=True),
        ])
        self.assertEqual(metrics["Semantic Duplicates Prevented"], 2)

    def test_avg_extracted_facts_per_turn(self):
        metrics = aggregate_metrics([
            _record(turn_index=1, extraction_updates=2),
            _record(turn_index=2, extraction_updates=4),
            _record(turn_index=3, extraction_updates=6),
        ])
        self.assertEqual(metrics["Avg Extracted Facts per Turn"], 4.0)

    def test_token_estimates(self):
        metrics = aggregate_metrics([
            _record(turn_index=1, prompt_tokens=100, response_tokens=10),
            _record(turn_index=2, prompt_tokens=200, response_tokens=20),
        ])
        self.assertEqual(metrics["Total Prompt Tokens (est.)"], 300)
        self.assertEqual(metrics["Avg Prompt Tokens per Turn (est.)"], 150.0)
        self.assertEqual(metrics["Total Response Tokens (est.)"], 30)
        self.assertEqual(metrics["Avg Response Tokens per Turn (est.)"], 15.0)

    def test_objective_switches(self):
        metrics = aggregate_metrics([
            _record(turn_index=1, objective="PERSONAS"),
            _record(turn_index=2, objective="PERSONAS"),
            _record(turn_index=3, objective="PROBLEMS"),
            _record(turn_index=4, objective="PROBLEMS"),
            _record(turn_index=5, objective="CURRENT_SOLUTIONS"),
        ])
        self.assertEqual(metrics["Objective Switches"], 2)

    def test_unanswered_and_skipped_objectives(self):
        metrics = aggregate_metrics([
            _record(turn_index=1, objective="PERSONAS", completed_fields=[]),
        ])
        self.assertEqual(metrics["Unanswered Objectives"], ["personas"])
        self.assertEqual(
            metrics["Skipped Objectives"],
            sorted(_REQUIRED_VALUES - {"personas"}),
        )

    def test_wrap_up_not_counted_as_targeted(self):
        metrics = aggregate_metrics([
            _record(turn_index=1, objective="PERSONAS", completed_fields=["personas"]),
            _record(
                turn_index=2,
                objective=Objective.WRAP_UP.value,
                completed_fields=sorted(_REQUIRED_VALUES),
            ),
        ])
        self.assertEqual(metrics["Objectives Completed"], len(_REQUIRED_VALUES))
        self.assertEqual(metrics["Unanswered Objectives"], [])
        self.assertEqual(metrics["Skipped Objectives"], sorted(_REQUIRED_VALUES - {"personas"}))

    def test_empty_records(self):
        metrics = aggregate_metrics([])
        self.assertEqual(metrics["Turns"], 0)
        self.assertEqual(metrics["Objectives Completed"], 0)
        self.assertEqual(metrics["Avg Turns per Objective"], 0)
        self.assertEqual(metrics["Repeated Question Attempts"], 0)
        self.assertEqual(metrics["Semantic Duplicates Prevented"], 0)
        self.assertEqual(metrics["Avg Extracted Facts per Turn"], 0)
        self.assertEqual(metrics["Checklist Completion Rate"], 0.0)
        self.assertEqual(metrics["Total Prompt Tokens (est.)"], 0)
        self.assertEqual(metrics["Objective Switches"], 0)
        self.assertEqual(metrics["Unanswered Objectives"], [])
        self.assertEqual(metrics["Skipped Objectives"], sorted(_REQUIRED_VALUES))

    def test_aggregate_is_json_serialisable(self):
        metrics = aggregate_metrics([_record(), _record(turn_index=2)])
        round_tripped = json.loads(json.dumps(metrics))
        self.assertEqual(round_tripped, metrics)


class TestBuildTurnMetric(unittest.TestCase):
    """Record projection from real pipeline objects (deterministic, no I/O)."""

    def test_record_projects_capture_and_objects(self):
        capture = {
            "family_plan": {
                "selected": "PERSONAS_WHO",
                "reask": True,
                "skip_reasons": [{"family": "PERSONAS_WHO", "reason": "x"}],
            },
            "question_family": "PERSONAS_WHO",
            "guard_blocked": True,
            "extraction_updates": [{"operation": "ADD", "field": "personas", "value": "students"}],
            "extraction_message_type": "MEANINGFUL",
            "rule_based_extraction": {"target_audience": "students"},
            "prompt": "ROLE: You are a design thinking mentor.",
        }
        session_data = SessionData()
        session_data.current_stage = "Empathize"
        objective = _objective(completed=["personas"], missing=["problems"])
        record = build_turn_metric(
            1,
            capture,
            objective,
            ResponseStrategy.ASK_QUESTION,
            LifecycleDecision.CONTINUE,
            session_data,
            "Who is the target audience?",
        )
        self.assertEqual(record["turn_index"], 1)
        self.assertEqual(record["objective"], Objective.PERSONAS.value)
        self.assertEqual(record["question_family"], "PERSONAS_WHO")
        self.assertTrue(record["family_reask"])
        self.assertEqual(record["family_skip_count"], 1)
        self.assertTrue(record["guard_blocked"])
        self.assertEqual(record["extraction_updates"], 1)
        self.assertTrue(record["rule_based_extraction"])
        self.assertEqual(record["completed_fields"], ["personas"])
        self.assertIsNotNone(record["prompt_tokens"])
        self.assertIsNotNone(record["response_tokens"])

    def test_missing_capture_keys_default_safely(self):
        record = build_turn_metric(
            1,
            {},
            _objective(),
            ResponseStrategy.ASK_QUESTION,
            LifecycleDecision.CONTINUE,
            SessionData(),
            None,
        )
        self.assertFalse(record["family_reask"])
        self.assertFalse(record["guard_blocked"])
        self.assertEqual(record["extraction_updates"], 0)
        self.assertIsNone(record["question_family"])
        self.assertIsNone(record["prompt_tokens"])
        self.assertIsNone(record["response_tokens"])


class TestBuildMetricsSection(unittest.TestCase):
    """The Developer Console section view over a session's turn log."""

    def test_fresh_session_has_zero_section(self):
        metrics = build_metrics_section(SessionData())
        self.assertEqual(metrics["Turns"], 0)
        self.assertIn("Unanswered Objectives", metrics)

    def test_session_log_aggregates(self):
        sd = SessionData()
        sd.turn_metrics = [
            _record(turn_index=1, objective="PERSONAS", completed_fields=["personas"]),
            _record(turn_index=2, objective="PROBLEMS", completed_fields=["personas", "problems"]),
        ]
        metrics = build_metrics_section(sd)
        self.assertEqual(metrics["Turns"], 2)
        self.assertEqual(metrics["Objectives Completed"], 2)


class TestMetricsLivePipeline(unittest.TestCase):
    """The live pipeline emits and accumulates the ConversationMetrics section."""

    AUDIT_USER = "audit_metrics"
    AUDIT_PROJ = "audit_metrics_proj"

    def setUp(self):
        from session_manager import get_session_manager
        get_session_manager().reset_runtime_state()

    def tearDown(self):
        from session_manager import get_session_manager
        try:
            get_session_manager().reset_runtime_state()
        except Exception:
            pass

    def test_two_turns_accumulate_and_round_trip(self):
        from unittest import mock

        with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama()}):
            with mock.patch(
                "mentor.MemoryExtractor.extract",
                return_value=ExtractionResult(MessageType.AMBIGUOUS, []),
            ):
                _, _, _, diag1 = mentor.process_mentor_turn(
                    "I want to help elderly people take their medications on time",
                    username=self.AUDIT_USER,
                    project_name=self.AUDIT_PROJ,
                    model_name="test-model",
                )
                _, _, _, diag2 = mentor.process_mentor_turn(
                    "It happens every single day",
                    username=self.AUDIT_USER,
                    project_name=self.AUDIT_PROJ,
                    model_name="test-model",
                )

        m1 = diag1["ConversationMetrics"]
        m2 = diag2["ConversationMetrics"]
        self.assertEqual(m1["Turns"], 1)
        self.assertEqual(m2["Turns"], 2)
        self.assertGreaterEqual(m2["Objectives Completed"], 1)
        self.assertIn("Unanswered Objectives", m2)
        self.assertIn("Skipped Objectives", m2)

        # Wire contract: diagnostics are json.dumps'd into X-Diagnostics.
        round_tripped = json.loads(json.dumps(diag2))
        self.assertIn("ConversationMetrics", round_tripped)
        self.assertEqual(round_tripped["ConversationMetrics"]["Turns"], 2)

    def test_observation_log_persists_per_turn(self):
        from unittest import mock
        from session_manager import get_session_manager

        with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama()}):
            with mock.patch(
                "mentor.MemoryExtractor.extract",
                return_value=ExtractionResult(MessageType.AMBIGUOUS, []),
            ):
                mentor.process_mentor_turn(
                    "hello",
                    username=self.AUDIT_USER,
                    project_name=self.AUDIT_PROJ,
                    model_name="test-model",
                )

        sd = get_session_manager().get_active_session_data()
        self.assertEqual(len(sd.turn_metrics), 1)
        self.assertEqual(sd.turn_metrics[0]["turn_index"], 1)
        self.assertIn("objective", sd.turn_metrics[0])
        self.assertIn("completed_fields", sd.turn_metrics[0])

    def test_metrics_do_not_alter_reply_or_state(self):
        """The metrics section is additive: the reply and checklist state are
        unchanged from the deterministic fallback pipeline (no new decisions)."""
        from unittest import mock
        from module3 import family_label

        with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama()}):
            with mock.patch(
                "mentor.MemoryExtractor.extract",
                return_value=ExtractionResult(MessageType.AMBIGUOUS, []),
            ):
                reply, _session, _timing, diag = mentor.process_mentor_turn(
                    "I want to help elderly people take their medications on time",
                    username=self.AUDIT_USER,
                    project_name=self.AUDIT_PROJ,
                    model_name="test-model",
                )
        self.assertTrue(reply and reply.strip())
        # The question family of the produced reply must still be
        # deterministically classified (metrics didn't disturb the guard).
        self.assertTrue(diag["QuestionFamilies"]["Question Family"])
        checklist = diag["ChecklistReasoning"]
        self.assertTrue(checklist["target_audience"]["complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
