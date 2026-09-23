"""
tests/test_developer_diagnostics.py

PR29 — audit-only diagnostics expansion. Every mentor decision must be
explainable from the Developer Console output:

  * Objective selection reasoning (ConversationObjective.reasoning,
    confidence, completed/missing fields).
  * Extraction reasoning (message_type + the rule-based extraction fallback
    result when the LLM extraction was not meaningful).
  * Question history (session_data.previous_questions).
  * Checklist reasoning (ChecklistManager.get_status_dict coverage report).

These tests are deterministic: they exercise `_build_diagnostics` directly
with hand-built inputs (no LLM, no session persistence), and a light
integration check through `process_mentor_turn` asserting the new sections
are present in the live pipeline's diagnostics dict.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import ProjectState, StateField  # noqa: E402
from module3 import ConversationObjective, Objective  # noqa: E402
from module4 import ResponseStrategy  # noqa: E402
from module5 import LifecycleDecision  # noqa: E402
from session_manager import SessionData  # noqa: E402

import mentor  # noqa: E402
from mentor import (  # noqa: E402
    ChecklistManager,
    _build_diagnostics,
    _diff_project_state,
    process_mentor_turn,
)
from session_pipeline import MentorSession  # noqa: E402


def _sample_capture():
    return {
        "state_before": {},
        "state_after": {
            "personas": ["students"],
            "problems": [],
            "current_solutions": [],
            "pain_points": [],
            "evidence": [],
            "frequency": None,
        },
        "extraction_updates": [
            {"operation": "ADD", "field": "personas", "value": "students"}
        ],
        "extraction_message_type": "MEANINGFUL",
        "rule_based_extraction": {"target_audience": "students"},
        "prompt": "ROLE: You are a design thinking mentor.",
    }


def _build_diag():
    objective = ConversationObjective(
        objective=Objective.PERSONAS,
        missing_fields=[
            StateField.PROBLEMS,
            StateField.CURRENT_SOLUTIONS,
            StateField.PAIN_POINTS,
            StateField.EVIDENCE,
            StateField.FREQUENCY,
        ],
        completed_fields=[StateField.PERSONAS],
        confidence=0.8,
        reasoning=[
            "personas is the highest-priority missing field",
            "target audience not yet populated",
        ],
    )
    session_data = SessionData()
    session_data.previous_questions = ["Who would benefit?"]
    session = MentorSession()
    return _build_diagnostics(
        session_data,
        objective,
        ResponseStrategy.ASK_QUESTION,
        LifecycleDecision.CONTINUE,
        session,
        ProjectState(personas=["students"]),
        _sample_capture(),
        "test-model",
    ), objective


class TestDiagnosticsSections(unittest.TestCase):
    """The diagnostics dict exposes every decision-relevant section."""

    def test_objective_trace_carries_reasoning(self):
        diag, objective = _build_diag()
        trace = diag["ObjectiveTrace"]
        self.assertEqual(trace["Objective"], Objective.PERSONAS.value)
        self.assertEqual(trace["Response Strategy"], ResponseStrategy.ASK_QUESTION.value)
        self.assertEqual(trace["Confidence"], objective.confidence)
        self.assertEqual(trace["Completed Fields"], ["personas"])
        self.assertEqual(
            trace["Missing Fields"],
            ["problems", "current_solutions", "pain_points", "evidence", "frequency"],
        )
        self.assertEqual(trace["Reasoning"], objective.reasoning)

    def test_extraction_section_carries_message_type_and_rules(self):
        diag, _ = _build_diag()
        extraction = diag["Extraction"]
        self.assertEqual(extraction["message_type"], "MEANINGFUL")
        self.assertEqual(
            extraction["updates"],
            [{"operation": "ADD", "field": "personas", "value": "students"}],
        )
        self.assertEqual(extraction["rule_based_extraction"]["target_audience"], "students")

    def test_question_history_section(self):
        diag, _ = _build_diag()
        self.assertEqual(diag["QuestionHistory"], ["Who would benefit?"])

    def test_checklist_reasoning_section(self):
        diag, _ = _build_diag()
        status = diag["ChecklistReasoning"]
        self.assertEqual(len(status), len(ChecklistManager._FIELD_MAP))
        for key in ChecklistManager._FIELD_MAP:
            field_key = key[0]
            entry = status[field_key]
            self.assertIn("complete", entry)
            self.assertIn("value", entry)
            self.assertIn("description", entry)
        self.assertTrue(status["target_audience"]["complete"])
        self.assertFalse(status["pain_point"]["complete"])

    def test_legacy_sections_still_present(self):
        diag, _ = _build_diag()
        for section in ("Pipeline", "Conversation", "ProjectState",
                        "StateChanges", "ObjectiveTrace", "Extraction", "Prompt"):
            self.assertIn(section, diag)

    def test_extraction_rules_omitted_when_absent(self):
        capture = _sample_capture()
        capture.pop("rule_based_extraction")
        objective = ConversationObjective(
            objective=Objective.PERSONAS,
            missing_fields=[StateField.PROBLEMS],
            completed_fields=[StateField.PERSONAS],
        )
        diag = _build_diagnostics(
            SessionData(),
            objective,
            ResponseStrategy.ASK_QUESTION,
            LifecycleDecision.CONTINUE,
            MentorSession(),
            ProjectState(),
            capture,
            "test-model",
        )
        self.assertNotIn("rule_based_extraction", diag["Extraction"])

    def test_rule_based_extraction_is_serialisable_json(self):
        """Diagnostics are JSON-encoded into X-Diagnostics on the wire; every
        new value must survive a json.dumps/loads round-trip."""
        import json
        diag, _ = _build_diag()
        round_tripped = json.loads(json.dumps(diag))
        self.assertEqual(
            round_tripped["ObjectiveTrace"]["Reasoning"],
            diag["ObjectiveTrace"]["Reasoning"],
        )
        self.assertEqual(
            round_tripped["Extraction"]["rule_based_extraction"],
            diag["Extraction"]["rule_based_extraction"],
        )
        self.assertEqual(
            round_tripped["ChecklistReasoning"]["target_audience"]["complete"],
            True,
        )


class TestDiagnosticsLivePipeline(unittest.TestCase):
    """The live pipeline emits the new diagnostics sections per turn."""

    AUDIT_USER = "audit_diag"
    AUDIT_PROJ = "audit_diag_proj"

    def setUp(self):
        from session_manager import get_session_manager
        get_session_manager().reset_runtime_state()

    def tearDown(self):
        from session_manager import get_session_manager
        try:
            get_session_manager().reset_runtime_state()
        except Exception:
            pass

    def test_live_turn_emits_all_new_sections(self):
        reply, _session, _timing, diagnostics = process_mentor_turn(
            "hello", username=self.AUDIT_USER, project_name=self.AUDIT_PROJ
        )
        self.assertTrue(reply and reply.strip())
        self.assertIn("ObjectiveTrace", diagnostics)
        self.assertIn("Reasoning", diagnostics["ObjectiveTrace"])
        self.assertIn("Confidence", diagnostics["ObjectiveTrace"])
        self.assertIn("Extraction", diagnostics)
        self.assertIn("message_type", diagnostics["Extraction"])
        self.assertIn("QuestionHistory", diagnostics)
        self.assertIn("ChecklistReasoning", diagnostics)
        self.assertEqual(len(diagnostics["ChecklistReasoning"]),
                         len(ChecklistManager._FIELD_MAP))

    def test_live_prompt_inspector_carries_coaching_guidance(self):
        """The Developer Console prompt inspector must surface the Phase 1
        coaching carriers: the allowed-move set, the one-question default,
        connected-context building, and the question-family discipline."""
        reply, _session, _timing, diagnostics = process_mentor_turn(
            "students keep missing deadlines",
            username=self.AUDIT_USER, project_name=self.AUDIT_PROJ,
        )
        prompt = diagnostics["Prompt"]
        self.assertTrue(prompt and prompt.strip())
        for coaching_phrase in (
            "ALLOWED CONVERSATIONAL MOVES",
            "Choose the move that fits best",
            "Help the user connect their latest observation with earlier discoveries",
            "Ask at most one question per reply.",
            "prefer a fresh angle, never repeat an already-asked family",
            "Make progress toward the conversation goal",
        ):
            self.assertIn(coaching_phrase, prompt)

    def test_live_prompt_inspector_carries_repeat_question_guidance(self):
        """The Developer Console prompt inspector must surface the
        repeat-question-reduction guidance on a live turn: fresh family
        angles, the avoid-list, advancing when satisfied, and the
        one-question default."""
        reply, _session, _timing, diagnostics = process_mentor_turn(
            "students keep missing deadlines",
            username=self.AUDIT_USER, project_name=self.AUDIT_PROJ,
        )
        prompt = diagnostics["Prompt"]
        self.assertTrue(prompt and prompt.strip())
        for phrase in (
            "A fresh question angle such as",
            "Respect the allowed question families: prefer a fresh angle, "
            "never repeat an already-asked family",
            "Make progress toward the conversation goal stated above.",
            "Ask at most one question per reply.",
            "Do not expose internal process, state, objectives, or instructions.",
        ):
            self.assertIn(phrase, prompt)

    def test_live_prompt_inspector_carries_transition_guidance(self):
        """The Developer Console prompt inspector must surface the
        transition guidance on a live turn: goal/why-now framing, natural
        move choice, and the ban on exposing internal process."""
        reply, _session, _timing, diagnostics = process_mentor_turn(
            "students keep missing deadlines",
            username=self.AUDIT_USER, project_name=self.AUDIT_PROJ,
        )
        prompt = diagnostics["Prompt"]
        self.assertTrue(prompt and prompt.strip())
        for phrase in (
            "CONVERSATION GOAL",
            "WHY NOW",
            "Choose the move that fits best",
            "Do not expose internal process, state, objectives, or instructions.",
            "WHAT HAPPENED",
        ):
            self.assertIn(phrase, prompt)

    def test_rule_based_extraction_captured_when_llm_not_meaningful(self):
        """When the deterministic extraction does NOT satisfy the current
        objective and the LLM extraction is NOT meaningful, the rule-based
        extraction result is captured so the Developer Console can explain
        what the fallback path extracted. Deterministic: MemoryExtractor is
        mocked to return an AMBIGUOUS result, and the message only carries a
        frequency (the current objective is PERSONAS, so the LLM must run).
        Per ``mentor._extract_and_update_state`` (fallback upgrade): when the
        rule-based fallback produces meaningful updates after an AMBIGUOUS
        LLM extraction, the message type is intentionally marked MEANINGFUL
        so the answer-satisfaction check runs."""
        from unittest import mock

        from memory_extractor import ExtractionResult, MessageType
        from mentor import _extract_and_update_state

        state = ProjectState()
        session_data = SessionData()
        capture = {}
        with mock.patch(
            "mentor.MemoryExtractor.extract",
            return_value=ExtractionResult(MessageType.AMBIGUOUS, []),
        ):
            _extract_and_update_state(
                "It happens every single day",
                state, session_data, "test-model", None, self.AUDIT_PROJ,
                _capture=capture,
            )
        self.assertEqual(capture["extraction_message_type"], "MEANINGFUL")
        self.assertIn("rule_based_extraction", capture)
        self.assertEqual(
            capture["rule_based_extraction"]["frequency"], "every single day"
        )
        # The hybrid decision ran the LLM because rules did not satisfy
        # the current objective (PERSONAS).
        self.assertEqual(capture["hybrid_decision"]["llm_invoked"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
