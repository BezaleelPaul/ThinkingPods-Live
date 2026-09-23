"""
tests/test_insight_detection.py — unit + integration tests for the
Insight Detection layer.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mentor  # noqa: E402
from insight_detection import (  # noqa: E402
    InsightConfidence,
    InsightType,
    detect_insight,
    insight_diagnostics_section,
    insight_instruction_bullet,
)


class _RaisingOllama:
    """Stub that forces the deterministic fallback path."""

    @staticmethod
    def chat(**kwargs):
        raise RuntimeError("ollama disabled in tests")


class _CountingOllama:
    """Stub that records LLM call count and returns a canned reply."""

    calls = 0

    @classmethod
    def chat(cls, **kwargs):
        cls.calls += 1
        return {"message": {"content": "That's a good point — tell me more."}}


def _meaningful(*updates):
    from memory_extractor import (
        ExtractionResult,
        ExtractionUpdate,
        MessageType,
        Operation,
        StateField,
    )
    return ExtractionResult(
        message_type=MessageType.MEANINGFUL,
        updates=[
            ExtractionUpdate(operation=Operation(u[0]), field=StateField(u[1]), value=u[2])
            for u in updates
        ],
    )


# ---------------------------------------------------------------------------
# Unit tests — detection rules
# ---------------------------------------------------------------------------


class TestDetectInsight(unittest.TestCase):
    """Every insight type is independently detectable via deterministic phrases."""

    def test_none_for_plain_statement(self):
        itype, conf, _expl, _signals = detect_insight(user_message="we want to help students")
        self.assertEqual(itype, InsightType.NONE)
        self.assertEqual(conf, InsightConfidence.LOW)

    def test_contradiction_from_recovery_category(self):
        itype, conf, _expl, signals = detect_insight(
            user_message="actually it's daily",
            recovery_category="CONTRADICTION",
        )
        self.assertEqual(itype, InsightType.CONTRADICTION)
        self.assertEqual(conf, InsightConfidence.HIGH)
        self.assertTrue(signals)

    def test_contradiction_from_phrasing(self):
        itype, _conf, _expl, _signals = detect_insight(
            user_message="this contradicts what I said earlier"
        )
        self.assertEqual(itype, InsightType.CONTRADICTION)

    def test_user_learning(self):
        itype, conf, _expl, signals = detect_insight(
            user_message="I hadn't thought about that until now"
        )
        self.assertEqual(itype, InsightType.USER_LEARNING)
        self.assertEqual(conf, InsightConfidence.HIGH)
        self.assertTrue(any("learning_phrase" in s for s in signals))

    def test_user_learning_real_issue(self):
        itype, _conf, _expl, _signals = detect_insight(
            user_message="I think the real issue is the onboarding"
        )
        self.assertEqual(itype, InsightType.USER_LEARNING)

    def test_surprising_observation(self):
        itype, conf, _expl, _signals = detect_insight(
            user_message="surprisingly nobody uses the mobile app"
        )
        self.assertEqual(itype, InsightType.SURPRISING_OBSERVATION)
        self.assertEqual(conf, InsightConfidence.MEDIUM)

    def test_new_pattern_phrase(self):
        itype, _conf, _expl, _signals = detect_insight(
            user_message="every time we release a feature the churn goes up"
        )
        self.assertEqual(itype, InsightType.NEW_PATTERN)

    def test_new_pattern_multi_field_extraction(self):
        extraction = [
            {"field": "personas", "operation": "ADD", "value": "students"},
            {"field": "problems", "operation": "ADD", "value": "churn"},
        ]
        itype, conf, _expl, signals = detect_insight(
            user_message="students keep churning when we onboard them",
            extraction_updates=extraction,
        )
        self.assertEqual(itype, InsightType.NEW_PATTERN)
        self.assertEqual(conf, InsightConfidence.MEDIUM)
        self.assertTrue(any("multi_field" in s for s in signals))

    def test_strong_evidence_percent(self):
        itype, conf, _expl, signals = detect_insight(
            user_message="we measured 80% of users abandon after signup"
        )
        self.assertEqual(itype, InsightType.STRONG_EVIDENCE)
        self.assertEqual(conf, InsightConfidence.HIGH)
        self.assertIn("numeric_evidence_detected", signals)

    def test_strong_evidence_survey(self):
        itype, conf, _expl, _signals = detect_insight(
            user_message="our survey showed users want this feature"
        )
        self.assertEqual(itype, InsightType.STRONG_EVIDENCE)
        self.assertEqual(conf, InsightConfidence.MEDIUM)

    def test_constraint(self):
        itype, conf, _expl, signals = detect_insight(
            user_message="we only have a small budget for this project"
        )
        self.assertEqual(itype, InsightType.CONSTRAINT)
        self.assertEqual(conf, InsightConfidence.HIGH)
        self.assertTrue(any("budget" in s for s in signals))

    def test_constraint_time(self):
        itype, _conf, _expl, _signals = detect_insight(
            user_message="we are limited to three months"
        )
        self.assertEqual(itype, InsightType.CONSTRAINT)

    def test_root_cause_hint(self):
        itype, conf, _expl, signals = detect_insight(
            user_message="the problem stems from a lack of feedback"
        )
        self.assertEqual(itype, InsightType.ROOT_CAUSE_HINT)
        self.assertEqual(conf, InsightConfidence.MEDIUM)
        self.assertTrue(any("root_cause" in s for s in signals))

    def test_root_cause_because(self):
        itype, _conf, _expl, _signals = detect_insight(
            user_message="it happens because the API times out"
        )
        self.assertEqual(itype, InsightType.ROOT_CAUSE_HINT)

    def test_root_cause_low_priority_behind_constraint(self):
        # Constraint keyword fires before root-cause on a mixed message.
        itype, _conf, _expl, _signals = detect_insight(
            user_message="the budget constraint is the reason we can't scale"
        )
        self.assertEqual(itype, InsightType.CONSTRAINT)


# ---------------------------------------------------------------------------
# Guidance rendering tests
# ---------------------------------------------------------------------------


class TestInsightGuidanceRendering(unittest.TestCase):
    """insight_instruction_bullet produces correct guidance text."""

    def test_bullet_contains_type_and_guidance(self):
        bullet = insight_instruction_bullet(
            InsightType.STRONG_EVIDENCE, InsightConfidence.HIGH
        )
        self.assertIsNotNone(bullet)
        self.assertIn("Insight detected: STRONG_EVIDENCE", bullet)
        self.assertIn("HIGH confidence", bullet)
        self.assertIn("Guidance:", bullet)

    def test_bullet_returns_none_for_none_type(self):
        self.assertIsNone(insight_instruction_bullet(None))

    def test_bullet_returns_none_for_none_insight(self):
        self.assertIsNone(insight_instruction_bullet(InsightType.NONE))

    def test_root_cause_hint_guidance_explores(self):
        bullet = insight_instruction_bullet(InsightType.ROOT_CAUSE_HINT)
        self.assertIn("root cause", bullet.lower())


# ---------------------------------------------------------------------------
# Diagnostics section tests
# ---------------------------------------------------------------------------


class TestInsightDiagnostics(unittest.TestCase):
    """insight_diagnostics_section produces the Developer Console section."""

    def test_diagnostics_with_insight(self):
        capture = {
            "insight_type": InsightType.STRONG_EVIDENCE,
            "insight_confidence": InsightConfidence.HIGH,
            "insight_explanation": "numeric evidence",
            "insight_signals": ["numeric_evidence_detected"],
        }
        section = insight_diagnostics_section(capture)
        self.assertEqual(section["Insight Type"], "STRONG_EVIDENCE")
        self.assertEqual(section["Confidence"], "HIGH")
        self.assertEqual(section["Signals"], ["numeric_evidence_detected"])

    def test_diagnostics_with_string_type(self):
        capture = {
            "insight_type": "CONSTRAINT",
            "insight_confidence": "HIGH",
            "insight_explanation": "budget",
            "insight_signals": ["budget"],
        }
        section = insight_diagnostics_section(capture)
        self.assertEqual(section["Insight Type"], "CONSTRAINT")

    def test_diagnostics_empty_for_none(self):
        self.assertEqual(insight_diagnostics_section({}), {})

    def test_diagnostics_empty_for_none_type(self):
        capture = {"insight_type": InsightType.NONE}
        self.assertEqual(insight_diagnostics_section(capture), {})


# ---------------------------------------------------------------------------
# Integration: insight appears in prompt
# ---------------------------------------------------------------------------


class TestInsightInPrompt(unittest.TestCase):
    """The insight instruction bullet surfaces in the LLM prompt."""

    def test_insight_bullet_in_ask_question_prompt(self):
        from module4.prompt_builder import build_prompt
        from module3 import ObjectiveEngine
        from module4 import ResponseStrategy
        from memory_extractor import ProjectState

        state = ProjectState()
        obj = ObjectiveEngine().determine_next(state)
        prompt = build_prompt(
            project_state=state,
            conversation_objective=obj,
            response_strategy=ResponseStrategy.ASK_QUESTION,
            insight_bullet=[
                insight_instruction_bullet(
                    InsightType.ROOT_CAUSE_HINT, InsightConfidence.MEDIUM
                )
            ],
        )
        self.assertIn("Instructions", prompt)
        self.assertIn("Insight detected: ROOT_CAUSE_HINT", prompt)
        self.assertIn("Guidance:", prompt)
        # Baseline prompt structure remains intact
        self.assertIn("Role", prompt)
        self.assertIn("Current Objective", prompt)
        self.assertIn("Known Project State", prompt)
        self.assertIn("Latest Conversation", prompt)

    def test_no_insight_bullet_means_no_insight_text(self):
        from module4.prompt_builder import build_prompt
        from module3 import ObjectiveEngine
        from module4 import ResponseStrategy
        from memory_extractor import ProjectState

        state = ProjectState()
        obj = ObjectiveEngine().determine_next(state)
        prompt = build_prompt(
            project_state=state,
            conversation_objective=obj,
            response_strategy=ResponseStrategy.ASK_QUESTION,
        )
        self.assertNotIn("Insight detected:", prompt)

    def test_coaching_and_insight_bullets_coexist(self):
        from module4.prompt_builder import build_prompt
        from module3 import ObjectiveEngine
        from module4 import ResponseStrategy
        from memory_extractor import ProjectState

        state = ProjectState()
        obj = ObjectiveEngine().determine_next(state)
        prompt = build_prompt(
            project_state=state,
            conversation_objective=obj,
            response_strategy=ResponseStrategy.ASK_QUESTION,
            coaching_bullet=["Current coaching strategy: DEEPEN"],
            insight_bullet=["Insight detected: STRONG_EVIDENCE"],
        )
        self.assertIn("Current coaching strategy: DEEPEN", prompt)
        self.assertIn("Insight detected: STRONG_EVIDENCE", prompt)


# ---------------------------------------------------------------------------
# Integration: insight appears in MW diagnostics
# ---------------------------------------------------------------------------


class TestInsightPipelineIntegration(unittest.TestCase):
    """Insight detection is computed during turn processing and surfaces
    in diagnostics."""

    def setUp(self):
        import uuid
        from session_manager import get_session_manager
        uniq = uuid.uuid4().hex[:8]
        self._username = f"insight_{uniq}"
        self._project = f"Insight_{uniq}"
        get_session_manager().reset_runtime_state(
            username=self._username, project_title=self._project
        )

    def tearDown(self):
        from session_manager import get_session_manager
        get_session_manager().reset_runtime_state(
            username=self._username, project_title=self._project
        )

    def _run_turn(self, user_message, patched_extract):
        raising = _RaisingOllama()
        with mock.patch.dict("sys.modules", {"ollama": raising}):
            with mock.patch(
                "mentor.MemoryExtractor.extract", side_effect=patched_extract
            ):
                reply, _session, _timing, diagnostics = mentor.process_mentor_turn(
                    user_message,
                    username=self._username,
                    project_name=self._project,
                )
        return reply, diagnostics

    def test_insight_section_appears_in_diagnostics(self):
        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "evidence", "80 percent signup drop"))
        _reply, diagnostics = self._run_turn(
            "we measured 80% of users abandon after signup",
            _patched,
        )
        self.assertIn("InsightDetection", diagnostics)
        section = diagnostics["InsightDetection"]
        self.assertEqual(section["Insight Type"], "STRONG_EVIDENCE")

    def test_no_insight_leaves_empty_section(self):
        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "personas", "students"))
        _reply, diagnostics = self._run_turn(
            "we want to help students",
            _patched,
        )
        self.assertIn("InsightDetection", diagnostics)
        self.assertEqual(diagnostics["InsightDetection"], {})

    def test_insight_guidance_in_prompt_inspector(self):
        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "problems", "churn"))
        _reply, diagnostics = self._run_turn(
            "the problem stems from a lack of feedback",
            _patched,
        )
        self.assertIn("Prompt", diagnostics or {})
        # Phase 1 carrier: insight shapes the Brief and Relevant Context
        # (WHAT HAPPENED / context snippets) instead of a dedicated bullet.
        prompt = diagnostics["Prompt"] or ""
        self.assertIn("WHAT HAPPENED", prompt)
        self.assertIn("RELEVANT CONTEXT", prompt)
        self.assertNotEqual(diagnostics["InsightDetection"], {})


class TestInsightNeverChangesPipeline(unittest.TestCase):
    """Insight detection is guidance-only — extraction, state, objectives,
    lifecycle, and LLM call count are all identical."""

    def setUp(self):
        import uuid
        from session_manager import get_session_manager
        uniq = uuid.uuid4().hex[:8]
        self._username = f"insight_ex_{uniq}"
        self._project = f"InsightEx_{uniq}"
        _CountingOllama.calls = 0
        get_session_manager().reset_runtime_state(
            username=self._username, project_title=self._project
        )

    def tearDown(self):
        from session_manager import get_session_manager
        get_session_manager().reset_runtime_state(
            username=self._username, project_title=self._project
        )

    def test_extraction_state_unchanged_and_single_llm_call(self):
        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "pain_points", "no feedback"))
        with mock.patch.dict("sys.modules", {"ollama": _CountingOllama}):
            with mock.patch(
                "mentor.MemoryExtractor.extract", side_effect=_patched
            ):
                _reply, _session, _timing, diagnostics = mentor.process_mentor_turn(
                    "it happens because the API times out",
                    username=self._username,
                    project_name=self._project,
                )
        from session_manager import get_session_manager
        state = get_session_manager().get_active_session_data().project_state.to_state_dict()
        # Extraction result unchanged — insight detection never mutates state.
        self.assertEqual(state["pain_points"], ["no feedback"])
        # Exactly one LLM call: insight detection adds no LLM round-trips.
        self.assertEqual(_CountingOllama.calls, 1)
        self.assertEqual(
            diagnostics["InsightDetection"]["Insight Type"],
            "ROOT_CAUSE_HINT",
        )

    def test_objective_and_lifecycle_unaffected_by_insight(self):
        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "personas", "students"))
        with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama}):
            with mock.patch(
                "mentor.MemoryExtractor.extract", side_effect=_patched
            ):
                _reply, _session, _timing, diagnostics = mentor.process_mentor_turn(
                    "we measured 80% of users abandon after signup",
                    username=self._username,
                    project_name=self._project,
                )
        pipeline = diagnostics["Pipeline"]
        self.assertIn(pipeline["Response Strategy"],
                      ("ASK_QUESTION", "GENERATE_SUMMARY"))
        self.assertIn(pipeline["Lifecycle Decision"],
                      ("CONTINUE", "CONTINUE_EMPATHIZE",
                       "PROCEED_TO_SOLUTION", "WRAP_UP"))


if __name__ == "__main__":
    unittest.main()
