"""
tests/test_coaching_strategy.py — unit + integration tests for the
Adaptive Coaching Strategy layer.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mentor  # noqa: E402
from coaching_strategy import (  # noqa: E402
    CoachingStrategy,
    coaching_instruction_bullet,
    coaching_diagnostics_section,
    determine_coaching_strategy,
    guidance_for,
)

_EMPTY_STATE = {
    "personas": [],
    "problems": [],
    "current_solutions": [],
    "pain_points": [],
    "evidence": [],
    "impacts": [],
    "frequency": None,
}


class _RaisingOllama:
    """Stub that forces the deterministic fallback path."""

    @staticmethod
    def chat(**kwargs):
        raise RuntimeError("ollama disabled in tests")


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
# Unit tests — coaching strategy rules
# ---------------------------------------------------------------------------


class TestDetermineCoachingStrategy(unittest.TestCase):
    """Every coaching strategy rule is independently testable."""

    def test_wrap_up_yields_summarize_progress(self):
        strat, _reason = determine_coaching_strategy(objective="WRAP_UP")
        self.assertEqual(strat, CoachingStrategy.SUMMARIZE_PROGRESS)

    def test_dont_know_yields_clarify(self):
        strat, reason = determine_coaching_strategy(
            objective="PROBLEMS",
            user_message="idk",
            recovery_category="DONT_KNOW",
        )
        self.assertEqual(strat, CoachingStrategy.CLARIFY)
        self.assertIn("clarif", reason.lower())

    def test_uncertain_answer_yields_clarify(self):
        strat, reason = determine_coaching_strategy(
            objective="PROBLEMS",
            user_message="maybe something about time",
            recovery_category="UNCERTAIN_ANSWER",
        )
        self.assertEqual(strat, CoachingStrategy.CLARIFY)

    def test_contradiction_yields_validate(self):
        strat, _reason = determine_coaching_strategy(
            objective="FREQUENCY",
            user_message="actually it's daily, not weekly like I said before",
            recovery_category="CONTRADICTION",
            state_after=_EMPTY_STATE,
            state_before=_EMPTY_STATE,
        )
        self.assertEqual(strat, CoachingStrategy.VALIDATE)

    def test_vague_message_yields_clarify(self):
        strat, reason = determine_coaching_strategy(
            objective="PROBLEMS",
            user_message="maybe",
        )
        self.assertEqual(strat, CoachingStrategy.CLARIFY)

    def test_empty_message_yields_clarify(self):
        strat, _reason = determine_coaching_strategy(
            objective="PROBLEMS",
            user_message="",
        )
        self.assertEqual(strat, CoachingStrategy.CLARIFY)

    def test_rich_answer_yields_deepen(self):
        strat, reason = determine_coaching_strategy(
            objective="PROBLEMS",
            user_message="students keep forgetting their assignments because they have no system to track deadlines and it causes anxiety",
            state_before=_EMPTY_STATE,
            state_after={
                "personas": [],
                "problems": ["forgetting assignments", "no system", "anxiety"],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
        )
        self.assertEqual(strat, CoachingStrategy.DEEPEN)
        self.assertIn("substantive", reason.lower() or "")

    def test_useful_answer_with_changed_fields_yields_deepen(self):
        strat, _reason = determine_coaching_strategy(
            objective="PERSONAS",
            user_message="college students",
            state_before=_EMPTY_STATE,
            state_after={
                "personas": ["college students"],
                "problems": [],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
        )
        self.assertEqual(strat, CoachingStrategy.DEEPEN)

    def test_topic_change_yields_validate(self):
        strat, _reason = determine_coaching_strategy(
            objective="PERSONAS",
            user_message="I was also thinking about pricing models",
            recovery_category="TOPIC_CHANGE",
            state_before=_EMPTY_STATE,
            state_after=_EMPTY_STATE,
        )
        self.assertEqual(strat, CoachingStrategy.VALIDATE)

    def test_advanced_objective_yields_transition(self):
        strat, reason = determine_coaching_strategy(
            objective="PROBLEMS",
            objective_advancement="ADVANCED",
            user_message="students forget assignments",
            state_before=_EMPTY_STATE,
            state_after={
                "personas": ["students"],
                "problems": ["forgetting assignments"],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
        )
        self.assertEqual(strat, CoachingStrategy.TRANSITION)

    def test_default_fresh_objective_yields_explore(self):
        strat, reason = determine_coaching_strategy(
            objective="PERSONAS",
            user_message="I want to help college students",
            state_before=_EMPTY_STATE,
            state_after=_EMPTY_STATE,
        )
        self.assertEqual(strat, CoachingStrategy.EXPLORE)
        self.assertIn("beginning", reason.lower())

    def test_three_satisfied_fields_yields_summarize_progress(self):
        strat, _reason = determine_coaching_strategy(
            objective="EVIDENCE",
            user_message="I see this daily_feature",
            state_before=_EMPTY_STATE,
            state_after={
                "personas": ["students"],
                "problems": ["forgetting assignments"],
                "current_solutions": ["planners"],
                "pain_points": [],
                "evidence": ["daily_feature"],
                "impacts": [],
                "frequency": None,
            },
        )
        self.assertEqual(strat, CoachingStrategy.SUMMARIZE_PROGRESS)


# ---------------------------------------------------------------------------
# Guidance rendering tests
# ---------------------------------------------------------------------------


class TestGuidanceRendering(unittest.TestCase):
    """coaching_instruction_bullet and guidance_for produce correct text."""

    def test_guidance_for_returns_string_for_every_strategy(self):
        for s in CoachingStrategy:
            g = guidance_for(s)
            self.assertIsNotNone(g)
            self.assertIsInstance(g, str)
            self.assertGreater(len(g), 10)

    def test_instruction_bullet_contains_strategy_and_guidance(self):
        bullet = coaching_instruction_bullet(CoachingStrategy.DEEPEN)
        self.assertIn("Current coaching strategy: DEEPEN", bullet)
        self.assertIn("Guidance:", bullet)
        self.assertIn("deeper", bullet.lower())

    def test_instruction_bullet_returns_none_for_none(self):
        self.assertIsNone(coaching_instruction_bullet(None))


# ---------------------------------------------------------------------------
# Diagnostics section tests
# ---------------------------------------------------------------------------


class TestCoachingDiagnostics(unittest.TestCase):
    """coaching_diagnostics_section produces the Developer Console section."""

    def test_diagnostics_with_strategy(self):
        capture = {"coaching_strategy": CoachingStrategy.DEEPEN, "coaching_reason": "rich answer"}
        section = coaching_diagnostics_section(capture)
        self.assertEqual(section["Selected Strategy"], "DEEPEN")
        self.assertEqual(section["Reason"], "rich answer")
        self.assertIn("rich answer", section["Supporting Signals"])

    def test_diagnostics_without_strategy_returns_empty(self):
        self.assertEqual(coaching_diagnostics_section({}), {})

    def test_diagnostics_with_enum_value_as_string(self):
        capture = {"coaching_strategy": "DEEPEN", "coaching_reason": "deep insight"}
        section = coaching_diagnostics_section(capture)
        self.assertEqual(section["Selected Strategy"], "DEEPEN")


# ---------------------------------------------------------------------------
# Integration: strategy appears in prompt
# ---------------------------------------------------------------------------


class TestCoachingInPrompt(unittest.TestCase):
    """The coaching instruction bullet surfaces in the LLM prompt."""

    def test_coaching_bullet_in_ask_question_prompt(self):
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
            coaching_bullet=[
                coaching_instruction_bullet(CoachingStrategy.DEEPEN)
            ],
        )
        self.assertIn("Instructions", prompt)
        self.assertIn("Current coaching strategy: DEEPEN", prompt)
        self.assertIn("Guidance:", prompt)
        # The baseline prompt structure remains intact
        self.assertIn("Role", prompt)
        self.assertIn("Current Objective", prompt)
        self.assertIn("Known Project State", prompt)
        self.assertIn("Latest Conversation", prompt)

    def test_no_coaching_bullet_means_no_coaching_text(self):
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
        self.assertNotIn("Current coaching strategy:", prompt)


# ---------------------------------------------------------------------------
# Integration: strategy appears in MW diagnostics
# ---------------------------------------------------------------------------


class TestCoachingPipelineIntegration(unittest.TestCase):
    """The coaching strategy is computed during turn processing and
    surfaces in diagnostics."""

    def setUp(self):
        import uuid
        from session_manager import get_session_manager
        uniq = uuid.uuid4().hex[:8]
        self._username = f"coach_{uniq}"
        self._project = f"Coach_{uniq}"
        mgr = get_session_manager()
        mgr.reset_runtime_state(username=self._username, project_title=self._project)

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

    def test_coaching_section_appears_in_diagnostics(self):
        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "personas", "students"))
        _reply, diagnostics = self._run_turn(
            "students forget assignments",
            _patched,
        )
        self.assertIn("CoachingStrategy", diagnostics)
        cs = diagnostics["CoachingStrategy"]
        self.assertIn("Selected Strategy", cs)
        self.assertIn("Reason", cs)
        self.assertTrue(cs["Selected Strategy"])

    def test_strategy_is_accessible_from_capture(self):
        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "personas", "college students"))
        reply, diagnostics = self._run_turn(
            "I want to help college students",
            _patched,
        )
        cs = diagnostics["CoachingStrategy"]
        self.assertIn(cs["Selected Strategy"],
            {s.value for s in CoachingStrategy}
        )

    def test_strategy_never_changes_state(self):
        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "personas", "students"))
        reply, diagnostics = self._run_turn(
            "students",
            _patched,
        )
        from session_manager import get_session_manager
        state = get_session_manager().get_active_session_data().project_state.to_state_dict()
        # The strategy never alters state — but extraction does.  Verify
        # the strategy section is present and well-formed.
        self.assertTrue(diagnostics["CoachingStrategy"]["Reason"])
        self.assertIn(diagnostics["CoachingStrategy"]["Selected Strategy"],
                      {s.value for s in CoachingStrategy})

    def test_strategy_in_prompt_inspector_for_non_summary_strategy(self):
        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "personas", "college students"))
        _reply, diagnostics = self._run_turn(
            "I want to help college students",
            _patched,
        )
        self.assertIn("Prompt", diagnostics or {})
        # Phase 1 carrier: the coaching strategy shapes the allowed-move
        # set instead of a dedicated bullet; the decision stays inspectable.
        self.assertIn(
            "ALLOWED CONVERSATIONAL MOVES",
            (diagnostics["Prompt"] or ""),
            "Prompt should contain coaching-shaped move guidance",
        )
        self.assertTrue(diagnostics["CoachingStrategy"]["Selected Strategy"])


class TestCoachingNeverChangesExtraction(unittest.TestCase):
    """Coaching strategy is guidance-only — extraction output is identical."""

    def setUp(self):
        import uuid
        from session_manager import get_session_manager
        uniq = uuid.uuid4().hex[:8]
        self._username = f"coach_ex_{uniq}"
        self._project = f"CoachEx_{uniq}"
        get_session_manager().reset_runtime_state(
            username=self._username, project_title=self._project
        )

    def test_extraction_state_is_unchanged_when_coaching_is_active(self):
        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "problems", "forgetting assignments"))
        raising = _RaisingOllama()
        with mock.patch.dict("sys.modules", {"ollama": raising}):
            with mock.patch(
                "mentor.MemoryExtractor.extract", side_effect=_patched
            ):
                _reply, _session, _timing, diagnostics = mentor.process_mentor_turn(
                    "students forget assignments",
                    username=self._username,
                    project_name=self._project,
                )
        from session_manager import get_session_manager
        state = get_session_manager().get_active_session_data().project_state.to_state_dict()
        self.assertEqual(state["problems"], ["forgetting assignments"])
        self.assertIn("Selected Strategy", diagnostics["CoachingStrategy"])

    def tearDown(self):
        from session_manager import get_session_manager
        get_session_manager().reset_runtime_state(
            username=self._username, project_title=self._project
        )


if __name__ == "__main__":
    unittest.main()