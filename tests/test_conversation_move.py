"""
tests/test_conversation_move.py — unit + integration tests for the
Conversational Move Planner.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mentor  # noqa: E402
from conversation_move import (  # noqa: E402
    ConversationMove,
    conversation_move_diagnostics_section,
    conversation_move_instruction_bullet,
    determine_conversation_move,
    guidance_for_move,
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

_THREE_FACTS_STATE = {
    "personas": ["students"],
    "problems": ["forgetting assignments"],
    "current_solutions": ["planners"],
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
# Unit tests — move rules
# ---------------------------------------------------------------------------


class TestDetermineConversationMove(unittest.TestCase):
    """Every conversational-move rule is independently testable."""

    def test_wrap_up_yields_summarize_progress(self):
        move, reason, signals = determine_conversation_move(
            current_objective="WRAP_UP",
        )
        self.assertEqual(move, ConversationMove.SUMMARIZE_PROGRESS)
        self.assertIn("recap", reason.lower())
        self.assertIn("objective=WRAP_UP", signals)

    def test_wrap_up_with_summary_presented_yields_close_topic(self):
        move, _reason, signals = determine_conversation_move(
            current_objective="WRAP_UP",
            summary_presented=True,
        )
        self.assertEqual(move, ConversationMove.CLOSE_TOPIC)
        self.assertIn("summary_already_presented", signals)

    def test_deepen_plus_root_cause_hint_yields_expand_idea(self):
        move, reason, signals = determine_conversation_move(
            current_objective="PROBLEMS",
            coaching_strategy="DEEPEN",
            insight_type="ROOT_CAUSE_HINT",
            user_message="the problem stems from a lack of feedback",
        )
        self.assertEqual(move, ConversationMove.EXPAND_IDEA)
        self.assertIn("root-cause hint", reason.lower())
        self.assertIn("coaching=DEEPEN", signals)
        self.assertIn("insight=ROOT_CAUSE_HINT", signals)

    def test_strong_evidence_yields_validate_discovery(self):
        move, _reason, signals = determine_conversation_move(
            current_objective="PROBLEMS",
            insight_type="STRONG_EVIDENCE",
        )
        self.assertEqual(move, ConversationMove.VALIDATE_DISCOVERY)
        self.assertIn("insight=STRONG_EVIDENCE", signals)

    def test_contradiction_yields_challenge_assumption(self):
        move, _reason, _signals = determine_conversation_move(
            current_objective="FREQUENCY",
            insight_type="CONTRADICTION",
        )
        self.assertEqual(move, ConversationMove.CHALLENGE_ASSUMPTION)

    def test_user_learning_yields_validate_discovery(self):
        move, _reason, _signals = determine_conversation_move(
            current_objective="PROBLEMS",
            insight_type="USER_LEARNING",
        )
        self.assertEqual(move, ConversationMove.VALIDATE_DISCOVERY)

    def test_surprising_observation_yields_expand_idea(self):
        move, _reason, _signals = determine_conversation_move(
            current_objective="PROBLEMS",
            insight_type="SURPRISING_OBSERVATION",
        )
        self.assertEqual(move, ConversationMove.EXPAND_IDEA)

    def test_new_pattern_yields_connect_information(self):
        move, _reason, _signals = determine_conversation_move(
            current_objective="PROBLEMS",
            insight_type="NEW_PATTERN",
        )
        self.assertEqual(move, ConversationMove.CONNECT_INFORMATION)

    def test_advanced_objective_yields_transition_topic(self):
        move, _reason, signals = determine_conversation_move(
            current_objective="PROBLEMS",
            objective_advancement="ADVANCED",
            state_before=_EMPTY_STATE,
            state_after=_EMPTY_STATE,
        )
        self.assertEqual(move, ConversationMove.TRANSITION_TOPIC)
        self.assertIn("advancement=ADVANCED", signals)

    def test_coaching_transition_yields_transition_topic(self):
        move, _reason, _signals = determine_conversation_move(
            current_objective="PROBLEMS",
            coaching_strategy="TRANSITION",
            state_before=_EMPTY_STATE,
            state_after=_EMPTY_STATE,
        )
        self.assertEqual(move, ConversationMove.TRANSITION_TOPIC)

    def test_satisfied_objective_yields_transition_topic(self):
        move, _reason, signals = determine_conversation_move(
            current_objective="PROBLEMS",
            objective_completed_fields=["personas"],
            objective_missing_fields=[],
            state_before=_EMPTY_STATE,
            state_after=_EMPTY_STATE,
        )
        self.assertEqual(move, ConversationMove.TRANSITION_TOPIC)
        self.assertIn("objective_satisfied", signals)

    def test_multiple_changed_fields_yields_connect_information(self):
        move, _reason, signals = determine_conversation_move(
            current_objective="PROBLEMS",
            changed_fields={"personas", "problems"},
            state_after=_EMPTY_STATE,
        )
        self.assertEqual(move, ConversationMove.CONNECT_INFORMATION)
        self.assertIn("changed_fields=2", signals)

    def test_three_populated_fact_fields_yields_connect_information(self):
        move, _reason, signals = determine_conversation_move(
            current_objective="PROBLEMS",
            state_after=_THREE_FACTS_STATE,
        )
        self.assertEqual(move, ConversationMove.CONNECT_INFORMATION)
        self.assertIn("populated_fact_fields=3", signals)

    def test_two_resolved_threads_yields_connect_information(self):
        move, _reason, _signals = determine_conversation_move(
            current_objective="PROBLEMS",
            memory_resolved_threads=[
                {"field": "personas"},
                {"field": "problems"},
            ],
            state_after=_EMPTY_STATE,
        )
        self.assertEqual(move, ConversationMove.CONNECT_INFORMATION)

    def test_coaching_summarize_progress_yields_summarize_progress(self):
        move, _reason, _signals = determine_conversation_move(
            current_objective="PROBLEMS",
            coaching_strategy="SUMMARIZE_PROGRESS",
            state_after=_EMPTY_STATE,
        )
        self.assertEqual(move, ConversationMove.SUMMARIZE_PROGRESS)

    def test_coaching_validate_yields_validate_discovery(self):
        move, _reason, _signals = determine_conversation_move(
            current_objective="PROBLEMS",
            coaching_strategy="VALIDATE",
            state_after=_EMPTY_STATE,
        )
        self.assertEqual(move, ConversationMove.VALIDATE_DISCOVERY)

    def test_coaching_deepen_without_insight_yields_expand_idea(self):
        move, _reason, _signals = determine_conversation_move(
            current_objective="PROBLEMS",
            coaching_strategy="DEEPEN",
            insight_type="NONE",
            state_after=_EMPTY_STATE,
        )
        self.assertEqual(move, ConversationMove.EXPAND_IDEA)

    def test_coaching_clarify_yields_elicit_information(self):
        move, _reason, _signals = determine_conversation_move(
            current_objective="PROBLEMS",
            coaching_strategy="CLARIFY",
            state_after=_EMPTY_STATE,
        )
        self.assertEqual(move, ConversationMove.ELICIT_INFORMATION)

    def test_coaching_explore_yields_elicit_information(self):
        move, _reason, _signals = determine_conversation_move(
            current_objective="PERSONAS",
            coaching_strategy="EXPLORE",
            state_after=_EMPTY_STATE,
        )
        self.assertEqual(move, ConversationMove.ELICIT_INFORMATION)

    def test_default_yields_elicit_information(self):
        move, _reason, signals = determine_conversation_move(
            current_objective="PROBLEMS",
            state_after=_EMPTY_STATE,
        )
        self.assertEqual(move, ConversationMove.ELICIT_INFORMATION)
        self.assertIn("default", signals)

    def test_every_move_is_reachable(self):
        produced = {
            determine_conversation_move(
                current_objective="WRAP_UP",
                summary_presented=True,
            )[0],
            determine_conversation_move(
                current_objective="WRAP_UP",
            )[0],
            determine_conversation_move(
                current_objective="PROBLEMS",
                coaching_strategy="DEEPEN",
                insight_type="ROOT_CAUSE_HINT",
            )[0],
            determine_conversation_move(
                current_objective="PROBLEMS",
                insight_type="STRONG_EVIDENCE",
            )[0],
            determine_conversation_move(
                current_objective="PROBLEMS",
                insight_type="CONTRADICTION",
            )[0],
            determine_conversation_move(
                current_objective="PROBLEMS",
                objective_advancement="ADVANCED",
            )[0],
            determine_conversation_move(
                current_objective="PROBLEMS",
                changed_fields={"personas", "problems"},
            )[0],
            determine_conversation_move(
                current_objective="PROBLEMS",
                state_after=_EMPTY_STATE,
            )[0],
        }
        self.assertEqual(len(produced), len(ConversationMove))


# ---------------------------------------------------------------------------
# Guidance rendering tests
# ---------------------------------------------------------------------------


class TestMoveGuidanceRendering(unittest.TestCase):
    """conversation_move_instruction_bullet produces correct text."""

    def test_guidance_for_move_returns_string_for_every_move(self):
        for m in ConversationMove:
            g = guidance_for_move(m)
            self.assertIsNotNone(g)
            self.assertIsInstance(g, str)
            self.assertGreater(len(g), 10)

    def test_instruction_bullet_contains_move_and_guidance(self):
        bullet = conversation_move_instruction_bullet(
            ConversationMove.CONNECT_INFORMATION
        )
        self.assertIn("Conversation move: CONNECT_INFORMATION", bullet)
        self.assertIn("Guidance:", bullet)
        self.assertIn("connect", bullet.lower())

    def test_instruction_bullet_returns_none_for_none(self):
        self.assertIsNone(conversation_move_instruction_bullet(None))


# ---------------------------------------------------------------------------
# Diagnostics section tests
# ---------------------------------------------------------------------------


class TestMoveDiagnostics(unittest.TestCase):
    """conversation_move_diagnostics_section produces the Developer Console
    section."""

    def test_diagnostics_with_move(self):
        capture = {
            "conversation_move": ConversationMove.EXPAND_IDEA,
            "conversation_move_reason": "expand the idea",
            "conversation_move_signals": ["coaching=DEEPEN"],
        }
        section = conversation_move_diagnostics_section(capture)
        self.assertEqual(section["Move"], "EXPAND_IDEA")
        self.assertEqual(section["Reason"], "expand the idea")
        self.assertEqual(section["Signals"], ["coaching=DEEPEN"])

    def test_diagnostics_with_string_value(self):
        capture = {
            "conversation_move": "TRANSITION_TOPIC",
            "conversation_move_reason": "bridge",
            "conversation_move_signals": [],
        }
        section = conversation_move_diagnostics_section(capture)
        self.assertEqual(section["Move"], "TRANSITION_TOPIC")

    def test_diagnostics_empty_without_move(self):
        self.assertEqual(conversation_move_diagnostics_section({}), {})


# ---------------------------------------------------------------------------
# Integration: move appears in prompt
# ---------------------------------------------------------------------------


class TestMoveInPrompt(unittest.TestCase):
    """The conversation-move instruction bullet surfaces in the LLM prompt."""

    def test_move_bullet_in_ask_question_prompt(self):
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
            move_bullet=[
                conversation_move_instruction_bullet(
                    ConversationMove.CONNECT_INFORMATION
                )
            ],
        )
        self.assertIn("Instructions", prompt)
        self.assertIn("Conversation move: CONNECT_INFORMATION", prompt)
        self.assertIn("Guidance:", prompt)
        # Baseline prompt structure remains intact
        self.assertIn("Role", prompt)
        self.assertIn("Current Objective", prompt)
        self.assertIn("Known Project State", prompt)
        self.assertIn("Latest Conversation", prompt)

    def test_no_move_bullet_means_no_move_text(self):
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
        self.assertNotIn("Conversation move:", prompt)

    def test_coaching_insight_and_move_bullets_coexist(self):
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
            insight_bullet=["Insight detected: ROOT_CAUSE_HINT"],
            move_bullet=["Conversation move: EXPAND_IDEA"],
        )
        self.assertIn("Current coaching strategy: DEEPEN", prompt)
        self.assertIn("Insight detected: ROOT_CAUSE_HINT", prompt)
        self.assertIn("Conversation move: EXPAND_IDEA", prompt)


# ---------------------------------------------------------------------------
# Integration: move appears in MW diagnostics
# ---------------------------------------------------------------------------


class TestMovePipelineIntegration(unittest.TestCase):
    """The conversation move is computed during turn processing and
    surfaces in diagnostics."""

    def setUp(self):
        import uuid
        from session_manager import get_session_manager
        uniq = uuid.uuid4().hex[:8]
        self._username = f"move_{uniq}"
        self._project = f"Move_{uniq}"
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

    def test_move_section_appears_in_diagnostics(self):
        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "problems", "churn"))
        _reply, diagnostics = self._run_turn(
            "students keep churning",
            _patched,
        )
        self.assertIn("ConversationMove", diagnostics)
        section = diagnostics["ConversationMove"]
        self.assertIn("Move", section)
        self.assertIn("Reason", section)
        self.assertIn("Signals", section)
        self.assertTrue(section["Move"])

    def test_move_is_valid_enum_value(self):
        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "personas", "students"))
        _reply, diagnostics = self._run_turn(
            "we want to help students",
            _patched,
        )
        section = diagnostics["ConversationMove"]
        self.assertIn(
            section["Move"],
            {m.value for m in ConversationMove},
        )

    def test_move_guidance_in_prompt_inspector(self):
        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "problems", "churn"))
        _reply, diagnostics = self._run_turn(
            "students keep churning",
            _patched,
        )
        self.assertIn("Prompt", diagnostics or {})
        self.assertIn(
            "ALLOWED CONVERSATIONAL MOVES",
            (diagnostics["Prompt"] or ""),
            "Prompt should contain move guidance",
        )


class TestMoveNeverChangesPipeline(unittest.TestCase):
    """The move planner is guidance-only — extraction, state, objectives,
    lifecycle, and LLM call count are all identical."""

    def setUp(self):
        import uuid
        from session_manager import get_session_manager
        uniq = uuid.uuid4().hex[:8]
        self._username = f"move_ex_{uniq}"
        self._project = f"MoveEx_{uniq}"
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
                    "the problem stems from a lack of feedback",
                    username=self._username,
                    project_name=self._project,
                )
        from session_manager import get_session_manager
        state = get_session_manager().get_active_session_data().project_state.to_state_dict()
        # Extraction result unchanged — the move planner never mutates state.
        self.assertEqual(state["pain_points"], ["no feedback"])
        # Exactly one LLM call: the move planner adds no LLM round-trips.
        self.assertEqual(_CountingOllama.calls, 1)
        # The move planner still resolved an intention for this turn.
        self.assertIn(diagnostics["ConversationMove"]["Move"],
                      {m.value for m in ConversationMove})

    def test_objective_and_lifecycle_unaffected_by_move(self):
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
