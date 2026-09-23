"""
tests/test_question_families.py

Lightweight semantic duplicate detection for mentor questions.

Regression target:
    Mentor keeps asking semantically identical questions:
        "How common is this?" / "How widespread is this?" /
        "What percentage experience this?" / "How frequently does this occur?"
    all collect the same information from the same angle.

The fix has three cooperating pieces, all tested here:

  1. A normalization layer (:mod:`module3.question_families`) groups questions
     into semantic families via priority-ordered regexes — NO embeddings, NO
     vector DBs.
  2. SessionData tracks the families already asked; the QuestionFamilyPlanner
     steers the next question to an unasked family, and only re-asks one when
     every family for the field has been asked AND the previous answers were
     still insufficient.
  3. The Developer Console surfaces the question family, the previously asked
     families, and the reason a family was skipped.
"""

import os
import sys
import unittest
from unittest import mock

# Allow importing from the project root regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import ExtractionResult, MessageType, ProjectState, StateField  # noqa: E402
from module3 import (  # noqa: E402
    FAMILIES_BY_FIELD,
    FAMILY_FALLBACK_QUESTIONS,
    QuestionFamily,
    QuestionFamilyPlanner,
    SufficiencyChecker,
    classify_question,
    family_label,
    field_of,
)
from module3.sufficiency import SufficiencyReport  # noqa: E402
from conversation_style_audit import analyze_reply  # noqa: E402
from conversation_pipeline import (  # noqa: E402
    _OBJECTIVE_FALLBACK_QUESTION,
    _build_journey_fallback,
    _build_summary_fallback,
)
from module3 import Objective, ObjectiveEngine  # noqa: E402
from module4 import ResponseStrategyEngine  # noqa: E402
from module5 import LifecycleDecision  # noqa: E402

class _RaisingOllama:
    """Stand-in for the ``ollama`` module that always fails, forcing the
    deterministic family-aware fallback path."""

    @staticmethod
    def chat(**kwargs):
        raise RuntimeError("ollama disabled in tests")


class _StubOllama:
    """Stand-in that returns canned replies in order, then raises."""

    def __init__(self, replies):
        self._replies = list(replies)

    def chat(self, **kwargs):
        if not self._replies:
            raise RuntimeError("no more stub replies")
        return {"message": {"content": self._replies.pop(0)}}


def _sufficiency(**values) -> SufficiencyReport:
    return SufficiencyChecker().evaluate(ProjectState(**values))


# ---------------------------------------------------------------------------
# Classification — question text -> semantic family
# ---------------------------------------------------------------------------


class TestQuestionFamilyClassification(unittest.TestCase):
    """The issue's exact phrasings must collapse into the four frequency
    families; semantically identical phrasings share a family."""

    def test_frequency_issue_phrases_group_by_family(self):
        self.assertEqual(
            classify_question("How common is this?"),
            QuestionFamily.FREQUENCY_PREVALENCE,
        )
        self.assertEqual(
            classify_question("How widespread is this?"),
            QuestionFamily.FREQUENCY_PREVALENCE,
        )
        self.assertEqual(
            classify_question("What percentage experience this?"),
            QuestionFamily.FREQUENCY_PERCENTAGE,
        )
        self.assertEqual(
            classify_question("How frequently does this occur?"),
            QuestionFamily.FREQUENCY_ESTIMATE,
        )

    def test_common_and_widespread_are_same_family(self):
        a = classify_question("How common is this?")
        b = classify_question("How widespread is this?")
        self.assertIsNotNone(a)
        self.assertEqual(a, b)

    def test_fallback_questions_classify_to_own_family(self):
        for family, text in FAMILY_FALLBACK_QUESTIONS.items():
            with self.subTest(family=family.value):
                self.assertEqual(
                    classify_question(text), family, text,
                )

    def test_non_question_returns_none(self):
        for text in (
            "",
            "   ",
            "So far I understand that you're focusing on students.",
            "Does that capture things accurately, or would you like to add anything?",
            "Thanks for confirming.",
        ):
            with self.subTest(text=text):
                self.assertIsNone(classify_question(text))

    def test_case_and_punctuation_insensitive(self):
        self.assertEqual(
            classify_question("HOW COMMON IS THIS?"),
            QuestionFamily.FREQUENCY_PREVALENCE,
        )

    def test_family_label_is_human_readable(self):
        self.assertEqual(family_label(QuestionFamily.FREQUENCY_ESTIMATE),
                         "Frequency estimate")
        self.assertEqual(family_label(QuestionFamily.PERSONAS_EXAMPLES),
                         "Personas examples")

    def test_field_of_maps_back_to_state_field(self):
        self.assertEqual(field_of(QuestionFamily.FREQUENCY_PERCENTAGE),
                         StateField.FREQUENCY)
        self.assertEqual(field_of(QuestionFamily.PERSONAS_WHO),
                         StateField.PERSONAS)
        self.assertIsNone(field_of(None))


# ---------------------------------------------------------------------------
# Planner — steer the next question to an unasked family
# ---------------------------------------------------------------------------


class TestQuestionFamilyPlanner(unittest.TestCase):
    """Rule: avoid asking another question from the same family unless the
    previous answers were insufficient."""

    def test_fresh_field_selects_first_candidate(self):
        plan = QuestionFamilyPlanner().plan(
            StateField.FREQUENCY, [], _sufficiency(),
        )
        self.assertEqual(plan.selected, QuestionFamily.FREQUENCY_ESTIMATE)
        self.assertFalse(plan.reask)
        self.assertEqual(plan.skip_reasons, ())

    def test_prefers_unasked_family_and_records_skip(self):
        plan = QuestionFamilyPlanner().plan(
            StateField.FREQUENCY,
            ["FREQUENCY_ESTIMATE"],
            _sufficiency(frequency="a lot"),
        )
        self.assertEqual(plan.selected, QuestionFamily.FREQUENCY_PERCENTAGE)
        self.assertFalse(plan.reask)
        self.assertEqual(
            [s["family"] for s in plan.skip_reasons],
            ["FREQUENCY_ESTIMATE"],
        )
        self.assertIn("insufficient", plan.skip_reasons[0]["reason"].lower())

    def test_all_families_asked_reenforces_reask(self):
        all_frequency = [f.value for f in FAMILIES_BY_FIELD[StateField.FREQUENCY]]
        plan = QuestionFamilyPlanner().plan(
            StateField.FREQUENCY, all_frequency, _sufficiency(frequency="a lot"),
        )
        self.assertTrue(plan.reask)
        # Re-ask the least-recently asked family (first in chronological order).
        self.assertEqual(plan.selected, QuestionFamily.FREQUENCY_ESTIMATE)
        self.assertIn("insufficient", plan.reask_reason.lower())

    def test_sufficient_field_selects_nothing(self):
        all_frequency = [f.value for f in FAMILIES_BY_FIELD[StateField.FREQUENCY]]
        plan = QuestionFamilyPlanner().plan(
            StateField.FREQUENCY, all_frequency,
            _sufficiency(frequency="every single day"),
        )
        self.assertIsNone(plan.selected)
        self.assertFalse(plan.reask)
        self.assertEqual(len(plan.skip_reasons), len(all_frequency))
        self.assertIn("sufficient", plan.skip_reasons[0]["reason"].lower())

    def test_no_field_returns_empty_plan(self):
        plan = QuestionFamilyPlanner().plan(
            None, [], _sufficiency(),
        )
        self.assertIsNone(plan.selected)
        self.assertFalse(plan.reask)

    def test_plan_is_deterministic(self):
        planner = QuestionFamilyPlanner()
        a = planner.plan(StateField.FREQUENCY, ["FREQUENCY_ESTIMATE"],
                         _sufficiency(frequency="a lot"))
        b = planner.plan(StateField.FREQUENCY, ["FREQUENCY_ESTIMATE"],
                         _sufficiency(frequency="a lot"))
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_every_field_has_candidate_families(self):
        for sf, families in FAMILIES_BY_FIELD.items():
            with self.subTest(field=sf.value):
                self.assertTrue(families)
                self.assertTrue(all(field_of(f) is sf for f in families))


# ---------------------------------------------------------------------------
# Integration — repeated conversations via process_mentor_turn
# ---------------------------------------------------------------------------


class TestRepeatedConversationIntegration(unittest.TestCase):
    """End-to-end: while the frequency answer stays insufficient the mentor
    varies the family; a repeated family is blocked; a sufficient answer stops
    frequency questions entirely."""

    USER = "qf_user"
    PROJ = "qf_proj"

    def setUp(self):
        from session_manager import get_session_manager
        get_session_manager().reset_runtime_state()

    def tearDown(self):
        from session_manager import get_session_manager
        try:
            get_session_manager().reset_runtime_state()
        except Exception:
            pass

    def _seed(self, personas=None, problems=None, frequency=None,
              asked_families=None):
        from session_manager import get_session_manager
        mgr = get_session_manager()
        sd = mgr.get_active_session_data()
        if personas is not None:
            sd.project_state.personas = personas
        if problems is not None:
            sd.project_state.problems = problems
        if frequency is not None:
            sd.project_state.frequency = frequency
        sd.asked_question_families = list(asked_families or [])
        mgr.save_session_data(mgr.get_active_session_id(), sd)

    def _run_turn(self, message, ollama=None):
        import mentor
        stub = ollama if ollama is not None else _RaisingOllama()
        with mock.patch.dict("sys.modules", {"ollama": stub}):
            with mock.patch(
                "mentor.MemoryExtractor.extract",
                return_value=ExtractionResult(MessageType.AMBIGUOUS, []),
            ):
                return mentor.process_mentor_turn(
                    message, username=self.USER, project_name=self.PROJ,
                )

    def _asked_after(self):
        from session_manager import get_session_manager
        return list(
            get_session_manager().get_active_session_data().asked_question_families
        )

    def test_frequency_families_vary_across_turns_while_insufficient(self):
        self._seed(personas=["students"], problems=["forgetting"])
        reply_a, *_ = self._run_turn("I think a lot of people")
        reply_b, *_ = self._run_turn("Hmm, hard to say exactly")
        fam_a = classify_question(reply_a)
        fam_b = classify_question(reply_b)
        self.assertIsNotNone(fam_a)
        self.assertIsNotNone(fam_b)
        # Semantic families differ across turns even though both pursue
        # frequency with insufficient answers.
        self.assertNotEqual(fam_a, fam_b)
        self.assertEqual(self._asked_after(),
                         [fam_a.value, fam_b.value])

    def test_llm_repeat_of_asked_family_is_blocked(self):
        self._seed(personas=["students"], problems=["forgetting"])
        # Turn 1: LLM asks an estimate question (fresh family).
        reply_a, *_ = self._run_turn(
            "I think a lot of people", ollama=_StubOllama(["How often does this occur?"]),
        )
        self.assertEqual(classify_question(reply_a),
                         QuestionFamily.FREQUENCY_ESTIMATE)
        # Turn 2: LLM repeats the SAME family; a fresh family is available, so
        # the guard substitutes the family-aware fallback.
        reply_b, *_ = self._run_turn(
            "Hmm, hard to say", ollama=_StubOllama(["How often does this occur?"]),
        )
        self.assertEqual(classify_question(reply_b),
                         QuestionFamily.FREQUENCY_PERCENTAGE)
        self.assertNotEqual(classify_question(reply_b),
                            QuestionFamily.FREQUENCY_ESTIMATE)

    def test_sufficient_answer_stops_frequency_repeats(self):
        self._seed(personas=["students"], problems=["forgetting"])
        reply, *_ = self._run_turn("It happens every single day.")
        # frequency is now SUFFICIENT -> objective advances past FREQUENCY.
        self.assertNotEqual(classify_question(reply),
                            QuestionFamily.FREQUENCY_ESTIMATE)
        self.assertNotIn("FREQUENCY_ESTIMATE", self._asked_after())
        self.assertNotIn("FREQUENCY_PERCENTAGE", self._asked_after())

    def test_reask_sanctioned_when_all_families_exhausted(self):
        all_frequency = [f.value for f in FAMILIES_BY_FIELD[StateField.FREQUENCY]]
        # frequency is left EMPTY (the user's answers were never captured), so
        # FREQUENCY stays the active objective and every family has been asked.
        self._seed(
            personas=["students"], problems=["forgetting"],
            asked_families=all_frequency,
        )
        reply, _s, _t, diag = self._run_turn("I keep telling you, a lot")
        # All families already asked and answers remain insufficient -> the
        # re-ask is allowed (not blocked by the guard).
        self.assertIsNotNone(classify_question(reply))
        self.assertEqual(
            field_of(classify_question(reply)), StateField.FREQUENCY,
        )
        self.assertTrue(diag["QuestionFamilies"]["Re-ask Sanctioned"])


# ---------------------------------------------------------------------------
# Diagnostics — Developer Console shows family tracking
# ---------------------------------------------------------------------------


class TestQuestionFamilyDiagnostics(unittest.TestCase):
    USER = "qf_diag"
    PROJ = "qf_diag_proj"

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
        with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama()}):
            with mock.patch(
                "mentor.MemoryExtractor.extract",
                return_value=ExtractionResult(MessageType.AMBIGUOUS, []),
            ):
                return mentor.process_mentor_turn(
                    message, username=self.USER, project_name=self.PROJ,
                )

    def _seed(self, **kwargs):
        from session_manager import get_session_manager
        mgr = get_session_manager()
        sd = mgr.get_active_session_data()
        if kwargs.get("personas"):
            sd.project_state.personas = kwargs["personas"]
        if kwargs.get("problems"):
            sd.project_state.problems = kwargs["problems"]
        if kwargs.get("frequency"):
            sd.project_state.frequency = kwargs["frequency"]
        sd.asked_question_families = list(kwargs.get("asked_families", []))
        mgr.save_session_data(mgr.get_active_session_id(), sd)

    def test_diagnostics_show_family_fields(self):
        self._seed(personas=["students"], problems=["forgetting"])
        reply, _s, _t, diag = self._run_turn("It's pretty common, I'd say")
        section = diag["QuestionFamilies"]
        self.assertIn("Question Family", section)
        self.assertIn("Planned Family", section)
        self.assertIn("Previously Asked Families", section)
        self.assertIn("Family Skip Reasons", section)
        self.assertIn("Re-ask Sanctioned", section)
        # The current question's family is reported in the console.
        expected = classify_question(reply)
        if expected is not None:
            self.assertEqual(section["Question Family"],
                             family_label(expected))

    def test_skip_reasons_surface_when_family_skipped(self):
        # frequency still missing (user's answer was not captured) and one
        # family already asked -> the planner skips it and selects a fresh one.
        self._seed(
            personas=["students"], problems=["forgetting"],
            asked_families=["FREQUENCY_ESTIMATE"],
        )
        _reply, _s, _t, diag = self._run_turn("Still pretty common")
        section = diag["QuestionFamilies"]
        # The already-asked family is shown in history and its skip reason is
        # reported.
        self.assertEqual(section["Previously Asked Families"],
                         ["Frequency estimate"])
        self.assertEqual(len(section["Family Skip Reasons"]), 1)
        self.assertEqual(
            section["Family Skip Reasons"][0]["family"],
            "FREQUENCY_ESTIMATE",
        )
        self.assertTrue(section["Family Skip Reasons"][0]["reason"])

    def test_question_family_is_json_serialisable(self):
        import json
        self._seed(personas=["students"])
        _reply, _s, _t, diag = self._run_turn("students")
        round_tripped = json.loads(json.dumps(diag["QuestionFamilies"]))
        self.assertEqual(round_tripped.keys(),
                         diag["QuestionFamilies"].keys())


class TestFallbackConversationalFlow(unittest.TestCase):
    """Deterministic fallback replies are written like an experienced Design
    Thinking facilitator: observation/acknowledgment first, one focused
    question, concise, no questionnaire filler.

    These tests pin two orthogonal properties:
    (a) the conversational-flow GOALS hold for every fallback reply, and
    (b) the fallback→family/objective DECISION mapping is byte-identical —
        only the wording changed, never which family/objective a fallback
        answers. The golden-conversation fixtures separately pin the full
        state/objective/checklist decisions per turn.
    """

    @staticmethod
    def _goal_failures(text):
        r = analyze_reply(reply=text)
        failures = {}
        if not r["begins_with_acknowledgment"]:
            failures["begins_with_acknowledgment"] = r["begins_with_acknowledgment"]
        if not r["has_observation"]:
            failures["has_observation"] = r["has_observation"]
        if r["immediately_asks_question"]:
            failures["immediately_asks_question"] = r["immediately_asks_question"]
        if text.count("?") != 1:
            failures["one_question"] = text.count("?")
        if r["word_count"] > 40:
            failures["concise"] = r["word_count"]
        if not 2 <= r["sentence_count"] <= 4:
            failures["sentence_span"] = r["sentence_count"]
        if "Thank you for sharing" in text or "Great question" in text:
            failures["filler"] = "filler phrase present"
        return failures

    def test_every_family_fallback_meets_conversational_flow_goals(self):
        for family, text in FAMILY_FALLBACK_QUESTIONS.items():
            with self.subTest(family=family.value):
                self.assertEqual(self._goal_failures(text), {}, text)

    # Documented objective→family decision encoded by the fallback wording.
    OBJECTIVE_FAMILY_MAP = {
        "PERSONAS": QuestionFamily.PERSONAS_WHO,
        "PROBLEMS": QuestionFamily.PROBLEMS_CORE,
        "CURRENT_SOLUTIONS": QuestionFamily.SOLUTIONS_CURRENT,
        "PAIN_POINTS": QuestionFamily.PAIN_MOTIVATION,
        "EVIDENCE": QuestionFamily.EVIDENCE_OBSERVATIONS,
        "FREQUENCY": QuestionFamily.FREQUENCY_ESTIMATE,
    }

    def test_objective_fallback_decision_mapping_preserved(self):
        """Each Objective's fallback must still answer the documented family
        (the decision), while also meeting the conversational-flow goals."""
        for name, family in self.OBJECTIVE_FAMILY_MAP.items():
            with self.subTest(objective=name):
                text = _OBJECTIVE_FALLBACK_QUESTION[Objective[name]]
                self.assertEqual(classify_question(text), family, text)
                self.assertEqual(self._goal_failures(text), {}, text)

    def test_summary_confirmation_transition_open_with_acknowledgment(self):
        state = ProjectState(
            personas=["students"],
            problems=["forgetting"],
            pain_points=["stress"],
        )
        summary = _build_summary_fallback(state)
        self.assertTrue(
            analyze_reply(reply=summary)["begins_with_acknowledgment"], summary
        )
        self.assertFalse(
            analyze_reply(reply=summary)["immediately_asks_question"], summary
        )
        self.assertEqual(summary.count("?"), 1, summary)

        objective = ObjectiveEngine().determine_next(state)
        strategy = ResponseStrategyEngine().determine_strategy(objective, state)
        confirmation = _build_journey_fallback(
            lifecycle_decision=LifecycleDecision.WAITING_FOR_CONFIRMATION,
            conversation_objective=objective,
            response_strategy=strategy,
            project_state=state,
        )
        transition = _build_journey_fallback(
            lifecycle_decision=LifecycleDecision.READY_FOR_TRANSITION,
            conversation_objective=objective,
            response_strategy=strategy,
            project_state=state,
        )
        for label, text in (("confirmation", confirmation),
                            ("transition", transition)):
            with self.subTest(label=label):
                self.assertTrue(
                    analyze_reply(reply=text)["begins_with_acknowledgment"], text
                )
                self.assertFalse(
                    analyze_reply(reply=text)["immediately_asks_question"], text
                )
                self.assertEqual(text.count("?"), 1, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
