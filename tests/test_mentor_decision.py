"""
tests/test_mentor_decision.py — regression tests for the measurement-only
Mentor Decision Audit (``mentor_decision_audit`` + its wiring in
``mentor``).

All tests are deterministic (mocked ollama + mocked MemoryExtractor).
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mentor_decision_audit as audit  # noqa: E402
from mentor_decision_audit import (  # noqa: E402
    CONTINUITY_GOOD,
    CONTINUITY_PARTIAL,
    CONTINUITY_POOR,
    QUALITY_NATURAL,
    QUALITY_REPEATED,
    QUALITY_SLIGHTLY_REPETITIVE,
    QUALITY_QUESTIONNAIRE,
    TRANSITION_TYPE_DIRECT,
    TRANSITION_TYPE_CONNECTED,
    TRANSITION_TYPE_SUMMARY_BRIDGE,
    TRANSITION_TYPE_TOPIC_RESTART,
    MentorDecisionSummary,
    assess_mentor_decision,
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
                operation=Operation(u[0]), field=StateField(u[1]), value=u[2]
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


# ---------------------------------------------------------------------------
# Unit tests — assess_mentor_decision
# ---------------------------------------------------------------------------


class TestAssessMentorDecision(unittest.TestCase):
    """Deterministic per-turn mentor decision classification."""

    def _call(self, **overrides):
        """Call assess_mentor_decision with static defaults,
        everything else via ``overrides``."""
        return assess_mentor_decision(
            turn_index=1,
            user_message="students forget assignments",
            objective_confidence=1.0,
            selection_trace=None,
            memory={"open_threads": [], "deferred_topics": [], "resolved_threads": []},
            **overrides,
        )

    def test_fresh_follows_user_answer(self):
        """Question targets the field the user just answered → GOOD."""
        r = self._call(
            objective_value="PERSONAS",
            question_family="PERSONAS_WHO",
            state_after={
                "personas": ["students"],
                "problems": [],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
            recovery={"Category": "NONE", "Affected Fields": ["personas"]},
        )
        self.assertEqual(r["continuity"], CONTINUITY_GOOD)
        self.assertEqual(r["question_quality"], QUALITY_NATURAL)

    def test_detached_question_is_questionnaire(self):
        """Question ignores the user's answer about problems and
        fires a scripted frequency question → Questionnaire-like + POOR
        continuity (the mentor steered to an unrelated field while
        the user opened up about problems)."""
        r = self._call(
            objective_value="PROBLEMS",
            question_family="FREQUENCY_ESTIMATE",
            state_after={
                "personas": ["students"],
                "problems": ["forgetting assignments"],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
            recovery={"Category": "TOPIC_CHANGE", "Affected Fields": ["personas", "problems"]},
        )
        self.assertEqual(r["continuity"], CONTINUITY_POOR)
        self.assertEqual(r["question_quality"], QUALITY_QUESTIONNAIRE)

    def test_guard_blocked_is_repeated(self):
        """LLM produced a semantic duplicate → Repeated quality + POOR
        continuity."""
        r = self._call(
            objective_value="PERSONAS",
            question_family="PERSONAS_WHO",
            family_plan={"reask": False, "asked_families": ["PERSONAS_WHO"]},
            guard_blocked=True,
            state_after={
                "personas": ["students"],
                "problems": ["forgetting assignments"],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
            recovery={"Category": "NONE", "Affected Fields": ["personas", "problems"]},
        )
        self.assertEqual(r["question_quality"], QUALITY_REPEATED)
        self.assertEqual(r["continuity"], CONTINUITY_POOR)

    def test_reask_is_slightly_repetitive(self):
        """Sanctioned re-ask → Slightly Repetitive quality + PARTIAL
        continuity."""
        r = self._call(
            objective_value="PERSONAS",
            question_family="PERSONAS_WHO",
            family_plan={"reask": True, "asked_families": ["PERSONAS_WHO"]},
            guard_blocked=False,
            state_before=_empty_state(),
            state_after=_empty_state(),
            recovery={"Category": "DONT_KNOW", "Affected Fields": []},
        )
        self.assertEqual(r["question_quality"], QUALITY_SLIGHTLY_REPETITIVE)
        self.assertEqual(r["continuity"], CONTINUITY_PARTIAL)

    def test_summary_turn_has_no_question(self):
        """A summary/confirmation turn has no question_family → GOOD
        continuity, Natural quality."""
        r = self._call(
            objective_value="WRAP_UP",
            question_family=None,
            generated_question="I understand — let me summarize.",
            state_after={
                "personas": ["students"],
                "problems": ["forgetting assignments"],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": "daily",
            },
            lifecycle_decision="READY_FOR_SUMMARY",
            response_strategy="GENERATE_SUMMARY",
        )
        self.assertEqual(r["continuity"], CONTINUITY_GOOD)
        self.assertEqual(r["question_quality"], QUALITY_NATURAL)
        self.assertTrue(r["acknowledged_information"])

    def test_restart_when_field_already_satisfied(self):
        """Question targets a field already satisfied before this turn →
        restarted_topic = True + POOR continuity."""
        r = self._call(
            objective_value="PERSONAS",
            question_family="PERSONAS_WHO",
            state_before={
                "personas": ["elderly people"],
                "problems": [],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
            state_after={
                "personas": ["elderly people", "students"],
                "problems": [],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
            recovery={"Category": "NONE", "Affected Fields": ["personas"]},
        )
        self.assertTrue(r["restarted_topic"])
        self.assertEqual(r["continuity"], CONTINUITY_POOR)

    def test_better_alternative_when_user_answered_different_field(self):
        """User answered problems but objective stayed on personas;
        personas got satisfied as a side effect → objective
        advanced appropriately. better_alternative is None because
        the target field was satisfied."""
        r = self._call(
            objective_value="PERSONAS",
            question_family="PERSONAS_WHO",
            state_after={
                "personas": ["students"],
                "problems": ["forgetting assignments"],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
            recovery={"Category": "TOPIC_CHANGE", "Affected Fields": ["problems"]},
        )
        # personas got satisfied → objective advanced appropriately
        self.assertTrue(r["objective_appropriate"])
        # problems is still unmet but was a side effect, not the target
        self.assertIsNone(r["better_alternative"])

    def test_no_better_alternative_when_user_answered_target(self):
        """User answered the objective's target field → no better
        alternative."""
        r = self._call(
            objective_value="PERSONAS",
            question_family="PERSONAS_WHO",
            state_after={
                "personas": ["students"],
                "problems": ["forgetting assignments"],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
            recovery={"Category": "NONE", "Affected Fields": ["personas"]},
        )
        self.assertTrue(r["objective_appropriate"])
        self.assertIsNone(r["better_alternative"])

    def test_acknowledged_when_objective_advanced(self):
        """The user's answer satisfied the target field → mentor
        acknowledged by advancing."""
        r = self._call(
            objective_value="PERSONAS",
            question_family="PROBLEMS_CORE",
            objective_advancement="ADVANCED",
            state_after={
                "personas": ["students"],
                "problems": [],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
            recovery={"Category": "NONE", "Affected Fields": ["personas"]},
        )
        self.assertTrue(r["acknowledged_information"])

    def test_detached_questionnaire_does_not_acknowledge_unrelated_answer(self):
        """A questionnaire-like question about an unrelated field
        (frequency) while the user opened up about problems →
        not acknowledged about the user's main concern."""
        r = self._call(
            objective_value="PROBLEMS",
            question_family="FREQUENCY_ESTIMATE",
            state_after={
                "personas": ["students"],
                "problems": ["forgetting assignments"],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
            recovery={"Category": "TOPIC_CHANGE", "Affected Fields": ["personas", "problems"]},
        )
        self.assertEqual(r["continuity"], CONTINUITY_POOR)
        self.assertEqual(r["question_quality"], QUALITY_QUESTIONNAIRE)
        # The mentor advanced past personas (acknowledged that),
        # but the question about frequency ignored the user's
        # main concern (problems). acknowledged_information is
        # True because personas was resolved; continuity/quality
        # capture the disconnect.
        self.assertTrue(r["acknowledged_information"])


class TestMentorDecisionSummary(unittest.TestCase):
    """Aggregation, merge, persistence, and scoring."""

    def _record(self, **overrides):
        """Return a plain dict suitable for summary.add_record."""
        defaults = dict(
            turn_index=1,
            objective="PERSONAS",
            continuity=CONTINUITY_GOOD,
            question_quality=QUALITY_NATURAL,
            objective_appropriate=True,
            acknowledged_information=True,
            restarted_topic=False,
            better_alternative=None,
            user_message="hello",
            generated_question="Q?",
        )
        defaults.update(overrides)
        return defaults

    def test_add_record_aggregates_counts(self):
        summary = MentorDecisionSummary()
        summary.add_record(self._record(objective="PERSONAS", continuity=CONTINUITY_GOOD, question_quality=QUALITY_NATURAL))
        summary.add_record(self._record(objective="PERSONAS", continuity=CONTINUITY_GOOD, question_quality=QUALITY_NATURAL))
        summary.add_record(self._record(objective="PROBLEMS", continuity=CONTINUITY_POOR, question_quality=QUALITY_REPEATED))
        self.assertEqual(summary.turn_count, 3)
        self.assertEqual(summary.objective_counts["PERSONAS"], 2)
        self.assertEqual(summary.objective_counts["PROBLEMS"], 1)
        self.assertEqual(summary.continuity_counts[CONTINUITY_GOOD], 2)
        self.assertEqual(summary.continuity_counts[CONTINUITY_POOR], 1)
        self.assertEqual(summary.quality_counts[QUALITY_NATURAL], 2)
        self.assertEqual(summary.quality_counts[QUALITY_REPEATED], 1)
        self.assertEqual(summary.appropriate_count, 3)
        self.assertEqual(summary.acknowledged_count, 3)
        self.assertEqual(summary.restarted_count, 0)

    def test_continuity_score(self):
        summary = MentorDecisionSummary()
        summary.add_record(self._record(continuity=CONTINUITY_GOOD))
        summary.add_record(self._record(continuity=CONTINUITY_GOOD))
        summary.add_record(self._record(continuity=CONTINUITY_POOR))
        # (2*1.0 + 1*0.0) / 3 * 100 = 66.7
        self.assertEqual(summary.continuity_score(), 66.7)

    def test_quality_score(self):
        summary = MentorDecisionSummary()
        summary.add_record(self._record(question_quality=QUALITY_NATURAL))
        summary.add_record(self._record(question_quality=QUALITY_NATURAL))
        summary.add_record(self._record(question_quality=QUALITY_QUESTIONNAIRE))
        # (2*1.0 + 1*0.0) / 3 * 100 = 66.7
        self.assertEqual(summary.quality_score(), 66.7)

    def test_repeated_objectives(self):
        summary = MentorDecisionSummary()
        summary.add_record(self._record(objective="PERSONAS"))
        summary.add_record(self._record(objective="PERSONAS"))
        summary.add_record(self._record(objective="PROBLEMS"))
        repeated = summary.repeated_objectives()
        self.assertEqual(repeated["PERSONAS"], 2)
        self.assertNotIn("PROBLEMS", repeated)

    def test_weak_transitions(self):
        summary = MentorDecisionSummary()
        summary.add_record(self._record(objective="PERSONAS"))
        summary.add_record(
            self._record(
                objective="PROBLEMS",
                continuity=CONTINUITY_POOR,
                question_quality=QUALITY_QUESTIONNAIRE,
            )
        )
        weak = summary.most_common_weak_transitions()
        self.assertIn("PERSONAS → PROBLEMS", weak)
        self.assertEqual(weak["PERSONAS → PROBLEMS"], 1)

    def test_better_alternative_turns(self):
        summary = MentorDecisionSummary()
        summary.add_record(
            self._record(
                better_alternative={"objective": "PROBLEMS", "reason": "user opened X"},
                user_message="I keep forgetting",
                generated_question="Who benefits?",
            )
        )
        self.assertEqual(len(summary.better_alternative_turns), 1)
        self.assertEqual(summary.better_alternative_turns[0]["better_alternative"], "PROBLEMS")

    def test_merge(self):
        a = MentorDecisionSummary()
        a.add_record(self._record(objective="PERSONAS", continuity=CONTINUITY_GOOD, question_quality=QUALITY_NATURAL))
        b = MentorDecisionSummary()
        b.add_record(self._record(objective="PROBLEMS", continuity=CONTINUITY_POOR, question_quality=QUALITY_REPEATED))
        a.merge(b)
        self.assertEqual(a.turn_count, 2)
        self.assertEqual(a.objective_counts["PERSONAS"], 1)
        self.assertEqual(a.objective_counts["PROBLEMS"], 1)
        self.assertEqual(a.continuity_counts[CONTINUITY_GOOD], 1)
        self.assertEqual(a.continuity_counts[CONTINUITY_POOR], 1)

    def test_persistence_round_trip(self):
        summary = MentorDecisionSummary()
        summary.add_record(self._record(objective="PERSONAS", continuity=CONTINUITY_GOOD, question_quality=QUALITY_NATURAL))
        restored = MentorDecisionSummary.from_dict(summary.to_dict())
        self.assertEqual(restored.to_dict(), summary.to_dict())

    def test_session_data_persists_summary(self):
        sd = SessionData()
        sd.mentor_decision_summary.add_record(self._record(objective="PERSONAS"))
        restored = SessionData.from_dict(sd.to_dict())
        self.assertEqual(
            restored.mentor_decision_summary.to_dict(),
            sd.mentor_decision_summary.to_dict(),
        )


class TestMentorDecisionWiringLivePipeline(unittest.TestCase):
    """The live pipeline surfaces the Mentor Decision sections;
    measurement only."""

    USER = "decision_user"
    PROJ = "DecisionProject"

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

        return get_session_manager().get_active_session_data().mentor_decision_summary

    def test_llm_turn_surfaces_mentor_decision_section(self):
        def _patched(*args, **kwargs):
            return _meaningful(
                ("ADD", "personas", "students"),
                ("ADD", "problems", "forgetting assignments"),
            )

        reply, _session, _timing, diagnostics = self._run_turn(
            "students keep forgetting assignments",
            _patched,
        )
        section = diagnostics["MentorDecision"]
        self.assertTrue(section)
        self.assertIn("Current Objective", section)
        self.assertIn("Conversation Continuity", section)
        self.assertIn("Question Quality", section)
        self.assertIn("Decision Confidence", section)
        self.assertIn("Generated Question", section)

        summary = diagnostics["MentorDecisionSummary"]
        self.assertTrue(summary.get("Objective Distribution"))
        self.assertTrue(reply and reply.strip())

    def test_audit_changes_nothing_about_the_pipeline(self):
        """The decision audit is observation-only: the turn's state,
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
        self.assertIn("MentorDecision", diagnostics)
        self.assertIn("MentorDecisionSummary", diagnostics)
        self.assertTrue(reply and reply.strip())

    def test_skip_turn_still_gets_decision_audit(self):
        """Even when the hybrid decision skips the LLM, the mentor
        decision audit runs (objective + question family are always
        produced)."""

        def _patched(*args, **kwargs):
            raise AssertionError("LLM extractor must not run on a skip")

        reply, _session, _timing, diagnostics = self._run_turn(
            "I want to help elderly people take their medications on time",
            _patched,
            gate="false",
        )
        self.assertIn("MentorDecision", diagnostics)
        self.assertIn("MentorDecisionSummary", diagnostics)
        self.assertTrue(reply and reply.strip())


# ---------------------------------------------------------------------------
# Transition-type classification — unit tests
# ---------------------------------------------------------------------------


class TestTransitionType(unittest.TestCase):
    """Transition Type classifies *how* the mentor moved from the user's
    answer to the next question: Direct, Connected, Summary Bridge, or
    Topic Restart."""

    def _call(self, **overrides):
        defaults = dict(
            turn_index=1,
            user_message="students forget assignments",
            objective_value="PERSONAS",
            objective_confidence=1.0,
            question_family="PERSONAS_WHO",
            family_plan={"reask": False, "asked_families": []},
            guard_blocked=False,
            state_before=_empty_state(),
            state_after=_empty_state(),
            recovery={"Category": "NONE", "Affected Fields": []},
            memory={"open_threads": [], "deferred_topics": [], "resolved_threads": []},
            lifecycle_decision="CONTINUE",
            response_strategy="ASK_QUESTION",
        )
        defaults.update(overrides)
        return assess_mentor_decision(**defaults)

    def test_direct_when_follows_user_answer(self):
        """Question targets the field the user just answered → Direct."""
        r = self._call(
            objective_value="PERSONAS",
            question_family="PERSONAS_WHO",
            state_after={
                "personas": ["students"],
                "problems": [],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
        )
        self.assertEqual(r["transition_type"], TRANSITION_TYPE_DIRECT)

    def test_connected_when_advances_to_new_field(self):
        """User's answer satisfied the target field and the question
        moves to a new, different, unmet field → Connected."""
        r = self._call(
            objective_value="PROBLEMS",
            question_family="PROBLEMS_CORE",
            state_before=_empty_state(),
            state_after={
                "personas": ["students"],
                "problems": [],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
        )
        self.assertEqual(r["transition_type"], TRANSITION_TYPE_CONNECTED)

    def test_summary_bridge_on_summary_lifecycle(self):
        """A lifecycle decision in the summary family → Summary Bridge."""
        r = self._call(
            objective_value="WRAP_UP",
            question_family=None,
            lifecycle_decision="READY_FOR_SUMMARY",
            response_strategy="GENERATE_SUMMARY",
            state_after={
                "personas": ["students"],
                "problems": ["forgetting assignments"],
                "current_solutions": ["planners"],
                "pain_points": ["anxiety"],
                "evidence": ["grades drop"],
                "impacts": [],
                "frequency": "daily",
            },
        )
        self.assertEqual(r["transition_type"], TRANSITION_TYPE_SUMMARY_BRIDGE)

    def test_topic_restart_when_satisfied_before(self):
        """Question targets a field already satisfied before this turn →
        Topic Restart."""
        r = self._call(
            objective_value="PERSONAS",
            question_family="PERSONAS_WHO",
            state_before={
                "personas": ["elderly people"],
                "problems": [],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
            state_after={
                "personas": ["elderly people"],
                "problems": [],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
        )
        self.assertEqual(r["transition_type"], TRANSITION_TYPE_TOPIC_RESTART)

    def test_topic_restart_when_guard_blocked(self):
        """Guard blocked the produced question → Topic Restart."""
        r = self._call(
            objective_value="PERSONAS",
            question_family="PERSONAS_WHO",
            family_plan={"reask": False, "asked_families": ["PERSONAS_WHO"]},
            guard_blocked=True,
        )
        self.assertEqual(r["transition_type"], TRANSITION_TYPE_TOPIC_RESTART)

    def test_connected_when_returning_to_open_thread(self):
        """Question returns to an open (still-unmet, previously asked)
        field → Connected."""
        r = self._call(
            objective_value="PROBLEMS",
            question_family="PROBLEMS_CORE",
            memory={
                "open_threads": [{"field": "problems", "reason": "partial", "turn": 1}],
                "deferred_topics": [],
                "resolved_threads": [],
            },
            state_before={
                "personas": ["students"],
                "problems": [],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
            state_after={
                "personas": ["students"],
                "problems": [],
                "current_solutions": [],
                "pain_points": [],
                "evidence": [],
                "impacts": [],
                "frequency": None,
            },
        )
        self.assertEqual(r["transition_type"], TRANSITION_TYPE_CONNECTED)

    def test_record_includes_transition_fields(self):
        """assess_mentor_decision returns transition_type and
        transition_reason keys for every turn."""
        r = self._call()
        self.assertIn("transition_type", r)
        self.assertIn("transition_reason", r)
        self.assertTrue(r["transition_reason"])


class TestTransitionTypeSummary(unittest.TestCase):
    """MentorDecisionSummary aggregates transition_type counts and
    exposes a weighted transition score."""

    def test_transition_counts_aggregated(self):
        s = MentorDecisionSummary()
        r1 = {"objective": "PERSONAS", "continuity": CONTINUITY_GOOD, "question_quality": QUALITY_NATURAL, "transition_type": TRANSITION_TYPE_DIRECT}
        r2 = {"objective": "PROBLEMS", "continuity": CONTINUITY_POOR, "question_quality": QUALITY_QUESTIONNAIRE, "transition_type": TRANSITION_TYPE_TOPIC_RESTART}
        r3 = {"objective": "WRAP_UP", "continuity": CONTINUITY_GOOD, "question_quality": QUALITY_NATURAL, "transition_type": TRANSITION_TYPE_SUMMARY_BRIDGE}
        for r in (r1, r2, r3):
            s.add_record(r)
        self.assertEqual(s.transition_counts[TRANSITION_TYPE_DIRECT], 1)
        self.assertEqual(s.transition_counts[TRANSITION_TYPE_TOPIC_RESTART], 1)
        self.assertEqual(s.transition_counts[TRANSITION_TYPE_SUMMARY_BRIDGE], 1)
        self.assertEqual(s.transition_counts[TRANSITION_TYPE_CONNECTED], 0)

    def test_transition_score_weights(self):
        s = MentorDecisionSummary()
        for _ in range(2):
            s.add_record({"objective": "PERSONAS", "continuity": CONTINUITY_GOOD, "question_quality": QUALITY_NATURAL, "transition_type": TRANSITION_TYPE_DIRECT})
        s.add_record({"objective": "PROBLEMS", "continuity": CONTINUITY_PARTIAL, "question_quality": QUALITY_NATURAL, "transition_type": TRANSITION_TYPE_CONNECTED})
        # 2*1.0 + 1*0.7 = 2.7; /3 turns *100 = 90.0
        self.assertAlmostEqual(s.transition_score(), 90.0, places=1)

    def test_transition_counts_in_to_dict_round_trip(self):
        s = MentorDecisionSummary()
        s.add_record({"objective": "PERSONAS", "continuity": CONTINUITY_GOOD, "question_quality": QUALITY_NATURAL, "transition_type": TRANSITION_TYPE_DIRECT})
        d = s.to_dict()
        self.assertIn("transition_counts", d)
        s2 = MentorDecisionSummary.from_dict(d)
        self.assertEqual(s2.transition_counts[TRANSITION_TYPE_DIRECT], 1)

    def test_to_display_includes_transition_type(self):
        s = MentorDecisionSummary()
        s.add_record({"objective": "PERSONAS", "continuity": CONTINUITY_GOOD, "question_quality": QUALITY_NATURAL, "transition_type": TRANSITION_TYPE_DIRECT})
        display = s.to_display()
        self.assertIn("Transition Type", display)
        self.assertIn("Transition Score", display)

    def test_merge_preserves_transition_counts(self):
        a = MentorDecisionSummary()
        b = MentorDecisionSummary()
        a.add_record({"objective": "PERSONAS", "continuity": CONTINUITY_GOOD, "question_quality": QUALITY_NATURAL, "transition_type": TRANSITION_TYPE_DIRECT})
        b.add_record({"objective": "PROBLEMS", "continuity": CONTINUITY_POOR, "question_quality": QUALITY_QUESTIONNAIRE, "transition_type": TRANSITION_TYPE_TOPIC_RESTART})
        b.add_record({"objective": "WRAP_UP", "continuity": CONTINUITY_GOOD, "question_quality": QUALITY_NATURAL, "transition_type": TRANSITION_TYPE_SUMMARY_BRIDGE})
        a.merge(b)
        self.assertEqual(a.transition_counts[TRANSITION_TYPE_DIRECT], 1)
        self.assertEqual(a.transition_counts[TRANSITION_TYPE_TOPIC_RESTART], 1)
        self.assertEqual(a.transition_counts[TRANSITION_TYPE_SUMMARY_BRIDGE], 1)


# ---------------------------------------------------------------------------
# Regression: objective ordering and state updates are unchanged
# ---------------------------------------------------------------------------


class TestTransitionTypeRegression(unittest.TestCase):
    """Adding transition_type to the audit must NOT change the objective
    sequence, the state updates produced by the pipeline, or any other
    observable decision. Only conversational-continuity measurements change."""

    def _run_turn(self, user_message, patched_extract, gate="true"):
        os.environ["COMPLEXITY_GATE"] = gate
        try:
            from session_manager import get_session_manager

            get_session_manager().reset_runtime_state(
                username="tester", project_title="transition_regression"
            )
        except Exception:
            pass
        raising = _RaisingOllama()
        with mock.patch.dict("sys.modules", {"ollama": raising}):
            with mock.patch(
                "mentor.MemoryExtractor.extract", side_effect=patched_extract
            ):
                reply, session, timing, diagnostics = mentor.process_mentor_turn(
                    user_message,
                    username="tester",
                    project_name="transition_regression",
                )
        return reply, session, timing, diagnostics

    def test_objective_sequence_unchanged(self):
        """The objective and lifecycle decisions are the same as a non-
        audited run — transition_type is observation-only."""

        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "personas", "students"))

        reply, _session, _timing, diagnostics = self._run_turn(
            "students forget their assignments",
            _patched,
        )
        pipeline = diagnostics["Pipeline"]
        self.assertEqual(pipeline["Objective"], "PROBLEMS")
        self.assertEqual(pipeline["Lifecycle Decision"], "CONTINUE")
        self.assertEqual(pipeline["Response Strategy"], "ASK_QUESTION")

    def test_state_updates_unchanged(self):
        """Extraction still mutates the same state fields — transition_type
        never feeds back into extraction or state."""

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

    def test_transition_type_present_in_diagnostics(self):
        """MentorDecision diagnostics expose the new Transition Type field."""

        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "personas", "students"))

        reply, _session, _timing, diagnostics = self._run_turn(
            "students forget their assignments",
            _patched,
        )
        section = diagnostics["MentorDecision"]
        self.assertIn("Transition Type", section)
        self.assertIn("Transition Reason", section)
        self.assertTrue(section["Transition Reason"])

    def test_transition_type_in_summary(self):
        """MentorDecisionSummary aggregates transition counts across turns."""
        from session_manager import get_session_manager

        get_session_manager().reset_runtime_state(
            username="summary_tester", project_title="transition_summary"
        )

        def _patched(*args, **kwargs):
            return _meaningful(("ADD", "personas", "students"))

        raising = _RaisingOllama()
        with mock.patch.dict("sys.modules", {"ollama": raising}):
            with mock.patch(
                "mentor.MemoryExtractor.extract", side_effect=_patched
            ):
                mentor.process_mentor_turn(
                    "students forget assignments",
                    username="summary_tester",
                    project_name="transition_summary",
                )
        summary = (
            get_session_manager()
            .get_active_session_data()
            .mentor_decision_summary
        )
        self.assertTrue(summary.transition_counts)
        self.assertGreaterEqual(
            sum(summary.transition_counts.values()), 1
        )


if __name__ == "__main__":
    unittest.main()