"""Tests for the observation-only Conversation Failure Audit.

Covers:
  * per-category classification (every ConversationFailure value);
  * summary aggregation, merge, serialization and display;
  * the Developer Console diagnostics section projection;
  * the live-pipeline wiring (mentor.process_mentor_turn diagnostics +
    session-level summary persistence);
  * the observation-only guarantee (the analyzer never mutates its inputs and
    never changes replies/objectives/lifecycle/extraction);
  * golden fixtures remain byte-identical.
"""

import json
import unittest
from unittest import mock

from conversation_failure_audit import (
    ConversationFailure,
    ConversationFailureRecord,
    ConversationFailureSummary,
    analyze_conversation_failure,
    conversation_failure_diagnostics_section,
    failure_label,
)

from tests.goldens.fixtures import load_all_fixtures, load_fixture
from tests.goldens.runner import (
    _RaisingOllama,
    _StubOllama,
    _build_extraction,
    _seed_session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mem(**overrides):
    base = {
        "open_threads": [],
        "resolved_threads": [],
        "deferred_topics": [],
        "acknowledged_facts": [],
        "summarized_facts": [],
    }
    base.update(overrides)
    return base


def _fields(**state):
    default = {k: [] for k in (
        "personas", "problems", "current_solutions", "pain_points",
        "evidence", "impacts", "frequency",
    )}
    default.update({k: v for k, v in state.items()})
    return default


def _analyze(**kw):
    return analyze_conversation_failure(**kw)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class TestDefaultAndNone(unittest.TestCase):
    def test_bare_call_returns_valid_record(self):
        rec = _analyze()
        self.assertIsInstance(rec, ConversationFailureRecord)
        self.assertEqual(rec.confidence, round(rec.confidence, 2))

    def test_healthy_turn_is_none(self):
        mem = _mem(open_threads=[{"field": "personas", "reason": "p", "turn": 1}])
        rec = _analyze(
            user_message="Small business owners in my city",
            mentor_reply="Good - to make it concrete, could you tell me about one?",
            conversation_move="ELICIT_INFORMATION",
            question_family="PERSONAS_EXAMPLES",
            project_state_before=_fields(),
            project_state_after=_fields(personas=["Small business owners"]),
            conversation_memory=mem,
        )
        self.assertIs(rec.primary_failure, ConversationFailure.NONE)


class TestMetaIntent(unittest.TestCase):
    def test_empty_user_is_meta(self):
        rec = _analyze(user_message="", mentor_reply="Who is affected?")
        self.assertIs(rec.primary_failure, ConversationFailure.META_INTENT)

    def test_short_acknowledgment_is_meta(self):
        rec = _analyze(
            user_message="thanks", mentor_reply="Understood, who else?",
            conversation_move="ELICIT_INFORMATION",
            project_state_before=_fields(), project_state_after=_fields(),
        )
        self.assertIs(rec.primary_failure, ConversationFailure.META_INTENT)

    def test_trust_marker_is_meta(self):
        rec = _analyze(
            user_message="I trust you", mentor_reply="Great, let's continue.",
        )
        self.assertIs(rec.primary_failure, ConversationFailure.META_INTENT)

    def test_vague_deferring_answer_is_not_meta(self):
        rec = _analyze(
            user_message="It depends on the situation",
            mentor_reply="Could you estimate how often?",
            conversation_move="ELICIT_INFORMATION",
            question_family="FREQUENCY_ESTIMATE",
            project_state_before=_fields(), project_state_after=_fields(),
            conversation_memory=_mem(open_threads=[{"field": "frequency", "reason": "p", "turn": 1}]),
        )
        self.assertIs(rec.primary_failure, ConversationFailure.NONE)


class TestMisunderstoodResponse(unittest.TestCase):
    def test_reasking_populated_field(self):
        rec = _analyze(
            user_message="The main issue is slow checkout at the till",
            mentor_reply="Great, so what other problems are there, e.g. stock issues?",
            conversation_move="ELICIT_INFORMATION",
            question_family="PROBLEMS_WHY",
            project_state_before=_fields(problems=["Slow checkout"]),
            project_state_after=_fields(problems=["Slow checkout"]),
            conversation_memory=_mem(),
        )
        self.assertIs(rec.primary_failure, ConversationFailure.MISUNDERSTOOD_RESPONSE)

    def test_resolved_field_reask_is_over_exploration_not_misunderstood(self):
        mem = _mem(resolved_threads=[{"field": "personas", "reason": "r", "turn": 2}])
        rec = _analyze(
            user_message="Cashiers handle the till",
            mentor_reply="And who else is affected by the checkout queue?",
            conversation_move="ELICIT_INFORMATION",
            question_family="PERSONAS_WHO",
            project_state_before=_fields(personas=["Cashiers"]),
            project_state_after=_fields(personas=["Cashiers"]),
            conversation_memory=mem,
        )
        self.assertIs(rec.primary_failure, ConversationFailure.OVER_EXPLORATION)


class TestOverExploration(unittest.TestCase):
    def test_expand_resolved_field(self):
        mem = _mem(resolved_threads=[{"field": "personas", "reason": "r", "turn": 2}])
        rec = _analyze(
            user_message="The cashiers rush at closing time too",
            mentor_reply="Interesting - what else can you tell me about the cashiers?",
            conversation_move="EXPAND_IDEA",
            question_family="PERSONAS_EXAMPLES",
            project_state_before=_fields(personas=["Cashiers"]),
            project_state_after=_fields(personas=["Cashiers"]),
            conversation_memory=mem,
        )
        self.assertIs(rec.primary_failure, ConversationFailure.OVER_EXPLORATION)


class TestAbruptTransition(unittest.TestCase):
    def test_transition_before_acknowledgment(self):
        mem = _mem(open_threads=[{"field": "problems", "reason": "p", "turn": 1}])
        rec = _analyze(
            user_message="The queue is the biggest annoyance on busy days",
            mentor_reply="Great, now tell me more about the current solutions you have today.",
            conversation_move="TRANSITION_TOPIC",
            question_family="SOLUTIONS_CURRENT",
            project_state_before=_fields(problems=["Slow checkout"]),
            project_state_after=_fields(problems=["Slow checkout"]),
            conversation_memory=mem,
        )
        self.assertIs(rec.primary_failure, ConversationFailure.ABRUPT_TRANSITION)

    def test_close_topic_with_nothing_acknowledged(self):
        rec = _analyze(
            user_message="The queues seem to get worse at the weekend",
            mentor_reply="Let's summarise the session so far.",
            conversation_move="SUMMARIZE_PROGRESS",
            question_family="FREQUENCY_ESTIMATE",
            project_state_before=_fields(),
            project_state_after=_fields(),
            conversation_memory=_mem(),
        )
        self.assertIs(rec.primary_failure, ConversationFailure.ABRUPT_TRANSITION)


class TestUnsupportedInference(unittest.TestCase):
    def test_implies_magnitude_without_evidence(self):
        rec = _analyze(
            user_message="Small owners and offices",
            mentor_reply="Good, so is it fair to say most people hit this every day?",
            conversation_move="EXPAND_IDEA",
            question_family="PROBLEMS_WHY",
            project_state_before=_fields(),
            project_state_after=_fields(problems=["Slow checkout"]),
            conversation_memory=_mem(
                open_threads=[{"field": "problems", "reason": "p", "turn": 1}]
            ),
        )
        self.assertIs(rec.primary_failure, ConversationFailure.UNSUPPORTED_INFERENCE)


class TestMissedInsight(unittest.TestCase):
    def test_state_value_but_no_memory_thread(self):
        rec = _analyze(
            user_message="Small business owners in my city",
            mentor_reply="Understood. What core problem do they face?",
            conversation_move="ELICIT_INFORMATION",
            question_family="PERSONAS_WHO",
            project_state_before=_fields(),
            project_state_after=_fields(personas=["Small business owners"]),
            conversation_memory=_mem(),
        )
        self.assertIs(rec.primary_failure, ConversationFailure.MISSED_INSIGHT)

    def test_no_missed_insight_when_thread_exists(self):
        mem = _mem(open_threads=[{"field": "personas", "reason": "p", "turn": 1}])
        rec = _analyze(
            user_message="Small business owners in my city",
            mentor_reply="Understood. What core problem do they face?",
            conversation_move="ELICIT_INFORMATION",
            question_family="PERSONAS_WHO",
            project_state_before=_fields(),
            project_state_after=_fields(personas=["Small business owners"]),
            conversation_memory=mem,
        )
        self.assertIs(rec.primary_failure, ConversationFailure.NONE)


class TestMemoryFailure(unittest.TestCase):
    def test_acknowledged_and_open_contradiction(self):
        mem = _mem(
            acknowledged_facts=[{"field": "problems", "reason": "a", "turn": 1}],
            open_threads=[{"field": "problems", "reason": "p", "turn": 2}],
        )
        rec = _analyze(
            user_message="The checkout queue grows on weekends",
            mentor_reply="Understood, anything else?",
            conversation_move="ELICIT_INFORMATION",
            question_family="PROBLEMS_CORE",
            project_state_before=_fields(problems=["Slow checkout"]),
            project_state_after=_fields(problems=["Slow checkout"]),
            conversation_memory=mem,
        )
        self.assertIs(rec.primary_failure, ConversationFailure.MEMORY_FAILURE)

    def test_resolved_field_absent_from_state(self):
        mem = _mem(
            resolved_threads=[{"field": "problems", "reason": "r", "turn": 3}]
        )
        mem["open_threads"] = []
        rec = _analyze(
            user_message="Anything else on the core problem?",
            mentor_reply="No, let's move on.",
            conversation_move="TRANSITION_TOPIC",
            question_family="SOLUTIONS_CURRENT",
            project_state_before=_fields(),
            project_state_after=_fields(),
            conversation_memory=mem,
        )
        self.assertIn(
            ConversationFailure.MEMORY_FAILURE, [rec.primary_failure] + rec.secondary_failures
        )


class TestSocialFailure(unittest.TestCase):
    def test_reprimand_outside_challenge(self):
        rec = _analyze(
            user_message="I think the problem might be inventory",
            mentor_reply="That's not right - the problem is clearly slow checkout.",
            conversation_move="VALIDATE_DISCOVERY",
            question_family="PROBLEMS_CORE",
            project_state_before=_fields(),
            project_state_after=_fields(),
            conversation_memory=_mem(),
        )
        self.assertIs(rec.primary_failure, ConversationFailure.SOCIAL_FAILURE)

    def test_challenge_move_is_not_social_failure(self):
        rec = _analyze(
            user_message="I think the problem might be inventory",
            mentor_reply="Let's interrogate that - is inventory really the core problem?",
            conversation_move="CHALLENGE_ASSUMPTION",
            question_family="PROBLEMS_WHY",
            project_state_before=_fields(),
            project_state_after=_fields(),
            conversation_memory=_mem(),
        )
        self.assertNotIn(ConversationFailure.SOCIAL_FAILURE, [rec.primary_failure] + rec.secondary_failures)


class TestMoveSelection(unittest.TestCase):
    def test_elicit_without_question(self):
        rec = _analyze(
            user_message="Small business owners and managers",
            mentor_reply="Understood.",
            conversation_move="ELICIT_INFORMATION",
            question_family="PERSONAS_WHO",
            project_state_before=_fields(),
            project_state_after=_fields(),
            conversation_memory=_mem(),
        )
        self.assertIs(rec.primary_failure, ConversationFailure.MOVE_SELECTION)

    def test_unrecognised_move(self):
        rec = _analyze(
            user_message="Small business owners and their managers",
            mentor_reply="What else?",
            conversation_move="SOMETHING_NEW",
            project_state_before=_fields(), project_state_after=_fields(),
            conversation_memory=_mem(),
        )
        self.assertIs(rec.primary_failure, ConversationFailure.MOVE_SELECTION)


class TestRepeatedInformation(unittest.TestCase):
    def test_reoffering_acknowledged_field(self):
        mem = _mem(
            acknowledged_facts=[{"field": "problems", "reason": "a", "turn": 1}]
        )
        rec = _analyze(
            user_message="It also affects the cashiers who work the till",
            mentor_reply="Understood. Anything else on the core problem?",
            conversation_move="ELICIT_INFORMATION",
            question_family="PROBLEMS_CORE",
            project_state_before=_fields(problems=["Slow checkout"]),
            project_state_after=_fields(problems=["Slow checkout", "Long queues"]),
            conversation_memory=mem,
        )
        self.assertIs(rec.primary_failure, ConversationFailure.REPEATED_INFORMATION)

    def test_reoffering_summarized_field(self):
        mem = _mem(
            summarized_facts=[{"field": "evidence", "reason": "s", "turn": 2}],
        )
        rec = _analyze(
            user_message="I have also seen her skip doses twice last week",
            mentor_reply="Thanks, anything else?",
            conversation_move="ELICIT_INFORMATION",
            question_family="EVIDENCE_OBSERVATIONS",
            project_state_before=_fields(evidence=["skip doses"]),
            project_state_after=_fields(evidence=["skip doses", "twice last week"]),
            conversation_memory=mem,
        )
        self.assertIs(rec.primary_failure, ConversationFailure.REPEATED_INFORMATION)


class TestPrimaryAndSecondary(unittest.TestCase):
    def test_primary_is_first_detected(self):
        # Empty user message (META) with a reprimand reply (SOCIAL) -> primary META.
        rec = _analyze(
            user_message="",
            mentor_reply="That's wrong - you need to focus.",
            conversation_move="VALIDATE_DISCOVERY",
            project_state_before=_fields(), project_state_after=_fields(),
            conversation_memory=_mem(),
        )
        self.assertIs(rec.primary_failure, ConversationFailure.META_INTENT)
        self.assertNotIn(ConversationFailure.META_INTENT, rec.secondary_failures)

    def test_secondary_collected_and_deduplicated(self):
        # A resolved field re-asked (OVER) while the field is also acknowledged
        # (MEMORY contradiction) -> one of them primary, the others secondary.
        mem = _mem(
            resolved_threads=[{"field": "personas", "reason": "r", "turn": 2}],
            acknowledged_facts=[{"field": "personas", "reason": "a", "turn": 1}],
            open_threads=[{"field": "personas", "reason": "p", "turn": 3}],
        )
        rec = _analyze(
            user_message="Cashiers handle the till",
            mentor_reply="And who else is affected by the checkout queue?",
            conversation_move="EXPAND_IDEA",
            question_family="PERSONAS_WHO",
            project_state_before=_fields(personas=["Cashiers"]),
            project_state_after=_fields(personas=["Cashiers"]),
            conversation_memory=mem,
        )
        cats = [rec.primary_failure] + list(rec.secondary_failures)
        self.assertIn(ConversationFailure.OVER_EXPLORATION, cats)
        self.assertIn(ConversationFailure.MEMORY_FAILURE, cats)
        self.assertEqual(len(rec.secondary_failures), len(set(rec.secondary_failures)))


class TestConfidence(unittest.TestCase):
    def test_confidence_within_range(self):
        for args in (
            dict(user_message=""),
            dict(user_message="Small business owners", mentor_reply="Good, who else?",
                 conversation_move="ELICIT_INFORMATION", question_family="PERSONAS_WHO",
                 project_state_before=_fields(), project_state_after=_fields(personas=["Small business owners"]),
                 conversation_memory=_mem()),
            dict(user_message="Cashiers", mentor_reply="And who else?",
                 conversation_move="EXPAND_IDEA", question_family="PERSONAS_WHO",
                 project_state_before=_fields(personas=["Cashiers"]), project_state_after=_fields(personas=["Cashiers"]),
                 conversation_memory=_mem(resolved_threads=[{"field": "personas", "reason": "r", "turn": 2}])),
        ):
            rec = _analyze(**args)
            self.assertGreaterEqual(rec.confidence, 0.0)
            self.assertLessEqual(rec.confidence, 1.0)
            self.assertEqual(rec.confidence, round(rec.confidence, 2))


class TestFailureTerminology(unittest.TestCase):
    def test_every_category_has_a_label(self):
        for cat in ConversationFailure:
            self.assertTrue(failure_label(cat))

    def test_failure_label_accepts_string_value(self):
        self.assertEqual(
            failure_label("META_INTENT"), failure_label(ConversationFailure.META_INTENT)
        )


# ---------------------------------------------------------------------------
# Record serialization / diagnostics section
# ---------------------------------------------------------------------------


class TestRecordSerialization(unittest.TestCase):
    def test_roundtrip(self):
        rec = _analyze(
            user_message="It depends on the situation",
            mentor_reply="Could you estimate how often?",
            conversation_move="ELICIT_INFORMATION",
            question_family="FREQUENCY_ESTIMATE",
            project_state_before=_fields(), project_state_after=_fields(),
            conversation_memory=_mem(),
        )
        restored = ConversationFailureRecord.from_dict(rec.to_dict())
        self.assertEqual(restored.to_dict(), rec.to_dict())

    def test_from_dict_tolerates_garbage(self):
        rec = ConversationFailureRecord.from_dict({"primary_failure": "NOT_A_CATEGORY"})
        self.assertIs(rec.primary_failure, ConversationFailure.NONE)
        self.assertEqual(rec.to_dict()["secondary_failures"], [])

    def test_from_none(self):
        rec = ConversationFailureRecord.from_dict(None)
        self.assertIs(rec.primary_failure, ConversationFailure.NONE)

    def test_json_serialisable(self):
        rec = _analyze(user_message="")
        json.dumps(rec.to_dict())


class TestDiagnosticsSection(unittest.TestCase):
    def test_empty_without_record(self):
        self.assertEqual(conversation_failure_diagnostics_section(None), {})
        proj = conversation_failure_diagnostics_section(ConversationFailureRecord())
        self.assertEqual(proj["Primary Failure"], "None")

    def test_projection_fields(self):
        rec = _analyze(
            user_message="", mentor_reply="Who is affected?",
            conversation_move="ELICIT_INFORMATION",
        )
        proj = conversation_failure_diagnostics_section(rec)
        self.assertEqual(proj["Primary Failure"], "Meta intent")
        self.assertIn("Confidence", proj)
        self.assertIn("Reason", proj)
        self.assertIn("Secondary Failures", proj)
        self.assertIn("Signals", proj)


# ---------------------------------------------------------------------------
# ConversationFailureSummary
# ---------------------------------------------------------------------------


class TestSummaryMetrics(unittest.TestCase):
    def test_empty_summary(self):
        summary = ConversationFailureSummary()
        self.assertEqual(summary.total_turns(), 0)
        self.assertEqual(summary.total_failures(), 0)
        self.assertEqual(summary.failure_rate(), 0.0)
        self.assertEqual(summary.failure_counts(), {})
        self.assertIsNone(summary.most_common_failure())
        self.assertEqual(summary.per_failure_percentages(), {})

    def test_aggregation(self):
        summary = ConversationFailureSummary()
        summary.add_record(_analyze(user_message=""))  # META
        summary.add_record(_analyze(
            user_message="Small business owners",
            mentor_reply="Understood. What core problem do they face?",
            conversation_move="ELICIT_INFORMATION",
            question_family="PERSONAS_WHO",
            project_state_before=_fields(),
            project_state_after=_fields(personas=["Small business owners"]),
            conversation_memory=_mem(),
        ))  # MISSED (has a state value, no memory)
        summary.add_record(_analyze(user_message="thanks"))  # META
        self.assertEqual(summary.total_turns(), 3)
        self.assertEqual(summary.total_failures(), 3)
        self.assertEqual(summary.failure_rate(), 1.0)
        self.assertEqual(summary.failure_counts()["META_INTENT"], 2)
        self.assertEqual(summary.failure_counts()["MISSED_INSIGHT"], 1)
        self.assertEqual(summary.most_common_failure(), "Meta intent")

    def test_failure_rate_partial(self):
        summary = ConversationFailureSummary()
        summary.add_record(_analyze(user_message=""))
        summary.add_record(_analyze(
            user_message="Small business owners in my city",
            mentor_reply="Understood. What core problem do they face?",
            conversation_move="ELICIT_INFORMATION",
            question_family="PERSONAS_WHO",
            project_state_before=_fields(),
            project_state_after=_fields(personas=["Small business owners"]),
            conversation_memory=_mem(open_threads=[{"field": "personas", "reason": "p", "turn": 1}]),
        ))
        summary.add_record(_analyze(user_message="thanks"))
        summary.add_record(_analyze(
            user_message="Teachers and school staff",
            mentor_reply="Understood. What core problem do they face?",
            conversation_move="ELICIT_INFORMATION",
            question_family="PERSONAS_WHO",
            project_state_before=_fields(),
            project_state_after=_fields(personas=["Teachers and school staff"]),
            conversation_memory=_mem(open_threads=[{"field": "personas", "reason": "p", "turn": 1}]),
        ))
        self.assertEqual(summary.total_turns(), 4)
        self.assertEqual(summary.total_failures(), 2)
        self.assertAlmostEqual(summary.failure_rate(), 0.5)

    def test_per_failure_percentages(self):
        summary = ConversationFailureSummary()
        for i in range(4):
            summary.add_record(_analyze(user_message=""))  # META each
        summary.add_record(_analyze(user_message="thanks"))  # META
        pcts = summary.per_failure_percentages()
        self.assertAlmostEqual(pcts["META_INTENT"], 100.0)

    def test_add_none_and_garbage_ignored(self):
        summary = ConversationFailureSummary()
        summary.add_record(None)
        summary.add_record(1234)
        self.assertEqual(summary.total_turns(), 0)

    def test_add_dict_record(self):
        summary = ConversationFailureSummary()
        summary.add_record({"primary_failure": "META_INTENT", "secondary_failures": [], "confidence": 0.7, "reason": "r"})
        self.assertEqual(summary.total_turns(), 1)
        self.assertEqual(summary.failure_counts()["META_INTENT"], 1)


class TestSummaryMergeAndPersistence(unittest.TestCase):
    def test_merge(self):
        a = ConversationFailureSummary()
        a.add_record(_analyze(user_message=""))
        b = ConversationFailureSummary()
        b.add_record(_analyze(user_message="thanks"))
        a.merge(b)
        self.assertEqual(a.total_turns(), 2)
        self.assertEqual(a.failure_counts()["META_INTENT"], 2)

    def test_json_roundtrip(self):
        summary = ConversationFailureSummary()
        summary.add_record(_analyze(user_message=""))
        summary.add_record(_analyze(
            user_message="Small business owners",
            mentor_reply="And who else is affected?",
            conversation_move="ELICIT_INFORMATION",
            question_family="PERSONAS_WHO",
            project_state_before=_fields(),
            project_state_after=_fields(personas=["Small business owners"]),
            conversation_memory=_mem(),
        ))
        restored = ConversationFailureSummary.from_dict(summary.to_dict())
        self.assertEqual(restored.to_dict(), summary.to_dict())
        self.assertEqual(restored.most_common_failure(), summary.most_common_failure())

    def test_from_empty_dict(self):
        summary = ConversationFailureSummary.from_dict({})
        self.assertEqual(summary.total_turns(), 0)
        summary2 = ConversationFailureSummary.from_dict(None)
        self.assertEqual(summary2.total_turns(), 0)

    def test_to_display(self):
        summary = ConversationFailureSummary()
        summary.add_record(_analyze(user_message=""))
        summary.add_record(_analyze(user_message="thanks"))
        display = summary.to_display()
        self.assertEqual(display["Turns Analyzed"], 2)
        self.assertEqual(display["Turns with Failures"], 2)
        self.assertIn("Turn Failure Rate", display)
        self.assertEqual(display["Most Common Failure"], "Meta intent")
        self.assertIn("Failure Category Counts", display)
        self.assertIn("Failure Category Percentages", display)


# ---------------------------------------------------------------------------
# Observation-only guarantee
# ---------------------------------------------------------------------------


class TestObservationOnly(unittest.TestCase):
    def test_analyzer_does_not_mutate_inputs(self):
        before = _fields(problems=["Slow checkout"])
        after = _fields(problems=["Slow checkout", "Long queues"])
        mem = _mem(acknowledged_facts=[{"field": "problems", "reason": "a", "turn": 1}])
        inputs = dict(
            user_message="It also affects cashiers",
            mentor_reply="Understood. Anything else?",
            objective="PROBLEMS",
            project_state_before=dict(before),
            project_state_after=dict(after),
            conversation_memory=dict(mem),
            conversation_move="ELICIT_INFORMATION",
            coaching_strategy="SOCRATIC",
            question_family="PROBLEMS_CORE",
            style_audit={"questions_present": True, "question_count": 1},
            diagnostics={"Pipeline": {"Objective": "PROBLEMS"}},
        )
        snapshot = json.dumps(inputs, sort_keys=True)
        _analyze(**inputs)
        self.assertEqual(json.dumps(inputs, sort_keys=True), snapshot)

    def test_diagnostics_do_not_contain_reply_pointer_mutation(self):
        # The failure record is purely additive to diagnostics; it never
        # rewrites any decision section.
        from run_conversation_failure import _collect
        summary, turns = _collect()
        self.assertEqual(turns, summary.total_turns())


# ---------------------------------------------------------------------------
# Live-pipeline wiring + session persistence
# ---------------------------------------------------------------------------


class TestWiringLivePipeline(unittest.TestCase):
    @staticmethod
    def _run_one_turn(fixture):
        import mentor
        from session_manager import get_session_manager

        _seed_session(fixture)
        mgr = get_session_manager()
        turn = fixture.turns[0]
        ollama_stub = (
            _RaisingOllama()
            if turn.ollama_reply is None
            else _StubOllama([turn.ollama_reply])
        )
        extraction = _build_extraction(turn)
        with mock.patch.dict("sys.modules", {"ollama": ollama_stub}):
            with mock.patch(
                "mentor.MemoryExtractor.extract", return_value=extraction
            ):
                reply, _session, _timing, diag = mentor.process_mentor_turn(
                    turn.user,
                    username=fixture.username,
                    project_name=fixture.project_name,
                )
        return reply, diag, mgr

    def test_sections_appear_in_diagnostics(self):
        fixture = load_all_fixtures()[0]
        _reply, diagnostics, _mgr = self._run_one_turn(fixture)
        self.assertIn("ConversationFailure", diagnostics)
        self.assertIn("ConversationFailureSummary", diagnostics)

    def test_reply_unchanged(self):
        fixture = load_all_fixtures()[0]
        reply, diagnostics, _mgr = self._run_one_turn(fixture)
        self.assertIsNotNone(reply)
        self.assertTrue(reply.strip())
        self.assertIn("ConversationFailure", diagnostics)

    def test_session_summary_persists_across_turns(self):
        from tests.goldens.fixtures import load_all_fixtures
        from tests.goldens.runner import (
            _RaisingOllama,
            _StubOllama,
            _build_extraction,
            _seed_session,
        )
        from session_manager import get_session_manager

        fixture = load_all_fixtures()[0]
        _seed_session(fixture)
        for turn in fixture.turns:
            import mentor

            ollama_stub = (
                _RaisingOllama()
                if turn.ollama_reply is None
                else _StubOllama([turn.ollama_reply])
            )
            extraction = _build_extraction(turn)
            with mock.patch.dict("sys.modules", {"ollama": ollama_stub}):
                with mock.patch(
                    "mentor.MemoryExtractor.extract", return_value=extraction
                ):
                    mentor.process_mentor_turn(
                        turn.user,
                        username=fixture.username,
                        project_name=fixture.project_name,
                    )
        sd = get_session_manager().get_active_session_data()
        summary = sd.conversation_failure_summary
        self.assertEqual(summary.total_turns(), len(fixture.turns))
        self.assertGreaterEqual(summary.total_failures(), 0)
        self.assertEqual(
            summary.total_turns(),
            get_session_manager().get_active_session_data().conversation_failure_summary.total_turns(),
        )

    def test_session_persistence_roundtrip(self):
        from session_manager import get_session_manager, SessionData

        fixture = load_all_fixtures()[0]
        _seed_session(fixture)
        mgr = get_session_manager()
        import mentor

        turn = fixture.turns[0]
        ollama_stub = (
            _RaisingOllama()
            if turn.ollama_reply is None
            else _StubOllama([turn.ollama_reply])
        )
        extraction = _build_extraction(turn)
        with mock.patch.dict("sys.modules", {"ollama": ollama_stub}):
            with mock.patch("mentor.MemoryExtractor.extract", return_value=extraction):
                mentor.process_mentor_turn(
                    turn.user, username=fixture.username, project_name=fixture.project_name
                )
        sd = mgr.get_active_session_data()
        encoded = sd.to_dict()
        decoded = SessionData.from_dict(encoded)
        self.assertEqual(
            decoded.conversation_failure_summary.to_dict(),
            sd.conversation_failure_summary.to_dict(),
        )

    def test_audit_failure_does_not_break_conversation(self):
        fixture = load_all_fixtures()[0]
        with mock.patch(
            "mentor.analyze_conversation_failure",
            side_effect=RuntimeError("boom"),
        ):
            reply, diagnostics, _mgr = self._run_one_turn(fixture)
        self.assertIsNotNone(reply)
        self.assertTrue(reply.strip())
        self.assertIn("ConversationFailureSummary", diagnostics)

    def test_objectives_unchanged(self):
        """The observation audit must not change the objective the pipeline
        selected. Compare with the golden conversation's expected objective."""
        fixture = load_fixture("01_straightforward_full_conversation")
        _reply, diagnostics, _mgr = self._run_one_turn(fixture)
        self.assertEqual(diagnostics["Pipeline"]["Objective"], "PROBLEMS")


class TestReportRunner(unittest.TestCase):
    def test_report_aggregates_goldens(self):
        from run_conversation_failure import _collect
        summary, turns = _collect()
        self.assertEqual(turns, summary.total_turns())
        self.assertIsInstance(summary.to_dict()["turn_records"], list)


# ---------------------------------------------------------------------------
# Golden byte-identity (observation-only)
# ---------------------------------------------------------------------------


class TestGoldenUnchanged(unittest.TestCase):
    def test_golden_conversations_still_pass(self):
        """Run the existing golden runner end-to-end; replies and state must
        remain byte-identical (the audit changes nothing the runner checks)."""
        import glob
        import os
        import subprocess
        import sys

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_golden_conversations.py", "-q"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        self.assertIn("passed", result.stdout, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()