"""
tests/test_conversation_style.py — regression tests for the measurement-only
Conversation Style Audit (``conversation_style_audit`` + its wiring in
``mentor.py`` / ``session_manager.py``).

Covers:

  * ``analyze_reply`` — deterministic per-turn classification of reply text
    (opening, length, acknowledgment, observation, bridge, question count,
    multiple unrelated questions, summary markers, references to prior
    info, topic restart).
  * ``ConversationStyleSummary`` — aggregation, all rates, opening
    diversity / most-common-openings, JSON round-trip, merge.
  * ``conversation_style_diagnostics_section`` — Developer Console
    projection.
  * Live-pipeline wiring — the ``ConversationStyle`` and
    ``ConversationStyleSummary`` sections appear in diagnostics and the
    session aggregate persists across turns.
  * Observation-only guarantee — adding the audit never changes the reply.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import conversation_style_audit as audit  # noqa: E402
from conversation_style_audit import (  # noqa: E402
    ConversationStyleSummary,
    analyze_reply,
    conversation_style_diagnostics_section,
)


def _analyze(text, **kwargs):
    kwargs.setdefault("turn_index", 1)
    kwargs.setdefault("reply", text)
    return analyze_reply(**kwargs)


class _PromptStyleOllama:
    """Prompt-reflective ``ollama`` stand-in used to prove the audit measures
    the instruction-driven conversational style.

    Emulates a model that follows the dynamic prompt's acknowledgment
    freedom: when the rendered prompt contains the allowed-moves block (the
    Phase 1 dynamic path) the reply opens with an acknowledgment/observation
    and gives a reason before the ONE question; otherwise it leads directly
    with the question. The QUESTION text is identical in both modes, so the
    comparison isolates exactly the opening behaviour the instructions steer.
    """

    def __init__(self, question):
        self.question = question
        self.last_prompt = ""

    def chat(self, **kwargs):
        prompt = "".join(
            (m.get("content") or "") for m in (kwargs.get("messages") or [])
        )
        self.last_prompt = prompt
        if "ALLOWED CONVERSATIONAL MOVES" in prompt:
            content = (
                "That's a helpful detail. "
                "You mentioned this affects people directly, "
                "so it's worth understanding how often it happens. "
                + self.question
            )
        else:
            content = self.question
        return {"message": {"content": content}}


class TestAnalyzeReplyLength(unittest.TestCase):
    def test_word_and_sentence_counts(self):
        r = _analyze("Thanks! That sounds helpful. How often does it happen?")
        self.assertEqual(r["word_count"], 9)
        self.assertEqual(r["sentence_count"], 3)

    def test_empty_reply(self):
        r = _analyze("")
        self.assertEqual(r["word_count"], 0)
        self.assertEqual(r["sentence_count"], 0)
        self.assertEqual(r["opening"], "(empty)")

    def test_opening_is_first_three_words(self):
        r = _analyze("A very long opening phrase here")
        self.assertEqual(r["opening"], "A very long")

    def test_opening_short_reply(self):
        r = _analyze("Hi")
        self.assertEqual(r["opening"], "Hi")


class TestAnalyzeOpening(unittest.TestCase):
    def test_begins_with_acknowledgment(self):
        for text in (
            "Thanks for sharing that.",
            "Thank you for explaining.",
            "I see, that makes sense.",
            "Great, let's dig in.",
            "Got it. Who benefits most?",
        ):
            self.assertTrue(
                _analyze(text)["begins_with_acknowledgment"], text
            )

    def test_not_acknowledgment(self):
        r = _analyze("Who specifically would benefit from this?")
        self.assertFalse(r["begins_with_acknowledgment"])
        self.assertTrue(r["immediately_asks_question"])

    def test_immediately_asks_question_detects_question_first(self):
        r = _analyze("How often does this happen to your team?")
        self.assertTrue(r["immediately_asks_question"])

    def test_not_immediate_question_when_ack_preamble(self):
        r = _analyze("Thanks for sharing. How often does this happen?")
        self.assertFalse(r["immediately_asks_question"])


class TestAnalyzeObservationAndSummary(unittest.TestCase):
    def test_observation_markers(self):
        for text in (
            "It sounds like a real pain point.",
            "That suggests the problem is widespread.",
            "You're describing a real constraint.",
            "I notice you care about speed.",
        ):
            self.assertTrue(_analyze(text)["has_observation"], text)

    def test_no_observation(self):
        r = _analyze("What is the core problem?")
        self.assertFalse(r["has_observation"])

    def test_summary_markers(self):
        for text in (
            "Here is a summary of what we know.",
            "Let's recap what we've covered so far.",
            "So far, I understand the audience well.",
            "Here's a summary of the pain points.",
        ):
            self.assertTrue(_analyze(text)["has_summary"], text)

    def test_no_summary(self):
        r = _analyze("How often does this happen?")
        self.assertFalse(r["has_summary"])


class TestAnalyzeQuestions(unittest.TestCase):
    def test_question_count(self):
        self.assertEqual(_analyze("Who? What? Where?")["question_count"], 3)
        self.assertEqual(_analyze("No questions here.")["question_count"], 0)

    def test_multiple_unrelated_questions(self):
        r = _analyze("Who would benefit? What about your budget?")
        self.assertTrue(r["multiple_unrelated_questions"])

    def test_multiple_related_questions(self):
        r = _analyze("How often does it happen? How often per week?")
        self.assertFalse(r["multiple_unrelated_questions"])

    def test_single_question_not_multiple(self):
        r = _analyze("Who would benefit?")
        self.assertFalse(r["multiple_unrelated_questions"])

    def test_zero_questions_not_multiple(self):
        r = _analyze("Just a statement.")
        self.assertFalse(r["multiple_unrelated_questions"])


class TestAnalyzePriorInfo(unittest.TestCase):
    def test_references_known_fact(self):
        r = _analyze(
            "You mentioned elderly people earlier. How often do they struggle?",
            known_facts=["elderly people are the target audience"],
        )
        self.assertTrue(r["references_prior_info"])

    def test_no_known_facts_means_no_reference(self):
        r = _analyze("How often does this happen?", known_facts=[])
        self.assertFalse(r["references_prior_info"])

    def test_bridge_via_prior_reply_overlap(self):
        r = _analyze(
            "How often does the pain point recur?",
            previous_reply="Let's understand the pain point better.",
        )
        self.assertTrue(r["has_bridge"])

    def test_bridge_via_explicit_marker(self):
        r = _analyze("As you mentioned, tell me more about the frequency.")
        self.assertTrue(r["has_bridge"])

    def test_no_bridge(self):
        r = _analyze(
            "What is the core problem?",
            previous_reply="",
            known_facts=[],
        )
        self.assertFalse(r["has_bridge"])

    def test_restarts_topic_asks_known_fact(self):
        r = _analyze(
            "Who specifically are the elderly people?",
            known_facts=["elderly people are the target audience"],
        )
        self.assertTrue(r["restarts_topic"])

    def test_not_restart_when_unknown(self):
        r = _analyze(
            "How much does it cost them?",
            known_facts=["elderly people are the target audience"],
        )
        self.assertFalse(r["restarts_topic"])


class TestDiagnosticsSection(unittest.TestCase):
    def test_empty_without_record(self):
        self.assertEqual(conversation_style_diagnostics_section(None), {})
        self.assertEqual(conversation_style_diagnostics_section({}), {})

    def test_projection_fields(self):
        r = analyze_reply(
            turn_index=3,
            reply="Thanks for sharing. It sounds painful. How often?",
        )
        section = conversation_style_diagnostics_section(r)
        self.assertEqual(section["Opening"], "Thanks for sharing")
        self.assertEqual(section["Reply Length"], "8 words / 3 sentences")
        self.assertEqual(section["Begins with Acknowledgment"], "Yes")
        self.assertEqual(section["Observation/Reflection"], "Yes")
        self.assertEqual(section["Immediately Asks Question"], "No")
        self.assertEqual(section["Questions"], 1)
        self.assertEqual(section["Contains Summary"], "No")

    def test_section_is_json_serialisable(self):
        import json
        r = analyze_reply(turn_index=1, reply="Who benefits?")
        json.dumps(conversation_style_diagnostics_section(r))


class TestSummaryAggregation(unittest.TestCase):
    def _records(self):
        return [
            analyze_reply(turn_index=1, reply="Thanks for sharing that."),
            analyze_reply(turn_index=2, reply="How often does it happen?"),
            analyze_reply(turn_index=3, reply="So far I see the pattern clearly."),
            analyze_reply(turn_index=4, reply="Thanks for sharing more."),
        ]

    def test_turn_count_and_length(self):
        s = ConversationStyleSummary()
        for r in self._records():
            s.add_record(r)
        self.assertEqual(s.turn_count, 4)
        self.assertGreater(s.average_reply_length(), 0.0)

    def test_most_common_openings(self):
        s = ConversationStyleSummary()
        for r in self._records():
            s.add_record(r)
        top = s.most_common_openings(limit=1)
        self.assertEqual(top[0][0], "Thanks for sharing")
        self.assertEqual(top[0][1], 2)

    def test_opening_diversity(self):
        s = ConversationStyleSummary()
        s.add_record(analyze_reply(reply="Same opening every time."))
        s.add_record(analyze_reply(reply="Same opening every day."))
        self.assertEqual(s.opening_diversity_score(), 0.0)
        s.add_record(analyze_reply(reply="A brand new opener."))
        self.assertGreater(s.opening_diversity_score(), 0.0)

    def test_rates(self):
        s = ConversationStyleSummary()
        s.add_record(analyze_reply(reply="Thanks! Who benefits?"))  # ack + q
        s.add_record(analyze_reply(reply="Here's a summary."))      # summary
        s.add_record(analyze_reply(reply="Who? What?"))             # multi q
        self.assertEqual(s.turn_count, 3)
        self.assertAlmostEqual(s.summary_rate(), 1 / 3, places=3)
        self.assertAlmostEqual(s.questionnaire_rate(), 1 / 3, places=3)
        self.assertAlmostEqual(s.multiple_question_rate(), 1 / 3, places=3)
        self.assertAlmostEqual(s.acknowledgment_rate(), 1 / 3, places=3)

    def test_zero_rates_for_empty_summary(self):
        s = ConversationStyleSummary()
        self.assertEqual(s.reflection_rate(), 0.0)
        self.assertEqual(s.summary_rate(), 0.0)
        self.assertEqual(s.bridge_rate(), 0.0)
        self.assertEqual(s.questionnaire_rate(), 0.0)
        self.assertEqual(s.topic_restart_rate(), 0.0)
        self.assertEqual(s.multiple_question_rate(), 0.0)
        self.assertEqual(s.average_reply_length(), 0.0)
        self.assertEqual(s.opening_diversity_score(), 0.0)

    def test_json_roundtrip(self):
        s = ConversationStyleSummary()
        for r in self._records():
            s.add_record(r)
        restored = ConversationStyleSummary.from_dict(s.to_dict())
        self.assertEqual(restored.to_dict(), s.to_dict())

    def test_merge(self):
        a = ConversationStyleSummary()
        b = ConversationStyleSummary()
        a.add_record(analyze_reply(reply="Thanks! Who?"))
        b.add_record(analyze_reply(reply="What is the problem?"))
        a.merge(b)
        self.assertEqual(a.turn_count, 2)
        self.assertEqual(a.opening_counts.get("Thanks Who"), 1)
        self.assertEqual(a.opening_counts.get("What is the"), 1)


class TestWiringLivePipeline(unittest.TestCase):
    def _run_one_turn(self, fixture):
        import mentor
        from tests.goldens.runner import (
            _RaisingOllama,
            _StubOllama,
            _build_extraction,
            _seed_session,
        )
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
        from tests.goldens.fixtures import load_all_fixtures
        fixture = load_all_fixtures()[0]
        reply, diag, mgr = self._run_one_turn(fixture)
        self.assertIn("ConversationStyle", diag)
        self.assertIn("ConversationStyleSummary", diag)
        self.assertIsNotNone(diag["ConversationStyle"])
        self.assertEqual(diag["ConversationStyleSummary"]["Turns Analyzed"], 1)

    def test_reply_unchanged_by_audit(self):
        """The audit must never alter the generated reply."""
        from tests.goldens.fixtures import load_all_fixtures
        fixture = load_all_fixtures()[0]
        reply, diag, _ = self._run_one_turn(fixture)
        self.assertTrue(reply)  # a real reply is still generated
        self.assertIsNotNone(diag["ConversationStyle"])

    def test_session_summary_persists_across_turns(self):
        from tests.goldens.fixtures import load_all_fixtures
        from session_manager import get_session_manager, SessionData
        fixture = load_all_fixtures()[0]
        import mentor
        from tests.goldens.runner import (
            _RaisingOllama,
            _StubOllama,
            _build_extraction,
            _seed_session,
        )

        _seed_session(fixture)
        mgr = get_session_manager()
        for turn in fixture.turns:
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
        sd = mgr.get_active_session_data()
        self.assertEqual(
            sd.conversation_style_summary.turn_count, len(fixture.turns)
        )

    def test_session_persistence_roundtrip(self):
        """SessionData.to_dict/from_dict preserves the aggregate."""
        from session_manager import SessionData
        sd = SessionData()
        sd.conversation_style_summary.add_record(
            analyze_reply(reply="Thanks! Who benefits?")
        )
        restored = SessionData.from_dict(sd.to_dict())
        self.assertEqual(
            restored.conversation_style_summary.to_dict(),
            sd.conversation_style_summary.to_dict(),
        )


# The pre-conversational-flow ASK_QUESTION instruction block. Counterfactual
# baseline: with these bullets (no observation-first directive) a question-led
# reply is the expected style.
LEGACY_ASK_QUESTION_INSTRUCTIONS = (
    "Acknowledge what the user just shared in one short sentence.",
    "Ask ONE natural follow-up question that builds on their most recent answer.",
    "Do not restart the topic - stay focused on the current objective.",
    "Do not ask for information already in the Known Project State or the Latest Conversation.",
    "Do not ask the same kind of question twice - avoid semantically identical questions.",
    "Do not summarize.",
    "Do not ask multiple questions.",
    "Do not invent information.",
    "Do not repeat previous questions.",
    "Keep the response concise and mentor-like - no filler, no excess words.",
    "Celebrate genuinely useful discoveries briefly - one short warm remark (for example, 'That is really helpful to know.') - then continue.",
    "Vary your sentence openings - never open two consecutive replies the same way.",
    "Avoid unnecessary transitions and filler such as 'Great question', 'Interesting', or 'Let me ask you'.",
    "Connect the new answer to something they already told you - reference a shared detail so it feels like one flowing conversation, not a questionnaire.",
    "Every few turns, briefly recap what you understand so far in one or two lines before asking the next question.",
    "Acknowledge any open threads, deferred topics, or previously resolved topics the user mentioned but you skipped earlier - gently reconnect them to the current question context.",
    "If the user shared new information that naturally advances the current objective, reference it specifically in your follow-up question to show direct engagement.",
)


class TestConversationalFlowInstructions(unittest.TestCase):
    """The dynamic prompt (Brief + Relevant Context + allowed moves) is the
    prompt lever for the conversational-flow goal. These tests prove:

      * the audit measures the difference the dynamic prompt makes
        (questionnaire rate drops, acknowledgment rate rises), and
      * the change is reply-style only — every decision (objective, stage,
        question family, checklist) is preserved identically.

    NOTE (Phase 1 intentional change): the live ``process_mentor_turn``
    path no longer consumes ``prompt_builder._INSTRUCTIONS`` — it renders
    the slim dynamic prompt instead. Patching ``_INSTRUCTIONS`` therefore
    no longer changes live replies; the legacy ``build_prompt`` (and its
    instruction wall) is preserved byte-identical for compatibility and
    direct unit tests. The old-vs-new-instructions comparison now lives
    at the prompt level: the dynamic prompt carries acknowledgment freedom
    ("ALLOWED CONVERSATIONAL MOVES" block) while a bare-question reply
    carries none.
    """

    QUESTION = "How often does the problem occur for you?"

    def _run_turn(self, stub):
        import mentor
        from tests.goldens.fixtures import load_all_fixtures
        from tests.goldens.runner import _build_extraction, _seed_session

        fixture = load_all_fixtures()[0]
        _seed_session(fixture)
        turn = fixture.turns[0]
        extraction = _build_extraction(turn)

        with mock.patch.dict("sys.modules", {"ollama": stub}):
            with mock.patch(
                "mentor.MemoryExtractor.extract", return_value=extraction
            ):
                reply, _session, _timing, diag = mentor.process_mentor_turn(
                    turn.user,
                    username=fixture.username,
                    project_name=fixture.project_name,
                )
        return reply, diag

    def _run_fixture(self, stub):
        import mentor
        from session_manager import get_session_manager
        from tests.goldens.fixtures import load_all_fixtures
        from tests.goldens.runner import _build_extraction, _seed_session

        fixture = load_all_fixtures()[0]
        _seed_session(fixture)

        for turn in fixture.turns:
            extraction = _build_extraction(turn)
            with mock.patch.dict("sys.modules", {"ollama": stub}):
                with mock.patch(
                    "mentor.MemoryExtractor.extract", return_value=extraction
                ):
                    mentor.process_mentor_turn(
                        turn.user,
                        username=fixture.username,
                        project_name=fixture.project_name,
                    )
        return get_session_manager().get_active_session_data().conversation_style_summary

    def test_dynamic_prompt_open_observation_first(self):
        """With the dynamic prompt's allowed-moves block the reply is
        acknowledged / observation-led; a style-blind reply leads with the
        question."""
        dyn_stub = _PromptStyleOllama(self.QUESTION)
        new_reply, _ = self._run_turn(dyn_stub)
        self.assertIn("ALLOWED CONVERSATIONAL MOVES", dyn_stub.last_prompt)

        class _BareOllama:
            def chat(self, **kwargs):
                return {"message": {"content": TestConversationalFlowInstructions.QUESTION}}

        old_reply, _ = self._run_turn(_BareOllama())

        new_record = analyze_reply(reply=new_reply)
        old_record = analyze_reply(reply=old_reply)

        self.assertTrue(new_record["begins_with_acknowledgment"], new_reply)
        self.assertTrue(new_record["has_observation"], new_reply)
        self.assertTrue(new_record["has_bridge"], new_reply)
        self.assertFalse(new_record["immediately_asks_question"], new_reply)
        self.assertEqual(new_record["question_count"], 1)

        self.assertFalse(old_record["begins_with_acknowledgment"], old_reply)
        self.assertTrue(old_record["immediately_asks_question"], old_reply)
        self.assertEqual(old_record["question_count"], 1)

    def test_same_decisions_across_reply_styles(self):
        """The style change must not move any decision the pipeline makes."""

        class _BareOllama:
            def chat(self, **kwargs):
                return {"message": {"content": TestConversationalFlowInstructions.QUESTION}}

        new_reply, new_diag = self._run_turn(_PromptStyleOllama(self.QUESTION))
        old_reply, old_diag = self._run_turn(_BareOllama())
        for key in ("Objective", "Stage"):
            self.assertEqual(
                new_diag["Pipeline"][key], old_diag["Pipeline"][key], key
            )
        self.assertEqual(
            new_diag["QuestionFamilies"]["Question Family"],
            old_diag["QuestionFamilies"]["Question Family"],
        )
        self.assertEqual(
            new_diag["ChecklistReasoning"], old_diag["ChecklistReasoning"]
        )
        self.assertNotEqual(new_reply, old_reply)

    def test_aggregate_style_improves_with_dynamic_prompt(self):
        """Across a whole fixture the dynamic prompt reduces questionnaire
        openings and raises acknowledgments, with the same number of turns."""
        new_summary = self._run_fixture(_PromptStyleOllama(self.QUESTION))

        class _BareOllama:
            def chat(self, **kwargs):
                return {"message": {"content": TestConversationalFlowInstructions.QUESTION}}

        old_summary = self._run_fixture(_BareOllama())
        self.assertEqual(new_summary.turn_count, old_summary.turn_count)
        self.assertLess(
            new_summary.questionnaire_count, old_summary.questionnaire_count
        )
        self.assertGreater(
            new_summary.acknowledgment_count, old_summary.acknowledgment_count
        )
        self.assertGreater(new_summary.acknowledgment_count, 0)


if __name__ == "__main__":
    unittest.main()