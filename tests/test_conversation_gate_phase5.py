"""
tests/test_conversation_gate_phase5.py — Phase 5 fixes for the Phase 4
findings.

Covers:
  P1  Hypothetical substantive-content guard: framing pauses steering,
      but extraction is suppressed ONLY when no DT content survives
      pivot-language stripping.
  P2  Person-scoped DONT_KNOW: third-person "don't know" statements are
      project facts (NORMAL_DT), first-person stays CONFUSED.
  P3  New meta-move frames (correction / topic shift / confusion),
      including context-dependent "Can you explain that?".
  P4  Pivot-language ingestion guard: paused TOPIC_SHIFT/CORRECTION turns
      can no longer turn "forget ..." into a ProjectState pain point.

Every pipeline test drives the real ``mentor.process_mentor_turn`` with a
stubbed ollama + mocked LLM extractor (deterministic, no network).
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation_context import (  # noqa: E402
    classify_conversation_context,
    strip_pivot_language,
)
from extraction_pipeline import RuleBasedExtractor  # noqa: E402
from hybrid_extraction import rule_update_dicts  # noqa: E402

_USER = "phase5_user"
_PROJECT = "Phase5Project"


def _reset_session():
    from session_manager import get_session_manager

    mgr = get_session_manager()
    mgr.reset_runtime_state(username=_USER, project_title=_PROJECT)
    return mgr.get_active_session_data()


def _active_session():
    from session_manager import get_session_manager

    return get_session_manager().get_active_session_data()


def _run_turn(user_message, ollama_reply=None):
    import mentor
    from memory_extractor import ExtractionResult, MessageType
    from tests.goldens.runner import _RaisingOllama, _StubOllama

    stub = (
        _RaisingOllama()
        if ollama_reply is None
        else _StubOllama([ollama_reply])
    )
    with mock.patch.dict("sys.modules", {"ollama": stub}):
        with mock.patch(
            "mentor.MemoryExtractor.extract",
            return_value=ExtractionResult(MessageType.AMBIGUOUS, []),
        ):
            reply, _session, _timing, diagnostics = mentor.process_mentor_turn(
                user_message, username=_USER, project_name=_PROJECT,
            )
    return reply, diagnostics, _active_session()


def _probe(msg):
    return rule_update_dicts(RuleBasedExtractor.extract(strip_pivot_language(msg)))


# ---------------------------------------------------------------------------
# P4 unit — pivot stripper
# ---------------------------------------------------------------------------


class TestStripPivotLanguage(unittest.TestCase):
    def test_pivot_heads_removed(self):
        self.assertEqual(
            _probe("Forget that, let's talk about civic sense instead."), [])
        self.assertEqual(_probe("Scratch that."), [])
        self.assertEqual(_probe("Never mind the old project."), [])

    def test_subject_led_facts_survive(self):
        self.assertNotEqual(_probe("I forget things daily"), [])
        cleaned = strip_pivot_language("Forget the coding thing. I care because deadlines slip weekly")
        self.assertIn("because", cleaned.lower())
        self.assertNotEqual(_probe(cleaned), [])

    def test_pure_function(self):
        msg = "Never mind that, let's switch topics."
        self.assertEqual(strip_pivot_language(msg), strip_pivot_language(msg))


# ---------------------------------------------------------------------------
# P1 — hypothetical substantive-content guard
# ---------------------------------------------------------------------------


class TestHypotheticalGuard(unittest.TestCase):
    def test_pure_frame_suppresses(self):
        sd = _reset_session()
        before = sd.project_state.to_state_dict()
        reply, diag, sd2 = _run_turn("Suppose I were a dog.", None)
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "HYPOTHETICAL")
        self.assertTrue(cc["DT Steering Paused"])
        self.assertTrue(cc["Extraction Suppressed"])
        self.assertEqual(reply, "Happy to explore that scenario with you. "
                         "Tell me more about the situation you have in mind.")
        self.assertEqual(sd2.project_state.to_state_dict(), before)

    def test_substantive_hypothetical_preserves_content(self):
        sd = _reset_session()
        reply, diag, sd2 = _run_turn(
            "Imagine badges as rewards for students.", None)
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "HYPOTHETICAL")
        self.assertTrue(cc["DT Steering Paused"])
        self.assertFalse(cc["Extraction Suppressed"])
        # The stakeholder noun survived the guard as a candidate fact.
        self.assertEqual(sd2.project_state.personas, ["college students"])

    def test_frequency_inside_hypothetical_not_lost(self):
        sd = _reset_session()
        _reply, diag, sd2 = _run_turn(
            "Suppose teachers review notes nightly.", None)
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "HYPOTHETICAL")
        self.assertTrue(cc["DT Steering Paused"])
        self.assertFalse(cc["Extraction Suppressed"])
        self.assertEqual(sd2.project_state.frequency, "nightly")


# ---------------------------------------------------------------------------
# P2 — person-scoped DONT_KNOW
# ---------------------------------------------------------------------------


class TestPersonScopedDontKnow(unittest.TestCase):
    def test_first_person_stays_confused(self):
        for msg in ["I don't understand.", "I don't know.",
                    "I can't figure this out.", "I don't get what you mean."]:
            ctx = classify_conversation_context(user_message=msg)
            self.assertEqual(ctx.mode.value, "CONFUSED", msg)

    def test_third_person_is_a_project_fact(self):
        for msg in ["The problem is that people don't know where to start.",
                    "Users don't know where to start.",
                    "Students don't know what to do."]:
            ctx = classify_conversation_context(user_message=msg)
            self.assertEqual(ctx.mode.value, "NORMAL_DT", msg)

    def test_self_implicit_idioms_stay_confused(self):
        ctx = classify_conversation_context(user_message="No idea.")
        self.assertEqual(ctx.mode.value, "CONFUSED")

    def test_pipeline_control_answer_now_normal(self):
        _reset_session()
        reply, diag, sd = _run_turn(
            "The problem is that people don't know where to start.",
            "What have they already tried to fix it?",
        )
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "NORMAL_DT")
        self.assertFalse(cc.get("DT Steering Paused", False))
        self.assertIn("A fresh question angle such as", diag["Prompt"])
        self.assertEqual(reply, "What have they already tried to fix it?")


# ---------------------------------------------------------------------------
# P3 — new meta-move frames
# ---------------------------------------------------------------------------


class TestNewCorrectionFrames(unittest.TestCase):
    def test_unit_table_high(self):
        cases = [
            "Actually, that's not what I'm talking about.",
            "No, I mean my own situation.",
            "That's not what I meant.",
        ]
        for msg in cases:
            ctx = classify_conversation_context(user_message=msg)
            self.assertEqual(ctx.mode.value, "CORRECTION", msg)
            self.assertEqual(ctx.confidence.value, "HIGH", msg)

    def test_plain_answer_with_no_i_mean_stays_normal(self):
        ctx = classify_conversation_context(
            user_message="No, I mean it happens weekly.")
        self.assertEqual(ctx.mode.value, "NORMAL_DT")

    def test_pipeline_pause_and_hygiene(self):
        sd = _reset_session()
        before_families = list(sd.asked_question_families)
        before_state = sd.project_state.to_state_dict()
        reply, diag, sd2 = _run_turn(
            "Actually, that's not what I'm talking about.",
            "Apologies - tell me what you'd like to focus on instead.",
        )
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "CORRECTION")
        self.assertEqual(cc["Confidence"], "HIGH")
        self.assertTrue(cc["DT Steering Paused"])
        self.assertEqual(list(sd2.asked_question_families), before_families)
        self.assertEqual(sd2.project_state.to_state_dict(), before_state)
        self.assertNotIn("Ask a NEW question angle", diag["Prompt"])


class TestNewTopicShiftFrames(unittest.TestCase):
    def test_unit_table(self):
        cases = [
            "I've changed the project completely.",
            "Let's switch topics.",
            "I want to talk about something else.",
            "Forget that, let's talk about civic sense instead.",
        ]
        for msg in cases:
            ctx = classify_conversation_context(user_message=msg)
            self.assertEqual(ctx.mode.value, "TOPIC_SHIFT", msg)
            self.assertEqual(ctx.confidence.value, "HIGH", msg)

    def test_forget_pivot_never_becomes_a_fact(self):
        sd = _reset_session()
        before = sd.project_state.to_state_dict()
        reply, diag, sd2 = _run_turn(
            "Forget that, let's talk about civic sense instead.", None)
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "TOPIC_SHIFT")
        self.assertTrue(cc["DT Steering Paused"])
        self.assertTrue(cc["Extraction Suppressed"])
        # The extractor's 'forget' fallback must not ingest the pivot.
        joined = json_safe = str(sd2.project_state.to_state_dict()).lower()
        self.assertNotIn("forget", joined)
        self.assertEqual(sd2.project_state.to_state_dict(), before)
        self.assertEqual(
            reply,
            "Understood - let's switch to the new direction. Tell me about "
            "what you're working on now.")

    def test_pivot_plus_real_fact_keeps_the_fact(self):
        sd = _reset_session()
        reply, diag, sd2 = _run_turn(
            "Forget the coding thing. I care because deadlines slip weekly",
            None)
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "TOPIC_SHIFT")
        self.assertTrue(cc["DT Steering Paused"])
        # Substantive clause survived stripping -> extraction preserved.
        self.assertFalse(cc["Extraction Suppressed"])
        self.assertTrue(any("because" in p for p in sd2.project_state.pain_points))


class TestNewConfusionFrames(unittest.TestCase):
    def test_get_what_you_mean_high_without_context(self):
        ctx = classify_conversation_context(user_message="I don't get what you mean.")
        self.assertEqual((ctx.mode.value, ctx.confidence.value),
                         ("CONFUSED", "HIGH"))

    def test_explain_request_is_context_dependent(self):
        cold = classify_conversation_context(user_message="Can you explain that?")
        self.assertEqual((cold.mode.value, cold.confidence.value),
                         ("CONFUSED", "MEDIUM"))
        warm = classify_conversation_context(
            user_message="Can you explain that?",
            previous_assistant_message=(
                "To focus the design on the right people, who specifically "
                "would benefit from this?"),
        )
        self.assertEqual((warm.mode.value, warm.confidence.value),
                         ("CONFUSED", "HIGH"))
        self.assertTrue(warm.pause_dt)

    def test_explain_request_does_not_fire_on_statements(self):
        ctx = classify_conversation_context(
            user_message="This explains the concept well.")
        self.assertEqual(ctx.mode.value, "NORMAL_DT")

    def test_pipeline_clarification_allowed_without_question(self):
        _reset_session()
        # Seed a real mentor question so the explanation request is
        # comprehension-directed (HIGH) rather than a cold topic request.
        _run_turn(
            "I want to help students manage assignment deadlines", None)
        reply, diag, _sd = _run_turn(
            "Can you explain that?",
            "Of course - I'll rephrase it more simply.",
        )
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "CONFUSED")
        self.assertEqual(cc["Confidence"], "HIGH")
        self.assertTrue(cc["DT Steering Paused"])
        self.assertEqual(reply, "Of course - I'll rephrase it more simply.")


# ---------------------------------------------------------------------------
# NORMAL_DT control smoke
# ---------------------------------------------------------------------------


class TestNormalControlSmoke(unittest.TestCase):
    def test_ordinary_answer_untouched(self):
        _reset_session()
        stub_q = "Since this matters, how often does it happen?"
        reply, diag, sd = _run_turn(
            "It happens almost every day.", stub_q)
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "NORMAL_DT")
        self.assertFalse(cc.get("DT Steering Paused", False))
        self.assertNotIn("Extraction Suppressed", cc)
        self.assertIn("A fresh question angle such as", diag["Prompt"])
        self.assertEqual(reply, stub_q)
        self.assertTrue(sd.asked_question_families)


if __name__ == "__main__":
    unittest.main(verbosity=2)
