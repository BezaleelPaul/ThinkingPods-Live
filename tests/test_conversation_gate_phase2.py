"""
tests/test_conversation_gate_phase2.py — Phase 2 activation tests for the
Conversational Context Gate.

Phase 2 contract under test:
  * NORMAL_DT turns are byte-for-byte behaviorally identical to pre-Phase 2
    (objective + family guidance + enforcement + family recording).
  * Activated pause modes (CORRECTION / TOPIC_SHIFT / DIRECT_QUESTION /
    CONFUSED) at HIGH confidence suppress DT question steering for THAT
    turn only: no family guidance, statement replies permitted, advice
    protection intact, asked_question_families unpolluted, ProjectState and
    the canonical Objective Engine untouched.
  * HYPOTHETICAL / AMBIGUOUS / UNSAFE remain observation-only.

Pipeline is driven end-to-end through mentor.process_mentor_turn with a
stubbed ollama module and a mocked MemoryExtractor (goldens-runner style),
so every assertion runs against the real production pipeline.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation_context import (  # noqa: E402
    ContextConfidence,
    ContextMode,
    classify_conversation_context,
    context_instruction_bullet,
    is_dt_paused,
    paused_turn_fallback,
)
from conversation_pipeline import enforce_mentor_reply  # noqa: E402

_USER = "phase2_user"
_PROJECT = "Phase2Project"


def _reset_session():
    from session_manager import get_session_manager

    mgr = get_session_manager()
    mgr.reset_runtime_state(username=_USER, project_title=_PROJECT)
    return mgr.get_active_session_data()


def _run_turn(user_message, ollama_reply=None):
    """One full pipeline turn; returns (reply, diagnostics, session_data)."""
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
    sd = _active_session()
    return reply, diagnostics, sd


def _active_session():
    from session_manager import get_session_manager

    return get_session_manager().get_active_session_data()


# Exact guidance texts mandated by the Phase 2 spec (bullet adds a prefix).
_SPEC_GUIDANCE = {
    ContextMode.CORRECTION: (
        "The user appears to be correcting your previous interpretation. "
        "Address the correction first and do not continue the previous "
        "Design Thinking question until their meaning is clear."
    ),
    ContextMode.TOPIC_SHIFT: (
        "The user appears to be intentionally changing topics. Follow the "
        "new topic naturally rather than forcing the previous Design "
        "Thinking objective into this reply."
    ),
    ContextMode.DIRECT_QUESTION: (
        "The user is directly asking for help. Acknowledge the request, "
        "but stay within the mentor role and first understand their "
        "situation rather than immediately prescribing a solution."
    ),
    ContextMode.CONFUSED: (
        "The user appears confused or uncertain. Clarify what they are "
        "asking about in simple language rather than pushing the planned "
        "Design Thinking question."
    ),
}

_ADVICE_MARKERS = (
    "you should", "you could", "i suggest", "i recommend",
    "try ", "build ", "implement", "add a feature", "the solution is",
)


# ---------------------------------------------------------------------------
# Unit tests — activation surface
# ---------------------------------------------------------------------------


class TestPauseDecision(unittest.TestCase):
    def test_only_activated_high_modes_pause(self):
        expectations = [
            (ContextMode.CORRECTION, ContextConfidence.HIGH, True),
            (ContextMode.TOPIC_SHIFT, ContextConfidence.HIGH, True),
            (ContextMode.DIRECT_QUESTION, ContextConfidence.HIGH, True),
            (ContextMode.CONFUSED, ContextConfidence.HIGH, True),
            (ContextMode.CORRECTION, ContextConfidence.MEDIUM, False),
            (ContextMode.CONFUSED, ContextConfidence.MEDIUM, False),
            # Phase 3: HYPOTHETICAL is now an activated pause mode.
            (ContextMode.HYPOTHETICAL, ContextConfidence.HIGH, True),
            (ContextMode.UNSAFE, ContextConfidence.HIGH, False),
            (ContextMode.AMBIGUOUS, ContextConfidence.LOW, False),
            (ContextMode.NORMAL_DT, ContextConfidence.HIGH, False),
        ]
        for mode, conf, expected in expectations:
            base = classify_conversation_context(user_message="x")
            from dataclasses import replace
            pause_dt = (
                conf is ContextConfidence.HIGH
                and mode not in (ContextMode.NORMAL_DT, ContextMode.AMBIGUOUS)
            )
            ctx = replace(
                base, mode=mode, confidence=conf, pause_dt=pause_dt
            )
            self.assertEqual(is_dt_paused(ctx), expected, f"{mode}/{conf}")
        self.assertFalse(is_dt_paused(None))

    def test_instruction_bullets_match_spec(self):
        for mode, text in _SPEC_GUIDANCE.items():
            bullet = context_instruction_bullet(mode)
            self.assertTrue(bullet.startswith("Conversational context: "))
            self.assertIn(text, bullet)
        self.assertIsNone(context_instruction_bullet(ContextMode.NORMAL_DT))
        self.assertIsNone(context_instruction_bullet(ContextMode.AMBIGUOUS))

    def test_fallbacks_respect_enforcement_constraints(self):
        for mode in _SPEC_GUIDANCE:
            fb = paused_turn_fallback(mode)
            self.assertIsNotNone(fb)
            lower = fb.lower()
            for marker in _ADVICE_MARKERS:
                self.assertNotIn(marker, lower, f"{mode}: {marker!r}")
            self.assertLessEqual(len(fb.split()), 55, mode)
            self.assertLessEqual(fb.count("?"), 1, mode)


class TestEnforceMentorReplyStatementOption(unittest.TestCase):
    FB = "fallback"

    def test_default_rejects_zero_question_reply(self):
        self.assertEqual(enforce_mentor_reply("Got it.", self.FB), self.FB)

    def test_allow_statement_accepts_zero_question_reply(self):
        self.assertEqual(
            enforce_mentor_reply("Got it.", self.FB, allow_statement=True), "Got it."
        )

    def test_advice_protection_active_even_with_statement(self):
        for reply in ("You should try X.", "I suggest starting smaller"):
            self.assertEqual(
                enforce_mentor_reply(reply, self.FB, allow_statement=True),
                self.FB,
            )

    def test_word_limit_active_even_with_statement(self):
        long_stmt = " ".join(["word"] * 60)
        self.assertEqual(
            enforce_mentor_reply(long_stmt, self.FB, allow_statement=True),
            self.FB,
        )

    def test_multi_question_truncation_with_statement(self):
        out = enforce_mentor_reply("A? B? C?", self.FB, allow_statement=True)
        self.assertEqual(out, "A?")

    def test_normal_question_reply_unchanged(self):
        q = "How often does this happen?"
        self.assertEqual(enforce_mentor_reply(q, self.FB), q)


# ---------------------------------------------------------------------------
# Integration — NORMAL_DT regression
# ---------------------------------------------------------------------------


class TestNormalTurnUnchanged(unittest.TestCase):
    def test_case_e_normal_answer_keeps_full_pipeline(self):
        _reset_session()
        stub_q = "Since this seems important, how often does it happen - daily or weekly?"
        reply, diag, sd = _run_turn("It happens almost every day.", stub_q)

        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "NORMAL_DT")
        self.assertEqual(cc["Confidence"], "HIGH")
        self.assertFalse(cc.get("DT Steering Paused", False))
        # Family guidance present (Phase 1 dynamic prompt carries it as a
        # fresh-angle suggestion), steering active, family recorded.
        self.assertIn("A fresh question angle such as", diag["Prompt"])
        self.assertNotIn("correcting your previous interpretation", diag["Prompt"])
        self.assertEqual(reply, stub_q)  # passed enforcement untouched
        self.assertIn("FREQUENCY_ESTIMATE", sd.asked_question_families)

    def test_zero_question_reply_still_rejected_on_normal_turns(self):
        _reset_session()
        # Message free of extractor keywords so the canonical objective is
        # PERSONAS on a fresh session.
        reply, diag, sd = _run_turn("Hmm interesting point.", "Nice.")
        self.assertFalse(diag["ConversationContext"].get("DT Steering Paused", False))
        self.assertTrue(reply.startswith("That's"))  # DT template fallback
        self.assertIn("PERSONAS_WHO", sd.asked_question_families)


# ---------------------------------------------------------------------------
# Integration — paused modes
# ---------------------------------------------------------------------------


class TestPausedCorrection(unittest.TestCase):
    def test_case_a(self):
        sd = _reset_session()
        families_before = list(sd.asked_question_families)
        state_before = sd.project_state.to_state_dict()

        stub = "Ah, got it - thanks for clarifying."
        reply, diag, sd2 = _run_turn("I'm not designing this for anyone.", stub)

        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "CORRECTION")
        self.assertEqual(cc["Confidence"], "HIGH")
        self.assertTrue(cc["DT Steering Paused"])
        # Family steering suppressed in the prompt.
        self.assertNotIn("Ask a NEW question angle", diag["Prompt"])
        self.assertIn(_SPEC_GUIDANCE[ContextMode.CORRECTION], diag["Prompt"])
        # Statement-shaped reply accepted verbatim.
        self.assertEqual(reply, stub)
        # Family hygiene: nothing recorded.
        self.assertEqual(list(sd2.asked_question_families), families_before)
        # Canonical engine untouched: objective still PERSONAS, not satisfied.
        self.assertEqual(diag["Pipeline"]["Objective"], "PERSONAS")
        self.assertEqual(sd2.project_state.to_state_dict(), state_before)


class TestPausedTopicShift(unittest.TestCase):
    def test_case_b(self):
        sd = _reset_session()
        before_state = sd.project_state.to_state_dict()
        before_families = list(sd.asked_question_families)

        stub = "Okay, we can look at education instead."
        reply, diag, sd2 = _run_turn(
            "Scratch that, let's talk about education instead.", stub
        )

        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "TOPIC_SHIFT")
        self.assertEqual(cc["Confidence"], "HIGH")
        self.assertTrue(cc["DT Steering Paused"])
        self.assertIn(_SPEC_GUIDANCE[ContextMode.TOPIC_SHIFT], diag["Prompt"])
        self.assertNotIn("Ask a NEW question angle", diag["Prompt"])
        self.assertEqual(reply, stub)
        self.assertEqual(list(sd2.asked_question_families), before_families)
        # The gate did not mutate ProjectState (old state preserved).
        self.assertEqual(sd2.project_state.to_state_dict(), before_state)


class TestPausedDirectQuestion(unittest.TestCase):
    def test_case_c_acknowledge_and_redirect(self):
        _reset_session()
        stub = ("I can help you work through it - what part of the logic "
                "feels hardest right now?")
        reply, diag, _sd = _run_turn(
            "I'm having trouble with coding logic. What should I do?", stub
        )
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "DIRECT_QUESTION")
        self.assertTrue(cc["DT Steering Paused"])
        self.assertIn(_SPEC_GUIDANCE[ContextMode.DIRECT_QUESTION], diag["Prompt"])
        self.assertEqual(reply, stub)

    def test_case_c_advice_protection_remains(self):
        _reset_session()
        reply, diag, _sd = _run_turn(
            "What should I do about this?",
            "You should break the problem into smaller functions.",
        )
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "DIRECT_QUESTION")
        self.assertTrue(cc["DT Steering Paused"])
        # Advice-marker reply rejected even on a paused turn -> conversational fallback.
        self.assertEqual(reply, paused_turn_fallback(ContextMode.DIRECT_QUESTION))


class TestPausedConfused(unittest.TestCase):
    def test_case_d(self):
        sd = _reset_session()
        before_families = list(sd.asked_question_families)

        stub = "Sorry - let me rephrase that more simply."
        reply, diag, sd2 = _run_turn("I don't understand.", stub)

        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "CONFUSED")
        self.assertEqual(cc["Confidence"], "HIGH")
        self.assertTrue(cc["DT Steering Paused"])
        self.assertIn(_SPEC_GUIDANCE[ContextMode.CONFUSED], diag["Prompt"])
        self.assertNotIn("Ask a NEW question angle", diag["Prompt"])
        self.assertEqual(reply, stub)  # zero-question clarification accepted
        self.assertEqual(list(sd2.asked_question_families), before_families)


# ---------------------------------------------------------------------------
# Integration — non-activated modes stay observation-only
# ---------------------------------------------------------------------------


class TestNonActivatedModesObservationOnly(unittest.TestCase):
    def test_case_f_low_confidence_ambiguous_does_not_pause(self):
        _reset_session()
        reply, diag, sd = _run_turn("I'm a dog.", "Nice.")
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "AMBIGUOUS")
        self.assertEqual(cc["Confidence"], "LOW")
        self.assertFalse(cc.get("DT Steering Paused", False))
        # Full normal pipeline active: family guidance + normal enforcement.
        self.assertIn("A fresh question angle such as", diag["Prompt"])
        self.assertTrue(reply.startswith("That's"))
        self.assertIn("PERSONAS_WHO", sd.asked_question_families)

    def test_hypothetical_pauses_since_phase_3(self):
        _reset_session()
        _reply, diag, _sd = _run_turn("Suppose students forgot everything.", None)
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "HYPOTHETICAL")
        self.assertTrue(cc.get("DT Steering Paused", False))


# ---------------------------------------------------------------------------
# Integration — hygiene + objective persistence across turns
# ---------------------------------------------------------------------------


class TestObjectivePersistenceAndHygiene(unittest.TestCase):
    def test_engine_resumes_canonically_after_pause(self):
        sd = _reset_session()
        before_families = list(sd.asked_question_families)

        # Turn 1 — paused correction.
        r1, d1, _sd = _run_turn("I'm not designing this for anyone.",
                                "Ah, got it - thanks for clarifying.")
        self.assertEqual(d1["Pipeline"]["Objective"], "PERSONAS")
        self.assertEqual(r1, "Ah, got it - thanks for clarifying.")

        # Turn 2 — ordinary answer: the Objective Engine resumes normally.
        stub_q = "Who specifically would benefit from this?"
        r2, d2, sd3 = _run_turn("It happens almost every day.", stub_q)
        self.assertEqual(d2["Pipeline"]["Objective"], "PERSONAS")
        self.assertEqual(d2["ConversationContext"]["Mode"], "NORMAL_DT")
        self.assertEqual(r2, stub_q)
        # Turn-2 reply registered exactly once; turn 1 contributed zero.
        self.assertEqual(sd3.asked_question_families.count("PERSONAS_WHO"), 1)
        self.assertEqual(
            [f for f in sd3.asked_question_families if f in before_families],
            before_families,
        )
        # The ordinary answer was extracted normally by the existing rules.
        self.assertTrue(sd3.project_state.frequency)


if __name__ == "__main__":
    unittest.main(verbosity=2)
