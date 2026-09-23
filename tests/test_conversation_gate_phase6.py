"""
tests/test_conversation_gate_phase6.py — MEDIUM-confidence acknowledge-first
framing.

Contract:
  * acknowledge_first = (mode != NORMAL_DT and confidence == MEDIUM)
  * MEDIUM turns stay FULLY canonical DT turns: extraction runs, objective
    unchanged, family planning + steering active, family recorded, guard
    and enforcement untouched.
  * Phase 1 dynamic coach: the framing reaches Module 4 as the Brief's
    acknowledge-first note inside the slim dynamic prompt (not the legacy
    framing bullet); the planned question remains canonical.
  * HIGH still pauses (no ack note), LOW ambiguity gets nothing.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation_context import (  # noqa: E402
    ContextConfidence,
    ContextMode,
    acknowledge_first_instruction_bullet,
    classify_conversation_context,
)

_USER = "phase6_user"
_PROJECT = "Phase6Project"


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
    # A raising stub still represents exactly one attempted reply call.
    calls = getattr(stub, "calls", 1)
    return reply, diagnostics, calls, _active_session()


_ACK_CORRECTION_TEXT = (
    "Briefly acknowledge that the user may be correcting or clarifying "
    "something. Do not assume what they meant. Continue with the planned "
    "question."
)
_ACK_CONFUSED_TEXT = (
    "Briefly acknowledge that the previous exchange may have been unclear. "
    "Do not assume what the user meant. Continue with the planned question."
)


# ---------------------------------------------------------------------------
# Data-model derivation
# ---------------------------------------------------------------------------


class TestAcknowledgeFirstDerivation(unittest.TestCase):
    def test_real_sentence_table(self):
        cases = [
            # (message, prev, mode, conf, ack)
            ("Actually parents are not affected.",
             "Who specifically would benefit from this?",
             "CORRECTION", "MEDIUM", True),
            ("I don't know, probably students.", None,
             "CONFUSED", "MEDIUM", True),
            ("That's not what I meant.", None,
             "CORRECTION", "HIGH", False),          # HIGH -> pause path
            ("I'm a dog.", None,
             "AMBIGUOUS", "LOW", False),            # LOW -> nothing
            ("It happens every single day.", None,
             "NORMAL_DT", "HIGH", False),
        ]
        for msg, prev, mode, conf, ack in cases:
            ctx = classify_conversation_context(
                user_message=msg, previous_assistant_message=prev)
            self.assertEqual(ctx.mode.value, mode, msg)
            self.assertEqual(ctx.confidence.value, conf, msg)
            self.assertEqual(ctx.acknowledge_first, ack, msg)
            self.assertEqual(ctx.pause_dt, mode == "CORRECTION" and conf == "HIGH", msg)

    def test_to_dict_exposes_flag(self):
        d = classify_conversation_context(user_message="No idea.").to_dict()
        self.assertTrue(d["Acknowledge First"])
        d2 = classify_conversation_context(user_message="It happens daily.").to_dict()
        self.assertFalse(d2["Acknowledge First"])


# ---------------------------------------------------------------------------
# Framing bullets
# ---------------------------------------------------------------------------


class TestAcknowledgmentBullets(unittest.TestCase):
    def test_texts_are_short_and_uncertain(self):
        for mode, text in (
            (ContextMode.CORRECTION, _ACK_CORRECTION_TEXT),
            (ContextMode.CONFUSED, _ACK_CONFUSED_TEXT),
        ):
            self.assertLessEqual(len(text.split()), 30, mode)
            self.assertIn("Do not assume what the", text)
            self.assertIn("Continue with the planned question", text)
            bullet = acknowledge_first_instruction_bullet(mode)
            self.assertTrue(bullet.startswith("Conversational framing: "))
            self.assertIn(text, bullet)

    def test_unmapped_modes_return_none(self):
        for mode in (ContextMode.NORMAL_DT, ContextMode.AMBIGUOUS,
                     ContextMode.HYPOTHETICAL, ContextMode.TOPIC_SHIFT,
                     ContextMode.DIRECT_QUESTION, ContextMode.UNSAFE):
            self.assertIsNone(acknowledge_first_instruction_bullet(mode))


# ---------------------------------------------------------------------------
# Pipeline — MEDIUM correction: acknowledge + planned question continues
# ---------------------------------------------------------------------------


class TestMediumCorrectionPipeline(unittest.TestCase):
    def setUp(self):
        _reset_session()
        # Seed a personas-family mentor question so the combo signal has
        # context on the following turn.
        _run_turn(
            "I want to help students manage assignment deadlines", None)

    def test_ack_bullet_and_steering_coexist(self):
        stub_q = ("Got it - let me make sure we're framing this right. "
                  "How often does this affect them?")
        reply, diag, calls, sd = _run_turn(
            "Actually parents are not affected.", stub_q)

        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "CORRECTION")
        self.assertEqual(cc["Confidence"], "MEDIUM")
        self.assertTrue(cc["Acknowledge First"])
        self.assertFalse(cc["DT Steering Paused"])

        prompt = diag["Prompt"]
        # Phase 1 carrier: Brief acknowledge-first note in WHAT HAPPENED.
        self.assertIn("acknowledge it lightly", prompt)
        self.assertIn("A fresh question angle such as", prompt)  # steering active

        # Canonical machinery fully intact on this turn: the objective is
        # whatever the canonical engine had after the seed (personas were
        # captured there), and the MEDIUM turn does not disturb it.
        self.assertEqual(diag["Pipeline"]["Objective"], "PROBLEMS")
        self.assertEqual(reply, stub_q)                        # 1 question, passed
        self.assertEqual(calls, 1)                             # no extra LLM call
        self.assertEqual(sd.asked_question_families[-1], "FREQUENCY_ESTIMATE")
        # Extraction ran normally: 'parents' captured as a persona fact.
        self.assertIn("parents", sd.project_state.personas)

    def test_advice_protection_active_on_medium_turns(self):
        reply, diag, _calls, _sd = _run_turn(
            "Actually parents are not affected.",
            "You should just survey parents directly.",
        )
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Confidence"], "MEDIUM")
        self.assertFalse(cc["DT Steering Paused"])
        # Advice marker rejected by UNCHANGED enforcement -> DT template.
        self.assertTrue(reply.startswith("That's"))


# ---------------------------------------------------------------------------
# Pipeline — MEDIUM confusion
# ---------------------------------------------------------------------------


class TestMediumConfusionPipeline(unittest.TestCase):
    def test_hedged_answer_gets_ack_and_canonical_question(self):
        _reset_session()
        _run_turn("I want to reduce missed deadlines at my university", None)
        stub_q = ("Understood - best guess is fine here. "
                  "How often do deadlines slip?")
        reply, diag, _calls, sd = _run_turn(
            "I don't know, probably students.", stub_q)
        cc = diag["ConversationContext"]
        self.assertEqual(cc["Mode"], "CONFUSED")
        self.assertEqual(cc["Confidence"], "MEDIUM")
        self.assertTrue(cc["Acknowledge First"])
        self.assertFalse(cc["DT Steering Paused"])
        self.assertIn("acknowledge it lightly", diag["Prompt"])
        self.assertIn("A fresh question angle such as", diag["Prompt"])
        self.assertEqual(reply, stub_q)
        self.assertEqual(sd.project_state.personas, ["college students"])


# ---------------------------------------------------------------------------
# NON-MEDIUM paths untouched
# ---------------------------------------------------------------------------


class TestNonMediumPaths(unittest.TestCase):
    def test_high_correction_still_pauses_without_ack(self):
        _reset_session()
        _reply, diag, _c, _sd = _run_turn("That's not what I meant.", None)
        cc = diag["ConversationContext"]
        self.assertTrue(cc["DT Steering Paused"])
        self.assertFalse(cc["Acknowledge First"])
        self.assertNotIn(_ACK_CORRECTION_TEXT, diag["Prompt"])

    def test_low_ambiguity_gets_nothing(self):
        _reset_session()
        _reply, diag, _c, _sd = _run_turn("I'm a dog.", None)
        cc = diag["ConversationContext"]
        self.assertFalse(cc["Acknowledge First"])
        self.assertNotIn("Conversational framing:", diag["Prompt"])

    def test_normal_turn_markers_preserved(self):
        # Phase 1 intentional change: the normal-turn prompt is the slim
        # dynamic prompt (not byte-identical to the legacy wall), but the
        # markers that matter are preserved: no framing note, family
        # steering present, same decisions and enforcement.
        _reset_session()
        stub_q = "Since this matters, how often does it happen?"
        reply, diag, _c, sd = _run_turn("It happens almost every day.", stub_q)
        cc = diag["ConversationContext"]
        self.assertEqual((cc["Mode"], cc["Confidence"]),
                         ("NORMAL_DT", "HIGH"))
        self.assertFalse(cc["Acknowledge First"])
        prompt = diag["Prompt"]
        self.assertNotIn("Conversational framing:", prompt)
        self.assertIn("A fresh question angle such as", prompt)
        self.assertEqual(reply, stub_q)
        self.assertEqual(sd.asked_question_families[-1], "FREQUENCY_ESTIMATE")

    def test_response_contains_at_most_one_question(self):
        _reset_session()
        _run_turn("I want to help students manage deadlines", None)
        reply, _diag, _c, _sd = _run_turn(
            "Actually parents are not affected.",
            "Noted. How often does this happen, and who else is involved?",
        )
        # Multi-question replies are truncated to the first question by the
        # UNCHANGED enforcement (normal-turn behaviour preserved).
        self.assertEqual(reply.count("?"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
