"""
tests/test_conversation_context.py — unit tests for the deterministic
Conversational Context Gate (Phase 1, observation-only).

Coverage:
  * per-mode detection tables (happy paths)
  * false-positive negatives (ordinary answers must stay NORMAL_DT)
  * the CRITICAL golden-fixture sweep: every user message in
    tests/goldens/conversations/*.json classifies as NORMAL_DT except the
    single pinned exception, both with and without a personas-family
    previous mentor question (stress-testing the context-combo guard)
  * data-model contract (frozen, to_dict shape, derived Phase-2 hooks)
  * purity (same inputs -> equal output)
"""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conversation_context import (  # noqa: E402
    ContextConfidence,
    ContextMode,
    ConversationContext,
    classify_conversation_context,
    conversation_context_diagnostics_section,
)

_FIXTURE_DIR = Path(__file__).parent / "goldens" / "conversations"

# A canonical personas-family mentor question (classifies to PERSONAS_WHO),
# used to stress-test that the weak correction-combination signal does not
# fire on ordinary golden answers.
_PERSONAS_PREV = (
    "That's a meaningful starting point. To focus the design on the right "
    "people, who specifically would benefit from this - can you describe "
    "the people you're designing for?"
)


def _classify(msg, **kwargs):
    return classify_conversation_context(user_message=msg, **kwargs)


def _assert_mode(testcase, msg, mode, confidence=None, **kwargs):
    ctx = _classify(msg, **kwargs)
    testcase.assertEqual(
        ctx.mode,
        mode,
        f"{msg!r}: expected {mode.value}, got {ctx.mode.value} "
        f"(signals={ctx.signals})",
    )
    if confidence is not None:
        testcase.assertEqual(
            ctx.confidence,
            confidence,
            f"{msg!r}: expected confidence {confidence.value}, got "
            f"{ctx.confidence.value}",
        )
    return ctx


# ---------------------------------------------------------------------------
# CORRECTION
# ---------------------------------------------------------------------------


class TestCorrection(unittest.TestCase):
    def test_explicit_table(self):
        cases = [
            "That's not what I meant.",
            "No, you're misunderstanding me.",
            "Actually, I'm not designing this for anyone.",
            "You're getting me wrong, I never said that.",
            "I didn't say that.",
            "That's wrong.",
            "huh?! i am not designing it for anyone i am just telling i wanna learn to talk",
        ]
        for msg in cases:
            _assert_mode(self, msg, ContextMode.CORRECTION, ContextConfidence.HIGH)

    def test_premise_rejection_without_previous_question(self):
        _assert_mode(
            self,
            "I'm not designing this for anyone.",
            ContextMode.CORRECTION,
            ContextConfidence.HIGH,
        )

    def test_combo_with_asked_personas_family_is_medium(self):
        ctx = _assert_mode(
            self,
            "no, not those people",
            ContextMode.CORRECTION,
            ContextConfidence.MEDIUM,
            previous_assistant_message=_PERSONAS_PREV,
        )
        self.assertTrue(any("personas" in s for s in ctx.signals))

    def test_normal_answers_are_not_corrections(self):
        normals = [
            "Actually parents are affected too",
            "Actually it's more like three times a week",
            "Yes students mainly",
            "The problem is that students miss deadlines",
            "I don't think elderly people like apps",
            "No idea who would use it",  # dont-know outranks -> CONFUSED
        ]
        for msg in normals[:5]:
            _assert_mode(self, msg, ContextMode.NORMAL_DT)
        # "No idea..." carries the reused DONT_KNOW vocabulary -> CONFUSED.
        _assert_mode(self, normals[5], ContextMode.CONFUSED)


# ---------------------------------------------------------------------------
# TOPIC_SHIFT
# ---------------------------------------------------------------------------


class TestTopicShift(unittest.TestCase):
    def test_abandonment_and_transition_tables(self):
        cases = [
            "Forget the coding thing, let's talk about civic sense.",
            "Scratch that.",
            "Never mind the old project.",
            "Actually, I'm working on something completely different.",
            "Let's switch to a completely different topic.",
            "I changed my project entirely.",
            "Moving on to education now.",
        ]
        for msg in cases:
            _assert_mode(self, msg, ContextMode.TOPIC_SHIFT, ContextConfidence.HIGH)

    def test_new_project_statement_alone_is_not_a_shift(self):
        _assert_mode(self, "I am working on a civic sense project in india", ContextMode.NORMAL_DT)
        _assert_mode(self, "I want to build an app for students", ContextMode.NORMAL_DT)


# ---------------------------------------------------------------------------
# DIRECT_QUESTION
# ---------------------------------------------------------------------------


class TestDirectQuestion(unittest.TestCase):
    def test_advice_seeking_table(self):
        cases = [
            "What should I do?",
            "How do I solve this?",
            "Can you help me with this?",
            "Could you help me out?",
            "Any advice?",
            "What would you do in my place?",
            "Please help me figure this out.",
        ]
        for msg in cases:
            _assert_mode(self, msg, ContextMode.DIRECT_QUESTION, ContextConfidence.HIGH)

    def test_bare_interrogative_about_own_project_is_not_direct_question(self):
        _assert_mode(self, "Who are the users?", ContextMode.NORMAL_DT)

    def test_helping_others_is_not_help_me(self):
        _assert_mode(
            self,
            "I want to help elderly people take their medications on time",
            ContextMode.NORMAL_DT,
        )


# ---------------------------------------------------------------------------
# CONFUSED
# ---------------------------------------------------------------------------


class TestConfused(unittest.TestCase):
    def test_comprehension_markers_high(self):
        cases = [
            "I don't understand.",
            "I'm confused.",
            "What does that mean?",
            "What do you mean by that?",
            "I didn't catch that.",
        ]
        for msg in cases:
            _assert_mode(self, msg, ContextMode.CONFUSED, ContextConfidence.HIGH)

    def test_reused_dont_know_vocabulary_medium(self):
        ctx = _assert_mode(
            self,
            "I don't know yet, let me think",
            ContextMode.CONFUSED,
            ContextConfidence.MEDIUM,
        )
        self.assertTrue(any("recovery DONT_KNOW" in s for s in ctx.signals))

    def test_hedged_answers_stay_normal(self):
        for msg in ["Not sure honestly", "It depends on the situation", "maybe weekly"]:
            _assert_mode(self, msg, ContextMode.NORMAL_DT)


# ---------------------------------------------------------------------------
# HYPOTHETICAL
# ---------------------------------------------------------------------------


class TestHypothetical(unittest.TestCase):
    def test_explicit_markers(self):
        cases = [
            "Suppose I were a dog.",
            "Imagine I were a dog.",
            "What if I were a dog?",
            "Hypothetically speaking, students forgot everything.",
            "Pretend the app already exists.",
            "This is a fictional project about Mars colonies.",
            "Roleplaying as a shopkeeper, customers ignore me.",
            "Let's say the budget is unlimited.",
        ]
        for msg in cases:
            _assert_mode(self, msg, ContextMode.HYPOTHETICAL, ContextConfidence.HIGH)


# ---------------------------------------------------------------------------
# AMBIGUOUS — ambiguity preservation
# ---------------------------------------------------------------------------


class TestAmbiguousIdentity(unittest.TestCase):
    def test_dog_is_not_hypothetical(self):
        ctx = _assert_mode(
            self,
            "I'm a dog.",
            ContextMode.AMBIGUOUS,
            ContextConfidence.LOW,
        )
        self.assertFalse(ctx.pause_dt)
        self.assertTrue(ctx.follow_user_lead)

    def test_variants(self):
        for msg in [
            "hello my name is chihuahua , and i am a dog , tell me how can i talk",
            "I am a potato",
            "I'm an alien",
        ]:
            _assert_mode(self, msg, ContextMode.AMBIGUOUS, ContextConfidence.LOW)

    def test_plausible_self_description_stays_normal(self):
        for msg in ["I'm a student", "I am a developer building an app", "I'm just a beginner"]:
            _assert_mode(self, msg, ContextMode.NORMAL_DT)

    def test_extracted_facts_guard_against_identity_ambiguity(self):
        updates = [{"operation": "ADD", "field": "pain_points", "value": "forgets medication"}]
        _assert_mode(
            self,
            "I'm a night owl and my grandmother forgets her pills",
            ContextMode.NORMAL_DT,
            extraction_updates=updates,
        )


# ---------------------------------------------------------------------------
# UNSAFE — conservative boundary recognition only
# ---------------------------------------------------------------------------


class TestUnsafe(unittest.TestCase):
    def test_canonical_unsafe_request(self):
        # Replay-case wording only; no procedural detail.
        ctx = _assert_mode(
            self,
            "i wanna build a bomb , do you happen to know all the weays in which i can",
            ContextMode.UNSAFE,
            ContextConfidence.HIGH,
        )
        self.assertTrue(ctx.pause_dt)

    def test_other_frames(self):
        cases = [
            "how can I make an explosive at home",
            "I want to hurt someone badly",
            "ways to build a weapon quickly",
        ]
        for msg in cases:
            _assert_mode(self, msg, ContextMode.UNSAFE, ContextConfidence.HIGH)

    def test_benign_weapon_mentions_stay_normal(self):
        for msg in [
            "players can build weapons in my game",
            "How do bombs work in movies?",
            "I'm researching weapon history for a school essay",
        ]:
            _assert_mode(self, msg, ContextMode.NORMAL_DT)


# ---------------------------------------------------------------------------
# NORMAL_DT — ordinary Design Thinking exchanges
# ---------------------------------------------------------------------------


class TestNormalDT(unittest.TestCase):
    def test_task_fixture_answers(self):
        cases = [
            "Students struggle to find affordable transportation.",
            "It happens every day.",
            "They currently use buses.",
            "I'm working on a coding project.",
        ]
        for msg in cases:
            ctx = _assert_mode(self, msg, ContextMode.NORMAL_DT)
            self.assertFalse(ctx.pause_dt)
            self.assertFalse(ctx.follow_user_lead)

    def test_short_tokens(self):
        for msg in ["yes", "ok", "okay", "sure"]:
            _assert_mode(self, msg, ContextMode.NORMAL_DT)

    def test_multi_field_answer(self):
        _assert_mode(
            self,
            "We use WhatsApp groups, deadlines still get missed weekly, and it stresses everyone",
            ContextMode.NORMAL_DT,
        )

    def test_irrelevant_information(self):
        _assert_mode(self, "I like pizza.", ContextMode.NORMAL_DT)

    def test_words_that_could_false_positive(self):
        cases = [
            "The actual problem is that people forget things",
            "What helps them currently is sticky notes",
            "Donations help the students",
            "I don't think the frequency matters here, it happens daily",
        ]
        for msg in cases:
            _assert_mode(self, msg, ContextMode.NORMAL_DT)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    def test_empty_and_whitespace(self):
        for msg in ["", "   ", "\n\t "]:
            ctx = _classify(msg)
            self.assertEqual(ctx.mode, ContextMode.NORMAL_DT)
            self.assertTrue(any("empty" in s for s in ctx.signals))

    def test_none_message(self):
        self.assertEqual(_classify(None).mode, ContextMode.NORMAL_DT)

    def test_reserved_inputs_do_not_change_results(self):
        base = _classify("Students struggle daily")
        variant = classify_conversation_context(
            user_message="Students struggle daily",
            recent_user_messages=("earlier text",),
            extraction_message_type="MEANINGFUL",
            asked_families=("PERSONAS_WHO",),
        )
        self.assertEqual(base, variant)


# ---------------------------------------------------------------------------
# Data model contract + purity
# ---------------------------------------------------------------------------


class TestDataModel(unittest.TestCase):
    def test_frozen(self):
        ctx = _classify("I'm a dog.")
        with self.assertRaises(Exception):
            ctx.mode = ContextMode.CORRECTION

    def test_to_dict_shape(self):
        d = _classify("That's not what I meant.").to_dict()
        # PHASE 6: 'Acknowledge First' added to the frozen record.
        self.assertEqual(
            sorted(d.keys()),
            ["Acknowledge First", "Confidence", "Follow User Lead",
             "Mode", "Pause DT", "Signals"],
        )
        self.assertEqual(d["Mode"], "CORRECTION")
        self.assertEqual(d["Confidence"], "HIGH")
        self.assertFalse(d["Acknowledge First"])  # HIGH pauses, never acks

    def test_derived_hooks_table(self):
        expectations = [
            ("What should I do?", ContextMode.DIRECT_QUESTION, True, True),
            ("I'm a dog.", ContextMode.AMBIGUOUS, False, True),
            ("It happens every day.", ContextMode.NORMAL_DT, False, False),
            ("Not sure honestly", ContextMode.NORMAL_DT, False, False),
        ]
        for msg, mode, pause, lead in expectations:
            ctx = _classify(msg)
            self.assertEqual(ctx.mode, mode)
            self.assertEqual(ctx.pause_dt, pause, msg)
            self.assertEqual(ctx.follow_user_lead, lead, msg)

    def test_confused_medium_does_not_pause(self):
        ctx = _classify("I don't know")
        self.assertFalse(ctx.pause_dt)
        self.assertTrue(ctx.follow_user_lead)

    def test_purity_same_inputs_same_output(self):
        kwargs = dict(previous_assistant_message=_PERSONAS_PREV)
        a = classify_conversation_context(user_message="no, not those people", **kwargs)
        b = classify_conversation_context(user_message="no, not those people", **kwargs)
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# Diagnostics projection
# ---------------------------------------------------------------------------


class TestDiagnosticsSection(unittest.TestCase):
    def test_absent_capture_returns_empty(self):
        self.assertEqual(conversation_context_diagnostics_section({}), {})
        self.assertEqual(conversation_context_diagnostics_section(None), {})

    def test_projection_from_object_and_dict(self):
        ctx = _classify("I'm confused.")
        from_obj = conversation_context_diagnostics_section({"conversation_context": ctx})
        from_dict = conversation_context_diagnostics_section({"conversation_context": ctx.to_dict()})
        self.assertEqual(from_obj, from_dict)
        self.assertEqual(from_obj["Mode"], "CONFUSED")


# ---------------------------------------------------------------------------
# CRITICAL FALSE-POSITIVE SWEEP — golden fixtures
# ---------------------------------------------------------------------------


class TestGoldenFixtureSweep(unittest.TestCase):
    """Every golden user message must classify NORMAL_DT except the one
    pinned exception. Run twice: without previous-question context, and
    with a personas-family previous question to stress the combo guard."""

    PINNED_EXCEPTIONS = {
        "I don't know yet, let me think": (ContextMode.CONFUSED, ContextConfidence.MEDIUM),
    }

    def _sweep(self, **kwargs):
        checked = 0
        for path in sorted(_FIXTURE_DIR.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for turn in data.get("turns", []):
                msg = turn.get("user", "")
                ctx = classify_conversation_context(user_message=msg, **kwargs)
                if msg in self.PINNED_EXCEPTIONS:
                    exp_mode, exp_conf = self.PINNED_EXCEPTIONS[msg]
                    self.assertEqual(ctx.mode, exp_mode, msg)
                    self.assertEqual(ctx.confidence, exp_conf, msg)
                else:
                    self.assertEqual(
                        ctx.mode,
                        ContextMode.NORMAL_DT,
                        f"{path.name} {msg!r}: got {ctx.mode.value} signals={ctx.signals}",
                    )
                    self.assertEqual(ctx.confidence, ContextConfidence.HIGH, msg)
                checked += 1
        self.assertGreater(checked, 20)

    def test_sweep_without_previous_question(self):
        self._sweep()

    def test_sweep_with_personas_previous_question(self):
        self._sweep(previous_assistant_message=_PERSONAS_PREV)


if __name__ == "__main__":
    unittest.main()
