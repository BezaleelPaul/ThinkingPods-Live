"""
tests/test_lifecycle_manager.py

Unit tests for module5.lifecycle_manager (Module 5 — Lifecycle Manager).

Covers (per the Module 5 spec test categories):
  ✓ Lifecycle decisions
  ✓ CONTINUE
  ✓ READY_FOR_SUMMARY
  ✓ WAITING_FOR_CONFIRMATION
  ✓ READY_FOR_TRANSITION
  ✓ Deterministic output
  ✓ Type rejections
  ✓ is_confirmation_message / is_summary_presented_marker helpers
  ✓ LifecycleDecision enum values
"""

import sys
import os
import unittest

# Allow importing from the project root regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import ProjectState  # noqa: E402

from module3 import ObjectiveEngine  # noqa: E402
from module4 import ResponseStrategy, ResponseStrategyEngine  # noqa: E402
from module5 import (  # noqa: E402
    LifecycleDecision,
    LifecycleManager,
    is_confirmation_message,
    summary_presented_marker,
    is_summary_presented_marker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_state() -> ProjectState:
    return ProjectState()


def _full_state() -> ProjectState:
    return ProjectState(
        personas=["x"], problems=["y"],
        current_solutions=["z"], pain_points=["q"],
        evidence=["w"], frequency="daily",
    )


def _full_state_tagged() -> ProjectState:
    """Full ProjectState with the summary-presented marker already installed."""
    ps = _full_state()
    ps.previous_assistant_message = summary_presented_marker("anything")
    return ps


def _full_state_tagged_with_confirmation(user_msg: str) -> ProjectState:
    """Full ProjectState with summary marker + a specific user message."""
    ps = _full_state()
    ps.previous_assistant_message = summary_presented_marker("anything")
    ps.previous_user_message = user_msg
    return ps


def _obj(state: ProjectState):
    return ObjectiveEngine().determine_next(state)


def _strat(obj, state: ProjectState):
    return ResponseStrategyEngine().determine_strategy(obj, state)


# ---------------------------------------------------------------------------
# LifecycleDecision enum
# ---------------------------------------------------------------------------


class TestLifecycleDecisionEnum(unittest.TestCase):

    def test_enum_members(self):
        for name in ("CONTINUE", "READY_FOR_SUMMARY",
                     "WAITING_FOR_CONFIRMATION", "READY_FOR_TRANSITION"):
            self.assertTrue(hasattr(LifecycleDecision, name))

    def test_enum_str_subclass(self):
        self.assertEqual(LifecycleDecision.CONTINUE.value, "CONTINUE")

    def test_values_are_unique(self):
        vals = {d.value for d in LifecycleDecision}
        self.assertEqual(len(vals), len(list(LifecycleDecision)))


# ---------------------------------------------------------------------------
# CONTINUE decisions
# ---------------------------------------------------------------------------


class TestDecisionContinue(unittest.TestCase):

    def test_empty_state_not_wrap_up_returns_continue(self):
        mgr = LifecycleManager()
        state = _empty_state()
        obj = _obj(state)
        strat = _strat(obj, state)
        self.assertFalse(obj.is_wrap_up)
        self.assertEqual(mgr.determine(state, obj, strat),
                         LifecycleDecision.CONTINUE)

    def test_partial_state_not_wrap_up_returns_continue(self):
        mgr = LifecycleManager()
        state = ProjectState(personas=["x"])
        obj = _obj(state)
        strat = _strat(obj, state)
        self.assertFalse(obj.is_wrap_up)
        self.assertEqual(mgr.determine(state, obj, strat),
                         LifecycleDecision.CONTINUE)

    def test_continue_is_not_terminal(self):
        self.assertFalse(LifecycleManager.is_terminal(LifecycleDecision.CONTINUE))

    def test_continue_does_not_expect_summary(self):
        self.assertFalse(
            LifecycleManager.expects_summary(LifecycleDecision.CONTINUE)
        )

    def test_continue_is_not_open_for_confirmation(self):
        self.assertFalse(
            LifecycleManager.is_open_for_confirmation(LifecycleDecision.CONTINUE)
        )


# ---------------------------------------------------------------------------
# READY_FOR_SUMMARY decisions
# ---------------------------------------------------------------------------


class TestDecisionReadyForSummary(unittest.TestCase):

    def test_full_state_no_marker_yields_ready_for_summary(self):
        """WRAP_UP, no prior summary marker → READY_FOR_SUMMARY."""
        mgr = LifecycleManager()
        state = _full_state()
        obj = _obj(state)
        strat = _strat(obj, state)
        self.assertTrue(obj.is_wrap_up)
        self.assertIsNone(state.previous_assistant_message)
        self.assertEqual(mgr.determine(state, obj, strat),
                         LifecycleDecision.READY_FOR_SUMMARY)

    def test_ready_for_summary_expects_summary(self):
        self.assertTrue(
            LifecycleManager.expects_summary(LifecycleDecision.READY_FOR_SUMMARY)
        )

    def test_ready_for_summary_not_terminal(self):
        self.assertFalse(
            LifecycleManager.is_terminal(LifecycleDecision.READY_FOR_SUMMARY)
        )


# ---------------------------------------------------------------------------
# WAITING_FOR_CONFIRMATION decisions
# ---------------------------------------------------------------------------


class TestDecisionWaitingForConfirmation(unittest.TestCase):

    def test_full_state_tagged_ambiguous_user_yields_waiting(self):
        """Summary presented, user message is NOT confirmation → WAITING."""
        mgr = LifecycleManager()
        state = _full_state_tagged_with_confirmation("i want to refine")
        obj = _obj(state)
        strat = _strat(obj, state)
        self.assertEqual(mgr.determine(state, obj, strat),
                         LifecycleDecision.WAITING_FOR_CONFIRMATION)

    def test_full_state_tagged_no_user_message_yields_waiting(self):
        mgr = LifecycleManager()
        state = _full_state_tagged()
        obj = _obj(state)
        strat = _strat(obj, state)
        self.assertEqual(mgr.determine(state, obj, strat),
                         LifecycleDecision.WAITING_FOR_CONFIRMATION)

    def test_waiting_is_not_terminal(self):
        self.assertFalse(
            LifecycleManager.is_terminal(
                LifecycleDecision.WAITING_FOR_CONFIRMATION
            )
        )

    def test_waiting_does_not_expect_summary(self):
        self.assertFalse(
            LifecycleManager.expects_summary(
                LifecycleDecision.WAITING_FOR_CONFIRMATION
            )
        )

    def test_waiting_is_open_for_confirmation(self):
        self.assertTrue(
            LifecycleManager.is_open_for_confirmation(
                LifecycleDecision.WAITING_FOR_CONFIRMATION
            )
        )


# ---------------------------------------------------------------------------
# READY_FOR_TRANSITION decisions
# ---------------------------------------------------------------------------


class TestDecisionReadyForTransition(unittest.TestCase):

    def test_tagged_with_yes_confirmation_yields_ready_for_transition(self):
        mgr = LifecycleManager()
        # Explicit spec example confirmation.
        for confirmation in ("yes", "correct", "that's right", "spot on",
                             "looks good", "go ahead", "proceed"):
            state = _full_state_tagged_with_confirmation(confirmation)
            obj = _obj(state)
            strat = _strat(obj, state)
            self.assertEqual(
                mgr.determine(state, obj, strat),
                LifecycleDecision.READY_FOR_TRANSITION,
                f"confirmation '{confirmation}' should yield "
                f"READY_FOR_TRANSITION"
            )

    def test_tagged_with_y_yield_ready_for_transition(self):
        """Edge: the bare 'y' confirmation should still trigger."""
        mgr = LifecycleManager()
        state = _full_state_tagged_with_confirmation("y")
        obj = _obj(state)
        strat = _strat(obj, state)
        self.assertEqual(mgr.determine(state, obj, strat),
                         LifecycleDecision.READY_FOR_TRANSITION)

    def test_ready_for_transition_is_terminal(self):
        self.assertTrue(
            LifecycleManager.is_terminal(LifecycleDecision.READY_FOR_TRANSITION)
        )

    def test_ready_for_transition_does_not_expect_summary(self):
        self.assertFalse(
            LifecycleManager.expects_summary(
                LifecycleDecision.READY_FOR_TRANSITION
            )
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism(unittest.TestCase):

    def test_same_state_yields_same_decision_twice(self):
        mgr = LifecycleManager()
        state = _full_state_tagged_with_confirmation("yes")
        obj = _obj(state)
        strat = _strat(obj, state)
        a = mgr.determine(state, obj, strat)
        b = mgr.determine(state, obj, strat)
        self.assertEqual(a, b)

    def test_two_manager_instances_agree(self):
        state = _full_state_tagged_with_confirmation("correct")
        obj = _obj(state)
        strat = _strat(obj, state)
        a = LifecycleManager().determine(state, obj, strat)
        b = LifecycleManager().determine(state, obj, strat)
        self.assertEqual(a, b)

    def test_same_waiting_state_repeated(self):
        mgr = LifecycleManager()
        state = _full_state_tagged_with_confirmation("maybe not")
        obj = _obj(state)
        strat = _strat(obj, state)
        d = mgr.determine(state, obj, strat)
        self.assertEqual(d, mgr.determine(state, obj, strat))
        self.assertEqual(d, LifecycleDecision.WAITING_FOR_CONFIRMATION)


# ---------------------------------------------------------------------------
# Type rejections
# ---------------------------------------------------------------------------


class TestTypeRejection(unittest.TestCase):

    def test_rejects_non_project_state(self):
        mgr = LifecycleManager()
        obj = _obj(_empty_state())
        strat = ResponseStrategy.ASK_QUESTION
        with self.assertRaises(TypeError):
            mgr.determine({"personas": []}, obj, strat)

    def test_rejects_non_conversation_objective(self):
        mgr = LifecycleManager()
        state = _empty_state()
        strat = ResponseStrategy.ASK_QUESTION
        with self.assertRaises(TypeError):
            mgr.determine(state, "WRAP_UP", strat)

    def test_rejects_non_response_strategy(self):
        mgr = LifecycleManager()
        state = _empty_state()
        obj = _obj(state)
        with self.assertRaises(TypeError):
            mgr.determine(state, obj, "ASK_QUESTION")


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestConfirmationLexicon(unittest.TestCase):

    def test_spec_examples_match(self):
        self.assertTrue(is_confirmation_message("yes"))
        self.assertTrue(is_confirmation_message("correct"))
        self.assertTrue(is_confirmation_message("looks good"))
        self.assertTrue(is_confirmation_message("that's right"))
        self.assertTrue(is_confirmation_message("spot on"))
        self.assertTrue(is_confirmation_message("confirmed"))
        self.assertTrue(is_confirmation_message("go ahead"))

    def test_non_confirmations_rejected(self):
        for msg in ("no", "maybe", "i want to refine", "actually",
                    "not really", "kind of", "we should edit",
                    "i don't think so", "", None):
            self.assertFalse(is_confirmation_message(msg),
                             f"'{msg}' must NOT count as confirmation")

    def test_whitespace_agnostic(self):
        self.assertTrue(is_confirmation_message("   yes   "))
        self.assertTrue(is_confirmation_message("\tyes"))

    def test_punctuation_tolerant(self):
        self.assertTrue(is_confirmation_message("yes."))
        self.assertTrue(is_confirmation_message("yes!"))

    def test_case_insensitive(self):
        self.assertTrue(is_confirmation_message("YES"))
        self.assertTrue(is_confirmation_message("Correct"))

    def test_single_char_y(self):
        self.assertTrue(is_confirmation_message("y"))
        self.assertTrue(is_confirmation_message("y."))

    def test_partial_match_at_start_fails(self):
        """'yes ok' is NOT a confirmation per the strict rules."""
        self.assertFalse(is_confirmation_message("yes ok"))

    def test_eager_match_is_exact(self):
        """'yes sir' is not an exact lexeme — rejected."""
        self.assertFalse(is_confirmation_message("yes sir"))


class TestSummaryMarkerHelpers(unittest.TestCase):

    def test_marker_is_predictable(self):
        self.assertEqual(
            summary_presented_marker("anything"),
            "<<empathize_summary_presented>>"
        )

    def test_marker_detection(self):
        self.assertTrue(
            is_summary_presented_marker("<<empathize_summary_presented>>hello")
        )

    def test_no_marker_on_plain_text(self):
        self.assertFalse(is_summary_presented_marker("hello"))
        self.assertFalse(is_summary_presented_marker(None))
        self.assertFalse(is_summary_presented_marker(""))


# ---------------------------------------------------------------------------
# Cross-turn sequencing integration (light integration)
# ---------------------------------------------------------------------------


class TestLifecycleSequencing(unittest.TestCase):
    """Walk the full CONTINUE → READY_FOR_SUMMARY → WAITING →
    READY_FOR_TRANSITION sequence entirely through the LifecycleManager,
    simulating the cross-turn state the integrator would feed."""

    def test_full_sequence_through_manager(self):
        mgr = LifecycleManager()
        obj_eng = ObjectiveEngine()
        rs_eng = ResponseStrategyEngine()

        # --- Step 1: empty partial state → CONTINUE ---
        # The integrator builds ProjectState incrementally, calling
        # determine() each turn with the CURRENT (not previous) state.
        ps1 = ProjectState(personas=["x"])
        obj1 = obj_eng.determine_next(ps1)
        strat1 = rs_eng.determine_strategy(obj1, ps1)
        self.assertEqual(mgr.determine(ps1, obj1, strat1),
                         LifecycleDecision.CONTINUE)

        # --- Step 2: full state (no marker) → READY_FOR_SUMMARY ---
        ps2 = _full_state()
        obj2 = obj_eng.determine_next(ps2)
        strat2 = rs_eng.determine_strategy(obj2, ps2)
        d2 = mgr.determine(ps2, obj2, strat2)
        self.assertEqual(d2, LifecycleDecision.READY_FOR_SUMMARY)

        # The integrator would now have built an EmpathizeSummary and
        # emitted a summary reply. We simulate the *next* turn:
        # Install the marker EXACTLY as the integrator would.
        ps3 = _full_state_tagged_with_confirmation("i am not sure about evidence")
        obj3 = obj_eng.determine_next(ps3)
        strat3 = rs_eng.determine_strategy(obj3, ps3)
        d3 = mgr.determine(ps3, obj3, strat3)
        self.assertEqual(d3, LifecycleDecision.WAITING_FOR_CONFIRMATION,
                         "Step 3: summary-presented marker + ambiguous user → WAITING")

        # --- Step 4: user says explicit confirmation → READY_FOR_TRANSITION --
        ps4 = _full_state_tagged_with_confirmation("yes")
        obj4 = obj_eng.determine_next(ps4)
        strat4 = rs_eng.determine_strategy(obj4, ps4)
        d4 = mgr.determine(ps4, obj4, strat4)
        self.assertEqual(d4, LifecycleDecision.READY_FOR_TRANSITION,
                         "Step 4: summary-presented marker + explicit yes → READY_FOR_TRANSITION")


if __name__ == "__main__":
    unittest.main(verbosity=2)