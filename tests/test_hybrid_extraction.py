"""
tests/test_hybrid_extraction.py

Regression tests for the objective-aware hybrid extraction decision layer
(``hybrid_extraction`` + its wiring in ``mentor._extract_and_update_state``):

  * rules satisfy the current objective  -> the LLM extractor is NOT invoked
    and the deterministic output is applied through the existing StateManager
    merge path (``merge_extracted_to_state``).
  * rules do NOT satisfy the objective   -> the LLM extractor runs exactly as
    before.
  * WRAP_UP / no active field objective  -> the LLM always runs (LLM-first at
    transition).
  * the "behavior identical after merge" guarantee: the skip path produces
    exactly the same ProjectState as calling ``merge_extracted_to_state`` with
    the same rule observation on a fresh, identical state.
  * the Developer Console ``HybridExtraction`` section is present with the
    expected fields.

All tests are deterministic (mocked ollama + mocked MemoryExtractor).
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hybrid_extraction import decide_hybrid_extraction, rule_update_dicts  # noqa: E402
from memory_extractor import ProjectState, StateField  # noqa: E402
from session_manager import SessionData  # noqa: E402

import mentor  # noqa: E402


class _RaisingOllama:
    """Stand-in for the ``ollama`` module that always fails, forcing the
    deterministic family-aware fallback path."""

    @staticmethod
    def chat(**kwargs):
        raise RuntimeError("ollama disabled in tests")


def _full_state() -> ProjectState:
    """A ProjectState with every required Empathize field populated."""
    state = ProjectState()
    state.personas = ["students"]
    state.problems = ["missing deadlines"]
    state.current_solutions = ["sticky notes"]
    state.pain_points = ["stress"]
    state.evidence = ["interviewed 5 students"]
    state.frequency = "weekly"
    return state


class TestDecisionLayer(unittest.TestCase):
    """Unit tests for ``decide_hybrid_extraction`` in isolation."""

    def setUp(self):
        from session_manager import get_session_manager

        get_session_manager().reset_runtime_state()

    def test_empty_state_audience_rules_satisfy_personas(self):
        state = ProjectState()
        session_data = SessionData()
        decision = decide_hybrid_extraction(
            {"target_audience": "elderly people"},
            state,
            session_data,
        )
        self.assertEqual(decision["objective"], "PERSONAS")
        self.assertEqual(decision["objective_field"], "personas")
        self.assertTrue(decision["satisfied"])
        self.assertFalse(decision["llm_invoked"])
        self.assertEqual(
            decision["reason"],
            "Current objective satisfied by deterministic extraction.",
        )

    def test_frequency_rules_do_not_satisfy_personas_objective(self):
        state = ProjectState()
        session_data = SessionData()
        decision = decide_hybrid_extraction(
            {"frequency": "every single day"},
            state,
            session_data,
        )
        self.assertEqual(decision["objective"], "PERSONAS")
        self.assertEqual(decision["objective_field"], "personas")
        self.assertFalse(decision["satisfied"])
        self.assertTrue(decision["llm_invoked"])
        self.assertEqual(decision["reason"], "Current objective still unresolved.")

    def test_frequency_rules_satisfy_frequency_objective(self):
        state = ProjectState()
        state.personas = ["students"]
        state.problems = ["missing deadlines"]
        session_data = SessionData()
        decision = decide_hybrid_extraction(
            {"frequency": "daily"},
            state,
            session_data,
        )
        self.assertEqual(decision["objective"], "FREQUENCY")
        self.assertEqual(decision["objective_field"], "frequency")
        self.assertTrue(decision["satisfied"])
        self.assertFalse(decision["llm_invoked"])

    def test_wrap_up_always_invokes_llm(self):
        state = _full_state()
        session_data = SessionData()
        decision = decide_hybrid_extraction(
            {"target_audience": "students"},
            state,
            session_data,
        )
        self.assertEqual(decision["objective"], "WRAP_UP")
        self.assertIsNone(decision["objective_field"])
        self.assertFalse(decision["satisfied"])
        self.assertTrue(decision["llm_invoked"])
        self.assertEqual(
            decision["reason"],
            "No active field objective to satisfy; preserving LLM-first behavior.",
        )

    def test_rule_update_dicts_maps_legacy_to_state_updates(self):
        updates = rule_update_dicts(
            {
                "target_audience": "students",
                "frequency": "daily",
                "confidences": {"target_audience": 0.98},
            }
        )
        by_field = {u["field"]: u for u in updates}
        self.assertEqual(by_field["personas"]["operation"], "ADD")
        self.assertEqual(by_field["personas"]["value"], "students")
        self.assertEqual(by_field["personas"]["confidence"], 0.98)
        self.assertEqual(by_field["frequency"]["operation"], "SET")
        self.assertEqual(by_field["frequency"]["value"], "daily")


class TestHybridWiringLivePipeline(unittest.TestCase):
    """Integration: the mentor pipeline applies the hybrid decision."""

    USER = "hybrid_user"
    PROJ = "HybridProject"

    def setUp(self):
        from session_manager import get_session_manager

        get_session_manager().reset_runtime_state()

    def tearDown(self):
        from session_manager import get_session_manager

        try:
            get_session_manager().reset_runtime_state()
        except Exception:
            pass

    def _run_turn(self, user_message, patch_extract):
        # COMPLEXITY_GATE=false keeps this suite pinned to the pure hybrid
        # skip/run decision (rules satisfy => always skip). The gate's own
        # behavior is covered in test_complexity_gate.py.
        with mock.patch.dict(os.environ, {"COMPLEXITY_GATE": "false"}):
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

    def _active_state(self):
        from session_manager import get_session_manager

        return get_session_manager().get_active_session_data().project_state

    def test_rules_satisfy_objective_skips_llm_and_applies_rules(self):
        calls = []

        def _patched(*args, **kwargs):
            calls.append(True)
            raise AssertionError("LLM extractor must not run when rules satisfy")

        reply, _session, _timing, diagnostics = self._run_turn(
            "I want to help elderly people take their medications on time",
            _patched,
        )
        self.assertEqual(calls, [])
        state = self._active_state()
        self.assertEqual(state.personas, ["elderly people"])
        self.assertTrue(reply and reply.strip())

        hybrid = diagnostics["HybridExtraction"]
        self.assertEqual(hybrid["Objective"], "PERSONAS")
        self.assertEqual(hybrid["Objective Field"], "personas")
        self.assertEqual(hybrid["Objective Satisfied By Rules"], "Yes")
        self.assertEqual(hybrid["LLM Invoked"], "No")
        self.assertEqual(
            hybrid["Reason"],
            "Current objective satisfied by deterministic extraction.",
        )
        self.assertEqual(hybrid["Rule Output"], ["Personas: elderly people"])

        # The Extraction section records the rule-based path as MEANINGFUL
        # with the deterministic updates.
        extraction = diagnostics["Extraction"]
        self.assertEqual(extraction["message_type"], "MEANINGFUL")
        self.assertEqual(
            extraction["rule_based_extraction"]["target_audience"],
            "elderly people",
        )
        self.assertEqual(extraction["updates"][0]["field"], "personas")

    def test_rules_do_not_satisfy_objective_invokes_llm(self):
        calls = []

        def _patched(*args, **kwargs):
            calls.append(True)
            from memory_extractor import (
                ExtractionResult,
                MessageType,
            )

            return ExtractionResult(MessageType.AMBIGUOUS, [])

        reply, _session, _timing, diagnostics = self._run_turn(
            "It happens every single day",
            _patched,
        )
        # The frequency-only message does not satisfy the PERSONAS objective,
        # so the LLM extractor must have been called.
        self.assertEqual(len(calls), 1)
        hybrid = diagnostics["HybridExtraction"]
        self.assertEqual(hybrid["Objective"], "PERSONAS")
        self.assertEqual(hybrid["Objective Satisfied By Rules"], "No")
        self.assertEqual(hybrid["LLM Invoked"], "Yes")
        self.assertEqual(hybrid["Reason"], "Current objective still unresolved.")
        self.assertTrue(reply and reply.strip())

    def test_skip_path_state_identical_to_direct_merge(self):
        """The hybrid skip path must produce EXACTLY the state that a direct
        ``merge_extracted_to_state`` of the same rule observation would."""
        def _patched(*args, **kwargs):
            raise AssertionError("LLM extractor must not run when rules satisfy")

        _reply, _session, _timing, diagnostics = self._run_turn(
            "I want to help elderly people take their medications on time",
            _patched,
        )
        rule_observation = diagnostics["Extraction"]["rule_based_extraction"]

        expected = ProjectState()
        from extraction_pipeline import merge_extracted_to_state

        merge_extracted_to_state(expected, SessionData(), rule_observation)

        self.assertEqual(
            self._active_state().to_state_dict(),
            expected.to_state_dict(),
        )
        self.assertEqual(
            self._active_state().personas,
            ["elderly people"],
        )


if __name__ == "__main__":
    unittest.main()
