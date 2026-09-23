"""
tests/test_complexity_gate.py

Regression tests for the deterministic Semantic Complexity Gate
(``complexity_gate`` + its wiring in ``hybrid_extraction`` and ``mentor``):

  * ``classify_complexity`` returns LOW / MEDIUM / HIGH from pure surface
    heuristics (causal / contrast / explanation / evidence / motivation /
    consequence cues, clause separators, candidate-category count) with no
    embeddings, LLM, or ML — and never touches state.
  * ``decision_for`` maps LOW -> SKIP_LLM and MEDIUM/HIGH -> RUN_LLM.
  * ``enabled()`` honours ``COMPLEXITY_GATE`` (default on, case-insensitive).
  * ``decide_hybrid_extraction`` gate flip: when the rules satisfy the current
    objective, a MEDIUM/HIGH message still invokes the LLM; a LOW message
    skips it exactly as before. The gate only ever affects the skip/run
    decision on the rules-satisfied path — never state, objectives, prompts,
    or lifecycle.
  * gate disabled -> the rules-satisfied path behaves byte-identically to the
    pre-gate behaviour (always skip).
  * the live pipeline surfaces the Developer Console ``SemanticComplexity``
    section (Complexity / Reasons / Decision / Gate Enabled).

All tests are deterministic (mocked ollama + mocked MemoryExtractor).
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from complexity_gate import (  # noqa: E402
    COMPLEXITY_HIGH,
    COMPLEXITY_LOW,
    COMPLEXITY_MEDIUM,
    DECISION_RUN,
    DECISION_SKIP,
    classify_complexity,
    decision_for,
    enabled,
)
from hybrid_extraction import decide_hybrid_extraction  # noqa: E402
from memory_extractor import ProjectState  # noqa: E402
from session_manager import SessionData  # noqa: E402

import mentor  # noqa: E402


class _EnvScope:
    """Context manager that pins COMPLEXITY_GATE for a block."""

    def __init__(self, value):
        self._value = value

    def __enter__(self):
        self._prev = os.environ.get("COMPLEXITY_GATE")
        if self._value is None:
            os.environ.pop("COMPLEXITY_GATE", None)
        else:
            os.environ["COMPLEXITY_GATE"] = self._value
        return self

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop("COMPLEXITY_GATE", None)
        else:
            os.environ["COMPLEXITY_GATE"] = self._prev


class TestClassifyComplexity(unittest.TestCase):
    """Pure-function unit tests for the deterministic classifier."""

    def test_low_single_explicit_statement(self):
        for msg, obs in (
            ("It happens every single day", {"frequency": "every single day"}),
            ("Students", {"target_audience": "students"}),
            ("She currently uses a simple pill box", {"existing_solution": "a simple pill box"}),
            ("yes", {}),
        ):
            with self.subTest(msg=msg):
                rec = classify_complexity(msg, obs)
                self.assertEqual(rec["complexity"], COMPLEXITY_LOW)
                self.assertEqual(rec["decision"], DECISION_SKIP)

    def test_medium_one_extra_semantic_dimension(self):
        for msg, obs in (
            ("I want to help elderly people take their medications on time", {"target_audience": "elderly people"}),
            ("My grandmother always forgets to take her pills", {"pain_point": "forgetting", "motivation": "personal", "frequency": "always"}),
            ("They all use sticky notes and a paper planner", {"existing_solution": "sticky notes"}),
            ("I have seen her skip doses at least twice last week", {"evidence": "skip doses"}),
            ("Actually parents are affected too", {"target_audience": "parents"}),
        ):
            with self.subTest(msg=msg):
                rec = classify_complexity(msg, obs)
                self.assertEqual(rec["complexity"], COMPLEXITY_MEDIUM)
                self.assertEqual(rec["decision"], DECISION_RUN)

    def test_high_multiple_semantic_concepts(self):
        for msg, obs in (
            ("It stresses me out because I worry she will miss a dose", {"pain_point": "stress", "motivation": "worry"}),
            ("I've been building this for students who keep missing deadlines - it stresses them out and I've interviewed ten of them", {"target_audience": "students", "pain_point": "deadlines", "motivation": "stress", "evidence": "interviewed"}),
        ):
            with self.subTest(msg=msg):
                rec = classify_complexity(msg, obs)
                self.assertEqual(rec["complexity"], COMPLEXITY_HIGH)
                self.assertEqual(rec["decision"], DECISION_RUN)

    def test_empty_message_is_low(self):
        rec = classify_complexity("", None)
        self.assertEqual(rec["complexity"], COMPLEXITY_LOW)
        self.assertEqual(rec["decision"], DECISION_SKIP)

    def test_none_message_is_low(self):
        rec = classify_complexity(None, None)
        self.assertEqual(rec["complexity"], COMPLEXITY_LOW)
        self.assertEqual(rec["decision"], DECISION_SKIP)

    def test_deterministic(self):
        msg = "It stresses me out because I worry she will miss a dose"
        obs = {"pain_point": "stress", "motivation": "worry"}
        first = classify_complexity(msg, obs)
        for _ in range(5):
            self.assertEqual(classify_complexity(msg, obs), first)

    def test_candidate_categories_add_to_score(self):
        # Two candidate categories bump the score to MEDIUM even without cues.
        rec = classify_complexity(
            "parents use a paper planner",
            {"target_audience": "parents", "existing_solution": "paper planner"},
        )
        self.assertEqual(rec["complexity"], COMPLEXITY_MEDIUM)
        self.assertEqual(rec["signals"]["candidate_categories"], 2)

    def test_three_candidate_categories_add_two(self):
        rec = classify_complexity(
            "grandma forgets pills daily out of habit",
            {"pain_point": "forgets", "motivation": "habit", "frequency": "daily"},
        )
        self.assertGreaterEqual(rec["signals"]["score"], 2)

    def test_record_shape(self):
        rec = classify_complexity("Students", {"target_audience": "students"})
        for key in ("complexity", "reasons", "decision", "signals"):
            self.assertIn(key, rec)
        for key in ("cues", "clauses", "candidate_categories", "score"):
            self.assertIn(key, rec["signals"])
        self.assertEqual(rec["signals"]["clauses"], 1)
        self.assertEqual(rec["signals"]["candidate_categories"], 1)


class TestDecisionFor(unittest.TestCase):
    def test_low_skips(self):
        self.assertEqual(decision_for(COMPLEXITY_LOW), DECISION_SKIP)

    def test_medium_runs(self):
        self.assertEqual(decision_for(COMPLEXITY_MEDIUM), DECISION_RUN)

    def test_high_runs(self):
        self.assertEqual(decision_for(COMPLEXITY_HIGH), DECISION_RUN)


class TestEnabled(unittest.TestCase):
    def test_default_enabled(self):
        with _EnvScope(None):
            self.assertTrue(enabled())

    def test_explicit_true(self):
        with _EnvScope("true"):
            self.assertTrue(enabled())

    def test_explicit_false(self):
        with _EnvScope("false"):
            self.assertFalse(enabled())

    def test_case_insensitive(self):
        with _EnvScope("TRUE"):
            self.assertTrue(enabled())
        with _EnvScope("False"):
            self.assertFalse(enabled())


class TestDecideHybridExtractionGateFlip(unittest.TestCase):
    """The gate only flips the skip/run decision on the rules-satisfied path."""

    def setUp(self):
        from session_manager import get_session_manager

        get_session_manager().reset_runtime_state()

    def test_low_rules_satisfied_still_skips_llm(self):
        with _EnvScope("true"):
            state = ProjectState()
            decision = decide_hybrid_extraction(
                {"target_audience": "students"},
                state,
                SessionData(),
                user_message="Students",
            )
        self.assertTrue(decision["satisfied"])
        self.assertFalse(decision["llm_invoked"])
        self.assertEqual(decision["complexity"], COMPLEXITY_LOW)
        self.assertTrue(decision["gate_enabled"])

    def test_medium_rules_satisfied_runs_llm(self):
        with _EnvScope("true"):
            state = ProjectState()
            decision = decide_hybrid_extraction(
                {"target_audience": "elderly people"},
                state,
                SessionData(),
                user_message="I want to help elderly people take their medications on time",
            )
        self.assertTrue(decision["satisfied"])
        self.assertTrue(decision["llm_invoked"])
        self.assertEqual(decision["complexity"], COMPLEXITY_MEDIUM)
        self.assertTrue(decision["gate_enabled"])
        self.assertEqual(
            decision["reason"],
            "Current objective satisfied by deterministic extraction, but "
            "message complexity requires the LLM extractor.",
        )

    def test_high_rules_satisfied_runs_llm(self):
        with _EnvScope("true"):
            state = ProjectState()
            # PERSONAS complete -> the current objective is PROBLEMS, which the
            # pain_point rule satisfies, letting the gate decide on the
            # rules-satisfied path.
            state.personas = ["students"]
            decision = decide_hybrid_extraction(
                {"pain_point": "stress", "motivation": "worry"},
                state,
                SessionData(),
                user_message="It stresses me out because I worry she will miss a dose",
            )
        self.assertTrue(decision["satisfied"])
        self.assertTrue(decision["llm_invoked"])
        self.assertEqual(decision["complexity"], COMPLEXITY_HIGH)

    def test_gate_disabled_rules_satisfied_always_skips(self):
        with _EnvScope("false"):
            state = ProjectState()
            decision = decide_hybrid_extraction(
                {"target_audience": "elderly people"},
                state,
                SessionData(),
                user_message="I want to help elderly people take their medications on time",
            )
        self.assertTrue(decision["satisfied"])
        self.assertFalse(decision["llm_invoked"])
        self.assertEqual(decision["complexity"], COMPLEXITY_MEDIUM)
        self.assertFalse(decision["gate_enabled"])
        self.assertEqual(
            decision["reason"],
            "Current objective satisfied by deterministic extraction.",
        )

    def test_rules_not_satisfied_llm_runs_regardless_of_complexity(self):
        with _EnvScope("true"):
            state = ProjectState()
            decision = decide_hybrid_extraction(
                {"frequency": "daily"},
                state,
                SessionData(),
                user_message="It happens every single day",
            )
        self.assertFalse(decision["satisfied"])
        self.assertTrue(decision["llm_invoked"])
        self.assertEqual(decision["complexity"], COMPLEXITY_LOW)

    def test_wrap_up_always_runs_llm(self):
        state = ProjectState()
        state.personas = ["students"]
        state.problems = ["missing deadlines"]
        state.current_solutions = ["sticky notes"]
        state.pain_points = ["stress"]
        state.evidence = ["interviewed 5 students"]
        state.frequency = "weekly"
        with _EnvScope("true"):
            decision = decide_hybrid_extraction(
                {"target_audience": "students"},
                state,
                SessionData(),
                user_message="I want to help students",
            )
        self.assertEqual(decision["objective"], "WRAP_UP")
        self.assertIsNone(decision["objective_field"])
        self.assertTrue(decision["llm_invoked"])

    def test_decision_record_carries_complexity_fields(self):
        with _EnvScope("true"):
            state = ProjectState()
            decision = decide_hybrid_extraction(
                {"target_audience": "elderly people"},
                state,
                SessionData(),
                user_message="I want to help elderly people take their medications on time",
            )
        for key in ("complexity", "complexity_reason", "complexity_decision", "gate_enabled"):
            self.assertIn(key, decision)
        self.assertEqual(decision["complexity_decision"], DECISION_RUN)
        self.assertTrue(decision["complexity_reason"])


class TestSemanticComplexityDiagnostics(unittest.TestCase):
    """The live pipeline surfaces the SemanticComplexity section."""

    USER = "gate_user"
    PROJ = "GateProject"

    def setUp(self):
        from session_manager import get_session_manager

        get_session_manager().reset_runtime_state()

    def tearDown(self):
        from session_manager import get_session_manager

        try:
            get_session_manager().reset_runtime_state()
        except Exception:
            pass

    def _run_turn(self, user_message, extraction_result, gate_value="true"):
        from memory_extractor import ExtractionResult, MessageType

        if extraction_result is None:
            extraction_result = ExtractionResult(MessageType.AMBIGUOUS, [])

        class _RaisingOllama:
            @staticmethod
            def chat(**kwargs):
                raise RuntimeError("ollama disabled in tests")

        with _EnvScope(gate_value):
            with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama()}):
                with mock.patch(
                    "mentor.MemoryExtractor.extract", return_value=extraction_result
                ):
                    return mentor.process_mentor_turn(
                        user_message,
                        username=self.USER,
                        project_name=self.PROJ,
                        model_name="test-model",
                    )

    def test_medium_message_has_complexity_section(self):
        from memory_extractor import (
            ExtractionResult,
            ExtractionUpdate,
            MessageType,
            Operation,
            StateField,
        )

        extraction = ExtractionResult(
            MessageType.MEANINGFUL,
            [
                ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "elderly people"),
                ExtractionUpdate(Operation.ADD, StateField.PROBLEMS, "missed doses"),
            ],
        )
        _reply, _sess, _timing, diagnostics = self._run_turn(
            "I want to help elderly people take their medications on time",
            extraction,
        )
        section = diagnostics["SemanticComplexity"]
        self.assertEqual(section["Complexity"], COMPLEXITY_MEDIUM)
        self.assertEqual(section["Decision"], DECISION_RUN)
        self.assertEqual(section["Gate Enabled"], "Yes")
        self.assertTrue(section["Reasons"])

        hybrid = diagnostics["HybridExtraction"]
        self.assertEqual(hybrid["Objective Satisfied By Rules"], "Yes")
        self.assertEqual(hybrid["LLM Invoked"], "Yes")

    def test_low_message_skips_and_complexity_low(self):
        from memory_extractor import (
            ExtractionResult,
            ExtractionUpdate,
            MessageType,
            Operation,
            StateField,
        )

        extraction = ExtractionResult(
            MessageType.MEANINGFUL,
            [ExtractionUpdate(Operation.SET, StateField.FREQUENCY, "every single day")],
        )
        # Seed personas + problems so FREQUENCY is the current objective.
        from session_manager import get_session_manager

        sd = get_session_manager().get_active_session_data()
        sd.project_state.personas = ["elderly people"]
        sd.project_state.problems = ["forgets her pills"]
        get_session_manager().save_session_data(get_session_manager().get_active_session_id(), sd)

        _reply, _sess, _timing, diagnostics = self._run_turn(
            "It happens every single day",
            extraction,
        )
        section = diagnostics["SemanticComplexity"]
        self.assertEqual(section["Complexity"], COMPLEXITY_LOW)
        self.assertEqual(section["Decision"], DECISION_SKIP)
        hybrid = diagnostics["HybridExtraction"]
        self.assertEqual(hybrid["Objective Satisfied By Rules"], "Yes")
        self.assertEqual(hybrid["LLM Invoked"], "No")


if __name__ == "__main__":
    unittest.main()
