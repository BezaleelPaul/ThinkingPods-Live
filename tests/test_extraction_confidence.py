"""
tests/test_extraction_confidence.py

Extraction confidence enrichment — each extracted fact now carries a
deterministic rule-strength confidence in ``[0.0, 1.0]``.

Scope (per the PR):
  * ``ExtractionUpdate`` gains ``confidence`` (default 1.0) — observation-only.
  * The rule-based extractor derives confidence from the strength of the rule
    that matched (explicit -> high, strong cadence -> high, evidence marker ->
    high, weak wording -> medium, ambiguous wording -> low).
  * Confidence is displayed in the Developer Console.
  * NO decision path consumes confidence: StateManager, ObjectiveEngine,
    lifecycle, and reply generation behave exactly as before. No ML is
    introduced; extraction behaviour (values, keys) is unchanged.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from extraction_pipeline import (  # noqa: E402
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_STRONG,
    CONFIDENCE_EVIDENCE,
    DEFAULT_CONFIDENCE,
    RuleBasedExtractor,
    merge_extracted_to_state,
)
from memory_extractor import (  # noqa: E402
    ExtractionResult,
    ExtractionUpdate,
    ExtractionValidator,
    MessageType,
    Operation,
    ProjectState,
    StateField,
)
from module3 import Objective, ObjectiveEngine  # noqa: E402
from session_manager import SessionData  # noqa: E402


class _RaisingOllama:
    """Stand-in for the ``ollama`` module that always fails, forcing the
    deterministic family-aware fallback path."""

    @staticmethod
    def chat(**kwargs):
        raise RuntimeError("ollama disabled in tests")


# ---------------------------------------------------------------------------
# ExtractionUpdate model
# ---------------------------------------------------------------------------


class TestExtractionUpdateConfidence(unittest.TestCase):
    """The typed fact carries a confidence, defaulting to 1.0."""

    def test_default_confidence_is_one(self):
        update = ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "students")
        self.assertEqual(update.confidence, 1.0)

    def test_explicit_confidence_is_preserved(self):
        update = ExtractionUpdate(
            Operation.ADD, StateField.PERSONAS, "students", confidence=0.98
        )
        self.assertEqual(update.confidence, 0.98)

    def test_to_dict_includes_confidence(self):
        update = ExtractionUpdate(
            Operation.SET, StateField.FREQUENCY, "every single day", confidence=0.82
        )
        d = update.to_dict()
        self.assertEqual(d["confidence"], 0.82)
        self.assertEqual(d["operation"], "SET")
        self.assertEqual(d["field"], "frequency")

    def test_result_to_dict_round_trips_confidence(self):
        result = ExtractionResult(
            message_type=MessageType.MEANINGFUL,
            updates=[
                ExtractionUpdate(Operation.ADD, StateField.EVIDENCE,
                                 "i have seen students lose marks", confidence=0.74),
            ],
        )
        d = result.to_dict()
        self.assertEqual(d["updates"][0]["confidence"], 0.74)


# ---------------------------------------------------------------------------
# ExtractionValidator — optional confidence parsing
# ---------------------------------------------------------------------------


class TestExtractionValidatorConfidence(unittest.TestCase):
    """Raw JSON may carry an optional confidence; invalid values are rejected."""

    def test_confidence_parsed_when_present(self):
        result = ExtractionValidator.validate({
            "message_type": "MEANINGFUL",
            "updates": [
                {"operation": "ADD", "field": "personas", "value": "students",
                 "confidence": 0.82},
            ],
        })
        self.assertEqual(result.updates[0].confidence, 0.82)

    def test_confidence_defaults_to_one_when_absent(self):
        result = ExtractionValidator.validate({
            "message_type": "MEANINGFUL",
            "updates": [
                {"operation": "ADD", "field": "personas", "value": "students"},
            ],
        })
        self.assertEqual(result.updates[0].confidence, 1.0)

    def test_out_of_range_confidence_rejected(self):
        for bad in (1.5, -0.1):
            with self.subTest(bad=bad):
                with self.assertRaises(Exception):
                    ExtractionValidator.validate({
                        "message_type": "MEANINGFUL",
                        "updates": [
                            {"operation": "ADD", "field": "personas", "value": "students",
                             "confidence": bad},
                        ],
                    })

    def test_non_numeric_confidence_rejected(self):
        for bad in ("abc", True, None, []):
            with self.subTest(bad=bad):
                with self.assertRaises(Exception):
                    ExtractionValidator.validate({
                        "message_type": "MEANINGFUL",
                        "updates": [
                            {"operation": "ADD", "field": "personas", "value": "students",
                             "confidence": bad},
                        ],
                    })


# ---------------------------------------------------------------------------
# Rule-based rule-strength confidence
# ---------------------------------------------------------------------------


class TestRuleBasedConfidence(unittest.TestCase):
    """Confidence mirrors rule strength for each extracted fact."""

    def test_problem_high_confidence(self):
        result = RuleBasedExtractor.extract("They struggle with forgetting medication.")
        self.assertEqual(result["pain_point"], "forgetting medication")
        self.assertEqual(result["confidences"]["pain_point"], CONFIDENCE_HIGH)

    def test_frequency_strong_cadence_confidence(self):
        result = RuleBasedExtractor.extract("It happens every single day.")
        self.assertEqual(result["frequency"], "every single day")
        self.assertEqual(result["confidences"]["frequency"], CONFIDENCE_STRONG)

    def test_frequency_quantitative_high_confidence(self):
        result = RuleBasedExtractor.extract("About three times a week.")
        self.assertEqual(result["frequency"], "three times a week")
        self.assertEqual(result["confidences"]["frequency"], CONFIDENCE_HIGH)

    def test_frequency_generic_token_medium_confidence(self):
        result = RuleBasedExtractor.extract("It happens often.")
        self.assertEqual(result["frequency"], "often")
        self.assertEqual(result["confidences"]["frequency"], CONFIDENCE_MEDIUM)

    def test_evidence_explicit_marker_confidence(self):
        result = RuleBasedExtractor.extract("I have seen students lose marks.")
        self.assertTrue(result["evidence"].startswith("i have seen students lose marks"))
        self.assertEqual(result["confidences"]["evidence"], CONFIDENCE_EVIDENCE)

    def test_ambiguous_motivation_low_confidence(self):
        result = RuleBasedExtractor.extract("This matters because it affects people.")
        self.assertEqual(result["confidences"]["motivation"], CONFIDENCE_LOW)

    def test_each_field_gets_its_own_confidence(self):
        result = RuleBasedExtractor.extract(
            "Elderly people struggle with forgetting medication every single day "
            "and I have seen many miss their doses."
        )
        conf = result["confidences"]
        self.assertEqual(conf["target_audience"], CONFIDENCE_HIGH)
        self.assertEqual(conf["pain_point"], CONFIDENCE_HIGH)
        self.assertEqual(conf["frequency"], CONFIDENCE_STRONG)
        self.assertEqual(conf["evidence"], CONFIDENCE_EVIDENCE)

    def test_no_confidences_when_no_facts(self):
        result = RuleBasedExtractor.extract("Hello!")
        self.assertNotIn("confidences", result)


# ---------------------------------------------------------------------------
# merge_extracted_to_state — confidence threads into ExtractionUpdate
# ---------------------------------------------------------------------------


class TestMergeConfidence(unittest.TestCase):
    """The rule-based dict's confidences flow into the typed ExtractionUpdate."""

    def test_confidence_threaded_into_update(self):
        state = ProjectState()
        with mock.patch(
            "extraction_pipeline.StateManager.apply_extraction"
        ) as apply_mock:
            merge_extracted_to_state(state, SessionData(), {
                "pain_point": "forgetting medication",
                "confidences": {"pain_point": 0.98},
            })
            result = apply_mock.call_args[0][0]
        self.assertIsInstance(result, ExtractionResult)
        self.assertEqual(result.updates[0].confidence, 0.98)
        self.assertEqual(result.updates[0].field, StateField.PROBLEMS)

    def test_default_confidence_when_confidences_absent(self):
        state = ProjectState()
        with mock.patch(
            "extraction_pipeline.StateManager.apply_extraction"
        ) as apply_mock:
            merge_extracted_to_state(state, SessionData(), {
                "frequency": "every single day",
            })
            result = apply_mock.call_args[0][0]
        self.assertEqual(result.updates[0].confidence, DEFAULT_CONFIDENCE)

    def test_state_unchanged_by_confidence_value(self):
        """The extracted VALUE (not the confidence) drives the state."""
        state = ProjectState()
        merge_extracted_to_state(state, SessionData(), {
            "target_audience": "elderly people",
            "confidences": {"target_audience": 0.98},
        })
        self.assertEqual(state.personas, ["elderly people"])
        self.assertNotIn("confidence", state.to_state_dict())


# ---------------------------------------------------------------------------
# ObjectiveEngine unaffected
# ---------------------------------------------------------------------------


class TestObjectiveEngineUnaffected(unittest.TestCase):
    """Confidence never feeds the Objective Engine (decision path unchanged)."""

    def test_objective_identical_with_and_without_confidence(self):
        def build(with_confidence):
            state = ProjectState()
            merge_extracted_to_state(state, SessionData(), {
                "target_audience": "elderly people",
                "pain_point": "forgetting medication",
                "confidences": (
                    {"target_audience": 0.98, "pain_point": 0.98}
                    if with_confidence else {}
                ),
            })
            return state

        state_conf = build(with_confidence=True)
        state_plain = build(with_confidence=False)
        self.assertEqual(state_conf.to_state_dict(), state_plain.to_state_dict())
        obj_conf = ObjectiveEngine().determine_next(state_conf)
        obj_plain = ObjectiveEngine().determine_next(state_plain)
        self.assertEqual(obj_conf, obj_plain)

    def test_confidence_not_part_of_objective_input(self):
        state = ProjectState()
        merge_extracted_to_state(state, SessionData(), {
            "target_audience": "students",
            "confidences": {"target_audience": 0.98},
        })
        obj = ObjectiveEngine().determine_next(state)
        self.assertEqual(obj.objective, Objective.PROBLEMS)


# ---------------------------------------------------------------------------
# Live pipeline — Developer Console view
# ---------------------------------------------------------------------------


class TestConfidenceLivePipeline(unittest.TestCase):
    """The Developer Console shows confidence for both extraction paths."""

    AUDIT_USER = "audit_conf"
    AUDIT_PROJ = "audit_conf_proj"

    def setUp(self):
        from session_manager import get_session_manager
        get_session_manager().reset_runtime_state()

    def tearDown(self):
        from session_manager import get_session_manager
        try:
            get_session_manager().reset_runtime_state()
        except Exception:
            pass

    def test_rule_based_fallback_shows_confidences(self):
        """When the LLM extraction is not meaningful, the rule-based fallback
        records per-fact confidence in the Developer Console."""
        import mentor

        with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama()}):
            with mock.patch(
                "mentor.MemoryExtractor.extract",
                return_value=ExtractionResult(MessageType.AMBIGUOUS, []),
            ):
                _reply, _session, _timing, diagnostics = mentor.process_mentor_turn(
                    "I want to help elderly people take their medications on time",
                    username=self.AUDIT_USER,
                    project_name=self.AUDIT_PROJ,
                    model_name="test-model",
                )
        extraction = diagnostics["Extraction"]
        self.assertIn("rule_based_extraction", extraction)
        conf = extraction["rule_based_extraction"]["confidences"]
        self.assertEqual(conf["target_audience"], CONFIDENCE_HIGH)
        self.assertEqual(
            extraction["rule_based_extraction"]["target_audience"], "elderly people"
        )

    def test_llm_path_updates_carry_default_confidence(self):
        """MEANINGFUL LLM updates surface confidence (default 1.0) in the
        Developer Console without touching the applied state. The message
        carries no deterministic facts for the current objective (PERSONAS),
        so the hybrid layer runs the LLM extraction."""
        import mentor

        updates = [
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "students"),
        ]
        with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama()}):
            with mock.patch(
                "mentor.MemoryExtractor.extract",
                return_value=ExtractionResult(MessageType.MEANINGFUL, updates),
            ):
                _reply, _session, _timing, diagnostics = mentor.process_mentor_turn(
                    "I really need some guidance",
                    username=self.AUDIT_USER,
                    project_name=self.AUDIT_PROJ,
                    model_name="test-model",
                )
        self.assertEqual(diagnostics["HybridExtraction"]["LLM Invoked"], "Yes")
        extraction = diagnostics["Extraction"]
        self.assertEqual(len(extraction["updates"]), 1)
        self.assertEqual(extraction["updates"][0]["field"], "personas")
        self.assertEqual(extraction["updates"][0]["confidence"], DEFAULT_CONFIDENCE)

    def test_diagnostics_are_json_serialisable_with_confidence(self):
        """The confidence values survive the X-Diagnostics wire contract."""
        import json
        import mentor

        with mock.patch.dict("sys.modules", {"ollama": _RaisingOllama()}):
            with mock.patch(
                "mentor.MemoryExtractor.extract",
                return_value=ExtractionResult(MessageType.AMBIGUOUS, []),
            ):
                _reply, _session, _timing, diagnostics = mentor.process_mentor_turn(
                    "I have seen elderly people struggle with medication daily",
                    username=self.AUDIT_USER,
                    project_name=self.AUDIT_PROJ,
                    model_name="test-model",
                )
        round_tripped = json.loads(json.dumps(diagnostics))
        conf = round_tripped["Extraction"]["rule_based_extraction"]["confidences"]
        self.assertIn("target_audience", conf)
        self.assertTrue(isinstance(conf["target_audience"], float))


if __name__ == "__main__":
    unittest.main(verbosity=2)
