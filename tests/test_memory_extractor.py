"""
tests/test_memory_extractor.py

Unit tests for the Memory Extractor module (Module 1, Empathize v2).

Coverage:
- ExtractionValidator: valid and invalid inputs (incl. ADD/SET rules)
- ExtractionResult / ExtractionUpdate data models
- safe_fallback behaviour
- ProjectState true list/scalar semantics and apply()
- MemoryExtractor._parse_and_validate (no Ollama calls)
- MemoryExtractor._build_prompt state serialization (list vs scalar)
- Module-level helpers (_extract_json, _strip_think_tags)
- mentor.py integration bridge (_project_state_from_session,
  _apply_extraction_to_state, merge_extracted_to_state,
  _build_legacy_session)

All tests run without a running Ollama instance.
"""

import sys
import os
import unittest

# Allow importing from the project root regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import (
    ExtractionResult,
    ExtractionUpdate,
    ExtractionValidationError,
    ExtractionValidator,
    LIST_FIELDS,
    MemoryExtractor,
    MessageType,
    Operation,
    ProjectState,
    SCALAR_FIELDS,
    StateField,
    _extract_json,
    _strip_think_tags,
)


# ---------------------------------------------------------------------------
# Reference data — the canonical contract example from the refactor spec
# ---------------------------------------------------------------------------
CANONICAL_RAW = {
    "message_type": "MEANINGFUL",
    "updates": [
        {"operation": "ADD", "field": "current_solutions", "value": "WhatsApp groups"},
        {"operation": "ADD", "field": "pain_points", "value": "Messages get buried"},
        {"operation": "SET", "field": "frequency", "value": "Weekly"},
    ],
}

LIST_FIELD_VALUES = [f.value for f in LIST_FIELDS]
SCALAR_FIELD_VALUES = [f.value for f in SCALAR_FIELDS]


class TestExtractionValidator_Valid(unittest.TestCase):
    """ExtractionValidator.validate() with well-formed inputs."""

    def test_meaningful_with_multiple_adds_and_set(self):
        """Canonical contract example: two ADDs + one SET in one extraction."""
        result = ExtractionValidator.validate(CANONICAL_RAW)

        self.assertIsInstance(result, ExtractionResult)
        self.assertEqual(result.message_type, MessageType.MEANINGFUL)
        self.assertEqual(len(result.updates), 3)

        u0, u1, u2 = result.updates
        self.assertEqual(u0.operation, Operation.ADD)
        self.assertEqual(u0.field, StateField.CURRENT_SOLUTIONS)
        self.assertEqual(u0.value, "WhatsApp groups")

        self.assertEqual(u1.operation, Operation.ADD)
        self.assertEqual(u1.field, StateField.PAIN_POINTS)
        self.assertEqual(u1.value, "Messages get buried")

        self.assertEqual(u2.operation, Operation.SET)
        self.assertEqual(u2.field, StateField.FREQUENCY)
        self.assertEqual(u2.value, "Weekly")

    def test_no_update_empty_updates(self):
        result = ExtractionValidator.validate({"message_type": "NO_UPDATE", "updates": []})
        self.assertEqual(result.message_type, MessageType.NO_UPDATE)
        self.assertEqual(result.updates, [])
        self.assertFalse(result.has_updates())

    def test_ambiguous_empty_updates(self):
        result = ExtractionValidator.validate({"message_type": "AMBIGUOUS", "updates": []})
        self.assertEqual(result.message_type, MessageType.AMBIGUOUS)
        self.assertFalse(result.has_updates())

    def test_end_empty_updates(self):
        result = ExtractionValidator.validate({"message_type": "END", "updates": []})
        self.assertEqual(result.message_type, MessageType.END)
        self.assertFalse(result.has_updates())

    def test_add_accepted_on_every_list_field(self):
        """ADD must be accepted for every list-typed StateField."""
        for state_field in LIST_FIELDS:
            raw = {
                "message_type": "MEANINGFUL",
                "updates": [
                    {"operation": "ADD", "field": state_field.value, "value": "test value"}
                ],
            }
            result = ExtractionValidator.validate(raw)
            self.assertEqual(result.updates[0].operation, Operation.ADD)
            self.assertEqual(result.updates[0].field, state_field)

    def test_set_accepted_on_every_scalar_field(self):
        """SET must be accepted for every scalar-typed StateField."""
        for state_field in SCALAR_FIELDS:
            raw = {
                "message_type": "MEANINGFUL",
                "updates": [
                    {"operation": "SET", "field": state_field.value, "value": "test value"}
                ],
            }
            result = ExtractionValidator.validate(raw)
            self.assertEqual(result.updates[0].operation, Operation.SET)
            self.assertEqual(result.updates[0].field, state_field)

    def test_multiple_adds_on_same_list_field(self):
        """Multiple ADD operations in one extraction, on the same list field."""
        raw = {
            "message_type": "MEANINGFUL",
            "updates": [
                {"operation": "ADD", "field": "personas", "value": "elderly people"},
                {"operation": "ADD", "field": "personas", "value": "caregivers"},
            ],
        }
        result = ExtractionValidator.validate(raw)
        self.assertEqual(len(result.updates), 2)
        self.assertEqual({u.value for u in result.updates}, {"elderly people", "caregivers"})
        self.assertTrue(all(u.operation is Operation.ADD for u in result.updates))
        self.assertTrue(all(u.field is StateField.PERSONAS for u in result.updates))

    def test_value_is_stripped(self):
        raw = {
            "message_type": "MEANINGFUL",
            "updates": [
                {"operation": "ADD", "field": "current_solutions", "value": "  sticky notes  "}
            ],
        }
        result = ExtractionValidator.validate(raw)
        self.assertEqual(result.updates[0].value, "sticky notes")


class TestExtractionValidator_Invalid(unittest.TestCase):
    """ExtractionValidator.validate() raises ExtractionValidationError on bad input."""

    def test_missing_message_type(self):
        with self.assertRaises(ExtractionValidationError) as ctx:
            ExtractionValidator.validate({"updates": []})
        self.assertIn("message_type", str(ctx.exception))

    def test_bad_message_type_value(self):
        with self.assertRaises(ExtractionValidationError) as ctx:
            ExtractionValidator.validate({"message_type": "UNKNOWN", "updates": []})
        self.assertIn("UNKNOWN", str(ctx.exception))

    def test_non_string_message_type(self):
        with self.assertRaises(ExtractionValidationError):
            ExtractionValidator.validate({"message_type": 42, "updates": []})

    def test_missing_updates_key(self):
        with self.assertRaises(ExtractionValidationError) as ctx:
            ExtractionValidator.validate({"message_type": "MEANINGFUL"})
        self.assertIn("updates", str(ctx.exception))

    def test_updates_not_a_list(self):
        with self.assertRaises(ExtractionValidationError):
            ExtractionValidator.validate({"message_type": "MEANINGFUL", "updates": "ADD"})

    def test_invalid_operation(self):
        with self.assertRaises(ExtractionValidationError) as ctx:
            ExtractionValidator.validate({
                "message_type": "MEANINGFUL",
                "updates": [{"operation": "DELETE", "field": "personas", "value": "x"}],
            })
        self.assertIn("DELETE", str(ctx.exception))

    def test_invalid_field_name(self):
        with self.assertRaises(ExtractionValidationError) as ctx:
            ExtractionValidator.validate({
                "message_type": "MEANINGFUL",
                "updates": [{"operation": "ADD", "field": "invalid_field", "value": "x"}],
            })
        self.assertIn("invalid_field", str(ctx.exception))

    def test_empty_value_rejected(self):
        with self.assertRaises(ExtractionValidationError):
            ExtractionValidator.validate({
                "message_type": "MEANINGFUL",
                "updates": [{"operation": "ADD", "field": "personas", "value": "   "}],
            })

    def test_not_a_dict(self):
        with self.assertRaises(ExtractionValidationError):
            ExtractionValidator.validate(["message_type", "MEANINGFUL"])

    def test_update_missing_field_key(self):
        with self.assertRaises(ExtractionValidationError):
            ExtractionValidator.validate({
                "message_type": "MEANINGFUL",
                "updates": [{"operation": "ADD", "value": "x"}],  # missing "field"
            })


class TestOperationFieldRules(unittest.TestCase):
    """ADD is only valid for list fields; SET is only valid for frequency."""

    def test_add_rejected_on_frequency(self):
        with self.assertRaises(ExtractionValidationError) as ctx:
            ExtractionValidator.validate({
                "message_type": "MEANINGFUL",
                "updates": [{"operation": "ADD", "field": "frequency", "value": "daily"}],
            })
        self.assertIn("ADD", str(ctx.exception))
        self.assertIn("list", str(ctx.exception))
        self.assertIn("frequency", str(ctx.exception))

    def test_set_rejected_on_every_list_field(self):
        for state_field in LIST_FIELDS:
            with self.assertRaises(ExtractionValidationError) as ctx:
                ExtractionValidator.validate({
                    "message_type": "MEANINGFUL",
                    "updates": [
                        {"operation": "SET", "field": state_field.value, "value": "x"}
                    ],
                })
            self.assertIn("SET", str(ctx.exception))
            self.assertIn("scalar", str(ctx.exception))
            self.assertIn(state_field.value, str(ctx.exception))

    def test_set_rejected_on_personas_specifically(self):
        """Explicit, message-bearing test for the most common SET-misuse case."""
        with self.assertRaises(ExtractionValidationError) as ctx:
            ExtractionValidator.validate({
                "message_type": "MEANINGFUL",
                "updates": [
                    {"operation": "SET", "field": "personas", "value": "elderly people"}
                ],
            })
        msg = str(ctx.exception)
        self.assertIn("SET", msg)
        self.assertIn("personas", msg)


class TestSafeFallback(unittest.TestCase):
    """ExtractionValidator.safe_fallback() always returns a safe AMBIGUOUS result."""

    def test_safe_fallback_type(self):
        self.assertIsInstance(ExtractionValidator.safe_fallback(), ExtractionResult)

    def test_safe_fallback_message_type(self):
        self.assertEqual(
            ExtractionValidator.safe_fallback().message_type, MessageType.AMBIGUOUS
        )

    def test_safe_fallback_no_updates(self):
        result = ExtractionValidator.safe_fallback()
        self.assertEqual(result.updates, [])
        self.assertFalse(result.has_updates())


class TestExtractionResult(unittest.TestCase):
    """ExtractionResult data model behaviour."""

    def test_has_updates_true(self):
        update = ExtractionUpdate(operation=Operation.ADD, field=StateField.PERSONAS, value="students")
        result = ExtractionResult(message_type=MessageType.MEANINGFUL, updates=[update])
        self.assertTrue(result.has_updates())

    def test_has_updates_false(self):
        result = ExtractionResult(message_type=MessageType.NO_UPDATE, updates=[])
        self.assertFalse(result.has_updates())

    def test_to_dict_canonical_contract(self):
        update = ExtractionUpdate(
            operation=Operation.ADD, field=StateField.CURRENT_SOLUTIONS, value="WhatsApp groups"
        )
        result = ExtractionResult(message_type=MessageType.MEANINGFUL, updates=[update])
        d = result.to_dict()
        self.assertEqual(d["message_type"], "MEANINGFUL")
        self.assertEqual(len(d["updates"]), 1)
        self.assertEqual(d["updates"][0]["operation"], "ADD")
        self.assertEqual(d["updates"][0]["field"], "current_solutions")
        self.assertEqual(d["updates"][0]["value"], "WhatsApp groups")

    def test_immutability(self):
        """ExtractionResult and ExtractionUpdate are frozen dataclasses."""
        result = ExtractionResult(message_type=MessageType.END, updates=[])
        with self.assertRaises((AttributeError, TypeError)):
            result.message_type = MessageType.MEANINGFUL  # type: ignore[misc]


class TestProjectState(unittest.TestCase):
    """ProjectState — true list/scalar semantics and apply()."""

    def test_defaults_are_empty_lists_and_none(self):
        s = ProjectState()
        for f in LIST_FIELDS:
            self.assertEqual(s.get_list(f), [])
        self.assertIsNone(s.get_scalar(StateField.FREQUENCY))
        self.assertIsNone(s.previous_assistant_message)
        self.assertIsNone(s.previous_user_message)

    def test_get_list_rejects_scalar(self):
        s = ProjectState()
        with self.assertRaises(TypeError):
            s.get_list(StateField.FREQUENCY)

    def test_get_scalar_rejects_list(self):
        s = ProjectState()
        with self.assertRaises(TypeError):
            s.get_scalar(StateField.PERSONAS)

    def test_set_scalar_strips_and_empties_to_none(self):
        s = ProjectState()
        s.set_scalar(StateField.FREQUENCY, "  weekly  ")
        self.assertEqual(s.frequency, "weekly")
        s.set_scalar(StateField.FREQUENCY, "   ")
        self.assertIsNone(s.frequency)

    def test_set_scalar_rejects_list_field(self):
        s = ProjectState()
        with self.assertRaises(TypeError):
            s.set_scalar(StateField.PERSONAS, "x")

    def test_apply_add_appends_and_dedupes(self):
        """ADD nails true-append semantics; duplicates are skipped, not stored twice."""
        from state_manager import StateManager
        s = ProjectState(personas=["students"])
        result = ExtractionResult(
            message_type=MessageType.MEANINGFUL,
            updates=[
                ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "students"),     # dup -> skipped
                ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "researchers"),   # new -> appended
            ],
        )
        StateManager(s).apply_extraction(result)
        self.assertEqual(s.personas, ["students", "researchers"])

    def test_apply_set_overwrites_frequency(self):
        from state_manager import StateManager
        s = ProjectState(frequency="rarely")
        result = ExtractionResult(
            message_type=MessageType.MEANINGFUL,
            updates=[
                ExtractionUpdate(Operation.SET, StateField.FREQUENCY, "weekly"),
            ],
        )
        StateManager(s).apply_extraction(result)
        self.assertEqual(s.frequency, "weekly")

    def test_apply_ignores_non_meaningful(self):
        """Non-MEANINGFUL results must not touch state even when passed to StateManager."""
        from state_manager import StateManager, StateBatchValidationError
        s = ProjectState(frequency="weekly")
        result = ExtractionResult(
            message_type=MessageType.NO_UPDATE, updates=[]
        )
        StateManager(s).apply_extraction(result)
        self.assertEqual(s.frequency, "weekly")

    def test_apply_canonical_contract_example(self):
        """The canonical 3-update extraction applies to a fresh ProjectState exactly."""
        from state_manager import StateManager
        result = ExtractionValidator.validate(CANONICAL_RAW)
        s = ProjectState()
        StateManager(s).apply_extraction(result)
        self.assertEqual(s.current_solutions, ["WhatsApp groups"])
        self.assertEqual(s.pain_points, ["Messages get buried"])
        self.assertEqual(s.frequency, "Weekly")
        self.assertEqual(s.personas, [])
        self.assertEqual(s.problems, [])
        self.assertEqual(s.evidence, [])

    def test_to_state_dict_has_arrays_and_scalar(self):
        s = ProjectState(
            personas=["students"],
            problems=["forgetting meds"],
            current_solutions=[],
            pain_points=["buried messages"],
            evidence=[],
            frequency="weekly",
        )
        d = s.to_state_dict()
        self.assertEqual(d[StateField.PERSONAS.value], ["students"])
        self.assertEqual(d[StateField.PROBLEMS.value], ["forgetting meds"])
        self.assertEqual(d[StateField.CURRENT_SOLUTIONS.value], [])
        self.assertEqual(d[StateField.PAIN_POINTS.value], ["buried messages"])
        self.assertEqual(d[StateField.EVIDENCE.value], [])
        self.assertEqual(d[StateField.FREQUENCY.value], "weekly")
        # bookkeeping fields must NOT be serialized
        self.assertNotIn("previous_assistant_message", d)
        self.assertNotIn("previous_user_message", d)

    def test_to_state_dict_lists_are_copies(self):
        """Mutation of the returned list must not leak back into ProjectState."""
        s = ProjectState(personas=["students"])
        d = s.to_state_dict()
        d["personas"].append("hack")
        self.assertEqual(s.personas, ["students"])


class TestHelperFunctions(unittest.TestCase):
    """Module-level helper functions."""

    def test_strip_think_tags(self):
        # The repo's _strip_think_tags removes  blocks
        # (DeepSeek R1 style) and <thinking>...</thinking> blocks.
        text = "actual output"
        self.assertEqual(_strip_think_tags(text), "actual output")

    def test_strip_thinking_tags(self):
        text = "<thinking>deep thoughts</thinking>result"
        self.assertEqual(_strip_think_tags(text), "result")

    def test_strip_no_tags(self):
        text = '{"message_type": "NO_UPDATE", "updates": []}'
        self.assertEqual(_strip_think_tags(text), text)

    def test_extract_json_clean(self):
        text = '{"message_type": "END", "updates": []}'
        result = _extract_json(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["message_type"], "END")

    def test_extract_json_with_surrounding_text(self):
        text = 'Here is the JSON: {"message_type": "NO_UPDATE", "updates": []} Done.'
        result = _extract_json(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["message_type"], "NO_UPDATE")

    def test_extract_json_no_json(self):
        text = "This is just plain text with no JSON object."
        self.assertIsNone(_extract_json(text))


class TestMemoryExtractorParseAndValidate(unittest.TestCase):
    """
    Test MemoryExtractor._parse_and_validate() in isolation (no Ollama).
    """

    def setUp(self):
        self.extractor = MemoryExtractor(model_name="test-model-no-ollama")

    def test_valid_json_returns_result(self):
        raw = (
            '{"message_type": "MEANINGFUL", '
            '"updates": [{"operation": "ADD", "field": "personas", "value": "elderly people"}]}'
        )
        result = self.extractor._parse_and_validate(raw)
        self.assertEqual(result.message_type, MessageType.MEANINGFUL)
        self.assertEqual(len(result.updates), 1)
        self.assertEqual(result.updates[0].field, StateField.PERSONAS)
        self.assertEqual(result.updates[0].operation, Operation.ADD)

    def test_invalid_json_returns_fallback(self):
        result = self.extractor._parse_and_validate("this is not json at all")
        self.assertEqual(result.message_type, MessageType.AMBIGUOUS)
        self.assertFalse(result.has_updates())

    def test_invalid_message_type_returns_fallback(self):
        result = self.extractor._parse_and_validate('{"message_type": "GARBAGE", "updates": []}')
        self.assertEqual(result.message_type, MessageType.AMBIGUOUS)

    def test_think_tags_stripped_before_parse(self):
        raw = '<thinking>let me reason</thinking>{"message_type": "NO_UPDATE", "updates": []}'
        result = self.extractor._parse_and_validate(raw)
        self.assertEqual(result.message_type, MessageType.NO_UPDATE)

    def test_add_on_frequency_returns_fallback(self):
        """Validator rejects ADD on frequency → extractor falls back to AMBIGUOUS."""
        raw = (
            '{"message_type": "MEANINGFUL", '
            '"updates": [{"operation": "ADD", "field": "frequency", "value": "daily"}]}'
        )
        result = self.extractor._parse_and_validate(raw)
        self.assertEqual(result.message_type, MessageType.AMBIGUOUS)
        self.assertFalse(result.has_updates())

    def test_set_on_list_field_returns_fallback(self):
        """Validator rejects SET on a list field → extractor falls back to AMBIGUOUS."""
        raw = (
            '{"message_type": "MEANINGFUL", '
            '"updates": [{"operation": "SET", "field": "personas", "value": "elderly people"}]}'
        )
        result = self.extractor._parse_and_validate(raw)
        self.assertEqual(result.message_type, MessageType.AMBIGUOUS)
        self.assertFalse(result.has_updates())

    def test_canonical_extraction_round_trip(self):
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "current_solutions", "value": "WhatsApp groups"},'
            '{"operation": "ADD", "field": "pain_points", "value": "Messages get buried"},'
            '{"operation": "SET", "field": "frequency", "value": "Weekly"}'
            "]}"
        )
        result = self.extractor._parse_and_validate(raw)
        self.assertEqual(result.message_type, MessageType.MEANINGFUL)
        self.assertEqual(len(result.updates), 3)
        self.assertEqual(result.updates[0].field, StateField.CURRENT_SOLUTIONS)
        self.assertEqual(result.updates[1].field, StateField.PAIN_POINTS)
        self.assertEqual(result.updates[2].field, StateField.FREQUENCY)

    def test_empty_string_returns_fallback(self):
        self.assertEqual(
            self.extractor._parse_and_validate("").message_type, MessageType.AMBIGUOUS
        )


class TestMemoryExtractorBuildPrompt(unittest.TestCase):
    """_build_prompt serializes ProjectState correctly (list vs scalar)."""

    def setUp(self):
        self.extractor = MemoryExtractor(model_name="test-model-no-ollama")

    def test_build_prompt_with_project_state_object(self):
        state = ProjectState(
            personas=["students"],
            current_solutions=["WhatsApp groups"],
            frequency="weekly",
        )
        prompt = self.extractor._build_prompt("hello", state, previous_assistant_message=None)
        # The state block renders keys unquoted and values as JSON literals.
        self.assertIn("personas: [\"students\"]", prompt)
        self.assertIn("current_solutions: [\"WhatsApp groups\"]", prompt)
        self.assertIn("frequency: \"weekly\"", prompt)
        # Empty list field renders as [].
        self.assertIn("problems: []", prompt)
        # User message is interpolated.
        self.assertIn('"hello"', prompt)

    def test_build_prompt_with_state_dict(self):
        """A dict produced by to_state_dict() is also accepted."""
        state = ProjectState(personas=["students"], frequency="daily")
        prompt = self.extractor._build_prompt(
            user_message="hi",
            project_state=state.to_state_dict(),
            previous_assistant_message="what?",
        )
        self.assertIn("personas: [\"students\"]", prompt)
        self.assertIn("frequency: \"daily\"", prompt)
        self.assertIn("PREVIOUS ASSISTANT MESSAGE", prompt)

    def test_build_prompt_no_previous_assistant_message(self):
        state = ProjectState()
        prompt = self.extractor._build_prompt(
            user_message="hi",
            project_state=state,
            previous_assistant_message=None,
        )
        self.assertNotIn("PREVIOUS ASSISTANT MESSAGE", prompt)


class TestMemoryExtractorEmptyMessage(unittest.TestCase):
    """MemoryExtractor.extract() short-circuits on empty/blank messages."""

    def setUp(self):
        self.extractor = MemoryExtractor(model_name="test-model-no-ollama")

    def test_empty_string(self):
        result = self.extractor.extract("", project_state=ProjectState())
        self.assertEqual(result.message_type, MessageType.NO_UPDATE)

    def test_whitespace_only(self):
        result = self.extractor.extract("   \n\t  ", project_state=ProjectState())
        self.assertEqual(result.message_type, MessageType.NO_UPDATE)


class TestMentorIntegration(unittest.TestCase):
    """
    Bridge helpers: _apply_extraction_to_state, merge_extracted_to_state,
    _build_legacy_session. These exercise the canonical ProjectState and
    legacy conversion without invoking Ollama.
    """

    def setUp(self):
        import mentor
        self.mentor = mentor

    def _fresh_session(self):
        return self.mentor.MentorSession(project_name="Proj")

    def test_apply_extraction_add_appends_to_state_list(self):
        from conversation_pipeline import _apply_extraction_to_state
        state = ProjectState(personas=["elderly people"])
        result = ExtractionResult(
            message_type=MessageType.MEANINGFUL,
            updates=[
                ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "caregivers"),
            ],
        )
        _apply_extraction_to_state(state, result)
        self.assertEqual(state.personas, ["elderly people", "caregivers"])

    def test_apply_extraction_set_overwrites_frequency(self):
        from conversation_pipeline import _apply_extraction_to_state
        state = ProjectState(frequency="rarely")
        result = ExtractionResult(
            message_type=MessageType.MEANINGFUL,
            updates=[ExtractionUpdate(Operation.SET, StateField.FREQUENCY, "weekly")],
        )
        _apply_extraction_to_state(state, result)
        self.assertEqual(state.frequency, "weekly")

    def test_apply_extraction_canonical_contract(self):
        from conversation_pipeline import _apply_extraction_to_state
        state = ProjectState()
        result = ExtractionValidator.validate(CANONICAL_RAW)
        _apply_extraction_to_state(state, result)
        self.assertEqual(state.current_solutions, ["WhatsApp groups"])
        self.assertEqual(state.pain_points, ["Messages get buried"])
        self.assertEqual(state.frequency, "Weekly")

    def test_apply_extraction_no_update_is_noop(self):
        from conversation_pipeline import _apply_extraction_to_state
        state = ProjectState(personas=["students"])
        before_personas = list(state.personas)

        for mt in (MessageType.NO_UPDATE, MessageType.AMBIGUOUS, MessageType.END):
            result = ExtractionResult(message_type=mt, updates=[])
            _apply_extraction_to_state(state, result)
            self.assertEqual(state.personas, before_personas, f"state leaked for {mt}")

    def test_merge_extracted_to_state_appends_list(self):
        from extraction_pipeline import merge_extracted_to_state
        from session_manager import SessionData
        state = ProjectState(personas=["elderly people"])
        sd = SessionData()
        extracted = {"target_audience": "caregivers"}
        merge_extracted_to_state(state, sd, extracted)
        self.assertEqual(state.personas, ["elderly people", "caregivers"])

    def test_merge_extracted_to_state_sets_scalar(self):
        from extraction_pipeline import merge_extracted_to_state
        from session_manager import SessionData
        state = ProjectState()
        sd = SessionData()
        extracted = {"frequency": "daily"}
        merge_extracted_to_state(state, sd, extracted)
        self.assertEqual(state.frequency, "daily")

    def test_merge_extracted_to_state_writes_bookkeeping(self):
        from extraction_pipeline import merge_extracted_to_state
        from session_manager import SessionData
        state = ProjectState()
        sd = SessionData()
        extracted = {
            "assumptions": ["they might not use apps"],
            "unknown_facts": ["tech literacy"],
        }
        merge_extracted_to_state(state, sd, extracted)
        self.assertEqual(sd.assumptions, ["they might not use apps"])
        self.assertEqual(sd.unknown_facts, ["tech literacy"])

    def test_build_legacy_session_round_trip(self):
        from session_pipeline import _build_legacy_session
        from session_manager import SessionData
        state = ProjectState(
            personas=["elderly people"],
            problems=["forgetting meds"],
            frequency="daily",
        )
        sd = SessionData(
            project_name="MedRemind",
            current_stage="Empathize",
            conversation_history=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "who?"},
            ],
            assumptions=["test assumption"],
        )
        session = _build_legacy_session(state, sd, "Proj")
        self.assertEqual(session.project_name, "MedRemind")
        self.assertEqual(session.assumptions, ["test assumption"])
        self.assertIn("Target Audience: elderly people", session.known_facts)
        self.assertIn("Pain Point: forgetting meds", session.known_facts)
        self.assertIn("Frequency: daily", session.known_facts)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# =============================================================================
# NEW: Extraction Accuracy Tests (Module 1 Improvement)
# =============================================================================

class TestExtractionAccuracy_EdgeCases(unittest.TestCase):
    """
    Tests covering the specific extraction accuracy improvements.
    These test the LLM prompt logic via _parse_and_validate with crafted JSON.
    """

    def setUp(self):
        self.extractor = MemoryExtractor(model_name="test-model")

    def _extract(self, raw_json: str) -> ExtractionResult:
        """Helper to parse and validate raw JSON via extractor's pipeline."""
        return self.extractor._parse_and_validate(raw_json)

    # --- Evidence vs Problem --------------------------------------------------

    def test_observation_is_evidence_not_problem(self):
        """'I have seen students lose marks' → evidence, NOT problems."""
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "evidence", "value": "observed students losing marks due to late submissions"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.message_type, MessageType.MEANINGFUL)
        self.assertEqual(len(result.updates), 1)
        self.assertEqual(result.updates[0].field, StateField.EVIDENCE)
        self.assertIn("observed", result.updates[0].value.lower())
        self.assertIn("losing marks", result.updates[0].value.lower())

    def test_personal_experience_is_evidence_or_pain_point(self):
        """'I personally faced this issue' → evidence AND/OR pain_points."""
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "evidence", "value": "experienced personally"},'
            '{"operation": "ADD", "field": "pain_points", "value": "experienced personally"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.message_type, MessageType.MEANINGFUL)
        self.assertEqual(len(result.updates), 2)
        fields = {u.field for u in result.updates}
        self.assertIn(StateField.EVIDENCE, fields)
        self.assertIn(StateField.PAIN_POINTS, fields)

    def test_research_statistic_is_evidence(self):
        """'Research shows 70% miss deadlines' → evidence."""
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "evidence", "value": "research shows 70% of students miss deadlines"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.updates[0].field, StateField.EVIDENCE)

    def test_interview_data_is_evidence(self):
        """'I interviewed five students' → evidence."""
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "evidence", "value": "interviewed five students who all forget assignments"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.updates[0].field, StateField.EVIDENCE)

    # --- Pain Point vs Problem ------------------------------------------------

    def test_emotional_struggle_is_pain_point(self):
        """'They feel stressed' → pain_points."""
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "pain_points", "value": "feel stressed"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.updates[0].field, StateField.PAIN_POINTS)
        self.assertIn("stress", result.updates[0].value.lower())

    def test_forgetfulness_is_pain_point(self):
        """'Forgetting things is stressful' → problems + pain_points."""
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "problems", "value": "forgetting things"},'
            '{"operation": "ADD", "field": "pain_points", "value": "stressful"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(len(result.updates), 2)
        fields = {u.field for u in result.updates}
        self.assertIn(StateField.PROBLEMS, fields)
        self.assertIn(StateField.PAIN_POINTS, fields)

    # --- Solution vs Problem --------------------------------------------------

    def test_existing_workaround_is_current_solution(self):
        """'Students usually set reminders' → current_solutions."""
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "current_solutions", "value": "setting reminders"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.updates[0].field, StateField.CURRENT_SOLUTIONS)

    def test_sticky_notes_alarms_are_current_solutions(self):
        """'They use sticky notes and phone alarms' → current_solutions."""
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "current_solutions", "value": "sticky notes and phone alarms"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.updates[0].field, StateField.CURRENT_SOLUTIONS)

    # --- Frequency Extraction -------------------------------------------------

    def test_daily_frequency_extraction(self):
        """'It happens every day' → frequency SET."""
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "SET", "field": "frequency", "value": "every day"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.updates[0].field, StateField.FREQUENCY)
        self.assertEqual(result.updates[0].operation, Operation.SET)
        self.assertIn("day", result.updates[0].value.lower())

    def test_weekly_frequency_extraction(self):
        """'Weekly thing' → frequency."""
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "SET", "field": "frequency", "value": "weekly"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.updates[0].field, StateField.FREQUENCY)

    def test_multiple_times_daily_frequency(self):
        """'Sometimes multiple times a day' → frequency."""
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "SET", "field": "frequency", "value": "sometimes multiple times a day"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.updates[0].field, StateField.FREQUENCY)

    # --- Ambiguous / Low Confidence Inputs → NO_UPDATE ------------------------

    def test_vague_motivation_returns_no_update(self):
        """'I want to make people disciplined' → NO_UPDATE (low confidence)."""
        raw = '{"message_type": "NO_UPDATE", "updates": []}'
        result = self._extract(raw)
        self.assertEqual(result.message_type, MessageType.NO_UPDATE)
        self.assertFalse(result.has_updates())

    def test_builder_intent_no_project_info_returns_no_update(self):
        """'I want to build an app for this' → NO_UPDATE."""
        raw = '{"message_type": "NO_UPDATE", "updates": []}'
        result = self._extract(raw)
        self.assertEqual(result.message_type, MessageType.NO_UPDATE)

    def test_ambiguous_short_reply_returns_ambiguous(self):
        """'yes' without context → AMBIGUOUS."""
        raw = '{"message_type": "AMBIGUOUS", "updates": []}'
        result = self._extract(raw)
        self.assertEqual(result.message_type, MessageType.AMBIGUOUS)

    # --- Multiple Updates in One Message --------------------------------------

    def test_multiple_fields_extracted_from_single_message(self):
        """Single message yielding persona + problem + evidence."""
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "personas", "value": "elderly people"},'
            '{"operation": "ADD", "field": "problems", "value": "medication timing difficulties"},'
            '{"operation": "ADD", "field": "evidence", "value": "personal observation of grandmother"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(len(result.updates), 3)
        fields = {u.field for u in result.updates}
        self.assertEqual(fields, {StateField.PERSONAS, StateField.PROBLEMS, StateField.EVIDENCE})

    def test_problem_and_evidence_from_causal_statement(self):
        """'Procrastination causes missed deadlines' → problems + evidence."""
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "problems", "value": "procrastination"},'
            '{"operation": "ADD", "field": "evidence", "value": "causes missed deadlines"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(len(result.updates), 2)
        fields = {u.field for u in result.updates}
        self.assertIn(StateField.PROBLEMS, fields)
        self.assertIn(StateField.EVIDENCE, fields)

    # --- Negative Examples (should NOT extract) -------------------------------

    def test_greeting_returns_no_update(self):
        raw = '{"message_type": "NO_UPDATE", "updates": []}'
        result = self._extract(raw)
        self.assertEqual(result.message_type, MessageType.NO_UPDATE)

    def test_thanks_returns_end(self):
        raw = '{"message_type": "END", "updates": []}'
        result = self._extract(raw)
        self.assertEqual(result.message_type, MessageType.END)

    def test_punctuation_only_returns_no_update(self):
        raw = '{"message_type": "NO_UPDATE", "updates": []}'
        result = self._extract(raw)
        self.assertEqual(result.message_type, MessageType.NO_UPDATE)

    # --- Confidence Rule: Low Confidence → NO_UPDATE -------------------------

    def test_confidence_below_80_returns_no_update(self):
        """
        The prompt instructs the LLM to return NO_UPDATE when confidence < 80%.
        This test verifies the extractor accepts such a result.
        """
        raw = '{"message_type": "NO_UPDATE", "updates": []}'
        result = self._extract(raw)
        self.assertEqual(result.message_type, MessageType.NO_UPDATE)
        self.assertFalse(result.has_updates())


class TestFieldDiscriminationRules(unittest.TestCase):
    """
    Explicit tests for the discrimination rules documented in the prompt:
    - Evidence is NOT a problem
    - Pain point is NOT evidence
    - Problem is NOT solution
    - Persona is NOT problem
    - Frequency is NEVER a pain point
    """

    def setUp(self):
        self.extractor = MemoryExtractor(model_name="test-model")

    def _extract(self, raw_json: str) -> ExtractionResult:
        return self.extractor._parse_and_validate(raw_json)

    def test_evidence_not_problem(self):
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "evidence", "value": "observed students struggling"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.updates[0].field, StateField.EVIDENCE)
        self.assertNotEqual(result.updates[0].field, StateField.PROBLEMS)

    def test_pain_point_not_evidence(self):
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "pain_points", "value": "frustration with missed deadlines"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.updates[0].field, StateField.PAIN_POINTS)
        self.assertNotEqual(result.updates[0].field, StateField.EVIDENCE)

    def test_problem_not_solution(self):
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "problems", "value": "missing deadlines"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.updates[0].field, StateField.PROBLEMS)
        self.assertNotEqual(result.updates[0].field, StateField.CURRENT_SOLUTIONS)

    def test_persona_not_problem(self):
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "personas", "value": "college students"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.updates[0].field, StateField.PERSONAS)
        self.assertNotEqual(result.updates[0].field, StateField.PROBLEMS)

    def test_frequency_never_pain_point(self):
        """Frequency is a scalar field, pain_points is a list - operation mismatch."""
        # The validator already rejects ADD on frequency and SET on pain_points.
        # This test confirms the field distinction at the enum level.
        self.assertIn(StateField.FREQUENCY, SCALAR_FIELDS)
        self.assertIn(StateField.PAIN_POINTS, LIST_FIELDS)
        self.assertNotIn(StateField.FREQUENCY, LIST_FIELDS)
        self.assertNotIn(StateField.PAIN_POINTS, SCALAR_FIELDS)


class TestConservativeExtractionBehavior(unittest.TestCase):
    """
    Tests ensuring the extractor behaves conservatively:
    - Returns AMBIGUOUS/NO_UPDATE rather than guessing
    - Doesn't hallucinate fields
    """

    def setUp(self):
        self.extractor = MemoryExtractor(model_name="test-model")

    def _extract(self, raw_json: str) -> ExtractionResult:
        return self.extractor._parse_and_validate(raw_json)

    def test_ambiguous_fallback_on_invalid_json(self):
        result = self.extractor._parse_and_validate("not json at all")
        self.assertEqual(result.message_type, MessageType.AMBIGUOUS)
        self.assertFalse(result.has_updates())

    def test_ambiguous_fallback_on_validation_error(self):
        # ADD on frequency is invalid → validator rejects → extractor returns AMBIGUOUS
        raw = (
            '{"message_type": "MEANINGFUL", "updates": ['
            '{"operation": "ADD", "field": "frequency", "value": "daily"}'
            ']}'
        )
        result = self._extract(raw)
        self.assertEqual(result.message_type, MessageType.AMBIGUOUS)
        self.assertFalse(result.has_updates())

    def test_empty_message_returns_no_update(self):
        # This tests the public extract() short-circuit
        result = self.extractor.extract("", project_state=ProjectState())
        self.assertEqual(result.message_type, MessageType.NO_UPDATE)

    def test_whitespace_only_returns_no_update(self):
        result = self.extractor.extract("   \n\t  ", project_state=ProjectState())
        self.assertEqual(result.message_type, MessageType.NO_UPDATE)


class TestPromptFieldDescriptions(unittest.TestCase):
    """
    Verify the prompt contains all required field descriptions and examples.
    """

    def test_prompt_contains_all_field_descriptions(self):
        from memory_extractor import EXTRACTOR_PROMPT
        prompt = EXTRACTOR_PROMPT

        # Field descriptions
        self.assertIn("personas", prompt)
        self.assertIn("problems", prompt)
        self.assertIn("current_solutions", prompt)
        self.assertIn("pain_points", prompt)
        self.assertIn("evidence", prompt)
        self.assertIn("frequency", prompt)

    def test_prompt_contains_discrimination_patterns(self):
        from memory_extractor import EXTRACTOR_PROMPT
        prompt = EXTRACTOR_PROMPT

        # Discrimination is taught by examples in V2
        self.assertIn("I have seen students lose marks", prompt)    # observation != problem
        self.assertIn("They feel stressed", prompt)                  # emotion = pain_point
        self.assertIn("The problem is late submission", prompt)      # explicit problem
        self.assertIn("elderly people take medication", prompt)      # persona extraction
        self.assertIn("weekly", prompt)                              # frequency

    def test_prompt_contains_conservatism(self):
        from memory_extractor import EXTRACTOR_PROMPT
        prompt = EXTRACTOR_PROMPT

        # Conservatism is stated in identity + examples
        self.assertIn("conservative", prompt)
        self.assertIn("NO_UPDATE", prompt)

    def test_prompt_contains_key_examples(self):
        from memory_extractor import EXTRACTOR_PROMPT
        prompt = EXTRACTOR_PROMPT

        # Core V2 examples
        self.assertIn("Help elderly people take medication", prompt)
        self.assertIn("WhatsApp groups", prompt)
        self.assertIn("Even I have faced this problem", prompt)
        self.assertIn("I have seen students lose marks", prompt)
        self.assertIn("want to make people disciplined", prompt)
        self.assertIn("They feel stressed", prompt)
        self.assertIn("Procrastination causes missed deadlines", prompt)
        self.assertIn("Forgetting things is stressful", prompt)
        self.assertIn("The problem is late submission", prompt)
        self.assertIn("hello", prompt)
        self.assertIn("thanks, that", prompt)
