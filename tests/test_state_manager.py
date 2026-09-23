"""
tests/test_state_manager.py

Unit tests for Module 2 of the Empathize v2 pipeline — ProjectState's
canonical state-management layer (StateValidator + StateManager).

Coverage (per the Module 2 spec):
  - ProjectState: default initialisation, serialisation, equality
  - Validator: valid / invalid message types, valid / invalid ADD,
    valid / invalid SET, malformed JSON, malformed updates, unknown
    fields, duplicate operations, atomic validation
  - StateManager: ADD, duplicate prevention, insertion order, SET
    overwrite, updating previous messages, multiple updates in one batch

All tests run with no Ollama, no network, no LLM — Module 2 is 100%
deterministic. The Memory Extractor is never called here.
"""

import copy
import json
import sys
import os
import unittest

# Allow importing from the project root regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_extractor import (
    ExtractionResult,
    ExtractionUpdate,
    ExtractionValidationError,
    ExtractionValidator,
    LIST_FIELDS,
    MessageType,
    Operation,
    ProjectState,
    SCALAR_FIELDS,
    StateField,
)
from state_manager import (
    StateBatchValidationError,
    StateManager,
    StateValidationError,
    StateValidator,
)


# ---------------------------------------------------------------------------
# Reference fixtures — the canonical contract example reused across tests
# ---------------------------------------------------------------------------

CANONICAL_RAW = {
    "message_type": "MEANINGFUL",
    "updates": [
        {"operation": "ADD", "field": "current_solutions", "value": "WhatsApp groups"},
        {"operation": "ADD", "field": "pain_points", "value": "Messages get buried"},
        {"operation": "SET", "field": "frequency", "value": "Weekly"},
    ],
}


def _meaningful(updates):
    """Build a typed ExtractionResult with message_type=MEANINGFUL."""
    return ExtractionResult(message_type=MessageType.MEANINGFUL, updates=list(updates))


def _mt(mt_value: str, raw_updates):
    """Build a raw dict (the LLM-shape input) from message_type + updates."""
    return {"message_type": mt_value, "updates": list(raw_updates)}


LIST_FIELD_VALUES = [f.value for f in LIST_FIELDS]
SCALAR_FIELD_VALUES = [f.value for f in SCALAR_FIELDS]


# ---------------------------------------------------------------------------
# ProjectState — re-exported from Module 1 but tested here because Module 2
# owns its lifecycle (defaults, serialisation, equality via == on dataclass).
# ---------------------------------------------------------------------------


class TestProjectState(unittest.TestCase):
    """ProjectState is the canonical state container tested via Module 2."""

    def test_default_initialisation_empty_lists(self):
        s = ProjectState()
        for f in LIST_FIELDS:
            self.assertEqual(s.get_list(f), [], f"{f.value} should default to []")
        self.assertEqual(s.get_list(StateField.PERSONAS), [])
        self.assertEqual(s.get_list(StateField.PROBLEMS), [])
        self.assertEqual(s.get_list(StateField.CURRENT_SOLUTIONS), [])
        self.assertEqual(s.get_list(StateField.PAIN_POINTS), [])
        self.assertEqual(s.get_list(StateField.EVIDENCE), [])

    def test_default_initialisation_scalar_is_none(self):
        s = ProjectState()
        self.assertIsNone(s.frequency)
        self.assertIsNone(s.previous_assistant_message)
        self.assertIsNone(s.previous_user_message)

    def test_default_initialisation_lists_are_independent_instances(self):
        """Each list field must be its own object — no shared default."""
        s = ProjectState()
        s.personas.append("students")
        # Mutating one list must not bleed into the other list fields.
        self.assertEqual(s.problems, [])
        self.assertEqual(s.evidence, [])

    def test_serialisation_round_trip_state_dict(self):
        """to_state_dict emits the extraction-relevant fields only."""
        s = ProjectState(
            personas=["Students"],
            problems=["Messages get buried"],
            current_solutions=["WhatsApp groups"],
            pain_points=["Feel overwhelmed"],
            evidence=["Saw students struggle"],
            impacts=["Students miss deadlines"],
            frequency="Weekly",
        )
        d = s.to_state_dict()
        self.assertEqual(
            sorted(d.keys()),
            sorted(["personas", "problems", "current_solutions", "pain_points", "evidence", "impacts", "frequency"]),
        )
        for f in LIST_FIELDS:
            self.assertEqual(d[f.value], s.get_list(f))
        self.assertEqual(d["frequency"], "Weekly")
        # previous_* bookkeeping MUST NOT leak into the extractor-facing dict.
        self.assertNotIn("previous_assistant_message", d)
        self.assertNotIn("previous_user_message", d)

    def test_serialisation_lists_are_defensive_copies(self):
        """Mutating to_state_dict output must not mutate the state."""
        s = ProjectState(personas=["A"])
        d = s.to_state_dict()
        d["personas"].append("B")
        self.assertEqual(s.personas, ["A"])

    def test_serialisation_frequency_none_round_trips(self):
        s = ProjectState()
        d = s.to_state_dict()
        self.assertIn("frequency", d)
        self.assertIsNone(d["frequency"])

    def test_equality_dataclass_semantics(self):
        a = ProjectState(personas=["x"], frequency="Weekly")
        b = ProjectState(personas=["x"], frequency="Weekly")
        c = ProjectState(personas=["x"], frequency="Daily")
        self.assertEqual(a, b, "two states with identical fields are equal")
        self.assertNotEqual(a, c, "differing frequency means not equal")

    def test_get_list_rejects_scalar_field(self):
        s = ProjectState()
        with self.assertRaises(TypeError):
            s.get_list(StateField.FREQUENCY)

    def test_get_scalar_rejects_list_field(self):
        s = ProjectState()
        with self.assertRaises(TypeError):
            s.get_scalar(StateField.PERSONAS)


# ---------------------------------------------------------------------------
# StateValidator — valid message types and valid operations/fields
# ---------------------------------------------------------------------------


class TestStateValidator_ValidMessageType(unittest.TestCase):
    """The four MessageType enum values are the only accepted types."""

    def test_valid_message_types_pass_batch_validation(self):
        for mt in (MessageType.MEANINGFUL, MessageType.NO_UPDATE,
                   MessageType.AMBIGUOUS, MessageType.END):
            result = ExtractionResult(message_type=mt, updates=[])
            # Should not raise — empty batch, valid type.
            StateValidator.validate_batch(result)

    def test_valid_raw_message_type_strings_parse(self):
        for mt_value in ("MEANINGFUL", "NO_UPDATE", "AMBIGUOUS", "END"):
            raw = _mt(mt_value, [])
            result = StateValidator.parse_and_validate(raw)
            self.assertEqual(result.message_type.value, mt_value)
            self.assertEqual(result.updates, [])


class TestStateValidator_ValidAdd(unittest.TestCase):
    """ADD is valid only for the 5 list-typed fields."""

    def test_add_each_list_field_passes_batch_validation(self):
        for sf in LIST_FIELDS:
            update = ExtractionUpdate(Operation.ADD, sf, "x")
            result = _meaningful([update])
            StateValidator.validate_batch(result)

    def test_canonical_contract_adds_pass(self):
        # Two ADDs on different fields in one batch.
        result = _meaningful([
            ExtractionUpdate(Operation.ADD, StateField.CURRENT_SOLUTIONS, "WhatsApp"),
            ExtractionUpdate(Operation.ADD, StateField.PAIN_POINTS, "Buried messages"),
        ])
        StateValidator.validate_batch(result)

    def test_multiple_adds_same_field_pass(self):
        result = _meaningful([
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Students"),
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Teachers"),
        ])
        StateValidator.validate_batch(result)


class TestStateValidator_ValidSet(unittest.TestCase):
    """SET is valid only for the scalar field 'frequency'."""

    def test_set_frequency_passes_batch_validation(self):
        result = _meaningful([ExtractionUpdate(Operation.SET, StateField.FREQUENCY, "Weekly")])
        StateValidator.validate_batch(result)

    def test_canonical_contract_set_passes(self):
        # Canonical batch: 2 ADDs + 1 SET — all compatible.
        result = StateValidator.parse_and_validate(CANONICAL_RAW)
        self.assertEqual(result.message_type, MessageType.MEANINGFUL)
        self.assertEqual(len(result.updates), 3)


# ---------------------------------------------------------------------------
# StateValidator — invalid operations / fields
# ---------------------------------------------------------------------------


class TestStateValidator_InvalidAdd(unittest.TestCase):
    """ADD on a scalar field must be rejected — atomic, with a clear error."""

    def test_add_on_frequency_rejected(self):
        result = _meaningful([ExtractionUpdate(Operation.ADD, StateField.FREQUENCY, "Daily")])
        with self.assertRaises(StateBatchValidationError) as ctx:
            StateValidator.validate_batch(result)
        self.assertTrue(any("ADD" in e and "frequency" in e for e in ctx.exception.errors))

    def test_add_on_frequency_via_raw_dict_rejected(self):
        raw = _mt("MEANINGFUL", [
            {"operation": "ADD", "field": "frequency", "value": "Daily"},
        ])
        with self.assertRaises(ExtractionValidationError):
            # Module 1's ExtractionValidator rejects the cross-type op first.
            StateValidator.parse_and_validate(raw)


class TestStateValidator_InvalidSet(unittest.TestCase):
    """SET on a list-typed field must be rejected by StateValidator."""

    def test_set_on_each_list_field_rejected(self):
        for sf in LIST_FIELDS:
            with self.subTest(field=sf.value):
                result = _meaningful([ExtractionUpdate(Operation.SET, sf, "x")])
                with self.assertRaises(StateBatchValidationError) as ctx:
                    StateValidator.validate_batch(result)
                self.assertTrue(
                    any("SET" in e and sf.value in e for e in ctx.exception.errors),
                    f"SET on {sf.value} should raise a SET-related error",
                )

    def test_set_on_each_list_field_via_raw_dict_rejected(self):
        for sf in LIST_FIELDS:
            with self.subTest(field=sf.value):
                raw = _mt("MEANINGFUL", [
                    {"operation": "SET", "field": sf.value, "value": "x"},
                ])
                with self.assertRaises(ExtractionValidationError):
                    StateValidator.parse_and_validate(raw)


# ---------------------------------------------------------------------------
# StateValidator — malformed JSON / malformed updates / unknown fields
# ---------------------------------------------------------------------------


class TestStateValidator_MalformedInput(unittest.TestCase):

    def test_non_dict_input_rejected_by_parse_and_validate(self):
        with self.assertRaises(ExtractionValidationError):
            StateValidator.parse_and_validate([1, 2, 3])  # type: ignore[arg-type]
        with self.assertRaises(ExtractionValidationError):
            StateValidator.parse_and_validate("not a dict")  # type: ignore[arg-type]

    def test_missing_message_type_rejected(self):
        with self.assertRaises(ExtractionValidationError):
            StateValidator.parse_and_validate({"updates": []})

    def test_missing_updates_key_rejected(self):
        with self.assertRaises(ExtractionValidationError):
            StateValidator.parse_and_validate({"message_type": "MEANINGFUL"})

    def test_updates_not_a_list_rejected(self):
        with self.assertRaises(ExtractionValidationError):
            StateValidator.parse_and_validate({
                "message_type": "MEANINGFUL",
                "updates": "not a list",
            })

    def test_updates_item_not_a_dict_rejected(self):
        with self.assertRaises(ExtractionValidationError):
            StateValidator.parse_and_validate({
                "message_type": "MEANINGFUL",
                "updates": ["not a dict"],
            })

    def test_update_missing_required_key_rejected(self):
        for missing_key in ("operation", "field", "value"):
            with self.subTest(missing=missing_key):
                broken = {"operation": "ADD", "field": "personas", "value": "x"}
                del broken[missing_key]
                raw = _mt("MEANINGFUL", [broken])
                with self.assertRaises(ExtractionValidationError):
                    StateValidator.parse_and_validate(raw)

    def test_invalid_message_type_string_rejected(self):
        with self.assertRaises(ExtractionValidationError):
            StateValidator.parse_and_validate({"message_type": "MAYBE", "updates": []})

    def test_invalid_operation_string_rejected(self):
        raw = _mt("MEANINGFUL", [
            {"operation": "APPEND", "field": "personas", "value": "x"},
        ])
        with self.assertRaises(ExtractionValidationError):
            StateValidator.parse_and_validate(raw)


class TestStateValidator_UnknownFields(unittest.TestCase):

    def test_unknown_field_rejected(self):
        raw = _mt("MEANINGFUL", [
            {"operation": "ADD", "field": "budget", "value": "100"},
        ])
        with self.assertRaises(ExtractionValidationError):
            StateValidator.parse_and_validate(raw)

    def test_empty_value_rejected(self):
        raw = _mt("MEANINGFUL", [
            {"operation": "ADD", "field": "personas", "value": "   "},
        ])
        with self.assertRaises(ExtractionValidationError):
            StateValidator.parse_and_validate(raw)

    def test_non_string_value_rejected(self):
        raw = _mt("MEANINGFUL", [
            {"operation": "ADD", "field": "personas", "value": 42},
        ])
        with self.assertRaises(ExtractionValidationError):
            StateValidator.parse_and_validate(raw)


# ---------------------------------------------------------------------------
# StateValidator — cross-update / atomic batch rules
# ---------------------------------------------------------------------------


class TestStateValidator_BatchAtomic(unittest.TestCase):
    """Atomic: one bad update in a batch rejects the whole batch."""

    def test_one_invalid_update_rejects_entire_batch(self):
        # A valid ADD followed by an invalid ADD-on-frequency.
        result = _meaningful([
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Students"),
            ExtractionUpdate(Operation.ADD, StateField.FREQUENCY, "Daily"),
        ])
        with self.assertRaises(StateBatchValidationError) as ctx:
            StateValidator.validate_batch(result)
        self.assertGreaterEqual(len(ctx.exception.errors), 1)

    def test_non_meaningful_with_updates_rejected(self):
        """NO_UPDATE / AMBIGUOUS / END must not carry updates — atomic reject."""
        for mt in (MessageType.NO_UPDATE, MessageType.AMBIGUOUS, MessageType.END):
            with self.subTest(message_type=mt.value):
                result = ExtractionResult(
                    message_type=mt,
                    updates=[ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "x")],
                )
                with self.assertRaises(StateBatchValidationError) as ctx:
                    StateValidator.validate_batch(result)
                self.assertTrue(any(mt.value in e for e in ctx.exception.errors))

    def test_duplicate_identical_updates_in_batch_rejected(self):
        """The exact same (op, field, value) appearing twice in one batch."""
        result = _meaningful([
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Students"),
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Students"),
        ])
        with self.assertRaises(StateBatchValidationError) as ctx:
            StateValidator.validate_batch(result)
        self.assertTrue(any("duplicate" in e for e in ctx.exception.errors))

    def test_conflicting_operations_on_same_field_rejected(self):
        """ADD and SET on the same field in one batch is a conflict."""
        # Use ADD on a list field + (invalid) SET on the same list field —
        # the conflict rule catches the cross-op combination.
        result = _meaningful([
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Students"),
            ExtractionUpdate(Operation.SET, StateField.PERSONAS, "x"),  # also invalid op×field
        ])
        with self.assertRaises(StateBatchValidationError):
            StateValidator.validate_batch(result)

    def test_atomic_validation_does_not_mutate_state_through_state_manager(self):
        """If a batch is rejected, the StateManager-owned ProjectState is unchanged.

        This is the real atomic guarantee: validation is read-only, so the
        partial-apply scenario never arises. StateManager's apply call
        validates FIRST and only mutates after validation passes."""
        manager = StateManager(ProjectState(personas=["existing"]))
        bad_result = _meaningful([
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Students"),
            ExtractionUpdate(Operation.ADD, StateField.FREQUENCY, "Daily"),  # invalid — breaks batch
        ])
        with self.assertRaises(StateBatchValidationError):
            manager.apply_extraction(bad_result,
                                     previous_assistant_message="hi",
                                     previous_user_message="hello")
        # State untouched by the failed batch.
        self.assertEqual(manager.state.personas, ["existing"])
        # Even previous_* bookkeeping is not written when validation failed.
        self.assertIsNone(manager.state.previous_assistant_message)
        self.assertIsNone(manager.state.previous_user_message)


class TestStateValidator_IsApplicable(unittest.TestCase):

    def test_meaningful_with_updates_is_applicable(self):
        result = StateValidator.parse_and_validate(CANONICAL_RAW)
        self.assertTrue(StateValidator.is_applicable(result))

    def test_meaningful_without_updates_not_applicable(self):
        result = ExtractionResult(message_type=MessageType.MEANINGFUL, updates=[])
        self.assertFalse(StateValidator.is_applicable(result))

    def test_conversational_types_not_applicable(self):
        for mt in (MessageType.NO_UPDATE, MessageType.AMBIGUOUS, MessageType.END):
            result = ExtractionResult(message_type=mt, updates=[])
            self.assertFalse(StateValidator.is_applicable(result))


# ---------------------------------------------------------------------------
# StateManager — ADD semantics
# ---------------------------------------------------------------------------


class TestStateManager_Add(unittest.TestCase):

    def test_add_appends_value_to_empty_list(self):
        manager = StateManager()
        result = _meaningful([ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Students")])
        manager.apply_extraction(result)
        self.assertEqual(manager.state.personas, ["Students"])

    def test_add_appends_to_existing_list(self):
        manager = StateManager(ProjectState(current_solutions=["WhatsApp"]))
        result = _meaningful([ExtractionUpdate(Operation.ADD, StateField.CURRENT_SOLUTIONS, "Google Calendar")])
        manager.apply_extraction(result)
        self.assertEqual(manager.state.current_solutions, ["WhatsApp", "Google Calendar"])

    def test_add_does_not_raise_on_duplicate(self):
        manager = StateManager(ProjectState(personas=["Students"]))
        result = _meaningful([ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Students")])
        # Must not raise.
        manager.apply_extraction(result)
        self.assertEqual(manager.state.personas, ["Students"])

    def test_add_duplicate_is_a_noop_returns_unchanged_state(self):
        manager = StateManager(ProjectState(personas=["Students", "Teachers"]))
        result = _meaningful([ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Students")])
        manager.apply_extraction(result)
        self.assertEqual(manager.state.personas, ["Students", "Teachers"])

    def test_add_preserves_insertion_order_across_multiple_batches(self):
        manager = StateManager()
        for value in ["Students", "Teachers", "Parents"]:
            result = _meaningful([
                ExtractionUpdate(Operation.ADD, StateField.PERSONAS, value),
            ])
            manager.apply_extraction(result)
        self.assertEqual(manager.state.personas, ["Students", "Teachers", "Parents"])

    def test_add_case_sensitive_duplicate_check(self):
        """Duplicate check is exact (case-sensitive) — "Students" vs "students" differ."""
        manager = StateManager(ProjectState(personas=["Students"]))
        result = _meaningful([ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "students")])
        manager.apply_extraction(result)
        self.assertEqual(manager.state.personas, ["Students", "students"])

    def test_add_strips_value_before_applying(self):
        """Module 1's ExtractionValidator strips values — Module 2 must honour that."""
        raw = _mt("MEANINGFUL", [
            {"operation": "ADD", "field": "personas", "value": "  Students  "},
        ])
        result = StateValidator.parse_and_validate(raw)
        manager = StateManager()
        manager.apply_extraction(result)
        self.assertEqual(manager.state.personas, ["Students"])

    def test_add_applies_to_each_list_field(self):
        manager = StateManager()
        result = _meaningful([
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Students"),
            ExtractionUpdate(Operation.ADD, StateField.PROBLEMS, "Messages get buried"),
            ExtractionUpdate(Operation.ADD, StateField.CURRENT_SOLUTIONS, "WhatsApp groups"),
            ExtractionUpdate(Operation.ADD, StateField.PAIN_POINTS, "Overwhelmed"),
            ExtractionUpdate(Operation.ADD, StateField.EVIDENCE, "Saw students struggle"),
        ])
        manager.apply_extraction(result)
        self.assertEqual(manager.state.personas, ["Students"])
        self.assertEqual(manager.state.problems, ["Messages get buried"])
        self.assertEqual(manager.state.current_solutions, ["WhatsApp groups"])
        self.assertEqual(manager.state.pain_points, ["Overwhelmed"])
        self.assertEqual(manager.state.evidence, ["Saw students struggle"])


class TestStateManager_Set(unittest.TestCase):

    def test_set_frequency_when_none(self):
        manager = StateManager()
        result = _meaningful([ExtractionUpdate(Operation.SET, StateField.FREQUENCY, "Weekly")])
        manager.apply_extraction(result)
        self.assertEqual(manager.state.frequency, "Weekly")

    def test_set_overwrites_existing_frequency(self):
        manager = StateManager(ProjectState(frequency="Weekly"))
        result = _meaningful([ExtractionUpdate(Operation.SET, StateField.FREQUENCY, "Daily")])
        manager.apply_extraction(result)
        self.assertEqual(manager.state.frequency, "Daily")

    def test_set_strips_value(self):
        raw = _mt("MEANINGFUL", [
            {"operation": "SET", "field": "frequency", "value": "  Daily  "},
        ])
        result = StateValidator.parse_and_validate(raw)
        manager = StateManager(ProjectState(frequency="Weekly"))
        manager.apply_extraction(result)
        self.assertEqual(manager.state.frequency, "Daily")


# ---------------------------------------------------------------------------
# StateManager — previous-message bookkeeping
# ---------------------------------------------------------------------------


class TestStateManager_PreviousMessages(unittest.TestCase):

    def test_set_previous_assistant_message(self):
        manager = StateManager()
        result = _meaningful([ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "x")])
        manager.apply_extraction(result, previous_assistant_message="What problem are you solving?")
        self.assertEqual(manager.state.previous_assistant_message, "What problem are you solving?")

    def test_set_previous_user_message(self):
        manager = StateManager()
        result = _meaningful([ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "x")])
        manager.apply_extraction(result, previous_user_message="I'm a student and I struggle with deadlines")
        self.assertEqual(manager.state.previous_user_message,
                         "I'm a student and I struggle with deadlines")

    def test_set_both_previous_messages(self):
        manager = StateManager()
        result = _meaningful([ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "x")])
        manager.apply_extraction(result,
                                previous_assistant_message="Who is your target audience?",
                                previous_user_message="Students mostly")
        self.assertEqual(manager.state.previous_assistant_message, "Who is your target audience?")
        self.assertEqual(manager.state.previous_user_message, "Students mostly")

    def test_rolls_forward_when_no_updates_in_batch(self):
        """Conversational / empty batches still update previous_* bookkeeping."""
        manager = StateManager(ProjectState(personas=["Students"]))
        result = ExtractionResult(message_type=MessageType.NO_UPDATE, updates=[])
        manager.apply_extraction(result,
                                previous_assistant_message="hello?",
                                previous_user_message="hi there")
        # State lists untouched.
        self.assertEqual(manager.state.personas, ["Students"])
        # But previous_* was rolled forward.
        self.assertEqual(manager.state.previous_assistant_message, "hello?")
        self.assertEqual(manager.state.previous_user_message, "hi there")

    def test_empty_string_previous_message_becomes_none(self):
        """Module 2 normalises whitespace-only previous messages to None."""
        manager = StateManager()
        result = _meaningful([ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "x")])
        manager.apply_extraction(result, previous_assistant_message="   ", previous_user_message="")
        self.assertIsNone(manager.state.previous_assistant_message)
        self.assertIsNone(manager.state.previous_user_message)

    def test_none_keeps_existing_previous_message(self):
        manager = StateManager()
        manager.state.previous_assistant_message = "first assistant"
        manager.state.previous_user_message = "first user"
        result = _meaningful([ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "x")])
        # Pass None for both — should NOT overwrite.
        manager.apply_extraction(result,
                                previous_assistant_message=None,
                                previous_user_message=None)
        self.assertEqual(manager.state.previous_assistant_message, "first assistant")
        self.assertEqual(manager.state.previous_user_message, "first user")


# ---------------------------------------------------------------------------
# StateManager — multiple updates in one batch (atomic behaviour)
# ---------------------------------------------------------------------------


class TestStateManager_MultiUpdateBatch(unittest.TestCase):

    def test_canonical_batch_applies_all_operations(self):
        manager = StateManager()
        result = StateValidator.parse_and_validate(CANONICAL_RAW)
        manager.apply_extraction(result,
                                previous_assistant_message="q",
                                previous_user_message="msg")
        self.assertEqual(manager.state.current_solutions, ["WhatsApp groups"])
        self.assertEqual(manager.state.pain_points, ["Messages get buried"])
        self.assertEqual(manager.state.frequency, "Weekly")
        self.assertEqual(manager.state.previous_assistant_message, "q")
        self.assertEqual(manager.state.previous_user_message, "msg")

    def test_multiple_adds_on_same_field_in_one_batch(self):
        manager = StateManager()
        result = _meaningful([
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Students"),
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Teachers"),
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Parents"),
        ])
        manager.apply_extraction(result)
        self.assertEqual(manager.state.personas, ["Students", "Teachers", "Parents"])

    def test_one_batch_two_adds_one_set(self):
        manager = StateManager(ProjectState(frequency="Rarely"))
        result = _meaningful([
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Students"),
            ExtractionUpdate(Operation.ADD, StateField.PROBLEMS, "Late deadlines"),
            ExtractionUpdate(Operation.SET, StateField.FREQUENCY, "Daily"),
        ])
        manager.apply_extraction(result)
        self.assertEqual(manager.state.personas, ["Students"])
        self.assertEqual(manager.state.problems, ["Late deadlines"])
        self.assertEqual(manager.state.frequency, "Daily")

    def test_failed_batch_leaves_state_untouched(self):
        """Atomicity: a batch with one bad update must not leave any partial state."""
        manager = StateManager(ProjectState(personas=["existing"], frequency="Weekly"))
        bad_result = _meaningful([
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "new"),
            ExtractionUpdate(Operation.SET, StateField.PERSONAS, "x"),  # invalid (SET on list field)
        ])
        with self.assertRaises(StateBatchValidationError):
            manager.apply_extraction(bad_result)
        # Nothing changed.
        self.assertEqual(manager.state.personas, ["existing"])
        self.assertEqual(manager.state.frequency, "Weekly")


# ---------------------------------------------------------------------------
# StateManager — additional contract behaviour
# ---------------------------------------------------------------------------


class TestStateManager_Contract(unittest.TestCase):

    def test_owned_state_is_the_one_passed_in(self):
        """StateManager stores a reference, not a copy, so the caller sees mutations."""
        state = ProjectState()
        manager = StateManager(state)
        self.assertIs(manager.state, state)
        result = _meaningful([ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "x")])
        manager.apply_extraction(result)
        # The caller's object was mutated in place.
        self.assertEqual(state.personas, ["x"])

    def test_default_state_when_none_passed(self):
        manager = StateManager()
        self.assertIsInstance(manager.state, ProjectState)
        for f in LIST_FIELDS:
            self.assertEqual(manager.state.get_list(f), [])
        self.assertIsNone(manager.state.frequency)

    def test_reset_clears_owned_state(self):
        manager = StateManager(ProjectState(personas=["x"], frequency="Daily"))
        manager.state.previous_assistant_message = "msg"
        manager.reset()
        # New fresh instance.
        self.assertEqual(manager.state.personas, [])
        self.assertIsNone(manager.state.frequency)
        self.assertIsNone(manager.state.previous_assistant_message)

    def test_state_dict_proxies_to_state_serialisation(self):
        manager = StateManager(ProjectState(personas=["x"], frequency="Weekly"))
        d = manager.state_dict()
        self.assertEqual(d["personas"], ["x"])
        self.assertEqual(d["frequency"], "Weekly")

    def test_apply_extraction_returns_owned_state(self):
        manager = StateManager()
        result = _meaningful([ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "x")])
        returned = manager.apply_extraction(result)
        self.assertIs(returned, manager.state)

    def test_conversational_batch_without_previous_messages_is_a_clean_noop(self):
        """NO_UPDATE with no previous-msg args changes nothing at all."""
        manager = StateManager(ProjectState(personas=["Students"]))
        result = ExtractionResult(message_type=MessageType.NO_UPDATE, updates=[])
        manager.apply_extraction(result)
        self.assertEqual(manager.state.personas, ["Students"])
        self.assertIsNone(manager.state.previous_assistant_message)
        self.assertIsNone(manager.state.previous_user_message)


# ---------------------------------------------------------------------------
# StateManager — error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy(unittest.TestCase):

    def test_batch_error_is_subclass_of_state_validation_error(self):
        self.assertTrue(issubclass(StateBatchValidationError, StateValidationError))

    def test_batch_error_is_subclass_of_value_error(self):
        self.assertTrue(issubclass(StateValidationError, ValueError))

    def test_batch_error_exposes_errors_list(self):
        result = _meaningful([
            ExtractionUpdate(Operation.ADD, StateField.FREQUENCY, "Daily"),
        ])
        try:
            StateValidator.validate_batch(result)
        except StateBatchValidationError as exc:
            self.assertIsInstance(exc.errors, list)
            self.assertGreaterEqual(len(exc.errors), 1)
            self.assertTrue(all(isinstance(e, str) for e in exc.errors))
        else:
            self.fail("expected StateBatchValidationError")

    def test_batch_error_message_includes_error_summary(self):
        result = _meaningful([
            ExtractionUpdate(Operation.ADD, StateField.FREQUENCY, "Daily"),
        ])
        with self.assertRaises(StateBatchValidationError) as ctx:
            StateValidator.validate_batch(result)
        self.assertIn("Update batch rejected", str(ctx.exception))
        self.assertIn("no state was changed", str(ctx.exception))


# ---------------------------------------------------------------------------
# Integration: parse raw LLM output through StateValidator → StateManager
# The end-to-end Module 2 path that the production pipeline uses.
# ---------------------------------------------------------------------------


class TestEndToEndPipeline(unittest.TestCase):
    """Memory Extractor output (raw dict) → StateValidator → StateManager."""

    def test_canonical_example_end_to_end(self):
        raw = copy.deepcopy(CANONICAL_RAW)
        result = StateValidator.parse_and_validate(raw)
        manager = StateManager()
        manager.apply_extraction(
            result,
            previous_assistant_message="Who uses it and how often?",
            previous_user_message="Students use WhatsApp groups and messages get buried weekly",
        )
        self.assertEqual(manager.state.current_solutions, ["WhatsApp groups"])
        self.assertEqual(manager.state.pain_points, ["Messages get buried"])
        self.assertEqual(manager.state.frequency, "Weekly")
        self.assertEqual(manager.state.previous_assistant_message, "Who uses it and how often?")

    def test_invalid_raw_rejected_before_state_mutation(self):
        raw = _mt("MEANINGFUL", [
            {"operation": "ADD", "field": "personas", "value": "Students"},
            {"operation": "ADD", "field": "frequency", "value": "Daily"},  # invalid op×field
        ])
        # Module 1's per-update validator rejects this at parse time.
        with self.assertRaises(ExtractionValidationError):
            StateValidator.parse_and_validate(raw)

    def test_duplicate_batch_rejected_and_state_untouched(self):
        """Identical (op, field, value) in one batch is a Module 2 cross-update error."""
        manager = StateManager()
        result = _meaningful([
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Students"),
            ExtractionUpdate(Operation.ADD, StateField.PERSONAS, "Students"),
        ])
        with self.assertRaises(StateBatchValidationError) as ctx:
            manager.apply_extraction(result)
        self.assertTrue(any("duplicate" in e for e in ctx.exception.errors))
        # State untouched.
        self.assertEqual(manager.state.personas, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
