"""
state_manager.py – Module 2 of the Empathize v2 Pipeline.

Architecture role
-----------------
User -> Memory Extractor -> **State Validator -> State Manager -> ProjectState**
      -> Objective Engine -> Prompt Builder -> Mentor

Module 2 is the application's deterministic state-management layer. It consumes
the structured output of Module 1 (the Memory Extractor), validates it as a
batch, and updates ProjectState. Module 2 owns ProjectState; no other component
mutates it directly.

This module is **100% deterministic**. No LLM calls anywhere.

Responsibilities
----------------
- ProjectState lives in Module 1 (`memory_extractor`) so the extractor can read
  it. Module 2 treats it as a pure data container — mutation goes through
  `StateManager`, never through `ProjectState.apply()` (which is retained in
  Module 1 for compatibility but is *not* called by the live pipeline).
- `StateValidator` validates a complete update batch atomically. If any one
  update is invalid, the entire batch is rejected — no partial updates, no
  silent failures.
- `StateManager` is the single component that mutates `ProjectState`:
    * ADD    -> append to a list field, prevent duplicates, preserve order.
    * SET    -> overwrite a scalar field.
    * After every successful batch, roll forward `previous_assistant_message`
      and `previous_user_message` for conversational continuity.

Non-responsibilities
--------------------
- No LLM use.
- No conversation-flow decisions (Objective Engine's job – Module 3).
- No backwards-compatibility shimming with the legacy `MentorSession`. A thin
  one-way adapter lives in `mentor.py` (mirroring `ProjectState` →
  `MentorSession` scalars) and is clearly marked `TODO(Module 3)` for removal.
- No automatic repair of malformed updates. Invalid input fails loudly and
  deterministically with a descriptive error.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

from timing import add_stage_ms
from memory_extractor import (
    ExtractionResult,
    ExtractionUpdate,
    ExtractionValidator,
    ExtractionValidationError,
    LIST_FIELDS,
    MessageType,
    Operation,
    ProjectState,
    SCALAR_FIELDS,
    StateField,
)



__all__ = [
    "StateValidationError",
    "StateBatchValidationError",
    "StateValidator",
    "StateManager",
]


# ---------------------------------------------------------------------------
# Deterministic application errors
# ---------------------------------------------------------------------------


class StateValidationError(ValueError):
    """
    Base class for all deterministic state-management errors.

    Module 2 never silently ignores invalid input and never auto-repairs
    malformed updates. Every validation failure raises a subclass of this
    exception with a descriptive, deterministic message.

    Inherits from ``ValueError`` so callers can still ``except ValueError``
    against it for broad cleanup, while Module 2 internals can target the
    specific subclasses below.
    """


class StateBatchValidationError(StateValidationError):
    """
    Raised when an entire update batch fails atomic validation.

    Module 2 validation is **atomic** — if any single update is invalid, or
    if any cross-update constraint is violated (duplicate operations on the
    same field, conflicting operations on the same field), the whole batch
    is rejected and ProjectState is left untouched.

    The ``errors`` attribute carries every individual problem discovered
    during the (read-only) validation pass, so callers can surface a full
    diagnostic instead of only the first failure.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors: list[str] = list(errors)
        msg = "Update batch rejected (no state was changed):"
        if errors:
            msg += "\n  - " + "\n  - ".join(errors)
        super().__init__(msg)


# ---------------------------------------------------------------------------
# StateValidator — atomic batch validation (deterministic, no LLM)
# ---------------------------------------------------------------------------


class StateValidator:
    """
    Deterministic validator for an ``ExtractionResult`` batch.

    Responsibilities (per the Module 2 spec):
      * validate the JSON structure            (delegated to Module 1)
      * validate the message type              (delegated to Module 1)
      * validate the operation                (delegated to Module 1)
      * validate the field names               (delegated to Module 1)
      * validate operation / field compatibility (delegated to Module 1)
      * validate the value types               (delegated to Module 1)
      * validate the batch as a whole          (owned by Module 2)
        - message-type / updates consistency (only MEANINGFUL mutates state)
        - duplicate operations on the same field within one batch
        - conflicting operations on the same field (ADD-then-SET, SET-then-ADD)

    All methods are static — no instance state, deterministic, no LLM, no
    randomness, no I/O. The same input always produces the same outcome.

    This validator does NOT mutate state. It only inspects and raises.
    """

    # Message types that are allowed to carry updates. NO_UPDATE / AMBIGUOUS /
    # END are conversational signals — any updates they carry are rejected
    # because they must not mutate state. Only MEANINGFUL drives state changes.
    _MUTATING_MESSAGE_TYPES: frozenset[MessageType] = frozenset({MessageType.MEANINGFUL})

    # ------------------------------------------------------------------
    # Raw-JSON entry point (re-uses Module 1's ExtractionValidator)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_and_validate(raw: dict[str, Any]) -> ExtractionResult:
        """
        Validate a raw JSON dict (the LLM's output) end-to-end into a typed,
        batch-validated ``ExtractionResult``.

        This re-uses Module 1's ``ExtractionValidator.validate`` for the
        per-update contract (JSON shape, enum membership, operation×field
        compatibility, value types) and then layers Module 2's cross-update
        batch validation on top. Any failure raises — no silent fallback
        here (the Memory Extractor itself owns the graceful AMBIGUOUS
        fallback for model-failure scenarios; Module 2 fails loudly).

        Raises
        ------
        ExtractionValidationError
            If the raw JSON violates Module 1's per-update contract.
        StateBatchValidationError
            If the parsed batch fails Module 2's cross-update validation.
        """
        result: ExtractionResult = ExtractionValidator.validate(raw)
        StateValidator.validate_batch(result)
        return result

    # ------------------------------------------------------------------
    # Batch validation entry point (operates on a typed ExtractionResult)
    # ------------------------------------------------------------------

    @staticmethod
    def validate_batch(result: ExtractionResult) -> None:
        """
        Atomically validate a typed ``ExtractionResult`` as one batch.

        Performs a read-only validation pass that collects every problem
        before raising. ProjectState is never touched by this method.

        Raises
        ------
        StateBatchValidationError
            If any single update or any cross-update constraint fails.
            The exception's ``errors`` list contains one message per problem
            found during the pass.
        """
        if not isinstance(result, ExtractionResult):
            raise StateBatchValidationError(
                [
                    f"Expected an ExtractionResult, got {type(result).__name__}"
                ]
            )

        errors: list[str] = []

        # 1. type-membership sanity (defence-in-depth; ExtractionResult is a
        #    frozen dataclass built by ExtractionValidator, but a hand-built
        #    instance could still carry foreign enum-looking strings).
        if not isinstance(result.message_type, MessageType):
            errors.append(
                f"message_type is not a MessageType (got "
                f"{type(result.message_type).__name__})"
            )
            # Cannot continue meaningfully if message_type is invalid.
            raise StateBatchValidationError(errors)

        for i, update in enumerate(result.updates):
            if not isinstance(update, ExtractionUpdate):
                errors.append(
                    f"updates[{i}]: expected ExtractionUpdate, got "
                    f"{type(update).__name__}"
                )
                continue
            if not isinstance(update.operation, Operation):
                errors.append(
                    f"updates[{i}].operation is not an Operation "
                    f"(got {type(update.operation).__name__})"
                )
            if not isinstance(update.field, StateField):
                errors.append(
                    f"updates[{i}].field is not a StateField "
                    f"(got {type(update.field).__name__})"
                )
            if not isinstance(update.value, str):
                errors.append(
                    f"updates[{i}].value is not a string "
                    f"(got {type(update.value).__name__})"
                )

        if errors:
            raise StateBatchValidationError(errors)

        # 2. operation / field compatibility (defence-in-depth against a
        #    hand-built ExtractionResult bypassing Module 1's validator).
        for i, update in enumerate(result.updates):
            op = update.operation
            fld = update.field
            if op is Operation.ADD and fld not in LIST_FIELDS:
                errors.append(
                    f"updates[{i}]: ADD is only valid for list fields "
                    f"{sorted(f.value for f in LIST_FIELDS)}; got "
                    f"'{fld.value}'"
                )
            elif op is Operation.SET and fld not in SCALAR_FIELDS:
                errors.append(
                    f"updates[{i}]: SET is only valid for scalar fields "
                    f"{sorted(f.value for f in SCALAR_FIELDS)}; got "
                    f"'{fld.value}'"
                )

        # 3. message-type / updates consistency. Only MEANINGFUL may carry
        #    updates; conversational types (NO_UPDATE / AMBIGUOUS / END)
        #    must be empty.
        if result.message_type not in StateValidator._MUTATING_MESSAGE_TYPES:
            if result.updates:
                errors.append(
                    f"message_type '{result.message_type.value}' is not "
                    f"allowed to carry updates, but {len(result.updates)} "
                    f"update(s) were provided"
                )
        else:
            # MEANINGFUL with empty updates is allowed (the user shared
            # something we recognised as information-bearing but the model
            # could not distil it into a structured update). The Objective
            # Engine handles this; we just accept it.
            pass

        # 4. duplicate operations within the batch — same (operation, field,
        #    value) appearing more than once is a model mistake and is
        #    rejected rather than silently collapsed. Note this is distinct
        #    from ADD deduplication at apply time: a batch that proposes
        #    the *exact same ADD twice in one turn* is malformed, while
        #    re-adding a value the state already holds is a no-op.
        seen: set[tuple[str, str, str]] = set()
        for i, update in enumerate(result.updates):
            key = (update.operation.value, update.field.value, update.value)
            if key in seen:
                errors.append(
                    f"updates[{i}]: duplicate operation "
                    f"({update.operation.value} '{update.field.value}' = "
                    f"{update.value!r}) within this batch"
                )
            else:
                seen.add(key)

        # 5. conflicting operations on the same field within one batch
        #    (e.g. ADD personas + SET personas, or SET frequency + ADD
        #    frequency). One batch must agree on the *kind* of operation
        #    per field.
        field_to_ops: dict[StateField, set[Operation]] = {}
        for update in result.updates:
            field_to_ops.setdefault(update.field, set()).add(update.operation)
        for fld, ops in field_to_ops.items():
            if len(ops) > 1:
                errors.append(
                    f"conflicting operations "
                    f"{sorted(op.value for op in ops)} on field "
                    f"'{fld.value}' within the same batch"
                )

        if errors:
            raise StateBatchValidationError(errors)

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @staticmethod
    def is_applicable(result: ExtractionResult) -> bool:
        """
        True iff applying this batch would actually change state.

        Returns False for non-MEANINGFUL message types and for MEANINGFUL
        batches with no updates. This is a *read-only* predicate — it
        performs no validation and does not raise.
        """
        return (
            result.message_type in StateValidator._MUTATING_MESSAGE_TYPES
            and bool(result.updates)
        )


# ---------------------------------------------------------------------------
# StateManager — the single mutator of ProjectState
# ---------------------------------------------------------------------------


class StateManager:
    """
    The application's single component authorised to mutate ``ProjectState``.

    Module 2's contract:
        * The Memory Extractor (Module 1) only *proposes* updates.
        * The StateValidator only *inspects* batches.
        * StateManager *applies* validated batches to ProjectState.

    StateManager owns one ``ProjectState`` instance (passed in or freshly
    created) and exposes a single mutating entry point —
    :meth:`apply_extraction` — which:

      1. Validates the batch atomically through :class:`StateValidator`.
         On failure, ProjectState is left untouched and
         :class:`StateBatchValidationError` is raised.
      2. Applies each update with strict ADD / SET semantics:
           * ADD  -> append to the field's list. Duplicates are NOT inserted
                    (case-sensitive, exact-string compare). Insertion order
                    is preserved. Duplicate ADDs do not raise.
           * SET  -> overwrite the field's scalar value.
      3. After a successful batch, rolls forward the previous-message
         bookkeeping fields. Both parameters are optional; passing ``None``
         keeps the previous value, but the intended usage is for the caller
         to pass the latest assistant / user message each turn so the state
         always reflects the most recent conversation.

    This class is deterministic: no LLM, no randomness, no I/O. The same
    ``(state, result, prev_assistant, prev_user)`` always produces the same
    resulting state or the same deterministic error.
    """

    def __init__(self, state: ProjectState | None = None) -> None:
        """
        Parameters
        ----------
        state:
            The ProjectState to own. If ``None``, a fresh empty
            ``ProjectState`` is created. StateManager stores a reference to
            this object (not a copy) and mutates it in place — callers that
            need a snapshot should copy the state before constructing the
            manager, or use :meth:`state` to access the live instance.
        """
        self._state: ProjectState = state if state is not None else ProjectState()

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def state(self) -> ProjectState:
        """The owned ProjectState instance (mutated in place by apply_extraction)."""
        return self._state

    def get_list(self, name: StateField) -> list[str]:
        """Read a list-typed field (delegates to ProjectState.get_list)."""
        return self._state.get_list(name)

    def get_scalar(self, name: StateField) -> str | None:
        """Read a scalar-typed field (delegates to ProjectState.get_scalar)."""
        return self._state.get_scalar(name)

    # ------------------------------------------------------------------
    # Mutating entry point
    # ------------------------------------------------------------------

    def apply_extraction(
        self,
        result: ExtractionResult,
        *,
        previous_assistant_message: str | None = None,
        previous_user_message: str | None = None,
        _timing: dict[str, float] | None = None,
    ) -> ProjectState:
        """
        Validate and apply one extraction batch to ProjectState.

        Validation is atomic — if any rule in :class:`StateValidator` fails,
        nothing is mutated and :class:`StateBatchValidationError` is raised.

        For non-mutating message types (NO_UPDATE / AMBIGUOUS / END) or
        MEANINGFUL batches with no updates, this is a no-op on the list /
        scalar fields but still rolls forward the previous-message
        bookkeeping (so the state always reflects the latest conversation).

        Parameters
        ----------
        result:
            The ``ExtractionResult`` produced by the Memory Extractor.
            Must be a properly typed instance; raw dicts should be fed
            through :meth:`StateValidator.parse_and_validate` first.
        previous_assistant_message:
            The assistant's most recent message, to record on the state
            for the next turn's contextual disambiguation. Pass ``None``
            to leave the existing value untouched.
        previous_user_message:
            The user's most recent message (the same message that was
            extracted), to record on the state. Pass ``None`` to leave
            the existing value untouched.

        Returns
        -------
        ProjectState
            The owned state (mutated in place), returned for chaining /
            convenience — same object as :attr:`state`.

        Raises
        ------
        StateBatchValidationError
            If the batch fails atomic validation. ProjectState is left
            unmodified (the validation pass is read-only; application only
            begins after validation succeeds).
        """
        # --- 1. atomic validation (read-only pass) -----------------------
        _v = time.perf_counter() if _timing is not None else None
        StateValidator.validate_batch(result)
        if _v is not None:
            add_stage_ms(_timing, "validation_batch", _v, time.perf_counter())

        # --- 2. apply updates with strict ADD/SET semantics ---------------
        # Only MEANINGFUL batches carry updates; the validator above already
        # guaranteed that non-MEANINGFUL batches have empty updates, so this
        # loop is naturally a no-op for them.
        _m = time.perf_counter() if _timing is not None else None
        for update in result.updates:
            if update.operation is Operation.ADD:
                self._apply_add(update.field, update.value)
            elif update.operation is Operation.SET:
                self._apply_set(update.field, update.value)

        # --- 3. roll forward previous-message bookkeeping ----------------
        self.set_previous_assistant_message(previous_assistant_message)
        self.set_previous_user_message(previous_user_message)
        if _m is not None:
            add_stage_ms(_timing, "merge_state", _m, time.perf_counter())

        return self._state

    # ------------------------------------------------------------------
    # Internal mutators
    # ------------------------------------------------------------------

    def _apply_add(self, field: StateField, value: str) -> None:
        """
        Append ``value`` to a list-typed field, ignoring exact duplicates
        and preserving insertion order. Duplicate ADDs are NOT errors.

        Precondition: ``field`` is in ``LIST_FIELDS`` and ``value`` is a
        non-empty string. The validator above already enforced this.
        """
        lst = self._state.get_list(field)  # raises TypeError if mis-used, defence-in-depth
        if value and value not in lst:
            lst.append(value)

    def _apply_set(self, field: StateField, value: str) -> None:
        """
        Overwrite a scalar-typed field with ``value``.

        Precondition: ``field`` is in ``SCALAR_FIELDS`` and ``value`` is a
        non-empty string. The validator above already enforced this.
        """
        # set_scalar handles the empty-string -> None normalisation, same as
        # Module 1's ProjectState.set_scalar — kept consistent so the v2
        # invariant (frequency is None or a non-empty string) holds.
        self._state.set_scalar(field, value)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset owned state to an empty ProjectState (replaces internal instance)."""
        self._state = ProjectState()

    def set_previous_assistant_message(self, message: str | None) -> None:
        """Set the previous assistant message (used for lifecycle marker injection)."""
        if message is not None:
            v = message.strip()
            self._state.previous_assistant_message = v if v else None

    def set_previous_user_message(self, message: str | None) -> None:
        """Set the previous user message."""
        if message is not None:
            v = message.strip()
            self._state.previous_user_message = v if v else None

    def state_dict(self) -> dict[str, Any]:
        """Return the extractor-facing serialisation of the owned state."""
        return self._state.to_state_dict()

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return (
            f"StateManager(state={self._state!r})"
        )
