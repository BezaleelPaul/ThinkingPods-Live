"""
memory_extractor.py – Module 1 of the Empathize v2 Pipeline.

Architecture role
-----------------
User → **Memory Extractor** → State Validator → ProjectState
     → Objective Engine → Prompt Builder → Mentor

Responsibilities
----------------
- Receive the latest user message, current project state, and (optionally)
  the previous assistant message.
- Perform semantic understanding of the user message.
- Return a structured ExtractionResult describing what was learned.

Non-responsibilities (intentional constraints)
----------------------------------------------
- Does NOT generate conversational replies.
- Does NOT ask follow-up questions.
- Does NOT update ProjectState directly.
- Does NOT choose the next objective.
- Does NOT make conversation-flow decisions.

The application owns state. The Memory Extractor only understands language.

Model
-----
Designed for Qwen 2.5 3B via Ollama, but model-agnostic by design.
Swap the model name without changing any public interface.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from timing import add_stage_ms

# NOTE: `ollama` is imported lazily inside MemoryExtractor._call_model so the
# data model, enums, validators, and ProjectState (Modules 1–3 import these)
# remain importable in environments without the Ollama client (CI, unit tests).
# The extractor never silently swallows a missing backend — _call_model returns
# None on any exception and the caller produces an AMBIGUOUS fallback.

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public Enums – strongly typed domain values
# ---------------------------------------------------------------------------


class MessageType(str, Enum):
    """Classification of the user's message for pipeline routing."""

    MEANINGFUL = "MEANINGFUL"
    """Message contains new project-relevant information."""

    NO_UPDATE = "NO_UPDATE"
    """Message carries no new project information (greetings, noise, etc.)."""

    AMBIGUOUS = "AMBIGUOUS"
    """Message cannot be confidently interpreted (short context-dependent replies)."""

    END = "END"
    """User is clearly ending or concluding the conversation."""


class Operation(str, Enum):
    """
    Supported state-update operations.

    Designed for extensibility — UPDATE, REMOVE can be added here later
    without changing the public ExtractionResult interface.

    Contract:
        ADD → only valid for list-typed StateFields (appends one item).
        SET → only valid for scalar-typed StateFields (overwrites value).
    """

    ADD = "ADD"
    """Append a single value to a list-typed StateField."""

    SET = "SET"
    """Overwrite the current value of a scalar-typed StateField."""
    # Future: UPDATE, REMOVE


class StateField(str, Enum):
    """
    Recognised ProjectState fields the extractor can populate.

    Names mirror the canonical ProjectState schema (see ``ProjectState`` below).
    Field *values* are the strings used in the LLM JSON contract — swapping
    them would require updating the prompt and validator together.
    """

    PERSONAS = "personas"
    """Who is affected: target audience(s) / user group(s). List-typed."""

    PROBLEMS = "problems"
    """The core pain point(s) / challenge(s) being experienced. List-typed."""

    CURRENT_SOLUTIONS = "current_solutions"
    """How people currently handle the problem (existing solutions / workarounds). List-typed."""

    PAIN_POINTS = "pain_points"
    """Why the builder cares, frustrations, impact. List-typed."""

    EVIDENCE = "evidence"
    """Observations, data, or research validating the problem. List-typed."""

    IMPACTS = "impacts"
    """Consequences / effects of the problem on those affected. List-typed.

    Supporting project knowledge only — this field does NOT gate Empathize
    completion (see ``REQUIRED_FIELDS`` / ``NON_REQUIRED_FIELDS``)."""

    FREQUENCY = "frequency"
    """How often the problem occurs. Scalar-typed (single value)."""


# Which fields are collections (append-only via ADD) vs. scalar (SET).
# Isolated here so the validator and the applier share a single source of
# truth; adding a new list/scalar field only requires updating these sets
# (and StateField itself).
LIST_FIELDS: frozenset[StateField] = frozenset({
    StateField.PERSONAS,
    StateField.PROBLEMS,
    StateField.CURRENT_SOLUTIONS,
    StateField.PAIN_POINTS,
    StateField.EVIDENCE,
    StateField.IMPACTS,
})
SCALAR_FIELDS: frozenset[StateField] = frozenset({StateField.FREQUENCY})

# Which fields gate Empathize completion (WRAP_UP / objective selection).
# ``NON_REQUIRED_FIELDS`` are captured as supporting project knowledge but
# deliberately excluded from the completeness contract so the mentor never
# asks for them and WRAP_UP is unaffected.
# Declared as ORDERED tuples so CoverageReport / objective-engine iteration
# follows StateField's declaration order deterministically (a frozenset
# would make iteration order arbitrary).
REQUIRED_FIELDS: tuple[StateField, ...] = (
    StateField.PERSONAS,
    StateField.PROBLEMS,
    StateField.CURRENT_SOLUTIONS,
    StateField.PAIN_POINTS,
    StateField.EVIDENCE,
    StateField.FREQUENCY,
)
NON_REQUIRED_FIELDS: tuple[StateField, ...] = (StateField.IMPACTS,)


# ---------------------------------------------------------------------------
# ProjectState — canonical Empathize v2 state model
# ---------------------------------------------------------------------------


@dataclass
class ProjectState:
    """
    Canonical project state for the Empathize v2 pipeline.

    This is the *single* state container owned by the application. The Memory
    Extractor reads from it (to avoid re-extracting known information) and
    proposes updates against it; the State Validator (Module 2) and downstream
    modules will consume it next.

    Type semantics:
        personas, problems, current_solutions, pain_points, evidence, impacts
            → list[str]   (append-only via Operation.ADD, de-duplicated).
        frequency
            → Optional[str]   (scalar; overwritten via Operation.SET).

    ``impacts`` is supporting project knowledge (consequences/effects of the
    problem). It is stored alongside the required fields but does NOT gate
    Empathize completion — see ``REQUIRED_FIELDS`` / ``NON_REQUIRED_FIELDS``.

    The trailing two fields (previous_assistant_message / previous_user_message)
    are conversation-flow bookkeeping owned by the application, not the
    extractor — the extractor only *reads* previous_assistant_message to
    disambiguate short replies.
    """

    personas: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    current_solutions: list[str] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    impacts: list[str] = field(default_factory=list)
    frequency: str | None = None

    previous_assistant_message: str | None = None
    previous_user_message: str | None = None

    # ---- typed access ------------------------------------------------

    def get_list(self, name: StateField) -> list[str]:
        """Return the list value for a LIST field (raises for scalar fields)."""
        if name not in LIST_FIELDS:
            raise TypeError(f"{name.value} is not a list-typed StateField")
        return getattr(self, name.value)

    def get_scalar(self, name: StateField) -> str | None:
        """Return the scalar value for a SCALAR field (raises for list fields)."""
        if name not in SCALAR_FIELDS:
            raise TypeError(f"{name.value} is not a scalar-typed StateField")
        return getattr(self, name.value)

    def set_scalar(self, name: StateField, value: str) -> None:
        if name not in SCALAR_FIELDS:
            raise TypeError(f"{name.value} is not a scalar-typed StateField")
        v = value.strip()
        setattr(self, name.value, v if v else None)

    # ---- serialization ------------------------------------------------

    def to_state_dict(self) -> dict[str, Any]:
        """
        Serialize the extraction-relevant fields into the dict format the
        extractor's prompt builder expects.

        List fields are emitted as JSON arrays; scalar ``frequency`` is
        emitted as a string or ``null``. The two ``previous_*`` bookkeeping
        fields are NOT included here — the extractor receives
        ``previous_assistant_message`` as a separate argument.
        """
        out: dict[str, Any] = {}
        for f in LIST_FIELDS:
            out[f.value] = list(self.get_list(f))
        for f in SCALAR_FIELDS:
            out[f.value] = self.get_scalar(f)
        return out


# ---------------------------------------------------------------------------
# Extraction output data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionUpdate:
    """
    A single proposed state change.

    Immutable by design — the extractor proposes; the application applies.

    ``confidence`` (in ``[0.0, 1.0]``) is the deterministic rule-strength
    score attached by the rule-based extractor (see
    ``extraction_pipeline.RuleBasedExtractor``). It is observation-only:
    it never influences StateManager, the Objective Engine, or any decision
    path. LLM-produced updates default to ``1.0`` (no rule signal).
    """

    operation: Operation
    field: StateField
    value: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "field": self.field.value,
            "value": self.value,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ExtractionResult:
    """
    Complete output of one Memory Extractor call.

    Immutable — the extractor produces this; downstream modules consume it.
    """

    message_type: MessageType
    updates: list[ExtractionUpdate] = field(default_factory=list)

    def has_updates(self) -> bool:
        """True when there is at least one proposed state change."""
        return bool(self.updates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_type": self.message_type.value,
            "updates": [u.to_dict() for u in self.updates],
        }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ExtractionValidationError(ValueError):
    """Raised when raw model output cannot be parsed into a valid ExtractionResult."""


class ExtractionValidator:
    """
    Validates raw JSON dicts returned by the LLM.

    All methods are static — no instance state needed. Deterministic:
    no LLM calls, no randomness, no I/O. The same input always produces
    the same outcome (a valid ``ExtractionResult`` or an
    ``ExtractionValidationError``).
    """

    _VALID_MESSAGE_TYPES: frozenset[str] = frozenset(m.value for m in MessageType)
    _VALID_OPERATIONS: frozenset[str] = frozenset(o.value for o in Operation)
    _VALID_FIELDS: frozenset[str] = frozenset(f.value for f in StateField)
    _LIST_FIELD_VALUES: frozenset[str] = frozenset(f.value for f in LIST_FIELDS)
    _SCALAR_FIELD_VALUES: frozenset[str] = frozenset(f.value for f in SCALAR_FIELDS)

    @staticmethod
    def validate(raw: dict[str, Any]) -> ExtractionResult:
        """
        Parse and validate a raw JSON dict into an ExtractionResult.

        Raises
        ------
        ExtractionValidationError
            If any required property is missing, has an invalid type, or
            contains an unrecognised enum value.
        """
        if not isinstance(raw, dict):
            raise ExtractionValidationError(
                f"Expected a JSON object, got {type(raw).__name__}"
            )

        # --- message_type ---
        if "message_type" not in raw:
            raise ExtractionValidationError("Missing required property: 'message_type'")
        raw_mt = raw["message_type"]
        if not isinstance(raw_mt, str):
            raise ExtractionValidationError(
                f"'message_type' must be a string, got {type(raw_mt).__name__}"
            )
        if raw_mt not in ExtractionValidator._VALID_MESSAGE_TYPES:
            raise ExtractionValidationError(
                f"Invalid message_type '{raw_mt}'. "
                f"Allowed: {sorted(ExtractionValidator._VALID_MESSAGE_TYPES)}"
            )
        message_type = MessageType(raw_mt)

        # --- updates ---
        if "updates" not in raw:
            raise ExtractionValidationError("Missing required property: 'updates'")
        raw_updates = raw["updates"]
        if not isinstance(raw_updates, list):
            raise ExtractionValidationError(
                f"'updates' must be an array, got {type(raw_updates).__name__}"
            )

        updates: list[ExtractionUpdate] = []
        for i, item in enumerate(raw_updates):
            update = ExtractionValidator._validate_update(item, index=i)
            updates.append(update)

        return ExtractionResult(message_type=message_type, updates=updates)

    @staticmethod
    def _validate_update(item: Any, index: int) -> ExtractionUpdate:
        if not isinstance(item, dict):
            raise ExtractionValidationError(
                f"updates[{index}]: expected an object, got {type(item).__name__}"
            )

        for key in ("operation", "field", "value"):
            if key not in item:
                raise ExtractionValidationError(
                    f"updates[{index}]: missing required property '{key}'"
                )

        raw_op = item["operation"]
        if not isinstance(raw_op, str):
            raise ExtractionValidationError(
                f"updates[{index}].operation must be a string"
            )
        if raw_op not in ExtractionValidator._VALID_OPERATIONS:
            raise ExtractionValidationError(
                f"updates[{index}]: invalid operation '{raw_op}'. "
                f"Allowed: {sorted(ExtractionValidator._VALID_OPERATIONS)}"
            )

        raw_field = item["field"]
        if not isinstance(raw_field, str):
            raise ExtractionValidationError(
                f"updates[{index}].field must be a string"
            )
        if raw_field not in ExtractionValidator._VALID_FIELDS:
            raise ExtractionValidationError(
                f"updates[{index}]: invalid field '{raw_field}'. "
                f"Allowed: {sorted(ExtractionValidator._VALID_FIELDS)}"
            )

        # Operation × field combination check (deterministic, rule-based).
        # ADD is valid only for list-typed fields; SET is valid only for
        # scalar-typed fields. Rejecting here keeps invalid proposals out
        # of the pipeline so the applier never sees a malformed update.
        if raw_op == Operation.ADD.value and raw_field not in ExtractionValidator._LIST_FIELD_VALUES:
            raise ExtractionValidationError(
                f"updates[{index}]: ADD is only valid for list fields "
                f"{sorted(ExtractionValidator._LIST_FIELD_VALUES)}; "
                f"got '{raw_field}'"
            )
        if raw_op == Operation.SET.value and raw_field not in ExtractionValidator._SCALAR_FIELD_VALUES:
            raise ExtractionValidationError(
                f"updates[{index}]: SET is only valid for scalar fields "
                f"{sorted(ExtractionValidator._SCALAR_FIELD_VALUES)}; "
                f"got '{raw_field}'"
            )

        raw_value = item["value"]
        if not isinstance(raw_value, str):
            raise ExtractionValidationError(
                f"updates[{index}].value must be a string, got {type(raw_value).__name__}"
            )
        if not raw_value.strip():
            raise ExtractionValidationError(
                f"updates[{index}].value must not be empty"
            )

        # Optional rule-strength confidence. Absent -> 1.0 (the LLM path has
        # no rule signal; only the rule-based extractor attaches real scores).
        # Rejected when out of range so malformed LLM output cannot smuggle an
        # invalid confidence into the typed update.
        raw_conf = item.get("confidence", 1.0)
        if isinstance(raw_conf, bool):
            raise ExtractionValidationError(
                f"updates[{index}].confidence must be a number, got bool"
            )
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            raise ExtractionValidationError(
                f"updates[{index}].confidence must be a number, "
                f"got {type(raw_conf).__name__}"
            )
        if not (0.0 <= confidence <= 1.0):
            raise ExtractionValidationError(
                f"updates[{index}].confidence must lie in [0.0, 1.0], "
                f"got {confidence!r}"
            )

        return ExtractionUpdate(
            operation=Operation(raw_op),
            field=StateField(raw_field),
            value=raw_value.strip(),
            confidence=confidence,
        )

    @staticmethod
    def safe_fallback() -> ExtractionResult:
        """
        Return a safe AMBIGUOUS result when the model output cannot be parsed.

        This allows the pipeline to continue gracefully — the State Validator
        and Objective Engine will simply see no new information.
        """
        return ExtractionResult(message_type=MessageType.AMBIGUOUS, updates=[])


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

# Designed for Qwen 2.5 3B. Isolated here so swapping models only requires
# updating this constant (or subclassing MemoryExtractor).
EXTRACTOR_PROMPT_V1 = """\

You are a semantic extraction unit. Your only task is to read a user message \
and extract structured project information into JSON.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: BE CONSERVATIVE. IF UNSURE (< 80% CONFIDENCE), RETURN NO_UPDATE.
HALLUCINATIONS ARE WORSE THAN MISSING DATA. WHEN IN DOUBT → NO_UPDATE.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STRICT RULES:
- Output ONLY valid JSON. No text before or after.
- Never produce conversational responses.
- Never explain your reasoning.
- Never ask questions.
- Return exactly one message_type value.
- Return an empty updates array if there is nothing to extract.

OUTPUT SCHEMA:
{{
  "message_type": "<one of: MEANINGFUL | NO_UPDATE | AMBIGUOUS | END>",
  "updates": [
    {{
      "operation": "<ADD or SET>",
      "field": "<field_name>",
      "value": "<extracted value>",
      "context": "<surrounding text providing context>",
      "domain": "<design thinking domain>",
      "keywords": ["<key_term_1>", "<key_term_2>"],
      "raw_text": "<original text span>",
      "extraction_method": "llm"
    }}
  ]
}}

FIELD NAMES — ONLY THESE ARE VALID:
- personas          : WHO experiences the problem (target audience, user group)          — LIST
- problems          : THE CORE PROBLEM / CHALLENGE being experienced                      — LIST
- current_solutions : HOW PEOPLE CURRENTLY HANDLE THE PROBLEM (existing workarounds)     — LIST
- pain_points       : FRUSTRATIONS, IMPACT, NEGATIVE FEELINGS, WHY IT HURTS             — LIST
- evidence          : OBSERVED PROOF, RESEARCH, DATA, PERSONAL OBSERVATIONS              — LIST
- frequency         : HOW OFTEN the problem occurs (single scalar value)                 — SCALAR

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD DEFINITIONS WITH EXAMPLES — STUDY THESE CAREFULLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▸ PERSONAS — WHO is affected (target audience, user group)
    Examples: "students", "elderly people", "college freshmen", "doctors", "parents",
              "remote workers", "small business owners", "teenagers", "caregivers"
    Keywords: "I want to help [X]", "for [X]", "[X] struggle with", "[X] need"
    ⚠️ NOT a problem, NOT a solution, NOT evidence.

▸ PROBLEMS — THE ACTUAL PROBLEM / CHALLENGE (what goes wrong)
    Examples: "forgetting assignments", "late submissions", "medication non-adherence",
              "missed appointments", "difficulty scheduling", "procrastination",
              "information overload", "poor time management"
    Keywords: "problem is", "issue is", "challenge is", "struggle with", "can't", "unable to"
    ⚠️ NOT a solution, NOT evidence, NOT a pain point (emotion), NOT a persona.

▸ CURRENT_SOLUTIONS — HOW PEOPLE CURRENTLY HANDLE IT (existing workarounds/tools)
    Examples: "sticky notes", "Google Calendar", "phone alarms", "Excel sheets",
              "WhatsApp groups", "manual reminders", "paper planners", "asking friends"
    Keywords: "currently use", "use", "using", "rely on", "workaround", "manage with"
    ⚠️ NOT a problem, NOT a pain point, NOT evidence.

▸ PAIN_POINTS — FRUSTRATIONS, EMOTIONAL IMPACT, NEGATIVE EXPERIENCES, WHY IT HURTS
    Examples: "stress", "anxiety", "feeling overwhelmed", "frustration", "guilt",
              "fear of forgetting", "loss of marks", "health risks", "wasted time",
              "embarrassment", "demotivation", "burnout", "helplessness"
    Keywords: "feel", "stressed", "anxious", "overwhelmed", "frustrated", "worried",
              "scared", "tired of", "exhausted", "hate", "painful"
    ⚠️ NOT a problem (the what), NOT evidence (proof), NOT a solution.
    ⚠️ "I faced this problem" → PAIN_POINT (personal frustration), NOT problem.
    ⚠️ "They feel stressed" → PAIN_POINT. "They miss deadlines" → PROBLEM.

▸ EVIDENCE — OBSERVED PROOF, RESEARCH, DATA, PERSONAL OBSERVATIONS, INTERVIEWS
    Examples: "I interviewed 5 students", "Research shows 70% forget",
              "I have seen many students lose marks", "My grandmother forgets daily",
              "Survey data indicates", "Studies confirm", "I observed", "I witnessed",
              "Personal experience", "My own struggle"
    Keywords: "I saw", "I observed", "I interviewed", "research shows", "data shows",
              "studies show", "statistics", "evidence", "proof", "witnessed",
              "have seen", "personally experienced"
    ⚠️ NOT a problem, NOT a pain point, NOT a solution.
    ⚠️ "Students lose marks" → EVIDENCE (observation). "Late submission" → PROBLEM.
    ⚠️ Personal experience ("I faced this") → EVIDENCE or PAIN_POINT, NOT problem.

▸ FREQUENCY — HOW OFTEN (single scalar value)
    Examples: "daily", "weekly", "monthly", "every semester", "rarely", "often",
              "multiple times a day", "once a week", "every morning"
    Keywords: "every day", "daily", "weekly", "monthly", "often", "rarely",
              "sometimes", "always", "constantly", "occasionally"
    ⚠️ SCALAR only — use SET operation. Never ADD.
    ⚠️ NOT a pain point, NOT evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL DIFFERENTIATION RULES — MEMORIZE THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ EVIDENCE IS NOT A PROBLEM.
   "Students lose marks" → EVIDENCE (observation of consequence)
   "Late submission" → PROBLEM

❌ PAIN POINT IS NOT EVIDENCE.
   "They feel stressed" → PAIN_POINT
   "I observed stress in 5 students" → EVIDENCE

❌ PROBLEM IS NOT A SOLUTION.
   "They use sticky notes" → CURRENT_SOLUTION
   "Sticky notes fall off" → PROBLEM

❌ PERSONA IS NOT A PROBLEM.
   "Students" → PERSONA
   "Students forget" → PROBLEM

❌ FREQUENCY IS NEVER A PAIN POINT.
   "Daily" → FREQUENCY
   "It happens daily" → FREQUENCY

❌ PERSONAL MOTIVATION ("I want to help") → PAIN_POINT (builder's frustration) or NO_UPDATE.
   "I want to make people disciplined" → PAIN_POINT: "lack of discipline" (if confident) else NO_UPDATE.

❌ PERSONAL EXPERIENCE ("I faced this") → EVIDENCE or PAIN_POINT, NOT PROBLEM.
   "I personally faced this issue" → EVIDENCE: "experienced personally" or PAIN_POINT: "experienced personally"

❌ OBSERVATION STATEMENTS ("I have seen...") → EVIDENCE, NOT PROBLEM.
   "I have seen students lose marks" → EVIDENCE: "observed students losing marks"

⚠️ AMBIGUOUS / LOW CONFIDENCE → RETURN NO_UPDATE WITH EMPTY UPDATES.
   Examples: "I want to help", "This is a problem", "It's difficult", "Make it better"
   These lack specific field information.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPERATION RULES (MUST FOLLOW EXACTLY):
- Use ADD only for list fields: personas, problems, current_solutions, pain_points, evidence.
- Use SET only for the scalar field: frequency.
- Never use ADD on frequency. Never use SET on a list field.
- Each ADD adds ONE value to that list. Emit multiple ADD operations for multiple distinct facts.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MESSAGE TYPE DEFINITIONS:
- MEANINGFUL : message contains new project-relevant information → include one or more updates
- NO_UPDATE  : greetings, punctuation, emoji, casual chat, noise, vague motivations → empty updates
- AMBIGUOUS  : "yes", "no", "maybe", "I don't know" without enough context → empty updates
- END        : user is clearly ending the session ("thanks, bye", "we're done") → empty updates

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPREHENSIVE EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User: "i want to build a system which will help old people to take their medications on time because my grandmother always forgets them"
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "ADD",
      "field": "personas",
      "value": "elderly people",
      "context": "elderly people who take medications",
      "domain": "personas",
      "keywords": ["elderly", "people", "medication"],
      "raw_text": "old people to take their medications",
      "extraction_method": "llm"
    }},
    {{
      "operation": "ADD",
      "field": "problems",
      "value": "forgetting to take medication on time",
      "context": "grandmother always forgets",
      "domain": "problems",
      "keywords": ["forgetting", "medication", "time"],
      "raw_text": "my grandmother always forgets them",
      "extraction_method": "llm"
    }},
    {{
      "operation": "ADD",
      "field": "pain_points",
      "value": "grandmother always forgets medication",
      "context": "personal concern for grandmother",
      "domain": "pain_points",
      "keywords": ["grandmother", "forgets", "medication"],
      "raw_text": "my grandmother always forgets them",
      "extraction_method": "llm"
    }},
    {{
      "operation": "ADD",
      "field": "evidence",
      "value": "personal experience with grandmother",
      "context": "firsthand observation of grandmother",
      "domain": "evidence",
      "keywords": ["personal", "experience", "grandmother"],
      "raw_text": "my grandmother always forgets them",
      "extraction_method": "llm"
    }}
  ]
}}

User: "students mostly use WhatsApp groups but messages get buried and it's a weekly thing"
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "ADD",
      "field": "personas",
      "value": "students",
      "context": "students communicating about assignments",
      "domain": "personas",
      "keywords": ["students", "assignments"],
      "raw_text": "students mostly use",
      "extraction_method": "llm"
    }},
    {{
      "operation": "ADD",
      "field": "current_solutions",
      "value": "WhatsApp groups",
      "context": "students using for assignment coordination",
      "domain": "current_solutions",
      "keywords": ["whatsapp", "groups", "assignments"],
      "raw_text": "students mostly use WhatsApp groups",
      "extraction_method": "llm"
    }},
    {{
      "operation": "ADD",
      "field": "pain_points",
      "value": "messages get buried",
      "context": "WhatsApp group communication for assignments",
      "domain": "pain_points",
      "keywords": ["messages", "buried", "communication"],
      "raw_text": "messages get buried",
      "extraction_method": "llm"
    }},
    {{
      "operation": "SET",
      "field": "frequency",
      "value": "weekly",
      "context": "assignment deadlines and communication",
      "domain": "frequency",
      "keywords": ["weekly", "assignments"],
      "raw_text": "it's a weekly thing",
      "extraction_method": "llm"
    }}
  ]
}}

User: "hello"
Output:
{{ "message_type": "NO_UPDATE", "updates": [] }}

User: "yes"
Output:
{{ "message_type": "AMBIGUOUS", "updates": [] }}

User: "they currently use sticky notes and phone alarms but it doesn't work well"
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "ADD",
      "field": "current_solutions",
      "value": "sticky notes and phone alarms",
      "context": "current reminder methods for medication",
      "domain": "current_solutions",
      "keywords": ["sticky", "notes", "phone", "alarms", "medication"],
      "raw_text": "they currently use sticky notes and phone alarms",
      "extraction_method": "llm"
    }},
    {{
      "operation": "ADD",
      "field": "pain_points",
      "value": "sticky notes and phone alarms do not work well",
      "context": "evaluation of current reminder methods",
      "domain": "pain_points",
      "keywords": ["sticky", "notes", "phone", "alarms", "work", "well"],
      "raw_text": "but it doesn't work well",
      "extraction_method": "llm"
    }}
  ]
}}

User: "it happens every single day, sometimes multiple times"
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "SET",
      "field": "frequency",
      "value": "every day, sometimes multiple times",
      "context": "frequency of the problem occurrence",
      "domain": "frequency",
      "keywords": ["every", "day", "multiple", "times"],
      "raw_text": "it happens every single day, sometimes multiple times",
      "extraction_method": "llm"
    }}
  ]
}}

User: "thanks, that's all I needed"
Output:
{{ "message_type": "END", "updates": [] }}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL DISCRIMINATION EXAMPLES (STUDY THESE PATTERNS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User: "This is the most commonly faced problem even I have faced it."
→ NOT a problem statement. Personal experience = EVIDENCE or PAIN_POINT.
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "ADD",
      "field": "evidence",
      "value": "experienced personally",
      "context": "personal experience with the problem",
      "domain": "evidence",
      "keywords": ["experienced", "personally"],
      "contributing_factors": [],
      "qualifiers": [],
      "raw_text": "I have faced it",
      "extraction_method": "llm"
    }},
    {{
      "operation": "ADD",
      "field": "pain_points",
      "value": "experienced personally",
      "context": "personal frustration with the problem",
      "domain": "pain_points",
      "keywords": ["experienced", "personally"],
      "contributing_factors": [],
      "qualifiers": [],
      "raw_text": "I have faced it",
      "extraction_method": "llm"
    }}
  ]
}}

User: "I have seen students do late submission and lose marks."
→ OBSERVATION = EVIDENCE. NOT a problem.
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "ADD",
      "field": "evidence",
      "value": "observed students losing marks due to late submissions",
      "context": "observation of student outcomes",
      "domain": "evidence",
      "keywords": ["observed", "students", "losing", "marks", "late", "submissions"],
      "contributing_factors": [],
      "qualifiers": [],
      "raw_text": "I have seen students do late submission and lose marks",
      "extraction_method": "llm"
    }}
  ]
}}

User: "I want to make people disciplined."
→ VAGUE MOTIVATION. Low confidence. → NO_UPDATE.
Output:
{{ "message_type": "NO_UPDATE", "updates": [] }}
(If highly confident: ADD pain_points "lack of discipline" — but default to NO_UPDATE)

User: "Students usually set reminders."
→ CURRENT_SOLUTION.
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "ADD",
      "field": "current_solutions",
      "value": "setting reminders",
      "context": "students using reminders for assignments",
      "domain": "current_solutions",
      "keywords": ["setting", "reminders", "students", "assignments"],
      "contributing_factors": [],
      "qualifiers": [],
      "raw_text": "Students usually set reminders",
      "extraction_method": "llm"
    }}
  ]
}}

User: "They feel stressed."
→ PAIN_POINT (emotional state).
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "ADD",
      "field": "pain_points",
      "value": "feel stressed",
      "context": "emotional state of students",
      "domain": "pain_points",
      "keywords": ["feel", "stressed"],
      "contributing_factors": [],
      "qualifiers": [],
      "raw_text": "They feel stressed",
      "extraction_method": "llm"
    }}
  ]
}}

User: "It happens almost every day."
→ FREQUENCY.
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "SET",
      "field": "frequency",
      "value": "almost every day",
      "context": "frequency of problem occurrence",
      "domain": "frequency",
      "keywords": ["almost", "every", "day"],
      "raw_text": "It happens almost every day",
      "extraction_method": "llm"
    }}
  ]
}}

User: "I interviewed five students and they all forget assignments."
→ EVIDENCE (interview data) + PROBLEM.
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "ADD",
      "field": "evidence",
      "value": "interviewed five students who all forget assignments",
      "context": "interview data from five students",
      "domain": "evidence",
      "keywords": ["interviewed", "five", "students", "forget", "assignments"],
      "raw_text": "I interviewed five students and they all forget assignments",
      "extraction_method": "llm"
    }},
    {{
      "operation": "ADD",
      "field": "problems",
      "value": "forgetting assignments",
      "context": "students forgetting assignments per interview data",
      "domain": "problems",
      "keywords": ["forgetting", "assignments", "students"],
      "raw_text": "they all forget assignments",
      "extraction_method": "llm"
    }}
  ]
}}

User: "My grandmother struggles with medication timing."
→ PERSONA + PROBLEM + EVIDENCE (personal observation).
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "ADD",
      "field": "personas",
      "value": "elderly people",
      "context": "grandmother needing medication timing help",
      "domain": "personas",
      "keywords": ["elderly", "people", "medication", "timing"],
      "raw_text": "My grandmother struggles with medication timing",
      "extraction_method": "llm"
    }},
    {{
      "operation": "ADD",
      "field": "problems",
      "value": "medication timing difficulties",
      "context": "grandmother's medication schedule",
      "domain": "problems",
      "keywords": ["medication", "timing", "difficulties", "grandmother"],
      "raw_text": "My grandmother struggles with medication timing",
      "extraction_method": "llm"
    }},
    {{
      "operation": "ADD",
      "field": "evidence",
      "value": "personal observation of grandmother",
      "context": "firsthand observation of grandmother's struggles",
      "domain": "evidence",
      "keywords": ["personal", "observation", "grandmother"],
      "raw_text": "My grandmother struggles with medication timing",
      "extraction_method": "llm"
    }}
  ]
}}

User: "Research shows 70% of students miss deadlines."
→ EVIDENCE (research statistic).
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "ADD",
      "field": "evidence",
      "value": "research shows 70% of students miss deadlines",
      "context": "research statistic on student deadline misses",
      "domain": "evidence",
      "keywords": ["research", "shows", "70%", "students", "miss", "deadlines"],
      "raw_text": "Research shows 70% of students miss deadlines",
      "extraction_method": "llm"
    }}
  ]
}}

User: "The problem is late submission and the solution is reminders."
→ PROBLEM + CURRENT_SOLUTION (if reminders exist) or just PROBLEM.
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "ADD",
      "field": "problems",
      "value": "late submission",
      "context": "problem statement with solution mention",
      "domain": "problems",
      "keywords": ["late", "submission"],
      "raw_text": "The problem is late submission",
      "extraction_method": "llm"
    }}
  ]
}}

User: "Forgetting things is stressful."
→ PAIN_POINT (stress) + PROBLEM (forgetting).
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "ADD",
      "field": "problems",
      "value": "forgetting things",
      "context": "forgetting causing stress",
      "domain": "problems",
      "keywords": ["forgetting", "things"],
      "raw_text": "Forgetting things is stressful",
      "extraction_method": "llm"
    }},
    {{
      "operation": "ADD",
      "field": "pain_points",
      "value": "stressful",
      "context": "forgetting things causes stress",
      "domain": "pain_points",
      "keywords": ["stressful"],
      "raw_text": "Forgetting things is stressful",
      "extraction_method": "llm"
    }}
  ]
}}

User: "I want to build an app for this."
→ NO_UPDATE (builder intent, no project info extracted).
Output:
{{ "message_type": "NO_UPDATE", "updates": [] }}

User: "Procrastination causes missed deadlines."
→ PROBLEM (procrastination) + EVIDENCE/RESULT (missed deadlines).
Output:
{{
  "message_type": "MEANINGFUL",
  "updates": [
    {{
      "operation": "ADD",
      "field": "problems",
      "value": "procrastination",
      "context": "procrastination leading to missed deadlines",
      "domain": "problems",
      "keywords": ["procrastination", "missed", "deadlines"],
      "raw_text": "Procrastination causes missed deadlines",
      "extraction_method": "llm"
    }},
    {{
      "operation": "ADD",
      "field": "evidence",
      "value": "causes missed deadlines",
      "context": "observation of procrastination consequence",
      "domain": "evidence",
      "keywords": ["causes", "missed", "deadlines"],
      "raw_text": "Procrastination causes missed deadlines",
      "extraction_method": "llm"
    }}
  ]
}}
      "domain": "evidence",
      "keywords": ["causes", "missed", "deadlines"],
      "contributing_factors": [],
      "qualifiers": [],
      "raw_text": "Procrastination causes missed deadlines",
      "extraction_method": "llm"
    }}
  ]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CURRENT PROJECT STATE:
{project_state}

{previous_context}

USER MESSAGE:
"{user_message}"

OUTPUT (JSON only):
"""

EXTRACTOR_PROMPT = """\\
You are a conservative semantic extraction unit.

STRICT RULES:
- Output ONLY valid JSON. No other text.
- Return exactly one message_type.
- Empty updates array if nothing to extract.
- Update entries contain only: operation, field, value.

VALID FIELDS:
- personas : WHO (ADD)
- problems : CHALLENGE (ADD)
- current_solutions : WORKAROUNDS (ADD)
- pain_points : FRUSTRATIONS (ADD)
- evidence : OBSERVED PROOF (ADD)
- impacts : CONSEQUENCES (ADD)
- frequency : HOW OFTEN (SET)

IMPACT RULE:
- An impact is a consequence of the problem, not the problem itself.

MESSAGE TYPES:
- MEANINGFUL : new info (updates)
- NO_UPDATE : greetings, noise, vague (empty)
- AMBIGUOUS : short reply, no context (empty)
- END : session end (empty)

EXAMPLES:

User: "Help elderly people take medication on time. Grandmother always forgets."
Output:
{{ "message_type": "MEANINGFUL", "updates": [
  {{ "operation": "ADD", "field": "personas", "value": "elderly people" }},
  {{ "operation": "ADD", "field": "problems", "value": "forgetting medication" }},
  {{ "operation": "ADD", "field": "pain_points", "value": "grandmother always forgets" }},
  {{ "operation": "ADD", "field": "evidence", "value": "grandmother always forgets" }}
] }}

User: "Students mostly use WhatsApp groups but messages get buried and it's a weekly thing."
Output:
{{ "message_type": "MEANINGFUL", "updates": [
  {{ "operation": "ADD", "field": "current_solutions", "value": "WhatsApp groups" }},
  {{ "operation": "ADD", "field": "pain_points", "value": "messages get buried" }},
  {{ "operation": "SET", "field": "frequency", "value": "weekly" }}
] }}

User: "Even I have faced this problem."
Output:
{{ "message_type": "MEANINGFUL", "updates": [
  {{ "operation": "ADD", "field": "evidence", "value": "experienced personally" }},
  {{ "operation": "ADD", "field": "pain_points", "value": "personal frustration" }}
] }}

User: "I have seen students lose marks from late submissions."
Output:
{{ "message_type": "MEANINGFUL", "updates": [ {{ "operation": "ADD", "field": "evidence", "value": "observed students losing marks" }} ] }}

User: "I want to make people disciplined."
Output:
{{ "message_type": "NO_UPDATE", "updates": [] }}

User: "They feel stressed."
Output:
{{ "message_type": "MEANINGFUL", "updates": [ {{ "operation": "ADD", "field": "pain_points", "value": "stressed" }} ] }}

User: "Procrastination causes missed deadlines."
Output:
{{ "message_type": "MEANINGFUL", "updates": [
  {{ "operation": "ADD", "field": "problems", "value": "procrastination" }},
  {{ "operation": "ADD", "field": "impacts", "value": "missed deadlines" }}
] }}

User: "Forgetting things is stressful."
Output:
{{ "message_type": "MEANINGFUL", "updates": [
  {{ "operation": "ADD", "field": "problems", "value": "forgetting things" }},
  {{ "operation": "ADD", "field": "pain_points", "value": "stressful" }}
] }}

User: "The problem is late submission and the solution is reminders."
Output:
{{ "message_type": "MEANINGFUL", "updates": [ {{ "operation": "ADD", "field": "problems", "value": "late submission" }} ] }}

User: "hello"
Output:
{{ "message_type": "NO_UPDATE", "updates": [] }}

User: "yes"
Output:
{{ "message_type": "AMBIGUOUS", "updates": [] }}

User: "thanks, that's all"
Output:
{{ "message_type": "END", "updates": [] }}


CURRENT PROJECT STATE:
{project_state}

{previous_context}

USER MESSAGE:
"{user_message}"

OUTPUT (JSON only):
"""


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> and <thinking>…</thinking> blocks from model output."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _extract_json(text: str) -> dict[str, Any] | None:
    """
    Extract the first valid JSON object from arbitrary model output.

    Tries strict parse first, then scans for the outermost { } block.
    Returns None if no valid JSON object is found.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# MemoryExtractor – public interface
# ---------------------------------------------------------------------------


class MemoryExtractor:
    """
    Semantic understanding component of the Empathize v2 pipeline.

    Responsibilities
    ----------------
    - Build the extraction prompt from current project state and user message.
    - Call the LLM (via Ollama) with temperature=0 for deterministic output.
    - Parse and validate the JSON response into a typed ExtractionResult.
    - Fall back to AMBIGUOUS gracefully on any failure.

    Non-responsibilities
    --------------------
    - Does NOT update ProjectState.
    - Does NOT route conversation flow.
    - Does NOT generate questions or replies.

    Model Independence
    ------------------
    Model-specific logic is isolated to the ``model_name`` constructor argument
    and the ``EXTRACTOR_PROMPT`` constant. To replace Qwen, pass a different
    model name and (optionally) override ``_build_prompt()``.

    Usage
    -----
    >>> state = ProjectState()
    >>> extractor = MemoryExtractor(model_name="qwen2.5:3b")
    >>> result = extractor.extract(
    ...     user_message="I want to help elderly people with medication.",
    ...     project_state=state,
    ... )
    >>> result.message_type
    <MessageType.MEANINGFUL: 'MEANINGFUL'>
    """

    # Ollama generation options — tuned for deterministic JSON output
    _GENERATION_OPTIONS: dict[str, Any] = {
        "temperature": 0.0,   # deterministic: no creativity needed for extraction
        "top_p": 1.0,
        "num_predict": 300,   # enough for up to 6 updates + overhead
        "repeat_penalty": 1.0,
    }

    def __init__(self, model_name: str) -> None:
        """
        Parameters
        ----------
        model_name:
            Ollama model to use for extraction (e.g. "qwen2.5:3b").
            The extractor is model-agnostic — swap freely without code changes.
        """
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        user_message: str,
        project_state: ProjectState | dict[str, Any],
        previous_assistant_message: str | None = None,
        _timing: dict[str, float] | None = None,
        _prof=None,
    ) -> ExtractionResult:
        """
        Extract structured information from one user message.

        Parameters
        ----------
        user_message:
            The latest message from the user. Required.
        project_state:
            The current ProjectState (preferred), or a dict produced by
            ``ProjectState.to_state_dict()``. Used so the model knows what has
            already been captured and avoids re-extracting known information.
        previous_assistant_message:
            The immediately preceding assistant message, if available.
            Used to resolve contextual replies like "yes" or "no".
        _timing:
            Optional read-only timing accumulator (see ``timing.add_stage_ms``).
            When provided, per-sub-stage elapsed ms is recorded into it.
            Never influences extraction behavior.
        _prof:
            Optional ``timing.TurnProfiler`` receiving one per-LLM-call
            record. Observation-only; ``None`` disables profiling with
            zero behavior change.

        Returns
        -------
        ExtractionResult
            Always returns a valid result. Falls back to AMBIGUOUS on failure.
        """
        _pre = time.perf_counter() if _timing is not None else None
        if not user_message or not user_message.strip():
            if _pre is not None:
                add_stage_ms(_timing, "preprocess", _pre, time.perf_counter())
            logger.debug("[MemoryExtractor] Empty user message → NO_UPDATE")
            return ExtractionResult(message_type=MessageType.NO_UPDATE, updates=[])
        if _pre is not None:
            add_stage_ms(_timing, "preprocess", _pre, time.perf_counter())

        prompt = self._build_prompt(
            user_message, project_state, previous_assistant_message, _timing=_timing
        )

        raw_output = self._call_model(prompt, _timing=_timing, _prof=_prof)
        if raw_output is None:
            logger.warning("[MemoryExtractor] Model call failed → AMBIGUOUS fallback")
            return ExtractionValidator.safe_fallback()

        return self._parse_and_validate(raw_output, _timing=_timing)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Static template cache — built once at import time.
    #
    # EXTRACTOR_PROMPT (~21 KB) is parsed by str.format() on every call.
    # Over 99% of it is completely static (role, rules, examples, schema).
    # Splitting at the first dynamic placeholder ({project_state}) and
    # pre-resolving the {{ → {  and  }} → } escape sequences on the head
    # lets us avoid re-parsing the full template each turn. Only the tiny
    # ~80-byte tail (two remaining placeholders) is format()ed per call.
    # ------------------------------------------------------------------

    # Head: everything before {project_state}, with format escapes resolved
    _EXTRACTOR_STATIC: str = (
        EXTRACTOR_PROMPT.split("{project_state}")[0]
        .replace("{{", "{")
        .replace("}}", "}")
    )
    """Template head (pre-formatted): role, rules, field definitions, examples, section headers."""

    _EXTRACTOR_DYNAMIC_TAIL: str = EXTRACTOR_PROMPT.split("{project_state}")[1]
    """Raw template tail after {project_state} — still carries {{/}} escapes and placeholders."""

    def _build_prompt(
        self,
        user_message: str,
        project_state: ProjectState | dict[str, Any],
        previous_assistant_message: str | None,
        _timing: dict[str, float] | None = None,
    ) -> str:
        """
        Render the extraction prompt with runtime context.

        List-typed StateFields are rendered as JSON arrays, scalar fields as
        a quoted string or ``null``. Accepts either a ``ProjectState``
        instance or the dict shape produced by ``ProjectState.to_state_dict()``
        so callers don't need to convert first.
        """
        _t = time.perf_counter() if _timing is not None else None
        if isinstance(project_state, ProjectState):
            state_dict = project_state.to_state_dict()
        else:
            state_dict = project_state

        state_lines = []
        for f in StateField:
            val = state_dict.get(f.value)
            if f in LIST_FIELDS:
                # Render collections as JSON arrays so the model sees list shape.
                arr = list(val) if isinstance(val, (list, tuple)) else []
                display = json.dumps(arr) if arr else "[]"
            else:
                display = f'"{val}"' if val else "null"
            state_lines.append(f"  {f.value}: {display}")
        state_str = "{\n" + ",\n".join(state_lines) + "\n}"

        # Inject previous assistant message only when available
        if previous_assistant_message and previous_assistant_message.strip():
            prev_ctx = (
                f"PREVIOUS ASSISTANT MESSAGE (use to interpret short replies like 'yes'/'no'):\n"
                f'"{previous_assistant_message.strip()}"'
            )
        else:
            prev_ctx = ""

        returned_tail = self._EXTRACTOR_DYNAMIC_TAIL.format(
            previous_context=prev_ctx,
            user_message=user_message.strip(),
        )

        if _t is not None:
            add_stage_ms(_timing, "llm_prompt", _t, time.perf_counter())
        return self._EXTRACTOR_STATIC + state_str + returned_tail

    def _call_model(self, prompt: str, _timing: dict[str, float] | None = None, _prof=None) -> str | None:
        """
        Call the Ollama model and return raw text output.

        Returns None on any exception so the caller can apply safe fallback.
        When ``_prof`` (a ``timing.TurnProfiler``) is given, one per-call
        record is appended; the call itself is byte-identical either way.
        """
        _t = time.perf_counter() if _timing is not None else None
        try:
            from timing import timed_ollama_chat  # local import: keeps module import-light

            response = timed_ollama_chat(
                _prof,
                purpose="extraction",
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options=self._GENERATION_OPTIONS,
                gated=True,  # hybrid gate decides whether this call runs at all
            )
            if _timing is not None and "prompt_eval_count" in response:
                _timing["prompt_tokens"] = response["prompt_eval_count"]
            return response["message"]["content"]
        except Exception as exc:
            logger.error("[MemoryExtractor] Ollama call failed: %s", exc)
            return None
        finally:
            if _t is not None:
                add_stage_ms(_timing, "llm_api", _t, time.perf_counter())

    def _parse_and_validate(self, raw_output: str, _timing: dict[str, float] | None = None) -> ExtractionResult:
        """
        Strip model artefacts, extract JSON, validate, and return a typed result.

        Falls back to AMBIGUOUS on any parse or validation error.
        """
        _t = time.perf_counter() if _timing is not None else None
        cleaned = _strip_think_tags(raw_output)
        parsed = _extract_json(cleaned)
        if _t is not None:
            add_stage_ms(_timing, "llm_parse", _t, time.perf_counter())

        if parsed is None:
            logger.warning(
                "[MemoryExtractor] Could not extract JSON from output: %r",
                cleaned[:200],
            )
            return ExtractionValidator.safe_fallback()

        _v = time.perf_counter() if _timing is not None else None
        try:
            result = ExtractionValidator.validate(parsed)
            if _v is not None:
                add_stage_ms(_timing, "validation_llm", _v, time.perf_counter())
            logger.debug(
                "[MemoryExtractor] Extracted: type=%s updates=%d",
                result.message_type.value,
                len(result.updates),
            )
            return result
        except ExtractionValidationError as exc:
            if _v is not None:
                add_stage_ms(_timing, "validation_llm", _v, time.perf_counter())
            logger.warning("[MemoryExtractor] Validation failed: %s", exc)
            return ExtractionValidator.safe_fallback()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def resolve_extractor_model() -> str:
    """
    Determine which Ollama model to use for extraction.

    Priority:
    1. EXTRACTOR_MODEL env var (explicit override)
    2. MENTOR_MODEL env var (project-wide model)
    3. "qwen2.5:3b" (default)

    Availability is NOT checked here — Ollama will surface an error at
    call time if the model is missing, which the extractor catches gracefully.
    """
    return (
        os.getenv("EXTRACTOR_MODEL")
        or os.getenv("MENTOR_MODEL")
        or "qwen2.5:3b"
    )
