"""
contracts.py — Architecture v1 shared backend contracts.

This module defines the canonical data models exchanged between pipeline stages.
All models are frozen dataclasses to ensure immutability and deterministic behavior.

Architecture role
-----------------
These contracts form the static type boundary between stages. The application
owns the logic; the LLM only produces natural language from structured plans.

Component ownership
-------------------
- Extractor → ExtractionResult, ExtractedFact
- Lifecycle Manager → LifecycleDecision

No component mutates another's contract. All data flows forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Shared Enums
# ---------------------------------------------------------------------------


class ExtractionType(str, Enum):
    """Classification of what the extractor identified in the user message."""

    ENTITY = "ENTITY"
    """A concrete entity: persona, problem, solution, pain point, evidence, frequency."""

    RELATION = "RELATION"
    """A relationship between entities (e.g., "persona X experiences problem Y")."""

    CONTRADICTION = "CONTRADICTION"
    """User statement contradicts previously extracted information."""

    AMBIGUITY = "AMBIGUITY"
    """User statement is too vague to extract with confidence."""

    NO_EXTRACTION = "NO_EXTRACTION"
    """Message contains no extractable design-thinking content (greetings, chatter)."""


class ConfidenceLevel(str, Enum):
    """Discrete confidence bands for deterministic reasoning."""

    HIGH = "HIGH"
    """≥ 0.8 — strong evidence, single clear interpretation."""

    MEDIUM = "MEDIUM"
    """0.5–0.79 — some evidence, but alternatives exist."""

    LOW = "LOW"
    """< 0.5 — weak evidence, highly ambiguous."""

    UNKNOWN = "UNKNOWN"
    """Confidence not yet assessed."""


class ResponseStrategy(str, Enum):
    """High-level strategy for the mentor's natural-language response.
    Owned by Response Planner."""

    ASK_QUESTION = "ASK_QUESTION"
    """Pursue a targeted hypothesis — ask ONE focused question."""

    ACKNOWLEDGE = "ACKNOWLEDGE"
    """Briefly acknowledge user input, then continue gathering."""

    CLARIFY = "CLARIFY"
    """Ask user to disambiguate a vague or contradictory statement."""

    GENERATE_SUMMARY = "GENERATE_SUMMARY"
    """Present synthesized understanding and ask for confirmation."""

    SUMMARIZE = "SUMMARIZE"
    """Present synthesized understanding and ask for confirmation."""

    CORRECT = "CORRECT"
    """Gently correct a contradiction the user introduced."""


class LifecycleAction(str, Enum):
    """Stage-level lifecycle action. Owned by Lifecycle Manager."""

    CONTINUE = "CONTINUE"
    """Remain in current stage, continue gathering."""

    TRANSITION = "TRANSITION"
    """Move to next stage/phase."""

    PAUSE = "PAUSE"
    """Temporarily suspend (e.g., waiting for external input)."""

    END_SESSION = "END_SESSION"
    """Conclude the entire session."""


# ---------------------------------------------------------------------------
# Core Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedFact:
    """
    A single atomic fact extracted from the user message.

    Immutable — the extractor proposes; the application applies.
    Owned by: Extractor

    Every field must be either:
    (1) explicitly stated by the user, or
    (2) a lossless normalization of explicitly stated information.

    If a field requires world knowledge or causal reasoning, it belongs in the
    InferenceEngine, not here.
    """

    extraction_type: ExtractionType
    """What kind of extraction this is."""

    field: str
    """Canonical field name (e.g., "personas", "problems", "frequency")."""

    value: str
    """The extracted text value, normalized (lowercase, trimmed)."""

    confidence: ConfidenceLevel
    """Extractor's confidence in this fact."""

    # --- Rich extraction fields (v1) ---
    # Only explicitly stated or losslessly normalized information.

    context: str = ""
    """The surrounding phrase from the user message that provides context.
    Must be a direct substring or minimal normalization (e.g., pronoun resolution)."""

    domain: str = ""
    """Design thinking domain this fact belongs to.
    One of: personas, problems, current_solutions, pain_points, evidence, frequency."""

    keywords: tuple[str, ...] = field(default_factory=tuple)
    """Key terms from the extraction (lowercase, from user's words only)."""

    raw_text: str = ""
    """Exact text span from user message that led to this extraction (verbatim)."""

    extraction_method: str = "llm"
    """How this was extracted: "llm" or "rule"."""

    source_span: tuple[int, int] | None = None
    """Optional (start, end) character indices in the user message."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Extensible metadata: {"source": "llm|rule", "model": "qwen2.5:3b", ...}."""


@dataclass(frozen=True)
class ExtractionResult:
    """
    Complete output of one extraction pass over a user message.

    The extractor does NOT update state — it only proposes facts.
    Owned by: Extractor
    """

    message_type: ExtractionType
    """Overall classification of the user message."""

    facts: tuple[ExtractedFact, ...] = field(default_factory=tuple)
    """All facts proposed for this turn. Empty if nothing extracted."""

    raw_output: str | None = None
    """Raw LLM output for debugging/auditing (optional)."""

    def has_extractions(self) -> bool:
        """True if at least one fact was proposed."""
        return bool(self.facts)


@dataclass(frozen=True)
class LifecycleDecision:
    """
    Stage-level lifecycle decision.

    Pure function of (inference_result, response_plan, project_state).
    Owned by: Lifecycle Manager
    """

    action: LifecycleAction
    """What the conversation should do next at the stage level."""

    target_stage: str | None = None
    """If TRANSITION: name of the next stage/phase (e.g., "Define", "Ideate")."""

    reason: str | None = None
    """Human-readable rationale for this decision."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Extensible: {"turn": 12, "completed_hypotheses": 6, ...}."""


# ---------------------------------------------------------------------------
# Convenience Type Aliases
# ---------------------------------------------------------------------------

FactSet = tuple[ExtractedFact, ...]