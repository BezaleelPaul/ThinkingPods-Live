"""
insight_detection.py — Insight Detection layer for the mentor pipeline.

Recognises when the user has shared a meaningful insight that deserves
acknowledgement or deeper exploration, separate from information gathering.

Purely deterministic.  NEVER changes extraction, lifecycle, ProjectState,
or ObjectiveEngine.  Insight detection is coaching guidance only — an
optional instruction bullet appended to the LLM prompt instructions.

Insight Types
-------------
NONE
    No significant insight detected.
NEW_PATTERN
    User connects two separate facts or ideas.
ROOT_CAUSE_HINT
    User explains *why* something happens ("because...", "due to...").
SURPRISING_OBSERVATION
    User shares something unexpected or counterintuitive.
USER_LEARNING
    User expresses a realization ("I hadn't thought about that",
    "Now I realise", "I think the real issue is...").
STRONG_EVIDENCE
    User provides numbers, research, interviews, measurements, or
    observed behavior.
CONSTRAINT
    User states a boundary: budget, time, technology, policy, resources.
CONTRADICTION
    User answer conflicts with prior knowledge or earlier statements.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

__all__ = [
    "InsightType",
    "InsightConfidence",
    "detect_insight",
    "insight_instruction_bullet",
    "insight_diagnostics_section",
]


class InsightType(str, Enum):
    NONE = "NONE"
    NEW_PATTERN = "NEW_PATTERN"
    ROOT_CAUSE_HINT = "ROOT_CAUSE_HINT"
    SURPRISING_OBSERVATION = "SURPRISING_OBSERVATION"
    USER_LEARNING = "USER_LEARNING"
    STRONG_EVIDENCE = "STRONG_EVIDENCE"
    CONSTRAINT = "CONSTRAINT"
    CONTRADICTION = "CONTRADICTION"


class InsightConfidence(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# ---------------------------------------------------------------------------
# Guidance strings rendered as instruction bullets
# ---------------------------------------------------------------------------

_INSIGHT_GUIDANCE: dict[InsightType, str] = {
    InsightType.NEW_PATTERN: (
        "The user has connected two distinct facts or ideas. "
        "Highlight this connection briefly before continuing."
    ),
    InsightType.ROOT_CAUSE_HINT: (
        "The user may have identified a possible root cause. "
        "Explore it before moving on."
    ),
    InsightType.SURPRISING_OBSERVATION: (
        "The user shared a surprising or unexpected observation. "
        "Acknowledge what makes it notable."
    ),
    InsightType.USER_LEARNING: (
        "The user experienced a realization or learning moment. "
        "Validate this insight before asking the next question."
    ),
    InsightType.STRONG_EVIDENCE: (
        "The user provided valuable evidence, data, or observed behavior. "
        "Acknowledge it briefly before continuing."
    ),
    InsightType.CONSTRAINT: (
        "The user stated a key constraint (budget, time, technology, or policy). "
        "Note how this boundary shapes the problem space."
    ),
    InsightType.CONTRADICTION: (
        "The user's statement conflicts with earlier information. "
        "Gently clarify the discrepancy."
    ),
}


def guidance_for_insight(insight_type: InsightType) -> Optional[str]:
    return _INSIGHT_GUIDANCE.get(insight_type)


# ---------------------------------------------------------------------------
# Keyword / pattern dictionaries for deterministic detection
# ---------------------------------------------------------------------------

_LEARNING_SIGNALS = (
    "i hadn't thought about",
    "now i realize",
    "now i realise",
    "i think the real issue",
    "it turns out",
    "makes me realize",
    "makes me realise",
    "i see now",
    "that's actually why",
)

_ROOT_CAUSE_SIGNALS = (
    "because",
    "due to",
    "the reason is",
    "root cause",
    "stems from",
    "comes down to",
    "triggers",
    "leads to",
    "caused by",
)

_EVIDENCE_SIGNALS = (
    "percent",
    "%",
    "data",
    "survey",
    "interview",
    "measured",
    "observed",
    "statistics",
    "metrics",
    "users reported",
    "times a day",
    "hours per",
    "dollars",
    "$",
    "cost",
)

_CONSTRAINT_SIGNALS = (
    "budget",
    "deadline",
    "time limit",
    "limited to",
    "cannot use",
    "policy",
    "tech stack",
    "resources",
    "only have",
    "restriction",
    "must use",
)

_SURPRISE_SIGNALS = (
    "surprisingly",
    "unexpectedly",
    "oddly",
    "funny thing is",
    "strangely",
    "ironically",
    "never expected",
)

_PATTERN_SIGNALS = (
    "every time",
    "whenever",
    "pattern",
    "correlation",
    "connected to",
    "linked with",
    "goes hand in hand",
)


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------


def detect_insight(
    *,
    user_message: str = "",
    current_objective: str = "",
    state: Optional[dict] = None,
    memory: Optional[dict] = None,
    recovery_category: Optional[str] = None,
    extraction_updates: Optional[list] = None,
) -> tuple[InsightType, InsightConfidence, str, list[str]]:
    """Detect whether the user's message contains a meaningful insight.

    Parameters
    ----------
    user_message:
        The latest raw user text.
    current_objective:
        The active objective value (e.g. ``"PROBLEMS"``).
    state:
        Current ``ProjectState`` dict.
    memory:
        Conversation memory dict.
    recovery_category:
        Recovery analysis category (e.g. ``"CONTRADICTION"``).
    extraction_updates:
        Read-only list of extraction updates produced this turn.

    Returns
    -------
    tuple[InsightType, InsightConfidence, str, list[str]]
        ``(insight_type, confidence, explanation, supporting_signals)``
    """
    msg_lower = (user_message or "").lower().strip()
    signals: list[str] = []

    # 1. Contradiction from recovery monitor or explicit contradiction phrasing
    if recovery_category == "CONTRADICTION" or "contradict" in msg_lower:
        signals.append(f"recovery_category={recovery_category}" if recovery_category else "contradiction phrasing")
        return (
            InsightType.CONTRADICTION,
            InsightConfidence.HIGH,
            "User statement conflicts with prior knowledge or statements.",
            signals,
        )

    # 2. User Learning / Realization
    for sig in _LEARNING_SIGNALS:
        if sig in msg_lower:
            signals.append(f"learning_phrase='{sig}'")
            return (
                InsightType.USER_LEARNING,
                InsightConfidence.HIGH,
                f"User expressed a realization using phrase '{sig}'.",
                signals,
            )

    # 3. Surprising Observation
    for sig in _SURPRISE_SIGNALS:
        if sig in msg_lower:
            signals.append(f"surprise_phrase='{sig}'")
            return (
                InsightType.SURPRISING_OBSERVATION,
                InsightConfidence.MEDIUM,
                f"User noted an unexpected observation using '{sig}'.",
                signals,
            )

    # 4. New Pattern (connecting facts or pattern keywords)
    for sig in _PATTERN_SIGNALS:
        if sig in msg_lower:
            signals.append(f"pattern_phrase='{sig}'")
            return (
                InsightType.NEW_PATTERN,
                InsightConfidence.MEDIUM,
                f"User connected ideas using pattern signal '{sig}'.",
                signals,
            )

    # Check for multi-fact extraction or connective "and... because"
    if extraction_updates and len(extraction_updates) >= 2:
        fields_touched = [u.get("field") for u in extraction_updates if isinstance(u, dict)]
        if len(set(fields_touched)) >= 2:
            signals.append(f"multi_field_extraction={fields_touched}")
            return (
                InsightType.NEW_PATTERN,
                InsightConfidence.MEDIUM,
                "User connected facts across multiple problem domains.",
                signals,
            )

    # 5. Strong Evidence (numbers, metrics, research, interview mentions)
    evidence_hits = [sig for sig in _EVIDENCE_SIGNALS if sig in msg_lower]
    # Also check for standalone numbers or percentages
    has_numbers = bool(re.search(r'\b\d+(\.\d+)?(?:%|(?:users|people|hours|days|dollars|dollar)\b)', msg_lower))
    if evidence_hits or has_numbers:
        if evidence_hits:
            signals.extend(f"evidence_keyword='{h}'" for h in evidence_hits)
        if has_numbers:
            signals.append("numeric_evidence_detected")
        conf = InsightConfidence.HIGH if has_numbers or len(evidence_hits) >= 2 else InsightConfidence.MEDIUM
        return (
            InsightType.STRONG_EVIDENCE,
            conf,
            "User provided quantifiable evidence, data, or observed behavior.",
            signals,
        )

    # 6. Constraint (budget, time, technology, policy)
    constraint_hits = [sig for sig in _CONSTRAINT_SIGNALS if sig in msg_lower]
    if constraint_hits:
        signals.extend(f"constraint_keyword='{h}'" for h in constraint_hits)
        return (
            InsightType.CONSTRAINT,
            InsightConfidence.HIGH,
            f"User stated a operational or technical boundary mentioning '{constraint_hits[0]}'.",
            signals,
        )

    # 7. Root Cause Hint (explaining why)
    root_cause_hits = [sig for sig in _ROOT_CAUSE_SIGNALS if sig in msg_lower]
    if root_cause_hits or "why" in msg_lower:
        if root_cause_hits:
            signals.extend(f"root_cause_keyword='{h}'" for h in root_cause_hits)
        else:
            signals.append("causal_query_or_explanation")
        return (
            InsightType.ROOT_CAUSE_HINT,
            InsightConfidence.MEDIUM,
            "User explained causal relationships or underlying mechanisms.",
            signals,
        )

    return (
        InsightType.NONE,
        InsightConfidence.LOW,
        "No significant insight patterns detected.",
        signals,
    )


# ---------------------------------------------------------------------------
# Prompt formatting & diagnostics helpers
# ---------------------------------------------------------------------------


def insight_instruction_bullet(
    insight_type: InsightType,
    confidence: InsightConfidence = InsightConfidence.MEDIUM,
) -> Optional[str]:
    """Render an optional instruction bullet when a meaningful insight is found."""
    if insight_type is None or insight_type == InsightType.NONE:
        return None
    guidance = _INSIGHT_GUIDANCE.get(insight_type)
    if not guidance:
        return None
    return (
        f"Insight detected: {insight_type.value} [{confidence.value} confidence]\n"
        f"Guidance: {guidance}"
    )


def insight_diagnostics_section(capture: dict) -> dict:
    """Developer Console projection of insight detection."""
    itype = capture.get("insight_type")
    if itype is None or itype == InsightType.NONE or itype == "NONE":
        return {}
    return {
        "Insight Type": itype.value if hasattr(itype, "value") else str(itype),
        "Confidence": capture.get("insight_confidence", "MEDIUM"),
        "Explanation": capture.get("insight_explanation", ""),
        "Signals": capture.get("insight_signals", []),
    }
