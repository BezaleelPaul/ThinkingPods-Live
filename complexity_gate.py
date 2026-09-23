"""
complexity_gate — deterministic Semantic Complexity Gate for hybrid extraction.

Sits on the "rules satisfy the current objective" path of the objective-aware
hybrid extraction decision (``hybrid_extraction.decide_hybrid_extraction``).
Before skipping the LLM extractor, the gate classifies the user message's
semantic complexity:

  LOW    — single, explicit, self-contained statement (e.g. a bare frequency)
  MEDIUM — one additional semantic dimension (e.g. frequency + explanation)
  HIGH   — multiple independent semantic concepts in one message

Decision (advisory, affects ONLY whether the LLM runs on the rules-satisfied
path):

  LOW    -> skip the LLM extractor (rules suffice)
  MEDIUM -> run the LLM extractor
  HIGH   -> run the LLM extractor

The detector is strictly deterministic:

  * no embeddings, no LLM, no ML — lightweight surface heuristics only
  * reads only the raw user message plus the rule extractor's candidate
    categories (``rule_observation``), mirroring ``extraction_comparison``
  * never mutates ``ProjectState``, objectives, lifecycle, prompts, or
    architecture — it only advises the skip/run decision

Indicators (each matched cue counts once; candidate-category and clause
signals are additive):

  * causal language        (because / so that / due to / leads to / results in)
  * contrast               (but / however / although / whereas / while)
  * explanation            (which means / in other words / the reason ...)
  * evidence language      (i saw / i observed / research / survey ...)
  * motivations            (want to / need to / worried / care about ...)
  * consequences           (affects / impact / leads to ...)
  * multiple clauses       (independent-idea separators)
  * multiple candidate categories (>= 2 rule-extraction categories)
"""

from __future__ import annotations

import os
import re

from extraction_comparison import _RULE_LEGACY_TO_STATE

__all__ = [
    "COMPLEXITY_HIGH",
    "COMPLEXITY_LOW",
    "COMPLEXITY_MEDIUM",
    "DECISION_RUN",
    "DECISION_SKIP",
    "classify_complexity",
    "decision_for",
    "enabled",
]

COMPLEXITY_LOW = "LOW"
COMPLEXITY_MEDIUM = "MEDIUM"
COMPLEXITY_HIGH = "HIGH"

DECISION_SKIP = "SKIP_LLM"
DECISION_RUN = "RUN_LLM"

# One flat list of cue phrases; a phrase matching more than one semantic
# category still counts as a single distinct signal (no double counting).
_CAUSAL_CUES = [
    "because", "so that", "due to", "leads to", "lead to",
    "result in", "results in", "caused", "causes", "cause",
    "therefore", "as a result", "which means", "since", "thus", "owing to",
]
_CONTRAST_CUES = [
    "but", "however", "although", "though", "yet", "whereas", "while",
    "despite", "even though", "on the other hand", "instead", "except",
]
_EXPLANATION_CUES = [
    "which means", "in other words", "basically", "the reason",
    "explain", "explains", "that's why", "that is why", "to elaborate",
]
_EVIDENCE_CUES = [
    "i saw", "i noticed", "i observed", "i've seen", "i have seen",
    "evidence", "research", "data", "survey", "interview", "interviewed",
    "observed", "statistics", "study", "studies", "reported", "witnessed",
]
_MOTIVATION_CUES = [
    "want to", "need to", "worried", "worry", "concerned", "care about",
    "motivated", "hoping", "hope to", "aim to", "goal", "trying to",
    "care", "concern",
]
_CONSEQUENCE_CUES = [
    "leads to", "results in", "causes", "affects", "affected",
    "impact", "consequence",
]

# Independent-idea separators (clause boundaries). Causal/contrast cues are
# deliberately excluded so a single relational word is not double-counted.
_CLAUSE_SEPARATORS = [",", ";", " and ", " or ", " - "]

_ALL_CUES: tuple[str, ...] = tuple(
    sorted(
        set(_CAUSAL_CUES + _CONTRAST_CUES + _EXPLANATION_CUES
            + _EVIDENCE_CUES + _MOTIVATION_CUES + _CONSEQUENCE_CUES)
    )
)

_CUE_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in _ALL_CUES) + r")\b")


def enabled() -> bool:
    """Whether the gate affects the skip/run decision (``COMPLEXITY_GATE``).

    Defaults to enabled. When disabled the rules-satisfied path behaves
    exactly as before (always skip the LLM) — the detector only ever reports.
    """
    return os.getenv("COMPLEXITY_GATE", "true").lower() == "true"


def decision_for(complexity: str) -> str:
    """Map a complexity level to the advisory skip/run decision."""
    if complexity == COMPLEXITY_LOW:
        return DECISION_SKIP
    return DECISION_RUN


def _candidate_categories(rule_observation: dict | None) -> int:
    """Count rule-extraction categories that would populate ProjectState."""
    if not rule_observation:
        return 0
    return sum(1 for legacy in _RULE_LEGACY_TO_STATE if rule_observation.get(legacy))


def _count_clauses(text: str) -> int:
    """Count independent-idea clauses using separator heuristics."""
    lower = text.lower()
    count = 1
    for sep in _CLAUSE_SEPARATORS:
        count += lower.count(sep)
    return max(1, count)


def classify_complexity(
    user_message: str,
    rule_observation: dict | None = None,
) -> dict:
    """Classify a user message's semantic complexity.

    Returns a JSON-serialisable record::

        {
            "complexity": "LOW" | "MEDIUM" | "HIGH",
            "reasons": [...],      # human-readable matched signals
            "decision": "SKIP_LLM" | "RUN_LLM",
            "signals": {           # diagnostic detail
                "cues": [...],
                "clauses": int,
                "candidate_categories": int,
                "score": int,
            },
        }

    Pure function — no state, no I/O, no randomness.
    """
    text = (user_message or "").strip()
    lower = text.lower()

    matched = sorted(set(_CUE_RE.findall(lower)))
    clauses = _count_clauses(text)
    categories = _candidate_categories(rule_observation)

    score = len(matched)
    if categories >= 3:
        score += 2
    elif categories >= 2:
        score += 1
    if clauses >= 2:
        score += 1

    if score <= 0:
        complexity = COMPLEXITY_LOW
    elif score <= 2:
        complexity = COMPLEXITY_MEDIUM
    else:
        complexity = COMPLEXITY_HIGH

    reasons: list[str] = []
    if matched:
        reasons.append(f"semantic cues: {', '.join(matched)}")
    if categories >= 2:
        reasons.append(f"{categories} candidate extraction categories")
    if clauses >= 2:
        reasons.append(f"{clauses} clauses")

    if not reasons:
        reasons.append("single, explicit, self-contained statement")

    return {
        "complexity": complexity,
        "reasons": reasons,
        "decision": decision_for(complexity),
        "signals": {
            "cues": matched,
            "clauses": clauses,
            "candidate_categories": categories,
            "score": score,
        },
    }
