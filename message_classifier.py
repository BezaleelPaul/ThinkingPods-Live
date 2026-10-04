"""Observation-only Message-Type Shadow Classifier for ReqGPT / ThinkingPods.

Pretrained HuggingFace zero-shot classification that independently votes on
the SAME four-way question Module 1's deterministic rules answer:

    MEANINGFUL / NO_UPDATE / AMBIGUOUS / END

The transformer's vote is recorded next to the rule extractor's decision so
the disagreement rate can be measured (``audits/run_classification_metrics``)
— it is a SHADOW VOTE and feeds nothing.

Observation-only contract (mirrors :mod:`conversation_style_audit` /
:mod:`conversation_failure_audit`):
    * module-level imports are pure standard library only — ``transformers``
      is imported lazily inside ``_get_pipeline``;
    * reads the raw user message only — never writes ProjectState, never
      calls any planner/extractor/validator, never changes prompts, never
      overrides the rule extractor's ``MessageType``;
    * computed AFTER the reply is finalised and only mounted into the
      Developer Console, so it can never influence the reply/objective/
      extraction/lifecycle/memory/recovery decisions;
    * ``classify_message`` NEVER raises: any missing dependency, missing
      model, or inference failure degrades to ``status="unavailable"``.

Environment:
    CLASSIFIER_ENABLED   ``true`` to run the model (default ``false`` —
                         diagnostics stay byte-identical when disabled).
    CLASSIFIER_MODEL     HF model id (default
                         ``valhalla/distilbart-mnli-12-1``).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


__all__ = [
    "MessageTypeLabel",
    "ClassificationResult",
    "enabled",
    "classify_message",
    "message_classification_diagnostics_section",
]

DEFAULT_MODEL = "valhalla/distilbart-mnli-12-1"

STATUS_OK = "ok"
STATUS_DISABLED = "disabled"
STATUS_UNAVAILABLE = "unavailable"

# The four labels Module 1's MessageType enum uses — kept as plain strings so
# this leaf never imports mentor/extraction modules.
CANDIDATE_LABELS = ("MEANINGFUL", "NO_UPDATE", "AMBIGUOUS", "END")

# Intent-style hypotheses (full statements, passed as candidate labels with
# hypothesis_template="{}"). Calibrated by replaying all 32 golden-fixture
# messages against the rule extractor's MessageType (offline benchmark):
# negation-framed phrases ("does not contain ...") collapsed everything into
# NO_UPDATE (agreement 14/32), premise-unverifiable phrasing ("...related to
# their project" — the premise carries no project context) scored 8/32, and
# these statement forms reached 18/32 on valhalla/distilbart-mnli-12-1.
_HYPOTHESIS_STATEMENTS = {
    "MEANINGFUL": "The user provides new information or an opinion.",
    "NO_UPDATE": "The user gives a brief acknowledgment.",
    "AMBIGUOUS": "The user is unsure or the meaning is unclear.",
    "END": "The user says goodbye to end the conversation.",
}
_STATEMENT_TO_LABEL = {v: k for k, v in _HYPOTHESIS_STATEMENTS.items()}


class MessageTypeLabel(str, Enum):
    """Shadow-vote label space (mirrors Module 1 ``MessageType`` values)."""

    MEANINGFUL = "MEANINGFUL"
    NO_UPDATE = "NO_UPDATE"
    AMBIGUOUS = "AMBIGUOUS"
    END = "END"


@dataclass(frozen=True)
class ClassificationResult:
    """JSON-safe per-turn shadow-vote record."""

    label: str = ""
    score: float = 0.0
    model: str = ""
    status: str = STATUS_DISABLED
    latency_ms: float = 0.0
    rule_label: str = ""
    agrees_with_rules: Optional[bool] = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Shadow Label": self.label,
            "Confidence": round(self.score, 4),
            "Rules Label": self.rule_label,
            "Agrees With Rules": (
                ""
                if self.agrees_with_rules is None
                else ("Yes" if self.agrees_with_rules else "No")
            ),
            "Model": self.model,
            "Status": self.status,
            "Latency": f"{self.latency_ms:.1f} ms",
            "Error": self.error,
        }


def enabled() -> bool:
    """Whether the shadow classifier runs this turn (``CLASSIFIER_ENABLED=true``)."""
    return os.getenv("CLASSIFIER_ENABLED", "false").lower() == "true"


def model_name() -> str:
    """Configured HF model id (``CLASSIFIER_MODEL``)."""
    return os.getenv("CLASSIFIER_MODEL", DEFAULT_MODEL)


# Lazy, cached pipeline (same sentinel pattern as sentiment._get_pipeline).
_pipeline: Any = None


def _get_pipeline() -> Any:
    """Load (once) and return the HF zero-shot pipeline, or ``None``."""
    global _pipeline
    if _pipeline == "_failed":
        return None
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline  # lazy: heavy import

        _pipeline = pipeline("zero-shot-classification", model=model_name())
    except Exception:
        _pipeline = "_failed"
        return None
    return _pipeline


def _reset_pipeline_for_tests() -> None:
    """Drop the cached pipeline (tests only)."""
    global _pipeline
    _pipeline = None


def classify_message(
    text: Optional[str],
    rule_label: str = "",
) -> ClassificationResult:
    """Zero-shot vote on the Module 1 message type. NEVER raises.

    ``rule_label`` is the deterministic extractor's decision for THIS turn;
    it is only recorded for comparison — it never biases the vote.
    """
    if not enabled():
        return ClassificationResult(status=STATUS_DISABLED, rule_label=rule_label)
    model = model_name()
    if not text or not text.strip():
        return ClassificationResult(
            model=model,
            status=STATUS_UNAVAILABLE,
            rule_label=rule_label,
            error="empty input",
        )
    try:
        pipe = _get_pipeline()
        if pipe is None:
            return ClassificationResult(
                model=model,
                status=STATUS_UNAVAILABLE,
                rule_label=rule_label,
                error="transformers/model unavailable",
            )
        statements = [_HYPOTHESIS_STATEMENTS[l] for l in CANDIDATE_LABELS]
        import torch

        t0 = time.perf_counter()
        with torch.inference_mode():
            raw = pipe(
                text.strip()[:2000],
                candidate_labels=statements,
                hypothesis_template="{}",
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        # Map the best-scoring statement back to the MessageType-style label.
        label = _STATEMENT_TO_LABEL.get(raw["labels"][0], "AMBIGUOUS")
        score = float(raw["scores"][0]) if raw.get("scores") else 0.0
        agrees: Optional[bool] = None
        if rule_label:
            agrees = label == rule_label
        return ClassificationResult(
            label=label,
            score=score,
            model=model,
            status=STATUS_OK,
            latency_ms=latency_ms,
            rule_label=rule_label,
            agrees_with_rules=agrees,
        )
    except Exception as exc:
        return ClassificationResult(
            model=model,
            status=STATUS_UNAVAILABLE,
            rule_label=rule_label,
            error=f"{type(exc).__name__}: {exc}"[:200],
        )


def message_classification_diagnostics_section(
    record: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Developer Console projection of the per-turn shadow-vote record.

    Read-only passthrough; ``{}`` for absent/empty records.
    """
    if not record:
        return {}
    return dict(record)
