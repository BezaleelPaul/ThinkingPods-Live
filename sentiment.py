"""Observation-only Sentiment Analysis for ReqGPT / ThinkingPods.

Pretrained HuggingFace text-classification (DistilBERT SST-2 by default) that
reads the emotional tone of the latest user message and surfaces it in the
Developer Console. No fine-tuning, no training data, no network at inference
time (model is cached locally after the first download).

Observation-only contract (mirrors :mod:`conversation_style_audit` /
:mod:`conversation_failure_audit`):
    * module-level imports are pure standard library only — ``transformers``
      is imported lazily inside ``_get_pipeline`` so importing this module
      costs nothing and never fails on environments without the package;
    * reads the raw user message only — never writes ProjectState, never
      calls any planner/extractor/validator, never changes prompts;
    * computed AFTER the reply is finalised and only mounted into the
      Developer Console, so it can never influence the reply/objective/
      extraction/lifecycle/memory/recovery decisions;
    * ``analyze_sentiment`` NEVER raises: any missing dependency, missing
      model, or inference failure degrades to ``status="unavailable"``.

Environment:
    SENTIMENT_ENABLED   ``true`` to run the model (default ``false`` —
                        diagnostics stay byte-identical to a non-sentiment
                        run when disabled).
    SENTIMENT_MODEL     HF model id (default
                        ``distilbert-base-uncased-finetuned-sst-2-english``).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


__all__ = [
    "SentimentLabel",
    "SentimentResult",
    "enabled",
    "analyze_sentiment",
    "sentiment_diagnostics_section",
]

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

STATUS_OK = "ok"
STATUS_DISABLED = "disabled"
STATUS_UNAVAILABLE = "unavailable"


class SentimentLabel(str, Enum):
    """Pretrained classifier output (SST-2 label space)."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class SentimentResult:
    """JSON-safe per-turn sentiment record. Never contains non-JSON values."""

    label: str = ""
    score: float = 0.0
    model: str = ""
    status: str = STATUS_DISABLED
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Label": self.label,
            "Score": round(self.score, 4),
            "Model": self.model,
            "Status": self.status,
            "Latency": f"{self.latency_ms:.1f} ms",
            "Error": self.error,
        }


def enabled() -> bool:
    """Whether the classifier runs this turn (``SENTIMENT_ENABLED=true``)."""
    return os.getenv("SENTIMENT_ENABLED", "false").lower() == "true"


def model_name() -> str:
    """Configured HF model id (``SENTIMENT_MODEL``)."""
    return os.getenv("SENTIMENT_MODEL", DEFAULT_MODEL)


# Lazy, cached pipeline. ``None`` = not loaded yet; a failed load is cached
# as the sentinel string "_failed" so we never retry a broken import/model
# on every turn.
_pipeline: Any = None


def _get_pipeline() -> Any:
    """Load (once) and return the HF sentiment pipeline, or ``None``."""
    global _pipeline
    if _pipeline == "_failed":
        return None
    if _pipeline is not None:
        return _pipeline
    try:
        from transformers import pipeline  # lazy: heavy import

        _pipeline = pipeline(
            "sentiment-analysis",
            model=model_name(),
            top_k=None,
        )
    except Exception:
        _pipeline = "_failed"
        return None
    return _pipeline


def _reset_pipeline_for_tests() -> None:
    """Drop the cached pipeline (tests only)."""
    global _pipeline
    _pipeline = None


def analyze_sentiment(text: Optional[str]) -> SentimentResult:
    """Classify the emotional tone of ``text``. NEVER raises.

    Disabled / empty input / missing dependency / inference failure all
    return a well-formed ``SentimentResult`` with the matching ``status``.
    """
    if not enabled():
        return SentimentResult(status=STATUS_DISABLED)
    model = model_name()
    if not text or not text.strip():
        return SentimentResult(
            model=model, status=STATUS_UNAVAILABLE, error="empty input"
        )
    try:
        pipe = _get_pipeline()
        if pipe is None:
            return SentimentResult(
                model=model,
                status=STATUS_UNAVAILABLE,
                error="transformers/model unavailable",
            )
        import torch

        t0 = time.perf_counter()
        with torch.inference_mode():
            raw = pipe(text.strip()[:2000])
        latency_ms = (time.perf_counter() - t0) * 1000
        # With top_k=None the pipeline returns [[{label, score}, ...]].
        best = raw[0][0] if raw and isinstance(raw[0], list) else raw[0]
        label = str(best.get("label", "")).upper()
        if label not in (SentimentLabel.POSITIVE.value, SentimentLabel.NEGATIVE.value):
            label = SentimentLabel.NEUTRAL.value
        return SentimentResult(
            label=label,
            score=float(best.get("score", 0.0)),
            model=model,
            status=STATUS_OK,
            latency_ms=latency_ms,
        )
    except Exception as exc:
        return SentimentResult(
            model=model,
            status=STATUS_UNAVAILABLE,
            error=f"{type(exc).__name__}: {exc}"[:200],
        )


def sentiment_diagnostics_section(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Developer Console projection of the per-turn sentiment record.

    Read-only passthrough; ``{}`` for absent/empty records (matches the
    ``ExtractionAccuracy``/``InsightDetection`` empty-section precedent).
    """
    if not record:
        return {}
    return dict(record)
