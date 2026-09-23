"""
TurnTiming — request-scoped timing telemetry for mentor pipeline.

Request-scoped only: never stored in SessionData, ProjectState,
conversation_history, or any persisted JSON.

Also hosts the extraction-stage accumulator used by the memory extractor /
rule-based extractor / state manager to record per-sub-stage elapsed time
(preprocessing, rules, llm, merge, validation, persistence). Observation-only:
no value is read back into any decision path.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


def add_stage_ms(accumulator: dict[str, float] | None, key: str, start: float, end: float) -> None:
    """Accumulate ``end - start`` (seconds) into ``accumulator[key]`` (ms).

    ``None`` accumulator (the default when no instrumentation is requested)
    is a no-op so existing callers/tests keep identical behavior."""
    if accumulator is None:
        return
    accumulator[key] = accumulator.get(key, 0.0) + (end - start) * 1000.0


# Ground-truth token count of the extraction prompt before the prompt-slimming
# simplification (measured via Ollama's prompt_eval_count on a representative
# mid-conversation state: template + 6 populated fields + previous context +
# user message). Displayed by the Developer Console so the size reduction is
# visible per turn; purely observational — never read by any decision path.
EXTRACTION_PROMPT_BASELINE_TOKENS: int = 2148


def summarize_extraction_timing(
    accumulator: dict[str, float], total_ms: float
) -> Dict[str, int]:
    """Project the raw per-step accumulator keys onto the six extraction
    sub-stages reported by the Developer Console.

    Every stage is always present (0 ms when unused), so the console can
    render a stable breakdown regardless of which path ran this turn. The
    LLM / validation / merge detail keys are the individual measurements
    behind the stage totals (kept separate so the console tree stays stable
    while the audit can still read prompt-vs-api-vs-parse split)."""

    def _sum(*keys: str) -> float:
        return sum(accumulator.get(k, 0.0) for k in keys)

    return {
        "preprocessing_ms": round(_sum("preprocess")),
        "rules_ms": round(_sum("rules")),
        "llm_ms": round(_sum("llm_prompt", "llm_api", "llm_parse")),
        "merge_ms": round(_sum("merge_state", "merge_session")),
        "validation_ms": round(_sum("validation_llm", "validation_batch")),
        "persistence_ms": round(_sum("persist")),
        # detail keys — the per-step measurements behind the stage totals
        "llm_prompt_ms": round(_sum("llm_prompt")),
        "llm_api_ms": round(_sum("llm_api")),
        "llm_parse_ms": round(_sum("llm_parse")),
        "validation_llm_ms": round(_sum("validation_llm")),
        "validation_batch_ms": round(_sum("validation_batch")),
        "merge_state_ms": round(_sum("merge_state")),
        "merge_session_ms": round(_sum("merge_session")),
        "preprocess_detail_ms": round(_sum("preprocess")),
        "rules_detail_ms": round(_sum("rules")),
        "persist_detail_ms": round(_sum("persist")),
        # prompt size telemetry — live tokens sent this turn (from Ollama's
        # prompt_eval_count) plus the reduction vs the pre-slimming baseline
        "prompt_tokens": int(accumulator.get("prompt_tokens", 0) or 0),
        "prompt_baseline_tokens": EXTRACTION_PROMPT_BASELINE_TOKENS,
        "prompt_reduction_pct": round(
            max(
                0.0,
                (
                    1.0
                    - (accumulator.get("prompt_tokens", 0) or 0)
                    / EXTRACTION_PROMPT_BASELINE_TOKENS
                )
                * 100.0,
            )
        ),
        "total_ms": round(total_ms),
    }


@dataclass
class TurnTiming:
    prepare_ms: float = 0.0
    extraction_ms: float = 0.0
    objective_ms: float = 0.0
    lifecycle_ms: float = 0.0
    prompt_ms: float = 0.0
    llm_ms: float = 0.0
    finalize_ms: float = 0.0
    extraction_breakdown: Dict[str, int] | None = None
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "prepare_ms": round(self.prepare_ms),
            "extraction_ms": round(self.extraction_ms),
            "objective_ms": round(self.objective_ms),
            "lifecycle_ms": round(self.lifecycle_ms),
            "prompt_ms": round(self.prompt_ms),
            "llm_ms": round(self.llm_ms),
            "finalize_ms": round(self.finalize_ms),
            "total_ms": round(self.total_ms),
        }
        if self.extraction_breakdown:
            d["extraction_breakdown"] = self.extraction_breakdown
        return d


def format_duration(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f} s"
    return f"{round(ms)} ms"


# ---------------------------------------------------------------------------
# TurnProfiler — lightweight per-turn latency profiler (Phase 1.5 audit).
#
# Reusable, stdlib-only, observation-only: two perf_counter reads per span
# (~100 ns) plus small list appends. Never persisted, never read by any
# decision path. Attach to a turn via an explicit ``_prof=None`` parameter;
# every instrumented call site treats ``None`` as "profiling off".
# ---------------------------------------------------------------------------


def _response_field(response: Any, name: str) -> Any:
    """Read an Ollama response metadata field from dict- or object-style
    responses (the real client returns a mapping-like ChatResponse; test
    stubs return plain dicts or simple objects)."""
    try:
        if hasattr(response, "get"):
            value = response.get(name)
            if value is not None:
                return value
    except Exception:
        pass
    return getattr(response, name, None)


class TurnProfiler:
    """Ordered per-stage spans + per-LLM-call records for one mentor turn."""

    def __init__(self) -> None:
        self._spans: List[Dict[str, Any]] = []
        self.llm_calls: List[Dict[str, Any]] = []

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        """Time one named stage; always appends exactly one span record."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self._spans.append(
                {"stage": name, "ms": round((time.perf_counter() - start) * 1000.0, 1)}
            )

    def record_llm_call(
        self,
        *,
        purpose: str,
        model: str,
        prompt_chars: int,
        duration_ms: float,
        prompt_tokens: Optional[int] = None,
        eval_count: Optional[int] = None,
        server_ms: Optional[Dict[str, Any]] = None,
        gated: bool = False,
        failed: bool = False,
    ) -> Dict[str, Any]:
        """Append one LLM-call record; returns the record."""
        record = {
            "purpose": purpose,
            "model": model,
            "prompt_chars": int(prompt_chars),
            "duration_ms": round(duration_ms, 1),
            "prompt_tokens": prompt_tokens,
            "eval_count": eval_count,
            "server_ms": server_ms or {},
            "gated": bool(gated),
            "failed": bool(failed),
        }
        self.llm_calls.append(record)
        return record

    def span_ms(self, name: str) -> float:
        """Total ms recorded under ``name`` (0.0 when absent)."""
        return round(sum(s["ms"] for s in self._spans if s["stage"] == name), 1)

    def to_dict(self) -> Dict[str, Any]:
        stages: Dict[str, float] = {}
        for span in self._spans:
            stages[span["stage"]] = round(stages.get(span["stage"], 0.0) + span["ms"], 1)
        return {
            "stages": stages,
            "stage_order": [s["stage"] for s in self._spans],
            "spans": [dict(s) for s in self._spans],
            "llm_calls": [dict(c) for c in self.llm_calls],
        }


def timed_ollama_chat(
    profiler: Optional[TurnProfiler],
    purpose: str,
    model: str,
    messages: list,
    options: Optional[dict] = None,
    gated: bool = False,
) -> Any:
    """Drop-in timed wrapper around ``ollama.chat`` (non-streaming).

    Identical call semantics to ``ollama.chat(model=..., messages=...,
    options=...)`` — same return value, same exceptions. When ``profiler``
    is given, records duration, prompt size, and any server-side metadata
    the response carries (``prompt_eval_count``, ``eval_count``,
    ``total_duration`` / ``load_duration`` / ``prompt_eval_duration`` /
    ``eval_duration`` in Ollama's nanosecond units, converted to ms).
    TTFT is NOT measurable on the non-streaming path by construction.
    """
    import ollama  # lazy: mirrors every existing call site's lazy import

    prompt_chars = sum(len((m or {}).get("content") or "") for m in (messages or []))
    start = time.perf_counter()
    try:
        if options is None:
            response = ollama.chat(model=model, messages=messages)
        else:
            response = ollama.chat(model=model, messages=messages, options=options)
    except Exception:
        if profiler is not None:
            profiler.record_llm_call(
                purpose=purpose,
                model=model,
                prompt_chars=prompt_chars,
                duration_ms=(time.perf_counter() - start) * 1000.0,
                gated=gated,
                failed=True,
            )
        raise
    duration_ms = (time.perf_counter() - start) * 1000.0
    if profiler is not None:
        server_ms: Dict[str, Any] = {}
        for key in (
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
        ):
            value = _response_field(response, key)
            if isinstance(value, (int, float)):
                server_ms[key] = round(value / 1e6, 1)
        profiler.record_llm_call(
            purpose=purpose,
            model=model,
            prompt_chars=prompt_chars,
            duration_ms=duration_ms,
            prompt_tokens=_response_field(response, "prompt_eval_count"),
            eval_count=_response_field(response, "eval_count"),
            server_ms=server_ms,
            gated=gated,
        )
    return response
