"""
conversation_metrics.py — developer-only, observation-only conversation metrics.

Aggregates per-turn pipeline observations into conversation-level quality
metrics surfaced in the Developer Console (app.py). This module is STRICTLY
observational:

  * It never influences mentor decisions (objective selection, reply
    generation, lifecycle, question-family planning, extraction).
  * The pure functions (``build_turn_metric`` / ``aggregate_metrics``) are
    deterministic and touch no I/O.
  * ``record_turn`` appends one immutable record to
    ``SessionData.turn_metrics`` — an append-only observation log that no
    decision path ever reads.

Metrics are computed from data the pipeline already produces (objective,
capture, lifecycle decision, reply) plus rough token estimates (chars/4
heuristic, "if available" — never a real tokeniser).
"""

from __future__ import annotations

from memory_extractor import REQUIRED_FIELDS, StateField
from module3 import Objective

__all__ = [
    "build_turn_metric",
    "aggregate_metrics",
    "build_metrics_section",
    "record_turn",
]


def _estimate_tokens(text) -> int | None:
    """Rough token estimate (chars/4 heuristic). ``None`` when unavailable."""
    if not text:
        return None
    return (len(text) + 3) // 4


def _objective_to_field(value):
    """Normalise an Objective value (e.g. ``"PERSONAS"``) to its StateField
    value (``"personas"``) so objective and completed-field names share one
    namespace. ``None`` for WRAP_UP / unknown values."""
    if not value or value == Objective.WRAP_UP.value:
        return None
    try:
        return StateField[value].value
    except KeyError:
        return value


def build_turn_metric(
    turn_index, capture, objective, response_strategy, lifecycle_decision,
    session_data, reply,
):
    """Project one turn's existing pipeline data into a JSON-safe record.

    Read-only: consumes objects produced by the pipeline, mutates nothing.
    """
    family_plan = capture.get("family_plan") or {}
    return {
        "turn_index": turn_index,
        "stage": session_data.current_stage,
        "objective": objective.objective.value,
        "advancement": objective.advancement,
        "strategy": response_strategy.value,
        "lifecycle": lifecycle_decision.value,
        "question_family": capture.get("question_family"),
        "family_reask": bool(family_plan.get("reask")),
        "family_skip_count": len(family_plan.get("skip_reasons") or []),
        "guard_blocked": bool(capture.get("guard_blocked")),
        "extraction_updates": len(capture.get("extraction_updates") or []),
        "rule_based_extraction": bool(capture.get("rule_based_extraction")),
        "completed_fields": [sf.value for sf in objective.completed_fields],
        "missing_fields": [sf.value for sf in objective.missing_fields],
        "prompt_tokens": _estimate_tokens(capture.get("prompt")),
        "response_tokens": _estimate_tokens(reply),
    }


def aggregate_metrics(records):
    """Conversation-level aggregate over turn records. Pure + deterministic."""
    turns = len(records)
    completed = {f for r in records for f in (r.get("completed_fields") or [])}
    objectives_completed = len(completed)
    targeted = {
        _objective_to_field(r.get("objective"))
        for r in records
        if _objective_to_field(r.get("objective"))
    }
    num_required = len(REQUIRED_FIELDS)
    total_prompt = sum(r.get("prompt_tokens") or 0 for r in records)
    total_response = sum(r.get("response_tokens") or 0 for r in records)
    return {
        "Turns": turns,
        "Objectives Completed": objectives_completed,
        "Avg Turns per Objective": (
            round(turns / objectives_completed, 2) if objectives_completed else 0
        ),
        "Repeated Question Attempts": sum(
            1 for r in records if r.get("family_reask")
        ),
        "Semantic Duplicates Prevented": sum(
            1 for r in records if r.get("guard_blocked")
        ),
        "Avg Extracted Facts per Turn": (
            round(
                sum(r.get("extraction_updates", 0) for r in records) / turns, 2
            )
            if turns else 0
        ),
        "Checklist Completion Rate": (
            round(objectives_completed / num_required, 2) if num_required else 0.0
        ),
        "Total Prompt Tokens (est.)": total_prompt,
        "Avg Prompt Tokens per Turn (est.)": (
            round(total_prompt / turns, 1) if turns else 0
        ),
        "Total Response Tokens (est.)": total_response,
        "Avg Response Tokens per Turn (est.)": (
            round(total_response / turns, 1) if turns else 0
        ),
        "Objective Switches": sum(
            1
            for a, b in zip(records, records[1:])
            if a.get("objective") != b.get("objective")
        ),
        "Unanswered Objectives": sorted(targeted - completed),
        "Skipped Objectives": sorted(
            {sf.value for sf in REQUIRED_FIELDS} - targeted
        ),
    }


def build_metrics_section(session_data):
    """Diagnostics section view: aggregate over the session's turn log."""
    return aggregate_metrics(getattr(session_data, "turn_metrics", []))


def record_turn(
    session_data, capture, objective, response_strategy, lifecycle_decision,
    reply,
):
    """Append the current turn's metric record to the session log (if any).

    Observation-only: failures are logged and swallowed so a metrics problem
    can never break the conversation.
    """
    try:
        if not hasattr(session_data, "turn_metrics"):
            session_data.turn_metrics = []
        turn_index = len(session_data.turn_metrics) + 1
        session_data.turn_metrics.append(
            build_turn_metric(
                turn_index,
                capture,
                objective,
                response_strategy,
                lifecycle_decision,
                session_data,
                reply,
            )
        )
        from session_manager import get_session_manager
        mgr = get_session_manager()
        session_id = mgr.get_active_session_id()
        if session_id is not None:
            mgr.save_session_data(session_id, session_data)
    except Exception as exc:
        print(f"[ConversationMetrics] failed to record turn: {exc}")
